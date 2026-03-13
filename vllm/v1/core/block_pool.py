# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import hashlib
import mmap
import os
import re
import struct
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.distributed.kv_events import (MEDIUM_GPU, AllBlocksCleared,
                                        BlockRemoved, BlockStored,
                                        KVCacheEvent)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHash, BlockHashWithGroupId,
                                         ExternalBlockHash,
                                         FreeKVCacheBlockQueue, KVCacheBlock,
                                         get_block_hash,
                                         make_block_hash_with_group_id,
                                         maybe_convert_block_hash)
from vllm.v1.request import Request

logger = init_logger(__name__)


class _SharedBlockAllocator:
    """Process-shared block allocator (experimental).

    This allocator provides a global view of free/used KV blocks across
    multiple server processes that map the same shared KV pool.
    """

    def __init__(self, path: str, key: str, num_gpu_blocks: int, reset: bool):
        self._num_gpu_blocks = int(num_gpu_blocks)
        self._refcnt_entry_size = 4
        self._meta_entry_size = 8
        self._meta_size = self._meta_entry_size * 3
        base = f"{path}.{key}"
        digest = hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()
        self._shm_dir = os.getenv("VLLM_SHARED_BLOCK_ALLOCATOR_SHM_DIR",
                                  "/dev/shm").strip() or "/dev/shm"
        self._base = os.path.join(self._shm_dir, f"vllm_sba_{digest}")
        self._meta_path = f"{self._base}.meta"
        self._lock_path = f"{self._base}.lock"
        self._bitmap_path = f"{self._base}.bitmap"
        self._refcnt_path = f"{self._base}.refcnt"
        self._req_dir = f"{self._base}.reqs"
        self._refcnt_entry_size = 4
        if reset:
            self._initialize_state()
        else:
            # Best effort lazy init for first process that comes up.
            if not os.path.exists(self._meta_path):
                self._initialize_state()

    def _initialize_state(self) -> None:
        os.makedirs(self._shm_dir, exist_ok=True)
        if os.path.isdir(self._req_dir):
            for name in os.listdir(self._req_dir):
                path = os.path.join(self._req_dir, name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        os.makedirs(self._req_dir, exist_ok=True)
        with open(self._lock_path, "a+b"):
            pass
        with open(self._meta_path, "wb") as f:
            f.truncate(self._meta_size)
        with open(self._meta_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            try:
                self._set_meta_num_gpu_blocks(mm, self._num_gpu_blocks)
                self._set_meta_free_count(mm, max(0, self._num_gpu_blocks - 1))
                self._set_meta_next_scan_start(mm, 1)
                mm.flush()
            finally:
                mm.close()
        bitmap_size = (self._num_gpu_blocks + 7) // 8
        with open(self._bitmap_path, "wb") as f:
            f.truncate(bitmap_size)
        with open(self._bitmap_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            try:
                if bitmap_size > 0:
                    mm[:] = b"\xff" * bitmap_size
                self._set_free_bit(mm, 0, False)
                # clear tail bits beyond num_gpu_blocks
                for block_id in range(self._num_gpu_blocks, bitmap_size * 8):
                    self._set_free_bit(mm, block_id, False)
                mm.flush()
            finally:
                mm.close()
        with open(self._refcnt_path, "wb") as f:
            f.truncate(self._num_gpu_blocks * self._refcnt_entry_size)
        with open(self._refcnt_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            try:
                self._set_refcnt(mm, 0, 1)
                mm.flush()
            finally:
                mm.close()

    def _load_state(self) -> dict[str, Any]:
        if not os.path.exists(self._meta_path):
            self._initialize_state()
        with open(self._meta_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            try:
                num_gpu_blocks = self._get_meta_num_gpu_blocks(mm)
                if num_gpu_blocks != self._num_gpu_blocks:
                    raise RuntimeError(
                        "shared allocator num_gpu_blocks mismatch: "
                        f"{num_gpu_blocks} vs {self._num_gpu_blocks}")
                return {
                    "num_gpu_blocks": num_gpu_blocks,
                    "free_count": self._get_meta_free_count(mm),
                    "next_scan_start": self._get_meta_next_scan_start(mm),
                }
            finally:
                mm.close()

    def _save_state(self, state: dict[str, Any], meta_mm: mmap.mmap) -> None:
        self._set_meta_num_gpu_blocks(meta_mm, int(state["num_gpu_blocks"]))
        self._set_meta_free_count(meta_mm, int(state["free_count"]))
        self._set_meta_next_scan_start(meta_mm, int(state["next_scan_start"]))

    def _with_lock(self, fn):
        with open(self._lock_path, "a+b") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                state = self._load_state()
                with open(self._meta_path, "r+b") as mf, \
                        open(self._bitmap_path, "r+b") as bf, \
                        open(self._refcnt_path, "r+b") as rf:
                    meta_mm = mmap.mmap(mf.fileno(), 0)
                    bitmap_mm = mmap.mmap(bf.fileno(), 0)
                    refcnt_mm = mmap.mmap(rf.fileno(), 0)
                    try:
                        out = fn(state, meta_mm, bitmap_mm, refcnt_mm)
                        self._save_state(state, meta_mm)
                        meta_mm.flush()
                        bitmap_mm.flush()
                        refcnt_mm.flush()
                    finally:
                        refcnt_mm.close()
                        bitmap_mm.close()
                        meta_mm.close()
                return out
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _with_shared_lock(self, fn):
        with open(self._lock_path, "a+b") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_SH)
            try:
                with open(self._meta_path, "r+b") as mf:
                    meta_mm = mmap.mmap(mf.fileno(), 0)
                    try:
                        state = {
                            "num_gpu_blocks": self._get_meta_num_gpu_blocks(meta_mm),
                            "free_count": self._get_meta_free_count(meta_mm),
                            "next_scan_start": self._get_meta_next_scan_start(meta_mm),
                        }
                        return fn(state)
                    finally:
                        meta_mm.close()
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _req_file_path(self, req_key: str) -> str:
        digest = hashlib.md5(req_key.encode(),
                             usedforsecurity=False).hexdigest()
        return os.path.join(self._req_dir, f"{digest}.bin")

    def _read_req_blocks(self, req_key: str) -> list[int]:
        path = self._req_file_path(req_key)
        if not os.path.exists(path):
            return []
        with open(path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            try:
                if mm.size() < 4:
                    return []
                count = int(struct.unpack_from("<I", mm, 0)[0])
                if count <= 0:
                    return []
                return list(struct.unpack_from(f"<{count}I", mm, 4))
            finally:
                mm.close()

    def _append_req_blocks(self, req_key: str, block_ids: list[int]) -> None:
        if not block_ids:
            return
        path = self._req_file_path(req_key)
        capacity = self._num_gpu_blocks
        total_size = 4 + capacity * self._refcnt_entry_size
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.truncate(total_size)
        with open(path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            try:
                count = int(struct.unpack_from("<I", mm, 0)[0])
                new_count = count + len(block_ids)
                if new_count > capacity:
                    raise RuntimeError(
                        f"shared allocator request block overflow: {new_count} > {capacity}"
                    )
                struct.pack_into(f"<{len(block_ids)}I", mm,
                                 4 + count * self._refcnt_entry_size,
                                 *block_ids)
                struct.pack_into("<I", mm, 0, new_count)
                mm.flush()
            finally:
                mm.close()

    def _delete_req_blocks(self, req_key: str) -> None:
        path = self._req_file_path(req_key)
        try:
            os.remove(path)
        except FileNotFoundError:
            return

    def _get_refcnt(self, mm: mmap.mmap, block_id: int) -> int:
        offset = int(block_id) * self._refcnt_entry_size
        return int(struct.unpack_from("<I", mm, offset)[0])

    def _set_refcnt(self, mm: mmap.mmap, block_id: int, value: int) -> None:
        offset = int(block_id) * self._refcnt_entry_size
        struct.pack_into("<I", mm, offset, int(value))

    def _get_meta_num_gpu_blocks(self, mm: mmap.mmap) -> int:
        return int(struct.unpack_from("<Q", mm, 0)[0])

    def _set_meta_num_gpu_blocks(self, mm: mmap.mmap, value: int) -> None:
        struct.pack_into("<Q", mm, 0, int(value))

    def _get_meta_free_count(self, mm: mmap.mmap) -> int:
        return int(struct.unpack_from("<Q", mm, 8)[0])

    def _set_meta_free_count(self, mm: mmap.mmap, value: int) -> None:
        struct.pack_into("<Q", mm, 8, int(value))

    def _get_meta_next_scan_start(self, mm: mmap.mmap) -> int:
        return int(struct.unpack_from("<Q", mm, 16)[0])

    def _set_meta_next_scan_start(self, mm: mmap.mmap, value: int) -> None:
        struct.pack_into("<Q", mm, 16, int(value))

    def _get_free_bit(self, mm: mmap.mmap, block_id: int) -> bool:
        if block_id < 0:
            return False
        byte_idx = block_id >> 3
        bit_mask = 1 << (block_id & 7)
        return (mm[byte_idx] & bit_mask) != 0

    def _set_free_bit(self, mm: mmap.mmap, block_id: int, is_free: bool) -> None:
        if block_id < 0:
            return
        byte_idx = block_id >> 3
        bit_mask = 1 << (block_id & 7)
        cur = mm[byte_idx]
        if is_free:
            mm[byte_idx:byte_idx + 1] = bytes([cur | bit_mask])
        else:
            mm[byte_idx:byte_idx + 1] = bytes([cur & (~bit_mask & 0xFF)])

    def _alloc_free_blocks(self,
                           state: dict[str, Any],
                           bitmap_mm: mmap.mmap,
                           refcnt_mm: mmap.mmap,
                           n: int) -> list[int]:
        want = int(n)
        if want <= 0:
            return []
        if int(state.get("free_count", 0)) < want:
            raise ValueError(
                f"Cannot get {want} free blocks from shared allocator")
        out: list[int] = []
        start = max(1, int(state.get("next_scan_start", 1)))
        block_id = start
        wrapped = False
        while len(out) < want:
            if block_id >= self._num_gpu_blocks:
                block_id = 1
                wrapped = True
            if wrapped and block_id >= start:
                break
            if self._get_free_bit(bitmap_mm, block_id):
                self._set_free_bit(bitmap_mm, block_id, False)
                self._set_refcnt(refcnt_mm, block_id, 1)
                out.append(int(block_id))
            block_id += 1
        if len(out) != want:
            raise ValueError(
                f"Cannot get {want} free blocks from shared allocator")
        state["free_count"] = int(state.get("free_count", 0)) - len(out)
        state["next_scan_start"] = int(block_id)
        return out

    @staticmethod
    def _num_free_from_intervals(intervals: list[list[int]]) -> int:
        total = 0
        for it in intervals:
            if isinstance(it, list) and len(it) == 2:
                s = int(it[0])
                e = int(it[1])
                if e >= s:
                    total += (e - s + 1)
        return int(total)

    @staticmethod
    def _take_from_intervals(intervals: list[list[int]], n: int) -> list[int]:
        if n <= 0:
            return []
        out: list[int] = []
        i = 0
        while i < len(intervals) and len(out) < n:
            s, e = int(intervals[i][0]), int(intervals[i][1])
            if e < s:
                intervals.pop(i)
                continue
            need = n - len(out)
            cnt = min(need, e - s + 1)
            out.extend(range(s, s + cnt))
            ns = s + cnt
            if ns <= e:
                intervals[i][0] = ns
                i += 1
            else:
                intervals.pop(i)
        return out

    @staticmethod
    def _add_ids_to_intervals(intervals: list[list[int]], ids: list[int]) -> None:
        vals = sorted(int(x) for x in ids if int(x) > 0)
        if not vals:
            return
        for b in vals:
            inserted = False
            for j, it in enumerate(intervals):
                s, e = int(it[0]), int(it[1])
                if s <= b <= e:
                    inserted = True
                    break
                if b == e + 1:
                    intervals[j][1] = b
                    inserted = True
                    # merge forward
                    if j + 1 < len(intervals) and int(intervals[j + 1][0]) <= b + 1:
                        intervals[j][1] = max(int(intervals[j][1]),
                                              int(intervals[j + 1][1]))
                        intervals.pop(j + 1)
                    break
                if b < s - 1:
                    intervals.insert(j, [b, b])
                    inserted = True
                    break
                if b == s - 1:
                    intervals[j][0] = b
                    inserted = True
                    break
            if not inserted:
                intervals.append([b, b])
        # Final merge pass.
        intervals.sort(key=lambda x: int(x[0]))
        k = 0
        while k + 1 < len(intervals):
            s1, e1 = int(intervals[k][0]), int(intervals[k][1])
            s2, e2 = int(intervals[k + 1][0]), int(intervals[k + 1][1])
            if s2 <= e1 + 1:
                intervals[k][1] = max(e1, e2)
                intervals.pop(k + 1)
            else:
                k += 1

    @staticmethod
    def _remove_specific_from_intervals(intervals: list[list[int]],
                                        want: list[int]) -> list[int]:
        out: list[int] = []
        for bid in sorted(int(x) for x in want if int(x) > 0):
            for j, it in enumerate(intervals):
                s, e = int(it[0]), int(it[1])
                if bid < s:
                    break
                if s <= bid <= e:
                    out.append(bid)
                    if s == e == bid:
                        intervals.pop(j)
                    elif bid == s:
                        intervals[j][0] = s + 1
                    elif bid == e:
                        intervals[j][1] = e - 1
                    else:
                        # split interval
                        left = [s, bid - 1]
                        right = [bid + 1, e]
                        intervals[j] = left
                        intervals.insert(j + 1, right)
                    break
        return out

    def allocate(self, num_blocks: int) -> list[int]:
        n = int(num_blocks)
        if n <= 0:
            return []

        def _op(state: dict[str, Any], meta_mm: mmap.mmap, bitmap_mm: mmap.mmap,
                refcnt_mm: mmap.mmap) -> list[int]:
            return self._alloc_free_blocks(state, bitmap_mm, refcnt_mm, n)

        return self._with_lock(_op)

    @staticmethod
    def canonical_request_id(request_id: str) -> str:
        rid = str(request_id)
        if rid.startswith("cmpl-"):
            rid = rid[len("cmpl-"):]
        had_marker = ("___prefill_addr_" in rid) or ("___decode_addr_" in rid)
        rid = rid.split("___prefill_addr_")[0].split("___decode_addr_")[0]
        # vLLM may append worker suffix like "-0" for the same logical req.
        if had_marker:
            rid = re.sub(r"-\d+$", "", rid)
        return rid

    @staticmethod
    def is_terminal_request_id(request_id: str) -> bool:
        rid = str(request_id)
        # In our handoff format, only non-last hops carry decode addr.
        return "___decode_addr_" not in rid

    @staticmethod
    def is_post_cutover_request_id(request_id: str) -> bool:
        # Post-cutover hops carry prefill_addr marker from previous server.
        return "___prefill_addr_" in str(request_id)

    @staticmethod
    def is_handoff_request_id(request_id: str) -> bool:
        rid = str(request_id)
        return ("___prefill_addr_" in rid) or ("___decode_addr_" in rid)

    def allocate_request_range(self, request_id: str, start: int,
                               end: int) -> list[int]:
        s = int(start)
        e = int(end)
        if e <= s:
            return []
        if s < 0:
            raise ValueError(f"invalid request range start={s}")
        req_key = self.canonical_request_id(request_id)

        def _op(state: dict[str, Any], meta_mm: mmap.mmap, bitmap_mm: mmap.mmap,
                refcnt_mm: mmap.mmap) -> list[int]:
            req_blocks = self._read_req_blocks(req_key)
            old_len = len(req_blocks)
            if old_len < e:
                picked = self._alloc_free_blocks(state, bitmap_mm, refcnt_mm,
                                                 e - old_len)
                req_blocks.extend(int(x) for x in picked)
                self._append_req_blocks(req_key, picked)
            for bid in req_blocks[s:min(e, old_len)]:
                self._set_refcnt(refcnt_mm, bid,
                                 self._get_refcnt(refcnt_mm, bid) + 1)
            return [int(x) for x in req_blocks[s:e]]

        return self._with_lock(_op)

    def reserve_specific(self, block_ids: list[int]) -> None:
        if not block_ids:
            return

        want = [int(x) for x in block_ids]

        def _op(state: dict[str, Any], meta_mm: mmap.mmap, bitmap_mm: mmap.mmap,
                refcnt_mm: mmap.mmap) -> None:
            for bid in want:
                cur = self._get_refcnt(refcnt_mm, bid)
                if cur == 0 and self._get_free_bit(bitmap_mm, bid):
                    self._set_free_bit(bitmap_mm, bid, False)
                    self._set_refcnt(refcnt_mm, bid, 1)
                    state["free_count"] = max(
                        0, int(state.get("free_count", 0)) - 1)
                else:
                    self._set_refcnt(refcnt_mm, bid, cur + 1)
            return None

        self._with_lock(_op)

    def release(self, block_ids: list[int]) -> None:
        if not block_ids:
            return

        free_ids = [int(x) for x in block_ids if int(x) > 0]

        def _op(state: dict[str, Any], meta_mm: mmap.mmap, bitmap_mm: mmap.mmap,
                refcnt_mm: mmap.mmap) -> None:
            for bid in free_ids:
                cur = self._get_refcnt(refcnt_mm, bid)
                if cur <= 1:
                    if cur == 1:
                        self._set_refcnt(refcnt_mm, bid, 0)
                        self._set_free_bit(bitmap_mm, bid, True)
                        state["free_count"] = int(state.get("free_count", 0)) + 1
                else:
                    self._set_refcnt(refcnt_mm, bid, cur - 1)
            return None

        self._with_lock(_op)

    def release_request_blocks(self,
                               request_id: str,
                               block_ids: list[int],
                               terminal: bool = False) -> None:
        if not block_ids and not terminal:
            return
        req_key = self.canonical_request_id(request_id)
        free_ids = [int(x) for x in block_ids if int(x) > 0]

        def _op(state: dict[str, Any], meta_mm: mmap.mmap, bitmap_mm: mmap.mmap,
                refcnt_mm: mmap.mmap) -> None:
            for bid in free_ids:
                cur = self._get_refcnt(refcnt_mm, bid)
                if cur <= 1:
                    if cur == 1:
                        self._set_refcnt(refcnt_mm, bid, 0)
                        self._set_free_bit(bitmap_mm, bid, True)
                        state["free_count"] = int(state.get("free_count", 0)) + 1
                else:
                    self._set_refcnt(refcnt_mm, bid, cur - 1)
            if terminal:
                self._delete_req_blocks(req_key)
            return None

        self._with_lock(_op)

    def num_free_blocks(self) -> int:
        def _op(state: dict[str, Any]) -> int:
            return int(state.get("free_count", 0))

        return int(self._with_shared_lock(_op))


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks 
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[BlockHashWithGroupId,
                          Union[KVCacheBlock, dict[int, KVCacheBlock]]] = {}

    def get_one_block(self,
                      key: BlockHashWithGroupId) -> Optional[KVCacheBlock]:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId,
            block_id: int) -> Optional[KVCacheBlock]:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        enable_kv_cache_events: Whether to enable kv cache events.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = \
            BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self._shared_allocator: Optional[_SharedBlockAllocator] = None
        shared_enable = os.getenv("VLLM_SHARED_BLOCK_ALLOCATOR_ENABLE",
                                  "").strip().lower() in {
                                      "1", "true", "yes", "on"
                                  }
        if shared_enable:
            # Keep key stable for processes that target the same physical GPU.
            key = os.getenv("VLLM_SHARED_BLOCK_ALLOCATOR_KEY", "").strip()
            if not key:
                cvd = os.getenv("CUDA_VISIBLE_DEVICES", "all")
                key = f"{cvd}__lr{os.getenv('LOCAL_RANK', '0')}"
                # Prefer runtime CUDA device identity when available.
                try:
                    import torch  # local import to avoid hard dependency here
                    if torch.cuda.is_available():
                        local_idx = int(torch.cuda.current_device())
                        phys = str(local_idx)
                        if cvd and cvd != "all":
                            parts = [x.strip() for x in cvd.split(",")]
                            if 0 <= local_idx < len(parts):
                                phys = parts[local_idx]
                        key = f"gpu{phys}"
                except Exception:
                    pass
            path = os.getenv("VLLM_SHARED_BLOCK_ALLOCATOR_PATH",
                             "/tmp/vllm_shared_block_allocator").strip()
            reset = os.getenv("VLLM_SHARED_BLOCK_ALLOCATOR_RESET",
                              "").strip().lower() in {"1", "true", "yes", "on"}
            try:
                self._shared_allocator = _SharedBlockAllocator(
                    path=path,
                    key=key,
                    num_gpu_blocks=num_gpu_blocks,
                    reset=reset,
                )
                logger.info("shared block allocator enabled key=%s path=%s",
                            key, path)
            except Exception as e:
                logger.warning(
                    "failed to enable shared block allocator, fallback to local pool: %s",
                    repr(e))
                self._shared_allocator = None

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        """Get the cached block by the block hash for each group in 
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id)
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id)
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks == num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        new_block_hashes = request.block_hashes[num_cached_blocks:]

        new_hashes: Optional[list[ExternalBlockHash]] = (
            [] if self.enable_kv_cache_events else None)
        for i, blk in enumerate(new_full_blocks):
            block_hash = new_block_hashes[i]
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id)

            # In shared-allocator / cross-process handoff mode, local block
            # metadata can be stale (block already carries an old hash).
            # Make cache insertion idempotent by evicting/resetting first.
            if blk.block_hash is not None:
                # Fast path: same hash already set (can happen when a block is
                # revisited across hops). Keep it and skip reinsertion.
                if blk.block_hash == block_hash_with_group_id:
                    continue
                self._maybe_evict_cached_block(blk)
                # Fallback hard reset if local map miss prevented eviction.
                if blk.block_hash is not None:
                    blk.reset_hash()

            # Update and added the full block to the cache.
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id,
                                                   blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: Optional[ExternalBlockHash] = None
            else:
                parent_block = blocks[num_cached_blocks - 1]
                assert parent_block.block_hash is not None
                parent_block_hash = maybe_convert_block_hash(
                    get_block_hash(parent_block.block_hash))

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.
                    all_token_ids[num_cached_blocks *
                                  block_size:num_full_blocks * block_size],
                    block_size=block_size,
                    lora_id=request.lora_request.id
                    if request.lora_request else None,
                    medium=MEDIUM_GPU,
                ))

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if self._shared_allocator is not None:
            block_ids = self._shared_allocator.allocate(num_blocks)
            ret: list[KVCacheBlock] = []
            for bid in block_ids:
                block = self.blocks[bid]
                if block.ref_cnt != 0:
                    raise RuntimeError(
                        "shared allocator selected a block that is still "
                        f"referenced locally: block_id={bid}, ref_cnt={block.ref_cnt}"
                    )
                if not block.is_null:
                    self._remove_from_local_free_queue_if_present(block)
                ret.append(block)
        else:
            if num_blocks > self.get_num_free_blocks():
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the pool")
            ret = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    def _get_new_blocks_local(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks > self.free_block_queue.num_free_blocks:
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        ret = self.free_block_queue.popleft_n(num_blocks)
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    def _use_shared_for_request(self, request_id: Optional[str]) -> bool:
        if self._shared_allocator is None:
            return False
        if not request_id:
            return False
        # Zero-copy shared-KV handoff requires producer and consumer to share
        # the exact same global block ids. Restricting the shared allocator to
        # post-cutover hops makes the producer allocate local-only ids while
        # the consumer allocates shared ids, which corrupts handoff decode.
        # Route every handoff-participating request through the shared block
        # table so both servers resolve the same canonical request range.
        return self._shared_allocator.is_handoff_request_id(request_id)

    def get_new_blocks_for_request(self, request_id: str, start: int,
                                   end: int) -> list[KVCacheBlock]:
        num = max(0, int(end) - int(start))
        if not self._use_shared_for_request(request_id):
            return self._get_new_blocks_local(num)
        block_ids = self._shared_allocator.allocate_request_range(
            request_id, int(start), int(end))
        ret: list[KVCacheBlock] = []
        for bid in block_ids:
            block = self.blocks[bid]
            # If this process has it in local free queue, remove it before use.
            if block.ref_cnt == 0 and not block.is_null:
                self._remove_from_local_free_queue_if_present(block)
            ret.append(block)

        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                block.ref_cnt += 1
        else:
            for block in ret:
                block.ref_cnt += 1
        return ret

    def _remove_from_local_free_queue_if_present(self,
                                                 block: KVCacheBlock) -> None:
        # In shared-allocator mode, a globally-free block may not be linked in
        # this process's local free queue (stale/local-only list). Remove only
        # when this node is actually linked to avoid RuntimeError.
        if block.prev_free_block is None or block.next_free_block is None:
            return
        try:
            self.free_block_queue.remove(block)
        except RuntimeError:
            # Best-effort safety for concurrent/stale local queue states.
            return

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash,
                                               block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed in single-process mode.
            # In shared-allocator mode, block metadata may originate from
            # another process and local cache map can miss this block id.
            # We must still clear stale hash before re-allocation.
            if self._shared_allocator is not None:
                block.reset_hash()
                return True
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(block_hashes=[
                    maybe_convert_block_hash(get_block_hash(block_hash))
                ],
                             medium=MEDIUM_GPU))
        return True

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    if self._shared_allocator is not None:
                        self._shared_allocator.reserve_specific([block.block_id])
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    def free_blocks(self,
                    ordered_blocks: Iterable[KVCacheBlock],
                    request_id: Optional[str] = None) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        released_blocks = [
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ]
        self.free_block_queue.append_n(released_blocks)
        if self._shared_allocator is not None and released_blocks and \
                self._use_shared_for_request(request_id):
            block_ids = [block.block_id for block in released_blocks]
            if request_id:
                self._shared_allocator.release_request_blocks(
                    request_id=request_id,
                    block_ids=block_ids,
                    terminal=self._shared_allocator.is_terminal_request_id(
                        request_id),
                )
            else:
                self._shared_allocator.release(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks - 1)
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        if self._shared_allocator is None:
            return self.free_block_queue.num_free_blocks
        try:
            return self._shared_allocator.num_free_blocks()
        except Exception:
            return self.free_block_queue.num_free_blocks

    def get_num_free_blocks_for_request(self, request_id: Optional[str]) -> int:
        """Request-aware free block count for tiered allocator mode.

        - pre-cutover/local requests: use local free queue only
        - post-cutover/shared requests: use shared allocator view
        """
        if not self._use_shared_for_request(request_id):
            return int(self.free_block_queue.num_free_blocks)
        try:
            return int(self._shared_allocator.num_free_blocks()
                       ) if self._shared_allocator is not None else int(
                           self.free_block_queue.num_free_blocks)
        except Exception:
            return int(self.free_block_queue.num_free_blocks)

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.
        
        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
