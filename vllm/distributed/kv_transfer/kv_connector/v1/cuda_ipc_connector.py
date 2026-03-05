# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.proxy_cluster.ipc_state_dict import (export_cuda_tensor_meta,
                                               rebuild_cuda_tensor_from_meta)
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    request_id: str
    block_ids: torch.Tensor
    num_tokens: int

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int],
                  block_size: int) -> "ReqMeta":
        del block_size
        return ReqMeta(
            request_id=request_id,
            block_ids=torch.tensor(block_ids),
            num_tokens=len(token_ids),
        )


@dataclass
class CudaIpcConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids, block_size))


class CudaIpcConnector(KVConnectorBase_V1):
    """CUDA IPC connector for same-node server handoff.

    Data-plane:
    - save_kv_layer: export CUDA IPC handle metadata to local files
    - start_load_kv: wait + import CUDA IPC handle metadata and inject KV
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        # Producer-side rolling state of full block ids for decode handoff.
        self._send_req_blocks: dict[str, list[int]] = {}
        # Base request IDs that should keep exporting decode KV across steps.
        self._send_enabled_bases: set[str] = set()
        # Keep partial prompt-preill state aligned with P2P connector logic.
        # Key: request_id, Val: (accumulated_block_ids, prompt_token_ids)
        self.chunked_prefill: dict[str, tuple[list[int], list[int]]] = {}
        transfer_config = vllm_config.kv_transfer_config
        self.can_send = transfer_config.is_kv_producer
        self.can_recv = transfer_config.is_kv_consumer

        self._ipc_meta_dir = transfer_config.get_from_extra_config(
            "ipc_meta_dir", "/tmp/vllm_kv_ipc")
        self._ipc_wait_timeout_s = float(
            transfer_config.get_from_extra_config("ipc_wait_timeout_s", 60.0))
        self._ipc_poll_interval_s = float(
            transfer_config.get_from_extra_config("ipc_poll_interval_s", 0.01))
        self._ipc_export_ttl_s = float(
            transfer_config.get_from_extra_config("ipc_export_ttl_s", 600.0))
        self._ipc_strict_load = bool(
            transfer_config.get_from_extra_config("ipc_strict_load", False))
        self._kv_owner_state_url = str(
            transfer_config.get_from_extra_config("kv_owner_state_url", "")).strip()
        self._kv_owner_state_timeout_s = float(
            transfer_config.get_from_extra_config("kv_owner_state_timeout_s", 1.0))
        self._shared_block_table_enable = bool(
            transfer_config.get_from_extra_config("shared_block_table_enable",
                                                  False))
        os.makedirs(self._ipc_meta_dir, exist_ok=True)

        # Keep exported tensors alive while peer imports IPC handles.
        self._inflight_exports: dict[str, tuple[torch.Tensor, float]] = {}
        self._warned_mismatch: set[str] = set()

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs: Any) -> None:
        del kwargs
        if not self.can_recv:
            return
        if self._connector_metadata is None:
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, CudaIpcConnectorMetadata)

        def _device_tag(layer: torch.Tensor) -> str:
            if layer.is_cuda:
                idx = layer.device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                return f"cuda{int(idx)}"
            return str(layer.device).replace(":", "_")

        def inject_kv_into_layer(layer: torch.Tensor, kv_cache: torch.Tensor,
                                 block_ids: torch.Tensor,
                                 request_id: str,
                                 layer_name: str) -> None:
            warn_key = f"{request_id}::{layer_name}"
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):
                num_block = kv_cache.shape[0]
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    n = min(int(len(block_ids)), int(num_block))
                    layer[block_ids[:n], ...] = kv_cache[:n, ...]
                    if warn_key not in self._warned_mismatch:
                        self._warned_mismatch.add(warn_key)
                        logger.warning(
                            "cuda_ipc kv_cache mismatch block_ids=%d num_block=%d req=%s layer=%s",
                            len(block_ids), num_block, request_id, layer_name)
            elif layer.shape[0] == 2:
                num_block = kv_cache.shape[1]
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    n = min(int(len(block_ids)), int(num_block))
                    layer[:, block_ids[:n], ...] = kv_cache[:, :n, ...]
                    if warn_key not in self._warned_mismatch:
                        self._warned_mismatch.add(warn_key)
                        logger.warning(
                            "cuda_ipc kv_cache mismatch block_ids=%d num_block=%d req=%s layer=%s",
                            len(block_ids), num_block, request_id, layer_name)

        for request in metadata.requests:
            req_id = request.request_id
            if not self.has_prefill_addr(req_id):
                continue

            base_req = self.base_request_id(req_id)
            load_errors: list[str] = []
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue
                layer_kv = kv_cache_attr[forward_context.virtual_engine]

                layer_key = f"{layer_name}@{_device_tag(layer_kv)}"
                tensor_key = f"{base_req}#{layer_key}"
                expected_blocks = min(
                    len(request.block_ids),
                    (request.num_tokens + self._block_size - 1) // self._block_size,
                )
                # Cross-engine scheduling can report slightly different token
                # counts for the same logical step. Token-based gating causes
                # false negatives ("insufficient_tokens") and breaks handoff.
                # Use block count as the hard validity check.
                min_tokens = 0
                # Use best-effort lookup for handoff continuity. Cross-engine
                # scheduling and different KV layouts can make strict block
                # count checks too strong in practice.
                rec = self._wait_owner_lookup_kv(
                    request_id=base_req,
                    layer_name=layer_key,
                    min_tokens=min_tokens,
                    min_blocks=0,
                )
                if rec and isinstance(rec.get("num_blocks"), int):
                    hinted_blocks = int(rec["num_blocks"])
                    if hinted_blocks > 0:
                        expected_blocks = min(expected_blocks, hinted_blocks)
                if expected_blocks <= 0:
                    # No usable remote KV blocks yet for this layer.
                    continue
                # Default path: inject into this worker's local block IDs that
                # correspond to currently valid tokens only.
                dst_block_ids = request.block_ids[:expected_blocks]
                if self._kv_owner_state_url:
                    # Prefer owner-state record; fallback to deterministic key
                    # for robustness when register/lookup races happen.
                    if rec and isinstance(rec.get("tensor_key"), str):
                        tensor_key = str(rec["tensor_key"])
                        # Experimental mode only: owner-state block IDs are
                        # used when a real shared block-table is enabled.
                        if self._shared_block_table_enable:
                            rec_block_ids = rec.get("block_ids")
                            if isinstance(rec_block_ids, list):
                                try:
                                    rec_ids = [int(x) for x in rec_block_ids]
                                    dst_block_ids = torch.tensor(
                                        rec_ids[:expected_blocks],
                                        dtype=request.block_ids.dtype,
                                    )
                                except Exception:
                                    dst_block_ids = request.block_ids[
                                        :expected_blocks]
                    else:
                        logger.warning(
                            "cuda_ipc owner lookup miss req=%s layer=%s fallback=local_key",
                            base_req, layer_key)
                elif rec and isinstance(rec.get("tensor_key"), str):
                    tensor_key = str(rec["tensor_key"])
                tensor_meta = self._wait_tensor_meta(
                    tensor_key,
                    min_tokens=min_tokens,
                    min_blocks=expected_blocks,
                )
                if tensor_meta is None:
                    logger.warning("cuda_ipc missing tensor meta: %s", tensor_key)
                    load_errors.append(f"missing_meta:{tensor_key}")
                    continue

                # Ensure current CUDA context matches this worker/layer device
                # before importing CUDA IPC memory.
                try:
                    if layer_kv.is_cuda:
                        torch.cuda.set_device(layer_kv.device)
                except Exception:
                    pass

                try:
                    remote_kv = rebuild_cuda_tensor_from_meta(tensor_meta)
                except Exception as e:
                    logger.warning(
                        "cuda_ipc rebuild failed req=%s layer=%s key=%s err=%s",
                        req_id, layer_key, tensor_key, repr(e))
                    load_errors.append(
                        f"rebuild_failed:{req_id}:{layer_key}:{tensor_key}")
                    continue
                # Destination indices must use this worker's local allocation
                # in default mode. In global shared-allocator mode, owner-state
                # may provide globally consistent block_ids for direct inject.
                inject_kv_into_layer(layer_kv, remote_kv, dst_block_ids,
                                     req_id, layer_key)
            if load_errors:
                logger.warning("cuda_ipc load degraded req=%s errors=%d",
                               req_id, len(load_errors))
                if self._ipc_strict_load:
                    raise RuntimeError(
                        "cuda_ipc strict load failed: " + load_errors[0])
        self._gc_inflight_exports()

    def _should_send_req(self, request_id: str) -> bool:
        if self.has_decode_addr(request_id):
            return True
        return self.base_request_id(request_id) in self._send_enabled_bases

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata",
                      **kwargs: Any) -> None:
        del kwargs
        if not self.can_send:
            return
        if self._connector_metadata is None:
            return

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, CudaIpcConnectorMetadata)

        def extract_kv_from_layer(layer: torch.Tensor,
                                  block_ids: torch.Tensor) -> torch.Tensor:
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):
                return layer[block_ids, ...]
            if layer.shape[0] == 2:
                return layer[:, block_ids, ...]
            return layer[block_ids, ...]

        def _device_tag(layer: torch.Tensor) -> str:
            if layer.is_cuda:
                idx = layer.device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                return f"cuda{int(idx)}"
            return str(layer.device).replace(":", "_")

        layer_key = f"{layer_name}@{_device_tag(kv_layer)}"

        for request in connector_metadata.requests:
            req_id = request.request_id
            if not self._should_send_req(req_id):
                continue
            valid_blocks = min(
                int(len(request.block_ids)),
                int((request.num_tokens + self._block_size - 1) //
                    self._block_size),
            )
            if valid_blocks <= 0:
                continue
            used_block_ids = request.block_ids[:valid_blocks]
            kv_cache = extract_kv_from_layer(kv_layer, used_block_ids)
            if not kv_cache.is_cuda:
                continue
            kv_cache = kv_cache.contiguous()
            if kv_cache.dim() == 0:
                continue

            base_req = self.base_request_id(req_id)
            # Deterministic tensor key per request/layer-device.
            # Old keys are guarded by request-id uniqueness + owner-state reset.
            tensor_key = f"{base_req}#{layer_key}"
            self._inflight_exports[tensor_key] = (kv_cache, time.time())
            exported_blocks = int(valid_blocks)
            exported_block_ids = [
                int(x) for x in used_block_ids[:exported_blocks].tolist()
            ]
            self._write_tensor_meta(
                tensor_key,
                export_cuda_tensor_meta(kv_cache),
                num_tokens=request.num_tokens,
                num_blocks=exported_blocks,
            )
            self._owner_register_kv(
                request_id=base_req,
                worker=self._decode_worker_from_req(req_id),
                hop=0,
                layer_name=layer_key,
                tensor_key=tensor_key,
                num_tokens=request.num_tokens,
                num_blocks=exported_blocks,
                block_ids=exported_block_ids,
            )
        self._gc_inflight_exports()

    def wait_for_save(self):
        return

    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs: Any) -> tuple[Optional[set[str]], Optional[set[str]]]:
        del kwargs
        # Do not eagerly free exports on producer request finish.
        # Consumer may still be importing CUDA IPC handles shortly after.
        del finished_req_ids
        self._gc_inflight_exports()
        return None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if not self.can_recv or not self.has_prefill_addr(request.request_id):
            return 0, False

        num_external_tokens = len(request.prompt_token_ids) - 1 - num_computed_tokens
        if num_external_tokens < 0:
            num_external_tokens = 0
        return num_external_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.can_recv and self.has_prefill_addr(
                request.request_id) and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = CudaIpcConnectorMetadata()
        seen_req_ids: set[str] = set()

        def _add_req_once(request_id: str, token_ids: list[int],
                          block_ids: list[int]) -> None:
            if request_id in seen_req_ids:
                return
            meta.add_request(
                request_id=request_id,
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
            )
            seen_req_ids.add(request_id)

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.can_send and self._should_send_req(new_req.req_id):
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                self._send_enabled_bases.add(self.base_request_id(new_req.req_id))
                # Initialize producer-side full block table for this request.
                self._send_req_blocks[new_req.req_id] = list(new_req.block_ids[0])
                # Prompt prefill may be chunked across steps.
                if num_tokens < len(new_req.prompt_token_ids):
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], new_req.prompt_token_ids)
                else:
                    # For decode handoff, use current total token count to
                    # export all valid KV blocks, not only original prompt.
                    _add_req_once(new_req.req_id, [0] * int(num_tokens),
                                  self._send_req_blocks[new_req.req_id])
            if self.can_recv and new_req.req_id in self._requests_need_load:
                _add_req_once(new_req.req_id, new_req.prompt_token_ids,
                              new_req.block_ids[0])
                self._requests_need_load.pop(new_req.req_id, None)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            if self.can_send and req_id in self.chunked_prefill:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = (self.chunked_prefill[req_id][0] + block_ids)
                prompt_token_ids = self.chunked_prefill[req_id][1]
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                else:
                    _add_req_once(req_id, prompt_token_ids, block_ids)
                    self.chunked_prefill.pop(req_id, None)
            if self.can_send and self._should_send_req(req_id):
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens).get(req_id, 0)
                total_tokens = int(num_computed_tokens + num_scheduled_tokens)
                full_blocks = self._send_req_blocks.get(req_id, []).copy()
                nb = new_block_ids[0] if new_block_ids else []
                if resumed_from_preemption:
                    full_blocks = list(nb)
                else:
                    full_blocks.extend(nb)
                self._send_req_blocks[req_id] = full_blocks
                if total_tokens > 0 and full_blocks:
                    _add_req_once(req_id, [0] * total_tokens, full_blocks)

            if not resumed_from_preemption:
                continue
            if self.can_recv and req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = cached_reqs.num_computed_tokens[i] + 1
                token_ids = request.all_token_ids[:total_tokens]
                block_ids = cached_reqs.new_block_ids[i][0]
                _add_req_once(req_id, token_ids, block_ids)

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        del block_ids
        self.chunked_prefill.pop(request.request_id, None)
        self._send_req_blocks.pop(request.request_id, None)
        self._send_enabled_bases.discard(self.base_request_id(request.request_id))
        return False, None

    # ==============================
    # Helpers
    # ==============================

    @staticmethod
    def normalize_request_id(request_id: str) -> str:
        # vLLM may append a worker/rank numeric suffix (e.g. "-0") to
        # request_id in some execution paths. Do not strip generic "-<digits>"
        # because user request IDs often naturally end with timestamps/random
        # numbers, and over-stripping causes cross-request KV state collisions.
        rid = request_id
        # Only strip a trailing rank suffix when the ID already contains our
        # injected peer-address markers and the suffix is appended at the end.
        m = re.search(
            r"(___(?:decode|prefill)_addr_[^:]+:\d+)(-\d+)$",
            rid,
        )
        if m:
            rid = rid[:-len(m.group(2))]
        return rid

    @staticmethod
    def base_request_id(request_id: str) -> str:
        rid = CudaIpcConnector.normalize_request_id(request_id)
        return rid.split("___prefill_addr_")[0].split("___decode_addr_")[0]

    @staticmethod
    def has_prefill_addr(request_id: str) -> bool:
        request_id = CudaIpcConnector.normalize_request_id(request_id)
        return re.search(r"___prefill_addr_([^:]+):(\d+)___",
                         request_id) is not None

    @staticmethod
    def has_decode_addr(request_id: str) -> bool:
        request_id = CudaIpcConnector.normalize_request_id(request_id)
        return re.search(r"___decode_addr_([^:]+):(\d+)$",
                         request_id) is not None

    def _meta_file_path(self, tensor_key: str) -> str:
        digest = hashlib.md5(tensor_key.encode(),
                             usedforsecurity=False).hexdigest()
        return os.path.join(self._ipc_meta_dir, f"{digest}.pkl")

    def _write_tensor_meta(self,
                           tensor_key: str,
                           tensor_meta: dict[str, Any],
                           num_tokens: int,
                           num_blocks: int) -> None:
        path = self._meta_file_path(tensor_key)
        payload = {
            "tensor_key": tensor_key,
            "created_at": time.time(),
            "meta": tensor_meta,
            "num_tokens": int(num_tokens),
            "num_blocks": int(num_blocks),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _wait_tensor_meta(self,
                          tensor_key: str,
                          min_tokens: int = 0,
                          min_blocks: int = 0) -> Optional[dict[str, Any]]:
        path = self._meta_file_path(tensor_key)
        deadline = time.time() + self._ipc_wait_timeout_s
        while time.time() < deadline:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        payload = pickle.load(f)
                    if payload.get("tensor_key") == tensor_key:
                        payload_tokens = int(payload.get("num_tokens", 0))
                        payload_blocks = int(payload.get("num_blocks", 0))
                        if payload_tokens and payload_tokens < min_tokens:
                            time.sleep(self._ipc_poll_interval_s)
                            continue
                        if payload_blocks and payload_blocks < min_blocks:
                            time.sleep(self._ipc_poll_interval_s)
                            continue
                        return payload.get("meta")
                except Exception:
                    pass
            time.sleep(self._ipc_poll_interval_s)
        return None

    def _gc_inflight_exports(self) -> None:
        if self._ipc_export_ttl_s <= 0:
            return
        now = time.time()
        to_del = [
            key for key, (_, created_at) in self._inflight_exports.items()
            if now - created_at > self._ipc_export_ttl_s
        ]
        for key in to_del:
            self._inflight_exports.pop(key, None)

    def _decode_worker_from_req(self, request_id: str) -> str:
        req = self.normalize_request_id(request_id)
        m = re.search(r"___decode_addr_([^:]+):(\d+)$", req)
        if m:
            return f"http://{m.group(1)}:{m.group(2)}"
        m = re.search(r"___prefill_addr_([^:]+):(\d+)___", req)
        if m:
            return f"http://{m.group(1)}:{m.group(2)}"
        return ""

    def _owner_post_json(self, path: str,
                         payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self._kv_owner_state_url:
            return None
        url = self._kv_owner_state_url.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req,
                                        timeout=self._kv_owner_state_timeout_s) as resp:
                body = resp.read()
                if not body:
                    return None
                obj = json.loads(body.decode("utf-8"))
                if isinstance(obj, dict):
                    return obj
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None
        return None

    def _owner_register_kv(
        self,
        request_id: str,
        worker: str,
        hop: int,
        layer_name: str,
        tensor_key: str,
        num_tokens: int,
        num_blocks: int,
        block_ids: list[int],
    ) -> None:
        _ = self._owner_post_json(
            "/register_kv",
            {
                "request_id": request_id,
                "worker": worker,
                "hop": int(hop),
                "layer_name": layer_name,
                "tensor_key": tensor_key,
                "num_tokens": int(num_tokens),
                "num_blocks": int(num_blocks),
                "block_ids": [int(x) for x in block_ids],
            },
        )

    def _owner_lookup_kv(
        self,
        request_id: str,
        layer_name: str,
        min_tokens: int,
        min_blocks: int,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        obj = self._owner_post_json(
            "/lookup_kv",
            {
                "request_id": request_id,
                "layer_name": layer_name,
                "min_tokens": int(min_tokens),
                "min_blocks": int(min_blocks),
            },
        )
        if not obj:
            return None, "empty"
        if not bool(obj.get("ok", False)):
            err = str(obj.get("error", ""))
            rec = obj.get("record")
            if isinstance(rec, dict):
                rec = dict(rec)
            else:
                rec = None
            return rec, err
        rec = obj.get("record")
        return (rec if isinstance(rec, dict) else None), None


    def _wait_owner_lookup_kv(
        self,
        request_id: str,
        layer_name: str,
        min_tokens: int,
        min_blocks: int,
    ) -> Optional[dict[str, Any]]:
        if not self._kv_owner_state_url:
            rec, _ = self._owner_lookup_kv(request_id, layer_name, min_tokens,
                                           min_blocks)
            return rec
        deadline = time.time() + self._ipc_wait_timeout_s
        last_partial: Optional[dict[str, Any]] = None
        last_warn_ts = 0.0
        warn_gap_s = 2.0
        while time.time() < deadline:
            rec, err = self._owner_lookup_kv(request_id, layer_name, min_tokens,
                                             min_blocks)
            if rec is not None:
                if err is None:
                    return rec
                # Owner may temporarily report insufficient_* while producer is
                # still exporting more blocks/tokens. Keep the newest partial
                # record and continue waiting; use partial as fallback on
                # timeout to avoid dead-loop.
                if err in {"insufficient_blocks", "insufficient_tokens"}:
                    last_partial = rec
                    now = time.time()
                    if now - last_warn_ts >= warn_gap_s:
                        last_warn_ts = now
                        logger.warning(
                            "cuda_ipc owner lookup pending req=%s layer=%s min_tokens=%d min_blocks=%d error=%s rec_tokens=%s rec_blocks=%s",
                            request_id, layer_name, min_tokens, min_blocks, err,
                            str(rec.get("num_tokens")), str(rec.get("num_blocks")))
                else:
                    now = time.time()
                    if now - last_warn_ts >= warn_gap_s:
                        last_warn_ts = now
                        logger.warning(
                            "cuda_ipc owner lookup reject req=%s layer=%s min_tokens=%d min_blocks=%d error=%s",
                            request_id, layer_name, min_tokens, min_blocks, err)
            time.sleep(self._ipc_poll_interval_s)
        if last_partial is not None:
            logger.warning(
                "cuda_ipc owner lookup fallback_partial req=%s layer=%s min_tokens=%d min_blocks=%d rec_tokens=%s rec_blocks=%s",
                request_id, layer_name, min_tokens, min_blocks,
                str(last_partial.get("num_tokens")),
                str(last_partial.get("num_blocks")))
            return last_partial
        return None
