# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import glob
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
        self._parallel_config = vllm_config.parallel_config
        self._tp_size = max(1, int(self._parallel_config.tensor_parallel_size))
        self._global_rank = int(getattr(self._parallel_config, "rank", 0))
        self._tp_rank = self._global_rank % self._tp_size
        self._tp_primary = self._tp_rank == 0
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
        self._resume_wait_timeout_s = float(
            transfer_config.get_from_extra_config("resume_wait_timeout_s", 30.0))
        self._shared_block_table_enable = bool(
            transfer_config.get_from_extra_config("shared_block_table_enable",
                                                  False))
        # In shared-kv-pool mode, export full layer KV handle and transfer
        # only block-index metadata to avoid large per-hop contiguous copies.
        self._shared_kv_pool_enable = bool(
            transfer_config.get_from_extra_config("shared_kv_pool_enable",
                                                  False))
        # True complete-shared-pool mode:
        # - owner allocates one global KV arena
        # - consumers only map shared tensors
        # - handoff skips per-hop KV export/import copy path
        self._zero_copy_shared_pool_mode = (
            self._shared_kv_pool_enable and self._shared_block_table_enable)
        os.makedirs(self._ipc_meta_dir, exist_ok=True)

        # Keep exported tensors alive while peer imports IPC handles.
        self._inflight_exports: dict[str, tuple[torch.Tensor, float]] = {}
        self._warned_mismatch: set[str] = set()
        self._warned_skip_reexport: set[str] = set()
        # Consumer-side: request is loaded from remote KV once.
        self._recv_loaded_once: set[str] = set()
        # Producer-side: delay send-path until near handoff cutover.
        self._send_activation_tokens: dict[str, int] = {}
        self._send_activation_margin_tokens = int(
            transfer_config.get_from_extra_config("send_activation_margin_tokens",
                                                  512))
        # Producer-side: publish KV metadata incrementally.
        self._send_publish_token_stride = int(
            transfer_config.get_from_extra_config("send_publish_token_stride",
                                                  64))
        self._last_published: dict[tuple[str, str], tuple[int, int]] = {}
        # Owner-state batch register queue.
        self._owner_batch_enabled = bool(
            transfer_config.get_from_extra_config("owner_batch_register_enable",
                                                  True))
        self._owner_batch_max_items = int(
            transfer_config.get_from_extra_config("owner_batch_max_items", 256))
        self._owner_flush_each_layer = bool(
            transfer_config.get_from_extra_config("owner_flush_each_layer",
                                                  False))
        self._owner_register_batch: list[dict[str, Any]] = []
        self._owner_lookup_grace_s = float(
            transfer_config.get_from_extra_config("owner_lookup_grace_s", 2.0))
        self._meta_wait_fallback_s = float(
            transfer_config.get_from_extra_config("meta_wait_fallback_s", 2.0))
        # Keep producer blocks alive briefly after request finish so relay hop
        # can consume shared KV before allocator reuse.
        self._handoff_pin_s = float(
            transfer_config.get_from_extra_config("handoff_pin_s", 1.5))
        self._deferred_finished_until: dict[str, float] = {}
        self._global_tensor_map_cache: Optional[dict[str, str]] = None
        logger.info(
            "cuda_ipc connector init can_send=%s can_recv=%s zero_copy=%s rank=%s tp_rank=%s tp_size=%s tp_primary=%s",
            self.can_send,
            self.can_recv,
            self._zero_copy_shared_pool_mode,
            self._global_rank,
            self._tp_rank,
            self._tp_size,
            self._tp_primary,
        )

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if not (self.can_send and self._zero_copy_shared_pool_mode):
            return

        def _device_tag(layer: torch.Tensor) -> str:
            if layer.is_cuda:
                idx = layer.device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                return f"cuda{int(idx)}"
            return str(layer.device).replace(":", "_")

        payload: dict[str, str] = {}
        for layer_name, kv_cache in kv_caches.items():
            if not kv_cache.is_cuda:
                continue
            layer_key = f"{layer_name}@{_device_tag(kv_cache)}"
            tensor_key = f"__global__#{layer_key}"
            meta_path = self._meta_file_path(tensor_key)
            if not os.path.exists(meta_path):
                self._write_tensor_meta(
                    tensor_key,
                    export_cuda_tensor_meta(kv_cache),
                    num_tokens=0,
                    num_blocks=0,
                )
            payload[layer_key] = tensor_key

        if not payload:
            return

        map_path = self._global_tensor_map_path()
        tmp_path = f"{map_path}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, map_path)
        self._global_tensor_map_cache = payload.copy()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs: Any) -> None:
        del kwargs
        if not self.can_recv:
            return
        if self._zero_copy_shared_pool_mode and not self._tp_primary:
            return
        if self._connector_metadata is None:
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, CudaIpcConnectorMetadata)
        global_tensor_map = (self._load_global_tensor_map()
                             if self._zero_copy_shared_pool_mode else {})

        if self._zero_copy_shared_pool_mode:
            for request in metadata.requests:
                req_id = request.request_id
                if not self.has_prefill_addr(req_id):
                    continue
                req_norm = self.normalize_request_id(req_id)
                if req_norm in self._recv_loaded_once:
                    continue
                base_req = self.base_request_id(req_id)
                logger.info(
                    "cuda_ipc zero-copy handoff req=%s; posting load_ack without tensor rebuild",
                    base_req)
                self._owner_load_ack(
                    request_id=base_req,
                    worker=self._decode_worker_from_req(req_id),
                    hop=1,
                )
                self._recv_loaded_once.add(req_norm)
            self._gc_inflight_exports()
            return

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
                                 layer_name: str,
                                 src_block_ids: Optional[torch.Tensor] = None
                                 ) -> None:
            warn_key = f"{request_id}::{layer_name}"
            if src_block_ids is None:
                src_block_ids = block_ids
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):
                n = min(int(len(block_ids)), int(len(src_block_ids)))
                if n > 0:
                    layer[block_ids[:n], ...] = kv_cache[src_block_ids[:n], ...]
                if n != int(len(block_ids)):
                    if warn_key not in self._warned_mismatch:
                        self._warned_mismatch.add(warn_key)
                        logger.warning(
                            "cuda_ipc kv_cache mismatch dst=%d src=%d req=%s layer=%s",
                            len(block_ids), len(src_block_ids), request_id,
                            layer_name)
            elif layer.shape[0] == 2:
                n = min(int(len(block_ids)), int(len(src_block_ids)))
                if n > 0:
                    layer[:, block_ids[:n], ...] = kv_cache[:, src_block_ids[:n],
                                                             ...]
                if n != int(len(block_ids)):
                    if warn_key not in self._warned_mismatch:
                        self._warned_mismatch.add(warn_key)
                        logger.warning(
                            "cuda_ipc kv_cache mismatch dst=%d src=%d req=%s layer=%s",
                            len(block_ids), len(src_block_ids), request_id,
                            layer_name)

        for request in metadata.requests:
            req_id = request.request_id
            if not self.has_prefill_addr(req_id):
                continue
            req_norm = self.normalize_request_id(req_id)
            if req_norm in self._recv_loaded_once:
                continue

            base_req = self.base_request_id(req_id)
            load_errors: list[str] = []
            loaded_layers = 0
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue
                layer_kv = kv_cache_attr[forward_context.virtual_engine]

                layer_key = f"{layer_name}@{_device_tag(layer_kv)}"
                tensor_key = global_tensor_map.get(
                    layer_key, f"{base_req}#{layer_key}")
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
                    fast_fail_on_miss=False,
                    timeout_s=self._owner_lookup_grace_s,
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
                src_block_ids = request.block_ids[:expected_blocks]
                owner_lookup_hit = False
                if self._kv_owner_state_url:
                    # Prefer owner-state record; fallback to deterministic key
                    # for robustness when register/lookup races happen.
                    if rec and isinstance(rec.get("tensor_key"), str):
                        owner_lookup_hit = True
                        tensor_key = str(rec["tensor_key"])
                        # Experimental mode only: owner-state block IDs are
                        # used when a real shared block-table is enabled.
                        if self._shared_block_table_enable:
                            rec_block_ids = rec.get("block_ids")
                            if isinstance(rec_block_ids, list):
                                try:
                                    rec_ids = [int(x) for x in rec_block_ids]
                                    expected_blocks = min(
                                        expected_blocks, len(rec_ids))
                                    src_block_ids = torch.tensor(
                                        rec_ids[:expected_blocks],
                                        dtype=request.block_ids.dtype,
                                    )
                                except Exception:
                                    src_block_ids = request.block_ids[
                                        :expected_blocks]
                    else:
                        if self._zero_copy_shared_pool_mode:
                            if layer_key not in global_tensor_map:
                                logger.warning(
                                    "cuda_ipc owner lookup miss req=%s layer=%s fallback=no_global_key",
                                    base_req, layer_key)
                        else:
                            logger.warning(
                                "cuda_ipc owner lookup miss req=%s layer=%s fallback=local_key",
                                base_req, layer_key)
                elif rec and isinstance(rec.get("tensor_key"), str):
                    tensor_key = str(rec["tensor_key"])
                # Re-slice destination after any expected_blocks adjustment.
                dst_block_ids = request.block_ids[:expected_blocks]
                tensor_meta = self._wait_tensor_meta(
                    tensor_key,
                    min_tokens=min_tokens,
                    min_blocks=expected_blocks,
                    timeout_s=(None if owner_lookup_hit else self._meta_wait_fallback_s),
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
                                     req_id, layer_key, src_block_ids)
                loaded_layers += 1
            if load_errors:
                logger.warning("cuda_ipc load degraded req=%s errors=%d",
                               req_id, len(load_errors))
                if self._ipc_strict_load:
                    raise RuntimeError(
                        "cuda_ipc strict load failed: " + load_errors[0])
            # In relay path, repeatedly re-loading the same request is costly.
            # Mark as loaded once we successfully injected at least one layer.
            if loaded_layers > 0 and not load_errors:
                logger.info(
                    "cuda_ipc load complete req=%s loaded_layers=%d; posting load_ack",
                    base_req, loaded_layers)
                self._owner_load_ack(
                    request_id=base_req,
                    worker=self._decode_worker_from_req(req_id),
                    hop=1,
                )
                self._wait_resume(request_id=base_req, hop=1)
                self._recv_loaded_once.add(req_norm)
        self._gc_inflight_exports()

    def _should_send_req(self,
                         request_id: str,
                         total_tokens: Optional[int] = None) -> bool:
        base_req = self.base_request_id(request_id)
        if base_req in self._send_enabled_bases:
            return True
        if not self.has_decode_addr(request_id):
            return False
        if total_tokens is None:
            return False
        activate = self._send_activation_tokens.get(request_id)
        if activate is None:
            return False
        if int(total_tokens) >= int(activate):
            self._send_enabled_bases.add(base_req)
            return True
        return False

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
            if self._shared_kv_pool_enable:
                # Full tensor export path: avoid per-hop contiguous slicing.
                return layer
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
            if not self._shared_kv_pool_enable:
                kv_cache = kv_cache.contiguous()
            if kv_cache.dim() == 0:
                continue

            base_req = self.base_request_id(req_id)
            # Deterministic tensor key per request/layer-device.
            # Old keys are guarded by request-id uniqueness + owner-state reset.
            tensor_key = f"{base_req}#{layer_key}"
            # For full-tensor export path, kv tensor is model-owned and
            # long-lived. No need to hold per-request in-flight refs.
            if not self._shared_kv_pool_enable:
                self._inflight_exports[tensor_key] = (kv_cache, time.time())
            exported_blocks = int(valid_blocks)
            # For shared-pool full-tensor export, keep meta permissive so
            # downstream hops can reuse the same tensor meta without re-export.
            if self._shared_kv_pool_enable and kv_cache.dim() > 0:
                if (isinstance(attn_metadata, MLACommonMetadata)
                        or kv_cache.shape[1] == 2):
                    exported_blocks = max(exported_blocks, int(kv_cache.shape[0]))
                elif kv_cache.shape[0] == 2:
                    exported_blocks = max(exported_blocks, int(kv_cache.shape[1]))
            exported_block_ids = [
                int(x) for x in used_block_ids[:exported_blocks].tolist()
            ]
            publish_key = (base_req, layer_key)
            last_pub = self._last_published.get(publish_key)
            if last_pub is not None:
                last_tokens, last_blocks = last_pub
                if (int(request.num_tokens) <= int(last_tokens)
                        and int(exported_blocks) <= int(last_blocks)):
                    continue
                if (self._send_publish_token_stride > 0
                        and int(request.num_tokens) < int(last_tokens) +
                        int(self._send_publish_token_stride)
                        and int(exported_blocks) <= int(last_blocks)):
                    continue
            wrote_meta = False
            meta_path = self._meta_file_path(tensor_key)
            if self._zero_copy_shared_pool_mode and os.path.exists(meta_path):
                # Complete shared-pool fast path:
                # tensor meta was already exported by an earlier hop; do not
                # re-export imported CUDA storages in relay hops.
                wrote_meta = False
            else:
                try:
                    self._write_tensor_meta(
                        tensor_key,
                        export_cuda_tensor_meta(kv_cache),
                        num_tokens=request.num_tokens,
                        num_blocks=exported_blocks,
                    )
                    wrote_meta = True
                except RuntimeError as e:
                    # CUDA tensors imported from another process cannot be
                    # re-shared. In shared-pool mode, reuse existing tensor
                    # meta and only refresh owner-state index metadata.
                    if self._shared_kv_pool_enable and (
                            "Attempted to send CUDA tensor received from another process"
                            in str(e)):
                        warn_key = f"{tensor_key}::{req_id}"
                        if warn_key not in self._warned_skip_reexport:
                            self._warned_skip_reexport.add(warn_key)
                            logger.warning(
                                "cuda_ipc skip re-export imported tensor req=%s layer=%s key=%s",
                                req_id, layer_key, tensor_key)
                    else:
                        raise
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
            self._last_published[publish_key] = (int(request.num_tokens),
                                                 int(exported_blocks))
            if not wrote_meta and self._shared_kv_pool_enable:
                # Keep in-flight refs untouched in this path.
                pass
        self._gc_inflight_exports()
        if self._owner_flush_each_layer:
            self._owner_flush_register_kv()

    def wait_for_save(self):
        if self._zero_copy_shared_pool_mode and not self._tp_primary:
            return
        self._owner_flush_register_kv()
        return

    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs: Any) -> tuple[Optional[set[str]], Optional[set[str]]]:
        del kwargs
        if self._zero_copy_shared_pool_mode and not self._tp_primary:
            return None, None
        # Only report producer-side async sends that have actually matured.
        # Ordinary completed requests must not be surfaced as
        # "finished_sending", otherwise the scheduler treats consumer-side
        # completions as KV-transfer callbacks and logs spurious warnings.
        now = time.time()
        done_send: set[str] = set()
        for req_id in finished_req_ids:
            if (self.can_send and self.has_decode_addr(req_id)
                    and self._handoff_pin_s > 0):
                self._deferred_finished_until.setdefault(
                    req_id, now + self._handoff_pin_s)
            else:
                pass
        matured = {
            req_id
            for req_id, due in list(self._deferred_finished_until.items())
            if due <= now
        }
        for req_id in matured:
            self._deferred_finished_until.pop(req_id, None)
        done_send |= matured
        self._owner_flush_register_kv()
        self._gc_inflight_exports()
        return (done_send if done_send else None), None

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
        del blocks
        if self.can_recv and self.has_prefill_addr(
                request.request_id) and num_external_tokens > 0:
            if self._zero_copy_shared_pool_mode and not self._tp_primary:
                self._recv_loaded_once.add(
                    self.normalize_request_id(request.request_id))
                return
            self._requests_need_load[request.request_id] = (
                request, None)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = CudaIpcConnectorMetadata()
        if self._zero_copy_shared_pool_mode and not self._tp_primary:
            return meta
        seen_req_ids: set[str] = set()

        def _first_group_block_ids(
                block_groups: Optional[list[list[int]]]) -> list[int]:
            if not block_groups:
                return []
            first = block_groups[0]
            if first is None:
                return []
            return list(first)

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
            num_scheduled_tokens = (
                scheduler_output.num_scheduled_tokens)[new_req.req_id]
            num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
            if self.can_send and self.has_decode_addr(new_req.req_id):
                prompt_len = int(len(new_req.prompt_token_ids))
                max_out = int(getattr(new_req.sampling_params, "max_tokens", 0))
                activate = prompt_len + max(
                    0, max_out - self._send_activation_margin_tokens)
                self._send_activation_tokens.setdefault(new_req.req_id, activate)
            if self.can_send and self.has_decode_addr(new_req.req_id):
                self._send_req_blocks.setdefault(new_req.req_id,
                                                 list(new_req.block_ids[0]))
            if self.can_send and self._should_send_req(new_req.req_id, num_tokens):
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
                block_ids = _first_group_block_ids(new_block_ids)
                if not resumed_from_preemption:
                    block_ids = (self.chunked_prefill[req_id][0] + block_ids)
                prompt_token_ids = self.chunked_prefill[req_id][1]
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                else:
                    _add_req_once(req_id, prompt_token_ids, block_ids)
                    self.chunked_prefill.pop(req_id, None)
            if self.can_send and (self.has_decode_addr(req_id)
                                  or self.base_request_id(req_id)
                                  in self._send_enabled_bases):
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens).get(req_id, 0)
                total_tokens = int(num_computed_tokens + num_scheduled_tokens)
                full_blocks = self._send_req_blocks.get(req_id, []).copy()
                nb = _first_group_block_ids(new_block_ids)
                if resumed_from_preemption:
                    full_blocks = list(nb)
                else:
                    full_blocks.extend(nb)
                self._send_req_blocks[req_id] = full_blocks
                if self._should_send_req(req_id, total_tokens):
                    self._send_enabled_bases.add(self.base_request_id(req_id))
                if (total_tokens > 0 and full_blocks
                        and self._should_send_req(req_id, total_tokens)):
                    _add_req_once(req_id, [0] * total_tokens, full_blocks)

            if not resumed_from_preemption:
                continue
            if self.can_recv and req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = cached_reqs.num_computed_tokens[i] + 1
                token_ids = request.all_token_ids[:total_tokens]
                block_ids = _first_group_block_ids(cached_reqs.new_block_ids[i])
                _add_req_once(req_id, token_ids, block_ids)

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if (self.can_send and self._zero_copy_shared_pool_mode
                and self.has_decode_addr(request.request_id)
                and self._tp_primary):
            total_tokens = len(getattr(request, "all_token_ids", []))
            base_req = self.base_request_id(request.request_id)
            global_tensor_map = self._load_global_tensor_map()
            exported_blocks = len(block_ids)
            exported_block_ids = [int(x) for x in block_ids]
            for layer_key, tensor_key in global_tensor_map.items():
                self._owner_register_kv(
                    request_id=base_req,
                    worker=self._decode_worker_from_req(request.request_id),
                    hop=0,
                    layer_name=layer_key,
                    tensor_key=tensor_key,
                    num_tokens=total_tokens,
                    num_blocks=exported_blocks,
                    block_ids=exported_block_ids,
                )
            self._owner_flush_register_kv()
            logger.info(
                "cuda_ipc publish_done req=%s tokens=%d blocks=%d layers=%d",
                base_req, total_tokens, exported_blocks,
                len(global_tensor_map))
            self._owner_publish_done(
                request_id=base_req,
                worker=self._decode_worker_from_req(request.request_id),
                hop=1,
            )
        self.chunked_prefill.pop(request.request_id, None)
        self._send_req_blocks.pop(request.request_id, None)
        self._send_activation_tokens.pop(request.request_id, None)
        req_norm = self.normalize_request_id(request.request_id)
        if not self.can_recv:
            self._recv_loaded_once.discard(req_norm)
        base_req = self.base_request_id(request.request_id)
        self._send_enabled_bases.discard(base_req)
        for key in [k for k in self._last_published if k[0] == base_req]:
            self._last_published.pop(key, None)
        # For handoff requests on producer side, delay block free briefly.
        should_delay_free = (
            self.can_send and self.has_decode_addr(request.request_id)
            and self._handoff_pin_s > 0
        )
        return should_delay_free, None

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
                          min_blocks: int = 0,
                          timeout_s: Optional[float] = None
                          ) -> Optional[dict[str, Any]]:
        path = self._meta_file_path(tensor_key)
        wait_s = self._ipc_wait_timeout_s if timeout_s is None else max(
            0.0, float(timeout_s))
        deadline = time.time() + wait_s
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

    def should_save_kv_layer(self) -> bool:
        """Fast path for attention hook to skip save-side work entirely.

        This avoids entering per-layer export logic on producer/relay steps
        before send-side handoff is actually activated.
        """
        if self._zero_copy_shared_pool_mode and self.can_send:
            return False
        if not self.can_send:
            return False
        if self._connector_metadata is None:
            return False
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, CudaIpcConnectorMetadata)
        for request in metadata.requests:
            if self._should_send_req(request.request_id):
                return True
        return False

    def should_build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> bool:
        if self._zero_copy_shared_pool_mode and not self._tp_primary:
            return False
        if self._zero_copy_shared_pool_mode and self.can_recv and not self._tp_primary:
            return False
        if self.can_recv and self._requests_need_load:
            return True
        if self._zero_copy_shared_pool_mode and self.can_send and not self.can_recv:
            return False
        if not self.can_send:
            return False

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            if not self.has_decode_addr(req_id):
                continue
            num_scheduled_tokens = (
                scheduler_output.num_scheduled_tokens).get(req_id, 0)
            num_tokens = int(num_scheduled_tokens + new_req.num_computed_tokens)
            if self._should_send_req(req_id, num_tokens):
                return True

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if not (self.has_decode_addr(req_id)
                    or self.base_request_id(req_id) in self._send_enabled_bases):
                continue
            num_scheduled_tokens = (
                scheduler_output.num_scheduled_tokens).get(req_id, 0)
            total_tokens = int(cached_reqs.num_computed_tokens[i] +
                               num_scheduled_tokens)
            if self._should_send_req(req_id, total_tokens):
                return True

        return False

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

    def _global_tensor_map_path(self) -> str:
        return os.path.join(self._ipc_meta_dir, "global_layer_map.pkl")

    def _load_global_tensor_map(self) -> dict[str, str]:
        if self._global_tensor_map_cache is not None:
            return self._global_tensor_map_cache

        merged: dict[str, str] = {}
        for path in sorted(
                glob.glob(
                    os.path.join(self._ipc_meta_dir, "global_layer_map*.pkl"))):
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(k, str) and isinstance(v, str):
                            merged[k] = v
            except Exception:
                continue
        self._global_tensor_map_cache = merged
        return merged

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
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None

    def _owner_publish_done(self, request_id: str, worker: str,
                            hop: int) -> None:
        self._owner_post_json(
            "/publish_done",
            {
                "request_id": request_id,
                "worker": worker,
                "hop": int(hop),
            },
        )

    def _owner_load_ack(self, request_id: str, worker: str, hop: int) -> None:
        self._owner_post_json(
            "/load_ack",
            {
                "request_id": request_id,
                "worker": worker,
                "hop": int(hop),
            },
        )

    def _wait_resume(self, request_id: str, hop: int) -> None:
        if not self._kv_owner_state_url:
            return
        if self._zero_copy_shared_pool_mode:
            return
        deadline = time.time() + self._resume_wait_timeout_s
        url = self._kv_owner_state_url.rstrip("/") + f"/state/{request_id}"
        logger.info("cuda_ipc wait resume req=%s hop>=%s timeout=%.1fs",
                    request_id, hop, self._resume_wait_timeout_s)
        last_obj: Optional[dict[str, Any]] = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        url, timeout=self._kv_owner_state_timeout_s) as resp:
                    body = resp.read()
                obj = json.loads(body.decode("utf-8")) if body else {}
                if isinstance(obj, dict):
                    last_obj = obj
                if (isinstance(obj, dict) and bool(obj.get("ok", False))
                        and int(obj.get("resume_hop", 0)) >= int(hop)):
                    logger.info("cuda_ipc resume satisfied req=%s state=%s",
                                request_id, obj)
                    return
                if (self._zero_copy_shared_pool_mode
                        and isinstance(obj, dict)
                        and bool(obj.get("ok", False))
                        and int(obj.get("publish_done_hop", 0)) >= int(hop)
                        and int(obj.get("load_ack_hop", 0)) >= int(hop)):
                    try:
                        self._owner_post_json(
                            "/resume",
                            {
                                "request_id": request_id,
                                "worker": self._local_worker_addr,
                                "hop": int(hop),
                            },
                        )
                    except Exception:
                        pass
                    logger.warning(
                        "cuda_ipc resume missing; auto-continue req=%s hop=%s state=%s",
                        request_id, hop, obj)
                    return
            except Exception:
                pass
            time.sleep(self._ipc_poll_interval_s)
        raise RuntimeError(
            f"cuda_ipc resume timeout req={request_id} hop={hop} state={last_obj}"
        )

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
        payload = {
            "request_id": request_id,
            "worker": worker,
            "hop": int(hop),
            "layer_name": layer_name,
            "tensor_key": tensor_key,
            "num_tokens": int(num_tokens),
            "num_blocks": int(num_blocks),
            "block_ids": [int(x) for x in block_ids],
        }
        if self._owner_batch_enabled:
            self._owner_register_batch.append(payload)
            return
        _ = self._owner_post_json("/register_kv", payload)

    def _owner_flush_register_kv(self) -> None:
        if not self._owner_register_batch:
            return
        if not self._owner_batch_enabled:
            self._owner_register_batch.clear()
            return
        pending = list(self._owner_register_batch)
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for item in pending:
            k = (str(item.get("request_id", "")), str(item.get("layer_name", "")))
            prev = by_key.get(k)
            if prev is None:
                by_key[k] = item
                continue
            if int(item.get("num_blocks", 0)) > int(prev.get("num_blocks", 0)):
                by_key[k] = item
                continue
            if int(item.get("num_tokens", 0)) > int(prev.get("num_tokens", 0)):
                by_key[k] = item
        payload = {"items": list(by_key.values())}
        obj = self._owner_post_json("/register_kv_batch", payload)
        if not obj:
            # Compatibility fallback when owner does not expose batch API yet.
            for item in by_key.values():
                _ = self._owner_post_json("/register_kv", item)
        self._owner_register_batch.clear()

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
        fast_fail_on_miss: bool = True,
        timeout_s: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        if not self._kv_owner_state_url:
            rec, _ = self._owner_lookup_kv(request_id, layer_name, min_tokens,
                                           min_blocks)
            return rec
        wait_s = self._ipc_wait_timeout_s if timeout_s is None else max(
            0.0, float(timeout_s))
        deadline = time.time() + wait_s
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
            elif fast_fail_on_miss and err in {"unknown_request", "unknown_layer", "empty"}:
                # Fast fallback path: do not block decode on control-plane miss.
                # Data-plane can still recover via deterministic local tensor key.
                return None
            time.sleep(self._ipc_poll_interval_s)
        if last_partial is not None:
            logger.warning(
                "cuda_ipc owner lookup fallback_partial req=%s layer=%s min_tokens=%d min_blocks=%d rec_tokens=%s rec_blocks=%s",
                request_id, layer_name, min_tokens, min_blocks,
                str(last_partial.get("num_tokens")),
                str(last_partial.get("num_blocks")))
            return last_partial
        return None
