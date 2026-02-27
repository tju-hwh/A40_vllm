# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import os
import pickle
import time
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
        transfer_config = vllm_config.kv_transfer_config
        self.can_send = transfer_config.is_kv_producer
        self.can_recv = transfer_config.is_kv_consumer

        self._ipc_meta_dir = transfer_config.get_from_extra_config(
            "ipc_meta_dir", "/tmp/vllm_kv_ipc")
        self._ipc_wait_timeout_s = float(
            transfer_config.get_from_extra_config("ipc_wait_timeout_s", 10.0))
        self._ipc_poll_interval_s = float(
            transfer_config.get_from_extra_config("ipc_poll_interval_s", 0.01))
        os.makedirs(self._ipc_meta_dir, exist_ok=True)

        # Keep exported tensors alive while peer imports IPC handles.
        self._inflight_exports: dict[str, torch.Tensor] = {}

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

        def inject_kv_into_layer(layer: torch.Tensor, kv_cache: torch.Tensor,
                                 block_ids: torch.Tensor,
                                 request_id: str) -> None:
            if (isinstance(attn_metadata, MLACommonMetadata)
                    or layer.shape[1] == 2):
                num_block = kv_cache.shape[0]
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    layer[block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "cuda_ipc kv_cache mismatch block_ids=%d num_block=%d req=%s",
                        len(block_ids), num_block, request_id)
            elif layer.shape[0] == 2:
                num_block = kv_cache.shape[1]
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    layer[:, block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "cuda_ipc kv_cache mismatch block_ids=%d num_block=%d req=%s",
                        len(block_ids), num_block, request_id)

        for request in metadata.requests:
            req_id = request.request_id
            if not self.has_prefill_addr(req_id):
                continue

            base_req = self.base_request_id(req_id)
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue
                layer_kv = kv_cache_attr[forward_context.virtual_engine]

                tensor_key = f"{base_req}#{layer_name}"
                tensor_meta = self._wait_tensor_meta(tensor_key)
                if tensor_meta is None:
                    logger.warning("cuda_ipc missing tensor meta: %s", tensor_key)
                    continue

                remote_kv = rebuild_cuda_tensor_from_meta(tensor_meta)
                inject_kv_into_layer(layer_kv, remote_kv, request.block_ids,
                                     req_id)

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

        for request in connector_metadata.requests:
            req_id = request.request_id
            if not self.has_decode_addr(req_id):
                continue

            kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            if not kv_cache.is_cuda:
                continue
            kv_cache = kv_cache.contiguous()

            base_req = self.base_request_id(req_id)
            tensor_key = f"{base_req}#{layer_name}"
            self._inflight_exports[tensor_key] = kv_cache
            self._write_tensor_meta(tensor_key, export_cuda_tensor_meta(kv_cache))

    def wait_for_save(self):
        return

    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs: Any) -> tuple[Optional[set[str]], Optional[set[str]]]:
        del kwargs
        # Best-effort cleanup for exported references.
        for req_id in finished_req_ids:
            base = self.base_request_id(req_id)
            prefix = f"{base}#"
            to_del = [k for k in self._inflight_exports if k.startswith(prefix)]
            for key in to_del:
                self._inflight_exports.pop(key, None)
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

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.can_send and self.has_decode_addr(new_req.req_id):
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
            if self.can_recv and new_req.req_id in self._requests_need_load:
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
                self._requests_need_load.pop(new_req.req_id, None)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]
            if not resumed_from_preemption:
                break
            if self.can_recv and req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = cached_reqs.num_computed_tokens[i] + 1
                token_ids = request.all_token_ids[:total_tokens]
                block_ids = cached_reqs.new_block_ids[i][0]
                meta.add_request(
                    request_id=req_id,
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        del request, block_ids
        return False, None

    # ==============================
    # Helpers
    # ==============================

    @staticmethod
    def normalize_request_id(request_id: str) -> str:
        return re.sub(r"-\d+$", "", request_id)

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

    def _write_tensor_meta(self, tensor_key: str, tensor_meta: dict[str, Any]) -> None:
        path = self._meta_file_path(tensor_key)
        payload = {
            "tensor_key": tensor_key,
            "created_at": time.time(),
            "meta": tensor_meta,
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _wait_tensor_meta(self, tensor_key: str) -> Optional[dict[str, Any]]:
        path = self._meta_file_path(tensor_key)
        deadline = time.time() + self._ipc_wait_timeout_s
        while time.time() < deadline:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        payload = pickle.load(f)
                    if payload.get("tensor_key") == tensor_key:
                        return payload.get("meta")
                except Exception:
                    pass
            time.sleep(self._ipc_poll_interval_s)
        return None
