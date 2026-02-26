# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import pickle
import time
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)
from vllm.proxy_cluster.ipc_state_dict import (export_module_ipc_meta,
                                               load_module_from_ipc_meta)

logger = init_logger(__name__)


def _find_meta_tensor_attrs(module: nn.Module) -> list[str]:
    names: list[str] = []
    for module_name, submodule in module.named_modules():
        param_names = set(submodule._parameters.keys())
        buffer_names = set(submodule._buffers.keys())
        for attr_name, value in vars(submodule).items():
            if attr_name in param_names or attr_name in buffer_names:
                continue
            if isinstance(value, torch.Tensor) and value.device.type == "meta":
                names.append(f"{module_name}.{attr_name}" if module_name else attr_name)
    return names


class IPCWeightShareModelLoader(BaseModelLoader):
    """Experimental loader: share CUDA weights via IPC meta file.

    Expected model_loader_extra_config keys:
    - ipc_role: "owner" | "consumer"
    - ipc_meta_path: filesystem path used for owner->consumer IPC meta handoff
    - ipc_wait_timeout_s: optional, default 600
    - ipc_poll_interval_s: optional, default 0.5
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        allowed = {
            "ipc_role",
            "ipc_meta_path",
            "ipc_wait_timeout_s",
            "ipc_poll_interval_s",
            # passthrough keys for DefaultModelLoader when role=owner
            "enable_multithread_load",
            "num_threads",
        }
        unexpected = set(extra.keys()) - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected extra config keys for load format {load_config.load_format}: {unexpected}"
            )

    @property
    def _extra(self) -> dict[str, Any]:
        return self.load_config.model_loader_extra_config or {}

    def _role(self) -> str:
        role = str(self._extra.get("ipc_role", "owner")).strip().lower()
        if role not in {"owner", "consumer"}:
            raise ValueError(f"ipc_role must be owner|consumer, got {role!r}")
        return role

    def _meta_path(self) -> str:
        path = str(self._extra.get("ipc_meta_path", "")).strip()
        if not path:
            raise ValueError("ipc_meta_path is required for ipc_weight_share loader")
        return path

    def _wait_timeout_s(self) -> float:
        return float(self._extra.get("ipc_wait_timeout_s", 600))

    def _poll_interval_s(self) -> float:
        return float(self._extra.get("ipc_poll_interval_s", 0.5))

    def _write_meta_file(self, meta_path: str, payload: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        tmp = f"{meta_path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, meta_path)

    def _wait_and_read_meta_file(self, meta_path: str) -> dict[str, Any]:
        deadline = time.time() + self._wait_timeout_s()
        while time.time() < deadline:
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    return pickle.load(f)
            time.sleep(self._poll_interval_s())
        raise TimeoutError(f"Timed out waiting for ipc meta file: {meta_path}")

    def download_model(self, model_config: ModelConfig) -> None:
        # Owner needs local weights. Consumer only needs model config/arch info.
        if self._role() == "owner":
            DefaultModelLoader(self._owner_default_load_config()).download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # This method is unused because this class overrides load_model.
        raise NotImplementedError("IPCWeightShareModelLoader uses custom load_model")

    def _owner_default_load_config(self) -> LoadConfig:
        """Build a sanitized LoadConfig for DefaultModelLoader.

        DefaultModelLoader does not understand ipc_* extra keys and also
        requires a standard load_format (e.g. auto/safetensors/pt).
        """
        cfg = self.load_config
        extra = cfg.model_loader_extra_config or {}
        default_extra = {}
        for k in ("enable_multithread_load", "num_threads"):
            if k in extra:
                default_extra[k] = extra[k]
        return LoadConfig(
            load_format="auto",
            download_dir=cfg.download_dir,
            safetensors_load_strategy=cfg.safetensors_load_strategy,
            model_loader_extra_config=default_extra,
            device=cfg.device,
            ignore_patterns=cfg.ignore_patterns,
            use_tqdm_on_load=cfg.use_tqdm_on_load,
            pt_load_map_location=cfg.pt_load_map_location,
        )

    def _load_owner_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> nn.Module:
        model = DefaultModelLoader(self._owner_default_load_config()).load_model(vllm_config, model_config)
        meta_path = self._meta_path()
        meta = export_module_ipc_meta(model)
        payload = {
            "model_name": model_config.model,
            "created_at": time.time(),
            "param_count": len(meta.get("params", {})),
            "buffer_count": len(meta.get("buffers", {})),
            "attr_count": len(meta.get("attrs", {})),
            "meta": meta,
        }
        self._write_meta_file(meta_path, payload)
        logger.info("ipc_weight_share owner exported meta file to %s (params=%s, buffers=%s, attrs=%s)",
                    meta_path, payload["param_count"], payload["buffer_count"], payload["attr_count"])
        return model

    def _load_consumer_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> nn.Module:
        load_config = vllm_config.load_config
        load_device = vllm_config.device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)

        meta_path = self._meta_path()
        payload = self._wait_and_read_meta_file(meta_path)
        module_meta = payload.get("meta", {})

        # Build model on meta device to avoid duplicate CUDA parameter allocation.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = initialize_model(vllm_config=vllm_config, model_config=model_config)

        stats = load_module_from_ipc_meta(model, module_meta)
        meta_params = [n for n, p in model.named_parameters(recurse=True) if p is not None and p.device.type == "meta"]
        meta_buffers = [n for n, b in model.named_buffers(recurse=True) if b is not None and b.device.type == "meta"]
        meta_attrs = _find_meta_tensor_attrs(model)
        if meta_params or meta_buffers or meta_attrs:
            sample_p = ",".join(meta_params[:8])
            sample_b = ",".join(meta_buffers[:8])
            sample_a = ",".join(meta_attrs[:8])
            raise RuntimeError(
                "ipc_weight_share consumer still has meta tensors after IPC load: "
                f"meta_params={len(meta_params)}[{sample_p}] "
                f"meta_buffers={len(meta_buffers)}[{sample_b}] "
                f"meta_attrs={len(meta_attrs)}[{sample_a}]"
            )
        process_weights_after_loading(model, model_config, target_device)
        logger.info("ipc_weight_share consumer loaded from %s (loaded_params=%s, loaded_buffers=%s, loaded_attrs=%s)",
                    meta_path, stats["loaded_param_count"], stats["loaded_buffer_count"],
                    stats.get("loaded_attr_count", "0"))
        return model.eval()

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> nn.Module:
        role = self._role()
        if role == "owner":
            return self._load_owner_model(vllm_config, model_config)
        return self._load_consumer_model(vllm_config, model_config)
