from __future__ import annotations

from typing import Any

import torch
import torch.multiprocessing.reductions as mp_reductions


def _resolve_parent_and_leaf(module: torch.nn.Module,
                             fq_name: str) -> tuple[torch.nn.Module, str]:
    parts = fq_name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _replace_parameter(module: torch.nn.Module, name: str,
                       rebuilt: torch.Tensor,
                       old_param: torch.nn.Parameter) -> None:
    parent, leaf = _resolve_parent_and_leaf(module, name)
    # For meta-initialized parameters, assigning .data to a CUDA tensor fails.
    # Recreate a parameter object and keep old python-side attributes when possible.
    cls = old_param.__class__
    try:
        new_param = cls.__new__(cls, rebuilt)
        if hasattr(old_param, "__dict__"):
            new_param.__dict__.update(old_param.__dict__)
    except Exception:
        new_param = torch.nn.Parameter(
            rebuilt, requires_grad=bool(old_param.requires_grad))
    parent._parameters[leaf] = new_param


def _replace_buffer(module: torch.nn.Module, name: str,
                    rebuilt: torch.Tensor) -> None:
    parent, leaf = _resolve_parent_and_leaf(module, name)
    parent._buffers[leaf] = rebuilt


def export_cuda_tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.device.type != "cuda":
        raise ValueError("export_cuda_tensor_meta requires a CUDA tensor")

    storage = tensor._typed_storage()
    (
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()

    return {
        "tensor_cls": type(tensor),
        "tensor_size": tuple(tensor.size()),
        "tensor_stride": tuple(tensor.stride()),
        "tensor_offset": tensor.storage_offset(),
        "storage_cls": type(storage),
        "dtype": tensor.dtype,
        "storage_device": device,
        "storage_handle": handle,
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "requires_grad": bool(tensor.requires_grad),
        "ref_counter_handle": ref_counter_handle,
        "ref_counter_offset": ref_counter_offset,
        "event_handle": event_handle,
        "event_sync_required": event_sync_required,
        "storage_handle_hex": handle.hex() if handle is not None else "",
    }


def rebuild_cuda_tensor_from_meta(meta: dict[str, Any]) -> torch.Tensor:
    return mp_reductions.rebuild_cuda_tensor(
        meta["tensor_cls"],
        meta["tensor_size"],
        meta["tensor_stride"],
        meta["tensor_offset"],
        meta["storage_cls"],
        meta["dtype"],
        meta["storage_device"],
        meta["storage_handle"],
        meta["storage_size_bytes"],
        meta["storage_offset_bytes"],
        meta["requires_grad"],
        meta["ref_counter_handle"],
        meta["ref_counter_offset"],
        meta["event_handle"],
        meta["event_sync_required"],
    )


def export_module_ipc_meta(module: torch.nn.Module) -> dict[str, Any]:
    params: dict[str, dict[str, Any]] = {}
    buffers: dict[str, dict[str, Any]] = {}
    attrs: dict[str, dict[str, Any]] = {}

    for name, p in module.named_parameters(recurse=True):
        if p is None:
            continue
        params[name] = export_cuda_tensor_meta(p.data)

    for name, b in module.named_buffers(recurse=True):
        if b is None:
            continue
        if b.device.type != "cuda":
            continue
        buffers[name] = export_cuda_tensor_meta(b)

    for module_name, submodule in module.named_modules():
        param_names = set(submodule._parameters.keys())
        buffer_names = set(submodule._buffers.keys())
        for attr_name, value in vars(submodule).items():
            if attr_name in param_names or attr_name in buffer_names:
                continue
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                fq_name = f"{module_name}.{attr_name}" if module_name else attr_name
                attrs[fq_name] = export_cuda_tensor_meta(value)

    return {"params": params, "buffers": buffers, "attrs": attrs}


def load_module_from_ipc_meta(module: torch.nn.Module, meta: dict[str, Any]) -> dict[str, str]:
    loaded_params: dict[str, str] = {}
    loaded_buffers: dict[str, str] = {}
    loaded_attrs: dict[str, str] = {}

    param_dict = dict(module.named_parameters(recurse=True))
    for name, tmeta in meta.get("params", {}).items():
        if name not in param_dict:
            continue
        rebuilt = rebuild_cuda_tensor_from_meta(tmeta)
        target = param_dict[name]
        if tuple(target.shape) != tuple(rebuilt.shape):
            raise ValueError(f"Parameter shape mismatch for {name}: {tuple(target.shape)} != {tuple(rebuilt.shape)}")
        if target.device.type == "meta":
            _replace_parameter(module, name, rebuilt, target)
        else:
            target.data = rebuilt
        loaded_params[name] = tmeta.get("storage_handle_hex", "")

    buffer_dict = dict(module.named_buffers(recurse=True))
    for name, tmeta in meta.get("buffers", {}).items():
        if name not in buffer_dict:
            continue
        rebuilt = rebuild_cuda_tensor_from_meta(tmeta)
        target = buffer_dict[name]
        if tuple(target.shape) != tuple(rebuilt.shape):
            raise ValueError(f"Buffer shape mismatch for {name}: {tuple(target.shape)} != {tuple(rebuilt.shape)}")
        if target.device.type == "meta":
            _replace_buffer(module, name, rebuilt)
        else:
            target.data = rebuilt
        loaded_buffers[name] = tmeta.get("storage_handle_hex", "")

    module_dict = dict(module.named_modules())
    for fq_name, tmeta in meta.get("attrs", {}).items():
        if "." in fq_name:
            module_name, attr_name = fq_name.rsplit(".", 1)
        else:
            module_name, attr_name = "", fq_name
        submodule = module_dict.get(module_name)
        if submodule is None or not hasattr(submodule, attr_name):
            continue
        current = getattr(submodule, attr_name)
        if not isinstance(current, torch.Tensor):
            continue
        rebuilt = rebuild_cuda_tensor_from_meta(tmeta)
        if tuple(current.shape) != tuple(rebuilt.shape):
            raise ValueError(f"Attr tensor shape mismatch for {fq_name}: {tuple(current.shape)} != {tuple(rebuilt.shape)}")
        setattr(submodule, attr_name, rebuilt)
        loaded_attrs[fq_name] = tmeta.get("storage_handle_hex", "")

    return {
        "loaded_param_count": str(len(loaded_params)),
        "loaded_buffer_count": str(len(loaded_buffers)),
        "loaded_attr_count": str(len(loaded_attrs)),
    }
