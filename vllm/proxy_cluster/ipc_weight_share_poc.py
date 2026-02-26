from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.proxy_cluster.ipc_state_dict import (
    export_cuda_tensor_meta,
    rebuild_cuda_tensor_from_meta,
)


@dataclass
class WorkerStats:
    server_id: int
    handled: int = 0
    storage_handle: bytes | None = None


def _storage_handle(t: torch.Tensor) -> bytes:
    # tuple: (device, handle, storage_size_bytes, storage_offset_bytes, ...)
    return t.untyped_storage()._share_cuda_()[1]


def _server_owner(
    server_id: int,
    consumer_init_qs: list[mp.Queue],
    task_q: mp.Queue,
    result_q: mp.Queue,
    in_features: int,
    out_features: int,
    dtype_name: str,
    seed: int,
) -> None:
    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    dtype = getattr(torch, dtype_name)
    with torch.inference_mode():
        weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
        bias = torch.randn(out_features, device="cuda", dtype=dtype)
    weight_meta = export_cuda_tensor_meta(weight)
    bias_meta = export_cuda_tensor_meta(bias)
    for q in consumer_init_qs:
        q.put({"kind": "weights_ipc_meta", "weight_meta": weight_meta, "bias_meta": bias_meta})
    result_q.put(
        {
            "kind": "ready",
            "server_id": server_id,
            "storage_handle_hex": _storage_handle(weight).hex(),
            "owner": True,
        }
    )
    _serve_loop(server_id, weight, bias, task_q, result_q)


def _server_consumer(
    server_id: int,
    init_q: mp.Queue,
    task_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    torch.cuda.set_device(0)
    msg = init_q.get()
    assert msg["kind"] == "weights_ipc_meta"
    weight_meta = msg["weight_meta"]
    weight: torch.Tensor = _rebuild_cuda_tensor_from_meta(weight_meta)
    bias: torch.Tensor = _rebuild_cuda_tensor_from_meta(msg["bias_meta"])
    result_q.put(
        {
            "kind": "ready",
            "server_id": server_id,
            "storage_handle_hex": weight_meta["storage_handle_hex"],
            "owner": False,
        }
    )
    _serve_loop(server_id, weight, bias, task_q, result_q)


def _serve_loop(
    server_id: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    task_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    while True:
        msg = task_q.get()
        if msg["kind"] == "stop":
            result_q.put({"kind": "stopped", "server_id": server_id})
            return
        if msg["kind"] == "probe":
            result_q.put(
                {
                    "kind": "probe_result",
                    "server_id": server_id,
                    "tag": msg.get("tag", ""),
                    "probe_value": float(weight.view(-1)[0].item()),
                }
            )
            continue
        if msg["kind"] == "mutate_owner_weight":
            if server_id == 1:
                delta = float(msg.get("delta", 1.0))
                with torch.inference_mode():
                    weight.view(-1)[0].add_(delta)
                torch.cuda.synchronize()
                result_q.put({"kind": "mutated", "server_id": server_id, "delta": delta})
            continue
        if msg["kind"] != "infer":
            continue

        req_id: int = msg["req_id"]
        decode_steps: int = msg["decode_steps"]
        x_cpu: torch.Tensor = msg["x"]
        x = x_cpu.to(device="cuda", dtype=weight.dtype, non_blocking=True)
        with torch.inference_mode():
            y = x
            for _ in range(decode_steps):
                y = torch.matmul(y, weight.t()) + bias
                y = torch.tanh(y)

        result_q.put(
            {
                "kind": "result",
                "server_id": server_id,
                "req_id": req_id,
                "sum": float(y.sum().item()),
            }
        )


def run_ipc_cluster(
    num_requests: int = 512,
    decode_steps: int = 8,
    in_features: int = 64,
    out_features: int = 64,
    dtype_name: str = "float16",
    seed: int = 42,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This PoC requires CUDA to validate IPC sharing.")

    mp.set_start_method("spawn", force=True)
    random.seed(seed)

    init_q_s2: mp.Queue = mp.Queue()
    init_q_s3: mp.Queue = mp.Queue()
    init_q_s4: mp.Queue = mp.Queue()
    task_qs = {sid: mp.Queue() for sid in (1, 2, 3, 4)}
    result_q: mp.Queue = mp.Queue()

    procs = [
        mp.Process(
            target=_server_owner,
            args=(1, [init_q_s2, init_q_s3, init_q_s4], task_qs[1], result_q, in_features, out_features, dtype_name, seed),
            daemon=True,
        ),
        mp.Process(target=_server_consumer, args=(2, init_q_s2, task_qs[2], result_q), daemon=True),
        mp.Process(target=_server_consumer, args=(3, init_q_s3, task_qs[3], result_q), daemon=True),
        mp.Process(target=_server_consumer, args=(4, init_q_s4, task_qs[4], result_q), daemon=True),
    ]
    for p in procs:
        p.start()

    # Wait ready messages from all servers and capture handles.
    handles: dict[int, str] = {}
    for _ in range(4):
        ready = result_q.get(timeout=60)
        assert ready["kind"] == "ready"
        handles[ready["server_id"]] = ready["storage_handle_hex"]

    for sid in (1, 2, 3, 4):
        task_qs[sid].put({"kind": "probe", "tag": "before"})
    probe_before: dict[int, float] = {}
    while len(probe_before) < 4:
        msg = result_q.get(timeout=60)
        if msg["kind"] == "probe_result" and msg.get("tag") == "before":
            probe_before[msg["server_id"]] = float(msg["probe_value"])

    delta = 1.0
    task_qs[1].put({"kind": "mutate_owner_weight", "delta": delta})
    while True:
        msg = result_q.get(timeout=60)
        if msg["kind"] == "mutated" and msg["server_id"] == 1:
            break

    for sid in (1, 2, 3, 4):
        task_qs[sid].put({"kind": "probe", "tag": "after"})
    probe_after: dict[int, float] = {}
    while len(probe_after) < 4:
        msg = result_q.get(timeout=60)
        if msg["kind"] == "probe_result" and msg.get("tag") == "after":
            probe_after[msg["server_id"]] = float(msg["probe_value"])

    t0 = time.perf_counter()
    for req_id in range(num_requests):
        sid = random.randint(1, 4)
        x = torch.randn(1, in_features)
        task_qs[sid].put({"kind": "infer", "req_id": req_id, "decode_steps": decode_steps, "x": x})

    stats = {sid: WorkerStats(server_id=sid) for sid in (1, 2, 3, 4)}
    received = 0
    while received < num_requests:
        try:
            msg = result_q.get(timeout=120)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out while waiting for results ({received}/{num_requests})") from exc
        if msg["kind"] != "result":
            continue
        sid = msg["server_id"]
        stats[sid].handled += 1
        received += 1

    elapsed = time.perf_counter() - t0

    for sid in (1, 2, 3, 4):
        task_qs[sid].put({"kind": "stop"})
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    all_handles_same = len(set(handles.values())) == 1
    all_probe_shifted = all(abs((probe_after[sid] - probe_before[sid]) - delta) < 1e-3 for sid in (1, 2, 3, 4))
    return {
        "num_requests": num_requests,
        "decode_steps": decode_steps,
        "elapsed_s": elapsed,
        "throughput_req_s": num_requests / elapsed if elapsed > 0 else 0.0,
        "handles": handles,
        "all_handles_same": all_handles_same,
        "probe_before": probe_before,
        "probe_after": probe_after,
        "all_probe_shifted": all_probe_shifted,
        "per_server_handled": {sid: stats[sid].handled for sid in (1, 2, 3, 4)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PoC: 4-process servers sharing one CUDA weight storage via IPC.")
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_ipc_cluster(
        num_requests=args.num_requests,
        decode_steps=args.decode_steps,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype_name=args.dtype,
        seed=args.seed,
    )
    print("\nIPC PoC Report")
    print(f"- num_requests: {report['num_requests']}")
    print(f"- decode_steps: {report['decode_steps']}")
    print(f"- elapsed_s: {report['elapsed_s']:.3f}")
    print(f"- throughput_req_s: {report['throughput_req_s']:.2f}")
    print(f"- all_handles_same: {report['all_handles_same']}")
    print(f"- all_probe_shifted: {report['all_probe_shifted']}")
    print(f"- per_server_handled: {report['per_server_handled']}")
    print("- handles:")
    for sid, h in sorted(report["handles"].items()):
        print(f"  server-{sid}: {h[:24]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
