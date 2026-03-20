#!/usr/bin/env python3
"""Launch multi-server cluster with DP+TP support for hop-vllm.

Each DP rank runs an independent TP group with server1 (owner) and server2 (shadow).
Requests are routed to a specific DP rank, and handoff happens within that TP group.

Example DP=2 TP=2 on 4 GPUs:
  DP Rank 0: GPUs 0,1  -> server1:8101, server2:8102
  DP Rank 1: GPUs 2,3  -> server1:8103, server2:8104
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _start(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    """Start a process in its own process group."""
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def _stop_all(children: list[subprocess.Popen]) -> None:
    for p in children:
        _stop_proc(p, timeout_s=15.0)


def _stop_proc(proc: subprocess.Popen, timeout_s: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            proc.kill()


def _build_ipc_extra_config(role: str, meta_path: str, timeout_s: float, poll_s: float) -> str:
    return json.dumps(
        {
            "ipc_role": role,
            "ipc_meta_path": meta_path,
            "ipc_wait_timeout_s": timeout_s,
            "ipc_poll_interval_s": poll_s,
        },
        separators=(",", ":"),
    )


def _build_vllm_cmd(
    host: str,
    port: int,
    model: str,
    role: str,
    meta_path: str,
    timeout_s: float,
    poll_s: float,
    extra_args: list[str],
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
    enforce_eager: bool | None = None,
    compilation_config: str | None = None,
    tensor_parallel_size: int | None = None,
    kv_transfer_config: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--load-format",
        "ipc_weight_share",
        "--model-loader-extra-config",
        _build_ipc_extra_config(role, meta_path, timeout_s, poll_s),
    ]
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if compilation_config:
        cmd += ["--compilation-config", compilation_config]
    if tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if kv_transfer_config:
        cmd += ["--kv-transfer-config", kv_transfer_config]
    cmd += extra_args
    return cmd


def _strip_overridden_args(args: list[str], flags: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token in flags:
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        matched_eq = False
        for f in flags:
            if token.startswith(f + "="):
                matched_eq = True
                break
        if matched_eq:
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _wait_server_ready(url: str, timeout_s: float) -> None:
    """Wait until server /v1/models is ready."""
    deadline = time.time() + timeout_s
    health_url = url.rstrip("/") + "/v1/models"
    host, port = url.rsplit(":", 1)
    host = host.split("://", 1)[-1]
    port_i = int(port)

    last_err = ""
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port_i), timeout=1.0):
                pass
        except OSError as e:
            last_err = f"tcp not ready: {e}"
            time.sleep(0.5)
            continue

        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return
                last_err = f"http status={resp.status}"
        except urllib.error.URLError as e:
            last_err = f"http not ready: {e}"
        time.sleep(0.5)
    raise TimeoutError(f"Server not ready within {timeout_s}s ({last_err})")


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _resolve_kv_transfer_config(
    raw: str | None,
    port: int,
    kv_owner_state_url: str = "",
    shared_kv_pool_enable: bool = False,
    shared_kv_pool_meta_path: str = "",
    shared_kv_pool_wait_timeout_s: float = 300.0,
    shared_kv_pool_poll_s: float = 0.1,
    shared_kv_pool_role: str = "",
    send_activation_margin_tokens: int | None = None,
    send_publish_token_stride: int | None = None,
    owner_flush_each_layer: bool | None = None,
) -> str | None:
    if not raw:
        return None
    replaced = raw.replace("{port}", str(port))
    try:
        obj = json.loads(replaced)
    except Exception:
        return replaced

    if not isinstance(obj, dict):
        return replaced
    connector = obj.get("kv_connector")
    if connector not in {"P2pNcclConnector", "CudaIpcConnector"}:
        return replaced

    extra = obj.get("kv_connector_extra_config")
    if not isinstance(extra, dict):
        extra = {}
        obj["kv_connector_extra_config"] = extra
    if connector == "P2pNcclConnector":
        extra.setdefault("http_port", int(port))
    if connector == "CudaIpcConnector" and kv_owner_state_url:
        extra.setdefault("kv_owner_state_url", kv_owner_state_url)
    if connector == "CudaIpcConnector" and shared_kv_pool_enable:
        extra.setdefault("shared_kv_pool_enable", True)
        extra.setdefault("shared_block_table_enable", True)
        if shared_kv_pool_role:
            extra.setdefault("shared_kv_pool_role", shared_kv_pool_role)
        if shared_kv_pool_meta_path:
            extra.setdefault("shared_kv_pool_meta_path", shared_kv_pool_meta_path)
        extra.setdefault("shared_kv_pool_wait_timeout_s", float(shared_kv_pool_wait_timeout_s))
        extra.setdefault("shared_kv_pool_poll_s", float(shared_kv_pool_poll_s))
    if connector == "CudaIpcConnector" and send_activation_margin_tokens is not None:
        extra.setdefault("send_activation_margin_tokens", int(send_activation_margin_tokens))
    if connector == "CudaIpcConnector" and send_publish_token_stride is not None:
        extra.setdefault("send_publish_token_stride", int(send_publish_token_stride))
    if connector == "CudaIpcConnector" and owner_flush_each_layer is not None:
        extra.setdefault("owner_flush_each_layer", bool(owner_flush_each_layer))
    return json.dumps(obj, separators=(",", ":"))


def _get_gpu_slice(dp_rank: int, tp_size: int, all_gpus: list[int]) -> list[int]:
    """Get GPU indices for a specific DP rank."""
    start_idx = dp_rank * tp_size
    end_idx = start_idx + tp_size
    return all_gpus[start_idx:end_idx]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch DP x TP vLLM servers with shadow consumers for hop handoff."
    )
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")

    # Parallelism config
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of DP ranks (each rank has independent TP group with owner+consumer)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size within each DP rank",
    )

    # Base ports - ports are allocated as:
    #   DP rank r: server1_port = base + 2*r, server2_port = base + 2*r + 1
    parser.add_argument(
        "--server1-base-port",
        type=int,
        default=8101,
        help="Base port for server1 (owner) of DP rank 0",
    )
    parser.add_argument(
        "--server2-base-port",
        type=int,
        default=8102,
        help="Base port for server2 (consumer) of DP rank 0",
    )
    parser.add_argument(
        "--server-kv-base-port",
        type=int,
        default=18101,
        help="Base KV port for connectors",
    )

    # GPU config
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of all GPU IDs to use",
    )

    # Owner (server1) config
    parser.add_argument("--owner-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--owner-max-num-seqs", type=int, default=128)
    parser.add_argument("--owner-max-model-len", type=int, default=None)
    parser.add_argument("--owner-compilation-config", default="")
    parser.add_argument("--owner-enforce-eager", action="store_true")

    # Consumer (server2) config
    parser.add_argument("--consumer-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--consumer-max-num-seqs", type=int, default=128)
    parser.add_argument("--consumer-max-model-len", type=int, default=None)
    parser.add_argument("--consumer-compilation-config", default='{"level":3,"use_inductor":true,"use_cudagraph":true}')
    parser.add_argument("--consumer-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--no-consumer-enforce-eager", action="store_true")

    # Shared KV pool config
    parser.add_argument("--shared-kv-pool-enable", action="store_true", help="Enable shared KV pool")
    parser.add_argument(
        "--shared-kv-pool-meta-path",
        default="/tmp/vllm_shared_kv_pool",
        help="Base path for shared KV pool metadata (will have .dp{N} suffix per rank)",
    )
    parser.add_argument("--shared-kv-pool-wait-timeout-s", type=float, default=300.0)
    parser.add_argument("--shared-kv-pool-poll-s", type=float, default=0.1)

    # KV transfer config
    parser.add_argument("--kv-owner-state-url", default="", help="KV owner state server URL")
    parser.add_argument("--kv-transfer-config-template", default="")
    parser.add_argument("--send-activation-margin-tokens", type=int, default=512)
    parser.add_argument("--send-publish-token-stride", type=int, default=64)
    parser.add_argument("--owner-flush-each-layer", action="store_true")

    # Timing
    parser.add_argument("--owner-startup-delay-s", type=float, default=2.0)
    parser.add_argument("--owner-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--ipc-wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--ipc-poll-interval-s", type=float, default=0.5)

    # MPS
    parser.add_argument("--enable-cuda-mps", action="store_true")
    parser.add_argument(
        "--mps-owner-percentage",
        type=int,
        default=100,
        help="MPS active thread percentage for owner servers",
    )
    parser.add_argument(
        "--mps-consumer-percentage",
        type=int,
        default=100,
        help="MPS active thread percentage for consumer servers",
    )

    args, vllm_extra = parser.parse_known_args()

    dp_size = args.data_parallel_size
    tp_size = args.tensor_parallel_size
    total_gpus_needed = dp_size * tp_size

    all_gpus = [int(x) for x in _parse_csv(args.cuda_visible_devices)]
    if len(all_gpus) < total_gpus_needed:
        print(
            f"ERROR: Need {total_gpus_needed} GPUs for DP={dp_size} x TP={tp_size}, "
            f"but only {len(all_gpus)} available: {all_gpus}",
            file=sys.stderr,
        )
        return 1

    print(f"Launching DP={dp_size} x TP={tp_size} cluster")
    print(f"Total GPUs needed: {total_gpus_needed}, available: {len(all_gpus)}")
    print(f"GPU assignments:")
    for r in range(dp_size):
        gpus = _get_gpu_slice(r, tp_size, all_gpus)
        s1_port = args.server1_base_port + 2 * r
        s2_port = args.server2_base_port + 2 * r
        print(f"  DP rank {r}: GPUs {gpus} -> server1:{s1_port}, server2:{s2_port}")

    # Prepare shared pool paths per DP rank for isolation
    shared_pool_paths: list[str] = []
    if args.shared_kv_pool_enable:
        base_path = args.shared_kv_pool_meta_path
        for r in range(dp_size):
            rank_path = f"{base_path}.dp{r}"
            shared_pool_paths.append(rank_path)
            # Clean up old files
            try:
                os.remove(rank_path)
            except FileNotFoundError:
                pass
            try:
                shutil.rmtree(f"{rank_path}.alloc")
            except FileNotFoundError:
                pass
    else:
        # Dummy paths
        shared_pool_paths = ["/tmp/vllm_dummy"] * dp_size

    # Clean up IPC meta
    ipc_meta_base = "/tmp/vllm_ipc_meta_dp"
    for r in range(dp_size):
        try:
            os.remove(f"{ipc_meta_base}.dp{r}")
        except FileNotFoundError:
            pass
    try:
        shutil.rmtree("/tmp/vllm_kv_ipc")
    except FileNotFoundError:
        pass

    children: list[tuple[subprocess.Popen, str]] = []  # (proc, role_desc)
    base_env = dict(os.environ)
    base_env.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _sig_handler(signum, frame):  # type: ignore[no-untyped-def]
        print("\nReceived signal, stopping all servers...")
        for proc, role in children:
            _stop_proc(proc)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Common stripped args
    owner_extra = _strip_overridden_args(
        vllm_extra,
        {
            "--gpu-memory-utilization",
            "--max-num-seqs",
            "--max-model-len",
            "--tensor-parallel-size",
            "--kv-transfer-config",
        },
    )
    consumer_extra = _strip_overridden_args(
        vllm_extra,
        {
            "--gpu-memory-utilization",
            "--max-num-seqs",
            "--max-model-len",
            "--compilation-config",
            "--tensor-parallel-size",
            "--kv-transfer-config",
        },
    )

    # Launch servers DP rank by DP rank (all owners first, then consumers)
    # This ensures owners export weights before consumers try to load

    # Phase 1: Launch all owners
    print("\n=== Phase 1: Launching all owner servers ===")
    owner_urls: list[str] = []
    for r in range(dp_size):
        port = args.server1_base_port + 2 * r
        gpus = _get_gpu_slice(r, tp_size, all_gpus)
        gpu_str = ",".join(str(g) for g in gpus)
        meta_path = f"{ipc_meta_base}.dp{r}"
        kv_port = args.server_kv_base_port + 2 * r

        kv_config = _resolve_kv_transfer_config(
            args.kv_transfer_config_template,
            kv_port,
            args.kv_owner_state_url,
            args.shared_kv_pool_enable,
            shared_pool_paths[r],
            args.shared_kv_pool_wait_timeout_s,
            args.shared_kv_pool_poll_s,
            "producer",
            args.send_activation_margin_tokens,
            args.send_publish_token_stride,
            args.owner_flush_each_layer,
        )

        cmd = _build_vllm_cmd(
            host=args.host,
            port=port,
            model=args.model,
            role="owner",
            meta_path=meta_path,
            timeout_s=args.ipc_wait_timeout_s,
            poll_s=args.ipc_poll_interval_s,
            extra_args=owner_extra,
            gpu_memory_utilization=args.owner_gpu_memory_utilization,
            max_num_seqs=args.owner_max_num_seqs,
            max_model_len=args.owner_max_model_len,
            enforce_eager=args.owner_enforce_eager,
            compilation_config=(args.owner_compilation_config or None),
            tensor_parallel_size=tp_size,
            kv_transfer_config=kv_config,
        )

        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        if args.enable_cuda_mps:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.mps_owner_percentage)
        if args.shared_kv_pool_enable:
            env["VLLM_SHARED_BLOCK_ALLOCATOR_ENABLE"] = "1"
            env["VLLM_SHARED_BLOCK_ALLOCATOR_PATH"] = f"{shared_pool_paths[r]}.alloc"
            env["VLLM_SHARED_BLOCK_ALLOCATOR_RESET"] = "1"

        url = f"http://{args.host}:{port}"
        owner_urls.append(url)
        print(f"  DP rank {r}: Starting owner on port {port}, GPUs {gpu_str}")
        proc = _start(cmd, env)
        children.append((proc, f"owner-dp{r}"))

    # Wait for all owners to be ready
    print("\n=== Waiting for all owners to be ready ===")
    time.sleep(args.owner_startup_delay_s)
    for r, url in enumerate(owner_urls):
        try:
            _wait_server_ready(url, args.owner_ready_timeout_s)
            print(f"  Owner DP rank {r} ready: {url}")
        except Exception as e:
            print(f"  ERROR: Owner DP rank {r} failed to become ready: {e}", file=sys.stderr)
            _stop_all([p for p, _ in children])
            return 1

    # Phase 2: Launch all consumers
    print("\n=== Phase 2: Launching all consumer (shadow) servers ===")
    consumer_urls: list[str] = []
    for r in range(dp_size):
        port = args.server2_base_port + 2 * r
        gpus = _get_gpu_slice(r, tp_size, all_gpus)
        gpu_str = ",".join(str(g) for g in gpus)
        meta_path = f"{ipc_meta_base}.dp{r}"
        kv_port = args.server_kv_base_port + 2 * r + 1

        kv_config = _resolve_kv_transfer_config(
            args.kv_transfer_config_template,
            kv_port,
            args.kv_owner_state_url,
            args.shared_kv_pool_enable,
            shared_pool_paths[r],
            args.shared_kv_pool_wait_timeout_s,
            args.shared_kv_pool_poll_s,
            "consumer",
            args.send_activation_margin_tokens,
            args.send_publish_token_stride,
            args.owner_flush_each_layer,
        )

        comp_cfg = args.consumer_compilation_config or None

        cmd = _build_vllm_cmd(
            host=args.host,
            port=port,
            model=args.model,
            role="consumer",
            meta_path=meta_path,
            timeout_s=args.ipc_wait_timeout_s,
            poll_s=args.ipc_poll_interval_s,
            extra_args=consumer_extra,
            gpu_memory_utilization=args.consumer_gpu_memory_utilization,
            max_num_seqs=args.consumer_max_num_seqs,
            max_model_len=args.consumer_max_model_len,
            enforce_eager=not args.no_consumer_enforce_eager,
            compilation_config=comp_cfg,
            tensor_parallel_size=tp_size,
            kv_transfer_config=kv_config,
        )

        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        if args.enable_cuda_mps:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.mps_consumer_percentage)
        if args.consumer_attention_backend:
            env["VLLM_ATTENTION_BACKEND"] = args.consumer_attention_backend
        if args.shared_kv_pool_enable:
            env["VLLM_SHARED_BLOCK_ALLOCATOR_ENABLE"] = "1"
            env["VLLM_SHARED_BLOCK_ALLOCATOR_PATH"] = f"{shared_pool_paths[r]}.alloc"
            env["VLLM_SHARED_BLOCK_ALLOCATOR_RESET"] = "0"

        url = f"http://{args.host}:{port}"
        consumer_urls.append(url)
        print(f"  DP rank {r}: Starting consumer on port {port}, GPUs {gpu_str}")
        proc = _start(cmd, env)
        children.append((proc, f"consumer-dp{r}"))

    # Wait for all consumers
    print("\n=== Waiting for all consumers to be ready ===")
    for r, url in enumerate(consumer_urls):
        try:
            _wait_server_ready(url, args.owner_ready_timeout_s)
            print(f"  Consumer DP rank {r} ready: {url}")
        except Exception as e:
            print(f"  ERROR: Consumer DP rank {r} failed to become ready: {e}", file=sys.stderr)
            _stop_all([p for p, _ in children])
            return 1

    print("\n" + "=" * 60)
    print(f"DP={dp_size} x TP={tp_size} cluster is READY")
    print("")
    print("Server URLs:")
    for r in range(dp_size):
        print(f"  DP rank {r}:")
        print(f"    Owner:    {owner_urls[r]}")
        print(f"    Consumer: {consumer_urls[r]}")
    print("")
    print("Router configuration:")
    print(f"  --server1-urls {','.join(owner_urls)}")
    print(f"  --server2-urls {','.join(consumer_urls)}")
    print("")
    print("Press Ctrl+C to stop all servers.")
    print("=" * 60)

    # Monitor loop
    while True:
        for proc, role in children:
            code = proc.poll()
            if code is not None:
                print(f"\nChild process exited unexpectedly (role={role}, code={code}). Stopping cluster.")
                _stop_all([p for p, _ in children])
                return 1
        time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())
