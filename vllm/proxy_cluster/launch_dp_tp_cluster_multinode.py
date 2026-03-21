#!/usr/bin/env python3
"""Launch multi-node DP x TP cluster for hop-vllm.

This script runs on each node and launches the DP ranks assigned to that node.
For TP4+DP4 across 2 nodes (8 GPUs each):
  Node 0 (172.24.79.15):
    - DP Rank 0: GPUs 0,1,2,3 -> server1:8101, server2:8102 (eth0)
    - DP Rank 1: GPUs 4,5,6,7 -> server1:8103, server2:8104 (eth1)
  Node 1 (172.24.79.13):
    - DP Rank 2: GPUs 0,1,2,3 -> server1:8105, server2:8106 (eth0)
    - DP Rank 3: GPUs 4,5,6,7 -> server1:8107, server2:8108 (eth1)

Example usage on Node 0:
  python -m vllm.proxy_cluster.launch_dp_tp_cluster_multinode \
      --model /root/model/Qwen3-8B \
      --node-id 0 \
      --node-ip 172.24.79.15 \
      --node-ranks 0,1 \
      --data-parallel-size 4 \
      --tensor-parallel-size 4

Example usage on Node 1:
  python -m vllm.proxy_cluster.launch_dp_tp_cluster_multinode \
      --model /root/model/Qwen3-8B \
      --node-id 1 \
      --node-ip 172.24.79.13 \
      --node-ranks 2,3 \
      --data-parallel-size 4 \
      --tensor-parallel-size 4
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


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _get_gpu_slice(rank_local: int, tp_size: int, all_gpus: list[int]) -> list[int]:
    """Get GPU indices for a local rank on this node."""
    start_idx = rank_local * tp_size
    end_idx = start_idx + tp_size
    return all_gpus[start_idx:end_idx]


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


def _resolve_kv_transfer_config(
    raw: str | None,
    port: int,
    kv_owner_state_url: str = "",
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
    if connector == "P2pNcclConnector" and kv_owner_state_url:
        extra.setdefault("kv_owner_state_url", kv_owner_state_url)
    if connector == "P2pNcclConnector" and send_activation_margin_tokens is not None:
        extra.setdefault("send_activation_margin_tokens", int(send_activation_margin_tokens))
    if connector == "P2pNcclConnector" and send_publish_token_stride is not None:
        extra.setdefault("send_publish_token_stride", int(send_publish_token_stride))
    if connector == "P2pNcclConnector" and owner_flush_each_layer is not None:
        extra.setdefault("owner_flush_each_layer", bool(owner_flush_each_layer))
    return json.dumps(obj, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch multi-node DP x TP vLLM servers."
    )
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")

    # Node config
    parser.add_argument("--node-id", type=int, required=True, help="Node ID (0, 1, 2, ...)")
    parser.add_argument("--node-ip", required=True, help="IP address of this node")
    parser.add_argument(
        "--node-ranks",
        required=True,
        help="Comma-separated list of global DP ranks this node should run (e.g., '0,1' or '2,3')",
    )
    parser.add_argument(
        "--master-ip",
        default="172.24.79.15",
        help="IP of master node (node 0) for distributed init",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Port for distributed init",
    )

    # Parallelism config
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=4,
        help="Total number of DP ranks across all nodes",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size within each DP rank",
    )

    # Base ports - global across all nodes
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

    # GPU config
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of all GPU IDs on this node",
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

    # KV transfer config (using P2pNcclConnector for cross-node)
    parser.add_argument("--kv-owner-state-url", default="", help="KV owner state server URL")
    parser.add_argument("--kv-transfer-config-template", default="")
    parser.add_argument("--send-activation-margin-tokens", type=int, default=512)
    parser.add_argument("--send-publish-token-stride", type=int, default=64)
    parser.add_argument("--owner-flush-each-layer", action="store_true")

    # Timing
    parser.add_argument("--owner-startup-delay-s", type=float, default=2.0)
    parser.add_argument("--owner-ready-timeout-s", type=float, default=300.0)

    args, vllm_extra = parser.parse_known_args()

    node_ranks = [int(x.strip()) for x in _parse_csv(args.node_ranks)]
    dp_size = args.data_parallel_size
    tp_size = args.tensor_parallel_size

    all_gpus = [int(x) for x in _parse_csv(args.cuda_visible_devices)]
    gpus_needed_per_node = len(node_ranks) * tp_size
    if len(all_gpus) < gpus_needed_per_node:
        print(
            f"ERROR: Node needs {gpus_needed_per_node} GPUs for ranks {node_ranks} x TP={tp_size}, "
            f"but only {len(all_gpus)} available: {all_gpus}",
            file=sys.stderr,
        )
        return 1

    print(f"Node {args.node_id} ({args.node_ip}): Launching DP ranks {node_ranks}")
    print(f"Total DP size: {dp_size}, TP size: {tp_size}")
    print(f"GPU assignments on this node:")
    for i, r in enumerate(node_ranks):
        gpus = _get_gpu_slice(i, tp_size, all_gpus)
        s1_port = args.server1_base_port + 2 * r
        s2_port = args.server2_base_port + 2 * r
        print(f"  Global DP rank {r}: local GPUs {gpus} -> server1:{s1_port}, server2:{s2_port}")

    # Clean up old files
    ipc_meta_base = "/tmp/vllm_ipc_meta_dp"
    for r in node_ranks:
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

    # NCCL network settings - use eth0-eth3 for cross-node communication
    base_env["NCCL_SOCKET_IFNAME"] = "eth0,eth1,eth2,eth3"
    base_env["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand, use Ethernet
    base_env["NCCL_DEBUG"] = "WARN"  # Enable NCCL debug warnings

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
            "--distributed-executor-backend",
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
            "--distributed-executor-backend",
        },
    )

    # Phase 1: Launch all owners on this node
    print("\n=== Phase 1: Launching owner servers on this node ===")
    owner_urls: list[str] = []
    for local_idx, r in enumerate(node_ranks):
        port = args.server1_base_port + 2 * r
        gpus = _get_gpu_slice(local_idx, tp_size, all_gpus)
        gpu_str = ",".join(str(g) for g in gpus)

        kv_config = _resolve_kv_transfer_config(
            args.kv_transfer_config_template,
            port,
            args.kv_owner_state_url,
            args.send_activation_margin_tokens,
            args.send_publish_token_stride,
            args.owner_flush_each_layer,
        )

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host", args.host,
            "--port", str(port),
            "--model", args.model,
            "--tensor-parallel-size", str(tp_size),
            "--distributed-executor-backend", "mp",  # Multi-process for TP
            "--gpu-memory-utilization", str(args.owner_gpu_memory_utilization),
            "--max-num-seqs", str(args.owner_max_num_seqs),
        ]
        if args.owner_max_model_len:
            cmd += ["--max-model-len", str(args.owner_max_model_len)]
        if args.owner_enforce_eager:
            cmd += ["--enforce-eager"]
        if args.owner_compilation_config:
            cmd += ["--compilation-config", args.owner_compilation_config]
        if kv_config:
            cmd += ["--kv-transfer-config", kv_config]
        cmd += owner_extra

        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str

        print(f"Starting owner for DP rank {r} on GPUs {gpu_str}, port {port}")
        print(f"  Command: {' '.join(cmd)}")
        proc = _start(cmd, env)
        children.append((proc, f"owner-dp{r}"))
        owner_urls.append(f"http://{args.node_ip}:{port}")
        time.sleep(args.owner_startup_delay_s)

    # Wait for all owners on this node to be ready
    print("\n=== Waiting for owners to be ready ===")
    for local_idx, r in enumerate(node_ranks):
        port = args.server1_base_port + 2 * r
        url = f"http://{args.host}:{port}"
        try:
            _wait_server_ready(url, args.owner_ready_timeout_s)
            print(f"Owner for DP rank {r} is ready at {url}")
        except TimeoutError as e:
            print(f"ERROR: Owner for DP rank {r} failed to start: {e}", file=sys.stderr)
            _stop_all([p for p, _ in children])
            return 1

    # Phase 2: Launch all consumers on this node
    print("\n=== Phase 2: Launching consumer servers on this node ===")
    consumer_urls: list[str] = []
    for local_idx, r in enumerate(node_ranks):
        port = args.server2_base_port + 2 * r
        gpus = _get_gpu_slice(local_idx, tp_size, all_gpus)
        gpu_str = ",".join(str(g) for g in gpus)

        kv_config = _resolve_kv_transfer_config(
            args.kv_transfer_config_template,
            port,
            args.kv_owner_state_url,
            args.send_activation_margin_tokens,
            args.send_publish_token_stride,
            args.owner_flush_each_layer,
        )

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host", args.host,
            "--port", str(port),
            "--model", args.model,
            "--tensor-parallel-size", str(tp_size),
            "--distributed-executor-backend", "mp",  # Multi-process for TP
            "--gpu-memory-utilization", str(args.consumer_gpu_memory_utilization),
            "--max-num-seqs", str(args.consumer_max_num_seqs),
        ]
        if args.consumer_max_model_len:
            cmd += ["--max-model-len", str(args.consumer_max_model_len)]
        if not args.no_consumer_enforce_eager:
            cmd += ["--enforce-eager"]
        if args.consumer_compilation_config:
            cmd += ["--compilation-config", args.consumer_compilation_config]
        if kv_config:
            cmd += ["--kv-transfer-config", kv_config]
        cmd += consumer_extra

        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        env["VLLM_ATTENTION_BACKEND"] = args.consumer_attention_backend

        print(f"Starting consumer for DP rank {r} on GPUs {gpu_str}, port {port}")
        print(f"  Command: {' '.join(cmd)}")
        proc = _start(cmd, env)
        children.append((proc, f"consumer-dp{r}"))
        consumer_urls.append(f"http://{args.node_ip}:{port}")

    # Wait for all consumers on this node to be ready
    print("\n=== Waiting for consumers to be ready ===")
    for local_idx, r in enumerate(node_ranks):
        port = args.server2_base_port + 2 * r
        url = f"http://{args.host}:{port}"
        try:
            _wait_server_ready(url, args.owner_ready_timeout_s)
            print(f"Consumer for DP rank {r} is ready at {url}")
        except TimeoutError as e:
            print(f"ERROR: Consumer for DP rank {r} failed to start: {e}", file=sys.stderr)
            _stop_all([p for p, _ in children])
            return 1

    print("\n=== All servers on this node are ready ===")
    print(f"Owner URLs: {owner_urls}")
    print(f"Consumer URLs: {consumer_urls}")

    # Keep running until interrupted
    print("\nPress Ctrl+C to stop all servers...")
    try:
        while True:
            for proc, role in children:
                ret = proc.poll()
                if ret is not None:
                    print(f"WARNING: {role} exited with code {ret}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping all servers...")
    finally:
        _stop_all([p for p, _ in children])

    return 0


if __name__ == "__main__":
    sys.exit(main())
