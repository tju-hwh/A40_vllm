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


def _start(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    # Put each launched server into its own process group so we can
    # terminate the full tree (API server + engine children) reliably.
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
            # Skip this flag and its value if present.
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


def _wait_owner_ready(owner_url: str, timeout_s: float) -> None:
    """Wait until owner OpenAI server is fully ready (engine initialized)."""
    deadline = time.time() + timeout_s
    health_url = owner_url.rstrip("/") + "/v1/models"
    host, port = owner_url.rsplit(":", 1)
    host = host.split("://", 1)[-1]
    port_i = int(port)

    last_err = ""
    while time.time() < deadline:
        # quick tcp probe first
        try:
            with socket.create_connection((host, port_i), timeout=1.0):
                pass
        except OSError as e:
            last_err = f"tcp not ready: {e}"
            time.sleep(0.5)
            continue

        # API ready probe
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return
                last_err = f"http status={resp.status}"
        except urllib.error.URLError as e:
            last_err = f"http not ready: {e}"
        time.sleep(0.5)
    raise TimeoutError(f"Owner server not ready within {timeout_s}s ({last_err})")


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    vals: list[int] = []
    for item in _parse_csv(raw):
        vals.append(int(item))
    return vals


def _resolve_kv_transfer_config(raw: str | None,
                                port: int,
                                kv_owner_state_url: str = "",
                                shared_kv_pool_enable: bool = False,
                                shared_kv_pool_meta_path: str = "",
                                shared_kv_pool_wait_timeout_s: float = 300.0,
                                shared_kv_pool_poll_s: float = 0.1,
                                shared_kv_pool_role: str = "",
                                send_activation_margin_tokens: int | None = None,
                                send_publish_token_stride: int | None = None,
                                owner_flush_each_layer: bool | None = None) -> str | None:
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
        # Enable request-level global block table when shared KV pool is on.
        extra.setdefault("shared_block_table_enable", True)
        if shared_kv_pool_role:
            extra.setdefault("shared_kv_pool_role", shared_kv_pool_role)
        if shared_kv_pool_meta_path:
            extra.setdefault("shared_kv_pool_meta_path",
                             shared_kv_pool_meta_path)
        extra.setdefault("shared_kv_pool_wait_timeout_s",
                         float(shared_kv_pool_wait_timeout_s))
        extra.setdefault("shared_kv_pool_poll_s",
                         float(shared_kv_pool_poll_s))
    if connector == "CudaIpcConnector" and send_activation_margin_tokens is not None:
        extra.setdefault("send_activation_margin_tokens",
                         int(send_activation_margin_tokens))
    if connector == "CudaIpcConnector" and send_publish_token_stride is not None:
        extra.setdefault("send_publish_token_stride",
                         int(send_publish_token_stride))
    if connector == "CudaIpcConnector" and owner_flush_each_layer is not None:
        extra.setdefault("owner_flush_each_layer",
                         bool(owner_flush_each_layer))
    return json.dumps(obj, separators=(",", ":"))


def _read_dynamic_active_upstream(control_path: str) -> str | None:
    try:
        with open(control_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    raw = obj.get("active_upstream")
    if raw is None:
        return None
    val = str(raw).strip()
    return val or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch real vLLM servers with experimental CUDA IPC shared parameters."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--num-servers",
        type=int,
        default=4,
        choices=[2, 4],
        help="Number of servers to launch: owner + 1 consumer, or owner + 3 consumers.",
    )
    parser.add_argument("--server1-port", type=int, default=8101, help="Owner vLLM server port.")
    parser.add_argument("--server2-port", type=int, default=8102, help="Consumer vLLM server port.")
    parser.add_argument("--server3-port", type=int, default=8103, help="Consumer vLLM server port.")
    parser.add_argument("--server4-port", type=int, default=8104, help="Consumer vLLM server port.")
    parser.add_argument(
        "--ipc-meta-path",
        default="/tmp/vllm_ipc_meta.pkl",
        help="Owner writes IPC metadata file, consumers read from it.",
    )
    parser.add_argument("--ipc-wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--ipc-poll-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--owner-gpu-memory-utilization",
        type=float,
        default=0.75,
        help="Owner server --gpu-memory-utilization (default 0.75).",
    )
    parser.add_argument(
        "--owner-max-num-seqs",
        type=int,
        default=128,
        help="Owner server --max-num-seqs (default 128).",
    )
    parser.add_argument(
        "--owner-max-model-len",
        type=int,
        default=None,
        help="Optional owner override for --max-model-len.",
    )
    parser.add_argument(
        "--owner-tensor-parallel-size",
        type=int,
        default=1,
        help="Owner server --tensor-parallel-size.",
    )
    parser.add_argument(
        "--owner-compilation-config",
        default="",
        help=(
            "Optional JSON string passed to owner --compilation-config. "
            "If empty, owner uses vLLM default compilation config."
        ),
    )
    parser.add_argument(
        "--consumer-gpu-memory-utilization",
        type=float,
        default=0.1,
        help="Consumer server --gpu-memory-utilization. Lower this on single GPU.",
    )
    parser.add_argument(
        "--consumer-max-num-seqs",
        type=int,
        default=32,
        help="Consumer server --max-num-seqs to reduce KV cache reservation.",
    )
    parser.add_argument(
        "--consumer-max-model-len",
        type=int,
        default=None,
        help="Optional consumer override for --max-model-len.",
    )
    parser.add_argument(
        "--consumer-tensor-parallel-size",
        type=int,
        default=1,
        help="Consumer server --tensor-parallel-size.",
    )
    parser.add_argument(
        "--no-consumer-enforce-eager",
        action="store_true",
        help="Disable --enforce-eager for consumers (default is enabled).",
    )
    parser.add_argument(
        "--consumer-compilation-config",
        default='{"level":0,"use_inductor":false,"use_cudagraph":false}',
        help="JSON string passed to consumer --compilation-config.",
    )
    parser.add_argument(
        "--server2-compilation-config",
        default="",
        help=(
            "Optional JSON for server2 --compilation-config. "
            "Overrides --consumer-compilation-config when set."
        ),
    )
    parser.add_argument(
        "--server3-compilation-config",
        default="",
        help=(
            "Optional JSON for server3 --compilation-config. "
            "Overrides --consumer-compilation-config when set."
        ),
    )
    parser.add_argument(
        "--server4-compilation-config",
        default="",
        help=(
            "Optional JSON for server4 --compilation-config. "
            "Overrides --consumer-compilation-config when set."
        ),
    )
    parser.add_argument(
        "--owner-cuda-visible-devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES for owner process.",
    )
    parser.add_argument(
        "--consumer-cuda-visible-devices",
        default="",
        help="Optional CSV, one entry per consumer (server2,3,4), e.g. '1,2,3'.",
    )
    parser.add_argument(
        "--consumer-cuda-visible-devices-all",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES applied to all consumer processes, e.g. '0,1'.",
    )
    parser.add_argument(
        "--consumer-attention-backend",
        default="TORCH_SDPA",
        help="Set VLLM_ATTENTION_BACKEND for consumers (e.g. TORCH_SDPA/FLASHINFER/FLASH_ATTN).",
    )
    parser.add_argument(
        "--enable-cuda-mps",
        action="store_true",
        help=(
            "Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE per launched server process. "
            "Requires an external CUDA MPS daemon to already be running."
        ),
    )
    parser.add_argument(
        "--mps-active-thread-percentages",
        default="",
        help=(
            "CSV for server1..N MPS percentages, "
            "for example '100,60' or '100,60,40,20'."
        ),
    )
    parser.add_argument(
        "--owner-kv-transfer-config",
        default="",
        help=(
            "Optional JSON string for owner --kv-transfer-config. "
            "Supports {port} placeholder."
        ),
    )
    parser.add_argument(
        "--consumer-kv-transfer-config",
        default="",
        help=(
            "Optional JSON string for all consumers --kv-transfer-config. "
            "Supports {port} placeholder."
        ),
    )
    parser.add_argument(
        "--kv-transfer-config-template",
        default="",
        help=(
            "Optional JSON string applied to owner+consumers when specific flags are empty. "
            "Supports {port} placeholder."
        ),
    )
    parser.add_argument(
        "--owner-startup-delay-s",
        type=float,
        default=2.0,
        help="Delay before launching consumer servers, letting owner start first.",
    )
    parser.add_argument(
        "--owner-ready-timeout-s",
        type=float,
        default=180.0,
        help="Max wait for owner /v1/models ready before starting consumers.",
    )
    parser.add_argument(
        "--enable-dynamic-consumer-kv",
        action="store_true",
        help=(
            "Enable dynamic consumer profile switching via control file. "
            "Only one of server2/3/4 will use active KV profile at a time."
        ),
    )
    parser.add_argument(
        "--dynamic-kv-control-path",
        default="/tmp/vllm_dynamic_kv_control.json",
        help="Control JSON path written by router. Example: {\"active_upstream\":\"http://127.0.0.1:8102\"}",
    )
    parser.add_argument(
        "--dynamic-kv-switch-timeout-s",
        type=float,
        default=120.0,
        help="Wait timeout for restarted consumer /v1/models readiness.",
    )
    parser.add_argument(
        "--active-consumer-gpu-memory-utilization",
        type=float,
        default=0.2,
        help="GPU memory utilization for active consumer (server2/3/4).",
    )
    parser.add_argument(
        "--active-consumer-max-num-seqs",
        type=int,
        default=16,
        help="max-num-seqs for active consumer (server2/3/4).",
    )
    parser.add_argument(
        "--dynamic-kv-auto-demote",
        action="store_true",
        help=(
            "Automatically demote previously active consumers back to inactive "
            "profile when active_upstream changes. Disabled by default to avoid "
            "killing CUDA-IPC source process before next hop consumes KV."
        ),
    )
    parser.add_argument(
        "--kv-owner-state-url",
        default="",
        help=(
            "Optional KV owner state server URL. When set and kv_connector is "
            "CudaIpcConnector, this URL is auto-injected into "
            "kv_connector_extra_config.kv_owner_state_url."
        ),
    )
    parser.add_argument(
        "--shared-kv-pool-enable",
        action="store_true",
        help=(
            "Enable experimental single shared KV pool export/import. "
            "Owner exports one KV pool, consumers map it via CUDA IPC."
        ),
    )
    parser.add_argument(
        "--shared-kv-pool-meta-path",
        default="/tmp/vllm_shared_kv_pool.pkl",
        help="Metadata path for experimental shared KV pool CUDA IPC handles.",
    )
    parser.add_argument(
        "--shared-kv-pool-wait-timeout-s",
        type=float,
        default=300.0,
        help="Consumer wait timeout for shared KV pool metadata.",
    )
    parser.add_argument(
        "--shared-kv-pool-poll-s",
        type=float,
        default=0.1,
        help="Poll interval for shared KV pool metadata file.",
    )
    parser.add_argument(
        "--send-activation-margin-tokens",
        type=int,
        default=512,
        help=(
            "Producer-side connector activation margin. Smaller values keep "
            "server1 on the native local decode path until closer to handoff."
        ),
    )
    parser.add_argument(
        "--send-publish-token-stride",
        type=int,
        default=64,
        help="Producer-side KV publish stride after send path is activated.",
    )
    parser.add_argument(
        "--owner-flush-each-layer",
        action="store_true",
        help=(
            "Flush owner-state registration every attention layer. Disabled by "
            "default to keep active decode closer to native vLLM."
        ),
    )
    args, vllm_extra = parser.parse_known_args()

    host = args.host
    all_ports = [args.server1_port, args.server2_port, args.server3_port, args.server4_port]
    ports = all_ports[: args.num_servers]
    urls = [f"http://127.0.0.1:{p}" for p in ports]

    try:
        os.remove(args.ipc_meta_path)
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree("/tmp/vllm_kv_ipc")
    except FileNotFoundError:
        pass

    children: list[subprocess.Popen] = []
    base_env = dict(os.environ)
    mps_percentages = _parse_csv_ints(args.mps_active_thread_percentages)
    if args.enable_cuda_mps:
        if len(mps_percentages) != args.num_servers:
            raise ValueError(
                "--enable-cuda-mps requires --mps-active-thread-percentages "
                f"with exactly {args.num_servers} comma-separated integers"
            )
        for pct in mps_percentages:
            if pct <= 0 or pct > 100:
                raise ValueError(
                    "MPS active thread percentages must be in the range 1..100"
                )

    def _sig_handler(signum, frame):  # type: ignore[no-untyped-def]
        _stop_all(children)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    owner_cmd = _build_vllm_cmd(
        host=host,
        port=args.server1_port,
        model=args.model,
        role="owner",
        meta_path=args.ipc_meta_path,
        timeout_s=args.ipc_wait_timeout_s,
        poll_s=args.ipc_poll_interval_s,
        extra_args=_strip_overridden_args(
            vllm_extra,
            {
                "--gpu-memory-utilization",
                "--max-num-seqs",
                "--max-model-len",
                "--tensor-parallel-size",
                "--kv-transfer-config",
            },
        ),
        gpu_memory_utilization=args.owner_gpu_memory_utilization,
        max_num_seqs=args.owner_max_num_seqs,
        max_model_len=args.owner_max_model_len,
        compilation_config=(args.owner_compilation_config or None),
        tensor_parallel_size=args.owner_tensor_parallel_size,
        kv_transfer_config=_resolve_kv_transfer_config(
            args.owner_kv_transfer_config or args.kv_transfer_config_template,
            args.server1_port,
            args.kv_owner_state_url,
            args.shared_kv_pool_enable,
            args.shared_kv_pool_meta_path,
            args.shared_kv_pool_wait_timeout_s,
            args.shared_kv_pool_poll_s,
            "producer",
            args.send_activation_margin_tokens,
            args.send_publish_token_stride,
            args.owner_flush_each_layer,
        ),
    )
    owner_env = dict(base_env)
    if args.owner_cuda_visible_devices:
        owner_env["CUDA_VISIBLE_DEVICES"] = args.owner_cuda_visible_devices
    if args.enable_cuda_mps:
        owner_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_percentages[0])
    if args.shared_kv_pool_enable:
        owner_env["VLLM_SHARED_BLOCK_ALLOCATOR_ENABLE"] = "1"
        owner_env["VLLM_SHARED_BLOCK_ALLOCATOR_PATH"] = (
            args.shared_kv_pool_meta_path + ".alloc")
        owner_env["VLLM_SHARED_BLOCK_ALLOCATOR_RESET"] = "1"
    children.append(_start(owner_cmd, owner_env))
    time.sleep(args.owner_startup_delay_s)
    owner_url = f"http://127.0.0.1:{args.server1_port}"
    try:
        _wait_owner_ready(owner_url, args.owner_ready_timeout_s)
    except Exception as e:
        print(f"Owner failed to become ready: {e}")
        _stop_all(children)
        return 1

    consumer_visible_devices = _parse_csv(args.consumer_cuda_visible_devices)
    consumer_ports = ports[1:]
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

    def _consumer_env(idx: int) -> dict[str, str]:
        env = dict(base_env)
        if args.consumer_cuda_visible_devices_all:
            env["CUDA_VISIBLE_DEVICES"] = args.consumer_cuda_visible_devices_all
        elif idx < len(consumer_visible_devices):
            env["CUDA_VISIBLE_DEVICES"] = consumer_visible_devices[idx]
        if args.enable_cuda_mps:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                mps_percentages[idx + 1]
            )
        if args.consumer_attention_backend:
            env["VLLM_ATTENTION_BACKEND"] = args.consumer_attention_backend
        if args.shared_kv_pool_enable:
            env["VLLM_SHARED_BLOCK_ALLOCATOR_ENABLE"] = "1"
            env["VLLM_SHARED_BLOCK_ALLOCATOR_PATH"] = (
                args.shared_kv_pool_meta_path + ".alloc")
            env["VLLM_SHARED_BLOCK_ALLOCATOR_RESET"] = "0"
        return env

    def _consumer_cmd(port: int, active_profile: bool) -> list[str]:
        gpu_util = args.active_consumer_gpu_memory_utilization if active_profile \
            else args.consumer_gpu_memory_utilization
        max_num_seqs = args.active_consumer_max_num_seqs if active_profile \
            else args.consumer_max_num_seqs
        per_server_comp_cfg = {
            args.server2_port: args.server2_compilation_config,
            args.server3_port: args.server3_compilation_config,
            args.server4_port: args.server4_compilation_config,
        }.get(port, "")
        comp_cfg = per_server_comp_cfg or args.consumer_compilation_config
        return _build_vllm_cmd(
            host=host,
            port=port,
            model=args.model,
            role="consumer",
            meta_path=args.ipc_meta_path,
            timeout_s=args.ipc_wait_timeout_s,
            poll_s=args.ipc_poll_interval_s,
            extra_args=consumer_extra,
            gpu_memory_utilization=gpu_util,
            max_num_seqs=max_num_seqs,
            max_model_len=args.consumer_max_model_len,
            enforce_eager=not args.no_consumer_enforce_eager,
            compilation_config=comp_cfg,
            tensor_parallel_size=args.consumer_tensor_parallel_size,
            kv_transfer_config=_resolve_kv_transfer_config(
                args.consumer_kv_transfer_config or args.kv_transfer_config_template,
                port,
                args.kv_owner_state_url,
                args.shared_kv_pool_enable,
                args.shared_kv_pool_meta_path,
                args.shared_kv_pool_wait_timeout_s,
                args.shared_kv_pool_poll_s,
                "consumer",
                args.send_activation_margin_tokens,
                args.send_publish_token_stride,
                args.owner_flush_each_layer,
            ),
        )

    # child slots: 0=owner, 1..N-1=consumers
    consumer_active_profile: dict[int, bool] = {}
    for idx, p in enumerate(consumer_ports):
        use_active = False
        children.append(_start(_consumer_cmd(p, use_active), _consumer_env(idx)))
        consumer_active_profile[p] = use_active

    dynamic_active_upstream: str | None = None

    server_lines = "".join(
        f"  server-{idx + 1} ({'owner' if idx == 0 else 'consumer'}): {url}\n"
        for idx, url in enumerate(urls)
    )
    print(
        f"\nStarted {args.num_servers} vLLM servers (experimental ipc_weight_share):\n"
        f"{server_lines}"
        f"IPC meta file: {args.ipc_meta_path}\n"
    )
    print("Press Ctrl+C to stop all.")

    if args.enable_dynamic_consumer_kv:
        print(f"Dynamic consumer KV enabled. control path: {args.dynamic_kv_control_path}")

    while True:
        if args.enable_dynamic_consumer_kv:
            desired_upstream = _read_dynamic_active_upstream(args.dynamic_kv_control_path)
            if desired_upstream != dynamic_active_upstream:
                dynamic_active_upstream = desired_upstream
                for i, port in enumerate(consumer_ports):
                    should_active = (dynamic_active_upstream == f"http://127.0.0.1:{port}")
                    if (not args.dynamic_kv_auto_demote
                            and not should_active
                            and consumer_active_profile[port]):
                        # Keep already-active consumers alive by default.
                        # This avoids invalidating CUDA IPC handles mid-handoff.
                        continue
                    if should_active == consumer_active_profile[port]:
                        continue
                    print(
                        f"[dynamic-kv] Switching server:{port} "
                        f"{'inactive->active' if should_active else 'active->inactive'}"
                    )
                    child_idx = i + 1
                    _stop_proc(children[child_idx])
                    children[child_idx] = _start(_consumer_cmd(port, should_active), _consumer_env(i))
                    consumer_active_profile[port] = should_active
                    try:
                        _wait_owner_ready(f"http://127.0.0.1:{port}",
                                          args.dynamic_kv_switch_timeout_s)
                    except Exception as e:
                        # Keep the cluster alive; only this consumer switch failed.
                        # Router readiness checks will surface the error for hops that
                        # require this upstream.
                        print(f"[dynamic-kv] server {port} failed to become ready after switch: {e}")
                        consumer_active_profile[port] = False

        for p in children:
            code = p.poll()
            if code is not None:
                print(f"Child process exited unexpectedly (pid={p.pid}, code={code}). Stopping cluster.")
                _stop_all(children)
                return 1
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
