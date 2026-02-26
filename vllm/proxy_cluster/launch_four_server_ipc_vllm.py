from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _start(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env)


def _stop_all(children: list[subprocess.Popen]) -> None:
    for p in children:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 15
    for p in children:
        if p.poll() is None:
            wait_s = max(0.0, deadline - time.time())
            try:
                p.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                p.kill()
    for p in children:
        if p.poll() is None:
            p.kill()


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch 4 real vLLM servers with experimental CUDA IPC shared parameters."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
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
        "--consumer-attention-backend",
        default="TORCH_SDPA",
        help="Set VLLM_ATTENTION_BACKEND for consumers (e.g. TORCH_SDPA/FLASHINFER/FLASH_ATTN).",
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
    args, vllm_extra = parser.parse_known_args()

    host = args.host
    ports = [args.server1_port, args.server2_port, args.server3_port, args.server4_port]
    urls = [f"http://127.0.0.1:{p}" for p in ports]

    try:
        os.remove(args.ipc_meta_path)
    except FileNotFoundError:
        pass

    children: list[subprocess.Popen] = []
    base_env = dict(os.environ)

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
            {"--gpu-memory-utilization", "--max-num-seqs", "--max-model-len"},
        ),
        gpu_memory_utilization=args.owner_gpu_memory_utilization,
        max_num_seqs=args.owner_max_num_seqs,
        max_model_len=args.owner_max_model_len,
    )
    owner_env = dict(base_env)
    if args.owner_cuda_visible_devices:
        owner_env["CUDA_VISIBLE_DEVICES"] = args.owner_cuda_visible_devices
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
    for idx, p in enumerate((args.server2_port, args.server3_port, args.server4_port)):
        consumer_extra = _strip_overridden_args(
            vllm_extra,
            {
                "--gpu-memory-utilization",
                "--max-num-seqs",
                "--max-model-len",
                "--compilation-config",
            },
        )
        cmd = _build_vllm_cmd(
            host=host,
            port=p,
            model=args.model,
            role="consumer",
            meta_path=args.ipc_meta_path,
            timeout_s=args.ipc_wait_timeout_s,
            poll_s=args.ipc_poll_interval_s,
            extra_args=consumer_extra,
            gpu_memory_utilization=args.consumer_gpu_memory_utilization,
            max_num_seqs=args.consumer_max_num_seqs,
            max_model_len=args.consumer_max_model_len,
            enforce_eager=not args.no_consumer_enforce_eager,
            compilation_config=args.consumer_compilation_config,
        )
        env = dict(base_env)
        if idx < len(consumer_visible_devices):
            env["CUDA_VISIBLE_DEVICES"] = consumer_visible_devices[idx]
        if args.consumer_attention_backend:
            env["VLLM_ATTENTION_BACKEND"] = args.consumer_attention_backend
        children.append(_start(cmd, env))

    print(
        "\nStarted 4 vLLM servers (experimental ipc_weight_share):\n"
        f"  server-1 (owner):    {urls[0]}\n"
        f"  server-2 (consumer): {urls[1]}\n"
        f"  server-3 (consumer): {urls[2]}\n"
        f"  server-4 (consumer): {urls[3]}\n"
        f"IPC meta file: {args.ipc_meta_path}\n"
    )
    print("Press Ctrl+C to stop all.")

    while True:
        for p in children:
            code = p.poll()
            if code is not None:
                print(f"Child process exited unexpectedly (pid={p.pid}, code={code}). Stopping cluster.")
                _stop_all(children)
                return 1
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
