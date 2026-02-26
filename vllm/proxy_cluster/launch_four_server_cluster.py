from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch 4-server cluster: 1 vLLM model server + 3 proxy servers."
    )
    parser.add_argument("--model", required=True, help="Model path/name for the real vLLM server (server-1).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--server1-port", type=int, default=8101, help="Real vLLM model server port.")
    parser.add_argument("--server2-port", type=int, default=8102, help="Ingress proxy (cutover policy) port.")
    parser.add_argument("--server3-port", type=int, default=8103, help="Relay proxy port.")
    parser.add_argument("--server4-port", type=int, default=8104, help="Relay proxy port.")
    parser.add_argument("--cutover-requests", type=int, default=1000)
    parser.add_argument(
        "--proxy-verbose-log",
        action="store_true",
        help="Enable proxy per-request logs for decode routing.",
    )
    args, vllm_extra = parser.parse_known_args()

    host = args.host
    s1 = f"http://127.0.0.1:{args.server1_port}"
    s3 = f"http://127.0.0.1:{args.server3_port}"
    s4 = f"http://127.0.0.1:{args.server4_port}"

    children: list[subprocess.Popen] = []

    def _sig_handler(signum, frame):  # type: ignore[no-untyped-def]
        _stop_all(children)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    base_env = dict(os.environ)

    # server-1: the only model-holding vLLM server.
    model_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(args.server1_port),
        "--model",
        args.model,
        *vllm_extra,
    ]
    children.append(_start(model_cmd, base_env))

    # server-3/server-4: pure relays to server-1, no model memory.
    for relay_port in (args.server3_port, args.server4_port):
        relay_env = dict(base_env)
        relay_env.update(
            {
                "PROXY_ROLE": "relay",
                "PRIMARY_UPSTREAM": s1,
                "ALT_UPSTREAMS": "",
                "CUTOVER_REQUESTS": str(args.cutover_requests),
                "PROXY_VERBOSE_LOG": "1" if args.proxy_verbose_log else "0",
            }
        )
        relay_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "vllm.proxy_cluster.proxy_server:create_app",
            "--factory",
            "--host",
            host,
            "--port",
            str(relay_port),
        ]
        children.append(_start(relay_cmd, relay_env))

    # server-2: ingress with cutover policy.
    ingress_env = dict(base_env)
    ingress_env.update(
        {
            "PROXY_ROLE": "ingress",
            "PRIMARY_UPSTREAM": s1,
            "ALT_UPSTREAMS": f"{s3},{s4}",
            "CUTOVER_REQUESTS": str(args.cutover_requests),
            "PROXY_VERBOSE_LOG": "1" if args.proxy_verbose_log else "0",
        }
    )
    ingress_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "vllm.proxy_cluster.proxy_server:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(args.server2_port),
    ]
    children.append(_start(ingress_cmd, ingress_env))

    print(
        "\nStarted 4 servers:\n"
        f"  server-1 (model):  {s1}\n"
        f"  server-2 (ingress cutover): http://127.0.0.1:{args.server2_port}\n"
        f"  server-3 (relay):  {s3}\n"
        f"  server-4 (relay):  {s4}\n"
        f"Cutover policy on server-2: first {args.cutover_requests} decode requests -> server-1, "
        "then round-robin -> server-3/server-4 (both relay to server-1).\n"
    )
    print("Press Ctrl+C to stop all.")

    while True:
        # Fail fast if any child exits unexpectedly.
        for p in children:
            code = p.poll()
            if code is not None:
                print(f"Child process exited unexpectedly (pid={p.pid}, code={code}). Stopping cluster.")
                _stop_all(children)
                return 1
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())

