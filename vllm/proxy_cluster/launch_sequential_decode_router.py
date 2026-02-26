from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_upstreams_ready(targets: list[str], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    pending = set(targets)
    while time.time() < deadline:
        done = set()
        for base in pending:
            url = base.rstrip("/") + "/v1/models"
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if 200 <= resp.status < 300:
                        done.add(base)
            except (urllib.error.URLError, TimeoutError):
                pass
        pending -= done
        if not pending:
            return
        time.sleep(0.5)
    raise RuntimeError(f"Upstreams not ready within {timeout_s}s: {sorted(pending)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a decode router with sequential block policy for 4 pre-started vLLM servers."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200, help="Ingress router port.")
    parser.add_argument("--server1-url", default="http://127.0.0.1:8101")
    parser.add_argument("--server2-url", default="http://127.0.0.1:8102")
    parser.add_argument("--server3-url", default="http://127.0.0.1:8103")
    parser.add_argument("--server4-url", default="http://127.0.0.1:8104")
    parser.add_argument("--block-size", type=int, default=128, help="Decode requests per server block.")
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--verbose-log", action="store_true")
    parser.add_argument(
        "--wait-upstreams-ready-timeout-s",
        type=float,
        default=120.0,
        help="Wait for all upstream /v1/models to become ready before starting router.",
    )
    args = parser.parse_args()

    env = dict(os.environ)
    env["PROXY_ROLE"] = "ingress"
    env["PRIMARY_UPSTREAM"] = args.server1_url.rstrip("/")
    env["ALT_UPSTREAMS"] = ",".join(
        [
            args.server2_url.rstrip("/"),
            args.server3_url.rstrip("/"),
            args.server4_url.rstrip("/"),
        ]
    )
    env["ROUTING_MODE"] = "sequential_blocks"
    env["SEQUENTIAL_BLOCK_SIZE"] = str(args.block_size)
    env["SEQUENTIAL_TARGETS"] = ",".join(
        [
            args.server1_url.rstrip("/"),
            args.server2_url.rstrip("/"),
            args.server3_url.rstrip("/"),
            args.server4_url.rstrip("/"),
        ]
    )
    env["REQUEST_TIMEOUT_S"] = str(args.request_timeout_s)
    env["CONNECT_TIMEOUT_S"] = str(args.connect_timeout_s)
    env["PROXY_VERBOSE_LOG"] = "1" if args.verbose_log else "0"

    upstreams = [
        args.server1_url.rstrip("/"),
        args.server2_url.rstrip("/"),
        args.server3_url.rstrip("/"),
        args.server4_url.rstrip("/"),
    ]
    _wait_upstreams_ready(upstreams, args.wait_upstreams_ready_timeout_s)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "vllm.proxy_cluster.proxy_server:create_app",
        "--factory",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    proc = subprocess.Popen(cmd, env=env)

    def _sig_handler(signum, frame):  # type: ignore[no-untyped-def]
        _stop(proc)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(
        "Started sequential decode router:\n"
        f"  ingress: http://127.0.0.1:{args.port}\n"
        f"  block size: {args.block_size}\n"
        f"  targets: {args.server1_url}, {args.server2_url}, {args.server3_url}, {args.server4_url}\n"
    )
    print("Policy: decode requests 1-128->server1, 129-256->server2, 257-384->server3, 385+->server4.")
    print("Press Ctrl+C to stop.")

    while True:
        code = proc.poll()
        if code is not None:
            print(f"Router exited unexpectedly (pid={proc.pid}, code={code}).")
            return 1
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
