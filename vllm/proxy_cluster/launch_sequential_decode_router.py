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
    print(
        "Waiting upstreams ready: "
        + ", ".join(sorted(pending))
        + f" (timeout={timeout_s:.1f}s)"
    )
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
        if done:
            print("Ready upstreams: " + ", ".join(sorted(done)))
        if not pending:
            print("All upstreams are ready.")
            return
        print("Still waiting: " + ", ".join(sorted(pending)))
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
    parser.add_argument(
        "--server-kv-ports",
        default="18101,18102,18103,18104",
        help="Comma-separated KV connector ports aligned with server1..server4 URLs.",
    )
    parser.add_argument(
        "--routing-mode",
        choices=["sequential_blocks", "sequential_handoff"],
        default="sequential_blocks",
        help="sequential_blocks: by request index; sequential_handoff: single request chained across servers.",
    )
    parser.add_argument("--block-size", type=int, default=128, help="Decode requests per server block.")
    parser.add_argument(
        "--decode-cutovers",
        default="1000,1000,1000",
        help="For sequential_handoff mode: decode token cutovers across server1/2/3, remaining on server4.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--verbose-log", action="store_true")
    parser.add_argument(
        "--require-kv-transfer",
        action="store_true",
        help=(
            "Fail handoff request if upstream does not return kv_transfer_params "
            "(prevents silent fallback to text-only continuation)."
        ),
    )
    parser.add_argument(
        "--wait-upstreams-ready-timeout-s",
        type=float,
        default=120.0,
        help="Wait for all upstream /v1/models to become ready before starting router.",
    )
    parser.add_argument(
        "--skip-wait-upstreams-ready",
        action="store_true",
        help="Start router immediately without waiting upstream /v1/models ready.",
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
    env["ROUTING_MODE"] = args.routing_mode
    env["SEQUENTIAL_BLOCK_SIZE"] = str(args.block_size)
    env["SEQUENTIAL_TARGETS"] = ",".join(
        [
            args.server1_url.rstrip("/"),
            args.server2_url.rstrip("/"),
            args.server3_url.rstrip("/"),
            args.server4_url.rstrip("/"),
        ]
    )
    env["SEQUENTIAL_TARGET_KV_PORTS"] = args.server_kv_ports
    env["SEQUENTIAL_DECODE_TOKENS"] = args.decode_cutovers
    env["REQUEST_TIMEOUT_S"] = str(args.request_timeout_s)
    env["CONNECT_TIMEOUT_S"] = str(args.connect_timeout_s)
    env["PROXY_VERBOSE_LOG"] = "1" if args.verbose_log else "0"
    env["REQUIRE_KV_TRANSFER"] = "1" if args.require_kv_transfer else "0"

    upstreams = [
        args.server1_url.rstrip("/"),
        args.server2_url.rstrip("/"),
        args.server3_url.rstrip("/"),
        args.server4_url.rstrip("/"),
    ]
    if not args.skip_wait_upstreams_ready:
        _wait_upstreams_ready(upstreams, args.wait_upstreams_ready_timeout_s)
    else:
        print("Skip waiting upstream readiness by --skip-wait-upstreams-ready.")

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
        f"  routing mode: {args.routing_mode}\n"
        f"  block size: {args.block_size}\n"
        f"  decode cutovers: {args.decode_cutovers}\n"
        f"  targets: {args.server1_url}, {args.server2_url}, {args.server3_url}, {args.server4_url}\n"
    )
    if args.routing_mode == "sequential_blocks":
        print("Policy: decode requests 1-128->server1, 129-256->server2, 257-384->server3, 385+->server4.")
    else:
        print("Policy: single request is chained by decode tokens: server1->server2->server3->server4.")
    print("Press Ctrl+C to stop.")

    while True:
        code = proc.poll()
        if code is not None:
            print(f"Router exited unexpectedly (pid={proc.pid}, code={code}).")
            return 1
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
