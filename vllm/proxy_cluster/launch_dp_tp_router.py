#!/usr/bin/env python3
"""Launch router for DP x TP hop cluster.

Supports routing to multiple DP ranks, where each rank has its own
server1 (owner) -> server2 (consumer) handoff chain.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Callable


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
        print("Still waiting: " + ".join(sorted(pending))")
        time.sleep(0.5)
    raise RuntimeError(f"Upstreams not ready within {timeout_s}s: {sorted(pending)}")


def _parse_url_list(raw: str) -> list[str]:
    """Parse comma-separated URLs, strip whitespace."""
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch router for DP x TP hop cluster with per-rank handoff."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200, help="Router port")

    # Multi-DP server URLs (comma-separated per rank)
    parser.add_argument(
        "--server1-urls",
        required=True,
        help="Comma-separated URLs for server1 (owner) of each DP rank",
    )
    parser.add_argument(
        "--server2-urls",
        required=True,
        help="Comma-separated URLs for server2 (consumer) of each DP rank",
    )
    parser.add_argument(
        "--server-kv-port-groups",
        default="",
        help="Semicolon-separated KV port groups, e.g., '18101,18102;18103,18104' for 2 DP ranks",
    )

    # Routing mode
    parser.add_argument(
        "--routing-mode",
        choices=["sequential_blocks", "sequential_handoff"],
        default="sequential_handoff",
        help="sequential_handoff: single request chains across servers within DP rank",
    )
    parser.add_argument(
        "--dp-routing-strategy",
        choices=["round_robin", "request_id_hash", "random"],
        default="request_id_hash",
        help="How to route requests to DP ranks",
    )

    # Handoff config
    parser.add_argument(
        "--decode-cutovers",
        default="4096",
        help="Decode token cutovers before moving to next server",
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--connect-timeout-s", type=float, default=60.0)
    parser.add_argument("--upstream-max-model-len", type=int, default=16384)
    parser.add_argument("--max-response-length", type=int, default=8192)

    # KV owner state
    parser.add_argument(
        "--kv-owner-state-url",
        default="",
        help="KV owner state server URL, e.g., http://127.0.0.1:8300",
    )
    parser.add_argument("--kv-handoff-wait-timeout-s", type=float, default=30.0)
    parser.add_argument("--kv-owner-state-strict", action="store_true")
    parser.add_argument(
        "--kv-handoff-global-phase-barrier",
        action="store_true",
        help="Enable global phase barrier for DP synchronization",
    )

    # Misc
    parser.add_argument("--verbose-log", action="store_true")
    parser.add_argument(
        "--wait-upstreams-ready-timeout-s",
        type=float,
        default=300.0,
    )
    parser.add_argument("--skip-wait-upstreams-ready", action="store_true")

    args = parser.parse_args()

    server1_urls = _parse_url_list(args.server1_urls)
    server2_urls = _parse_url_list(args.server2_urls)

    if len(server1_urls) != len(server2_urls):
        print(
            f"ERROR: server1-urls ({len(server1_urls)}) and server2-urls ({len(server2_urls)}) "
            "must have the same length",
            file=sys.stderr,
        )
        return 1

    dp_size = len(server1_urls)
    if dp_size == 0:
        print("ERROR: No server URLs provided", file=sys.stderr)
        return 1

    print(f"Router configured for DP={dp_size}")
    print("Server mapping:")
    for r in range(dp_size):
        print(f"  DP rank {r}: server1={server1_urls[r]}, server2={server2_urls[r]}")

    # Build sequential targets for handoff
    # For each DP rank, the handoff chain is: server1 -> server2
    all_targets = []
    for r in range(dp_size):
        all_targets.append(server1_urls[r])
        all_targets.append(server2_urls[r])

    # Build KV ports
    kv_ports = []
    if args.server_kv_port_groups:
        # Format: "18101,18102;18103,18104" for DP=2
        groups = args.server_kv_port_groups.split(";")
        for g in groups:
            ports = [int(x.strip()) for x in g.split(",") if x.strip()]
            kv_ports.extend(ports)
    else:
        # Default: auto-generate based on DP size
        base_port = 18101
        for r in range(dp_size):
            kv_ports.append(base_port + 2 * r)      # server1 kv port
            kv_ports.append(base_port + 2 * r + 1)  # server2 kv port

    env = dict(os.environ)
    env["PROXY_ROLE"] = "ingress"
    env["PRIMARY_UPSTREAM"] = server1_urls[0]  # Primary is first rank's owner
    env["ALT_UPSTREAMS"] = ",".join(all_targets[1:])  # Rest are alternates
    env["ROUTING_MODE"] = args.routing_mode
    env["SEQUENTIAL_TARGETS"] = ",".join(all_targets)
    env["SEQUENTIAL_TARGET_KV_PORTS"] = ",".join(str(p) for p in kv_ports)
    env["SEQUENTIAL_DECODE_TOKENS"] = args.decode_cutovers
    env["REQUEST_TIMEOUT_S"] = str(args.request_timeout_s)
    env["CONNECT_TIMEOUT_S"] = str(args.connect_timeout_s)
    env["MAX_RESPONSE_LENGTH"] = str(max(1, int(args.max_response_length)))
    env["UPSTREAM_MAX_MODEL_LEN"] = str(max(1, int(args.upstream_max_model_len)))
    env["PROXY_VERBOSE_LOG"] = "1" if args.verbose_log else "0"
    env["KV_OWNER_STATE_URL"] = args.kv_owner_state_url
    env["KV_OWNER_STATE_STRICT"] = "1" if args.kv_owner_state_strict else "0"
    env["KV_HANDOFF_WAIT_TIMEOUT_S"] = str(args.kv_handoff_wait_timeout_s)
    env["KV_HANDOFF_GLOBAL_PHASE_BARRIER"] = "1" if args.kv_handoff_global_phase_barrier else "0"

    # New env vars for DP routing
    env["DP_ROUTING_STRATEGY"] = args.dp_routing_strategy
    env["SERVER1_URLS"] = ",".join(server1_urls)
    env["SERVER2_URLS"] = ",".join(server2_urls)

    # Wait for upstreams
    upstreams = server1_urls + server2_urls
    if not args.skip_wait_upstreams_ready:
        try:
            _wait_upstreams_ready(upstreams, args.wait_upstreams_ready_timeout_s)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    else:
        print("Skip waiting upstream readiness by --skip-wait-upstreams-ready.")

    # Start router
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

    print("\n" + "=" * 60)
    print(f"Router started at http://{args.host}:{args.port}")
    print(f"  Routing mode: {args.routing_mode}")
    print(f"  DP routing: {args.dp_routing_strategy}")
    print(f"  Decode cutovers: {args.decode_cutovers}")
    print(f"  KV owner state: {args.kv_owner_state_url or '(disabled)'}")
    print("=" * 60)
    print("Press Ctrl+C to stop.")

    while True:
        code = proc.poll()
        if code is not None:
            print(f"\nRouter exited unexpectedly (code={code}).")
            return 1
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
