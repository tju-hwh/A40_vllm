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
        description="Launch a decode router with sequential block policy for pre-started vLLM servers."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200, help="Ingress router port.")
    parser.add_argument(
        "--num-servers",
        type=int,
        default=4,
        choices=[2, 4],
        help="Number of upstream servers to use in sequential routing.",
    )
    parser.add_argument("--server1-url", default="http://127.0.0.1:8101")
    parser.add_argument("--server2-url", default="http://127.0.0.1:8102")
    parser.add_argument("--server3-url", default="http://127.0.0.1:8103")
    parser.add_argument("--server4-url", default="http://127.0.0.1:8104")
    parser.add_argument(
        "--server1-urls",
        default="",
        help="Optional CSV of shard URLs for logical server1 stage.",
    )
    parser.add_argument(
        "--server2-urls",
        default="",
        help="Optional CSV of shard URLs for logical server2 stage.",
    )
    parser.add_argument(
        "--server3-urls",
        default="",
        help="Optional CSV of shard URLs for logical server3 stage.",
    )
    parser.add_argument(
        "--server4-urls",
        default="",
        help="Optional CSV of shard URLs for logical server4 stage.",
    )
    parser.add_argument(
        "--server-kv-ports",
        default="18101,18102,18103,18104",
        help="Comma-separated KV connector ports aligned with enabled server URLs.",
    )
    parser.add_argument(
        "--server-kv-port-groups",
        default="",
        help="Optional semicolon-separated KV port groups aligned with logical stages.",
    )
    parser.add_argument(
        "--server-dp-sizes",
        default="1,1,1,1",
        help="Comma-separated DP sizes aligned with enabled server URLs.",
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
        help="For sequential_handoff mode: decode token cutovers before moving to next server.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--upstream-max-model-len",
        type=int,
        default=3072,
        help="Per-upstream max_model_len used for per-hop max_tokens clamp.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=4096,
        help="Hard upper bound for per-request max_tokens at router ingress.",
    )
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
    parser.add_argument(
        "--dynamic-kv-control-path",
        default="",
        help=(
            "Optional control JSON path for dynamic consumer KV switching. "
            "When set, router writes active_upstream before each handoff hop."
        ),
    )
    parser.add_argument(
        "--dynamic-kv-wait-timeout-s",
        type=float,
        default=120.0,
        help="Wait timeout for upstream ready after dynamic KV switch.",
    )
    parser.add_argument(
        "--dynamic-kv-settle-s",
        type=float,
        default=2.0,
        help="Delay after writing dynamic KV control to avoid restart race.",
    )
    parser.add_argument(
        "--kv-owner-state-url",
        default="",
        help=(
            "Optional KV owner state server URL, e.g. http://127.0.0.1:8300. "
            "When set, router sends acquire/commit/release control events per hop."
        ),
    )
    parser.add_argument(
        "--kv-owner-state-strict",
        action="store_true",
        help="Fail request if KV owner state server acquire/commit fails.",
    )
    args = parser.parse_args()

    env = dict(os.environ)
    env["PROXY_ROLE"] = "ingress"
    def _parse_group(urls_value: str, fallback_url: str) -> list[str]:
        raw = urls_value.strip()
        if raw:
            return [x.strip().rstrip("/") for x in raw.split(",") if x.strip()]
        return [fallback_url.rstrip("/")]

    grouped_targets = [
        _parse_group(args.server1_urls, args.server1_url),
        _parse_group(args.server2_urls, args.server2_url),
        _parse_group(args.server3_urls, args.server3_url),
        _parse_group(args.server4_urls, args.server4_url),
    ]
    env["PRIMARY_UPSTREAM"] = grouped_targets[0][0]
    all_targets = [group[0] for group in grouped_targets]
    active_targets = all_targets[: args.num_servers]
    env["ALT_UPSTREAMS"] = ",".join(active_targets[1:])
    env["ROUTING_MODE"] = args.routing_mode
    env["SEQUENTIAL_BLOCK_SIZE"] = str(args.block_size)
    env["SEQUENTIAL_TARGETS"] = ",".join(active_targets)
    active_groups = grouped_targets[: args.num_servers]
    env["SEQUENTIAL_TARGET_GROUPS"] = ";".join(
        ",".join(group) for group in active_groups
    )

    if args.server_kv_port_groups.strip():
        env["SEQUENTIAL_TARGET_KV_PORT_GROUPS"] = args.server_kv_port_groups
    else:
        env["SEQUENTIAL_TARGET_KV_PORTS"] = args.server_kv_ports

    if args.server_dp_sizes.strip():
        env["SEQUENTIAL_TARGET_DP_SIZES"] = args.server_dp_sizes
    else:
        env["SEQUENTIAL_TARGET_DP_SIZES"] = ",".join(
            str(len(group)) for group in active_groups
        )
    env["SEQUENTIAL_DECODE_TOKENS"] = args.decode_cutovers
    env["REQUEST_TIMEOUT_S"] = str(args.request_timeout_s)
    env["CONNECT_TIMEOUT_S"] = str(args.connect_timeout_s)
    env["MAX_RESPONSE_LENGTH"] = str(max(1, int(args.max_response_length)))
    env["UPSTREAM_MAX_MODEL_LEN"] = str(max(1, int(args.upstream_max_model_len)))
    env["PROXY_VERBOSE_LOG"] = "1" if args.verbose_log else "0"
    env["REQUIRE_KV_TRANSFER"] = "1" if args.require_kv_transfer else "0"
    env["DYNAMIC_KV_CONTROL_PATH"] = args.dynamic_kv_control_path
    env["DYNAMIC_KV_WAIT_TIMEOUT_S"] = str(args.dynamic_kv_wait_timeout_s)
    env["DYNAMIC_KV_SETTLE_S"] = str(args.dynamic_kv_settle_s)
    env["KV_OWNER_STATE_URL"] = args.kv_owner_state_url
    env["KV_OWNER_STATE_STRICT"] = "1" if args.kv_owner_state_strict else "0"

    upstreams = []
    for group in active_groups:
        upstreams.extend(group)
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
        f"  max response length: {args.max_response_length}\n"
        f"  upstream max model len: {args.upstream_max_model_len}\n"
        f"  dynamic kv control path: {args.dynamic_kv_control_path or '(disabled)'}\n"
        f"  kv owner state url: {args.kv_owner_state_url or '(disabled)'}\n"
        f"  targets: {'; '.join(', '.join(group) for group in active_groups)}\n"
    )
    if args.routing_mode == "sequential_blocks":
        print(f"Policy: decode requests are assigned across {args.num_servers} servers in sequential blocks.")
    else:
        print(f"Policy: single request is chained by decode tokens across {args.num_servers} servers.")
    print("Press Ctrl+C to stop.")

    while True:
        code = proc.poll()
        if code is not None:
            print(f"Router exited unexpectedly (pid={proc.pid}, code={code}).")
            return 1
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
