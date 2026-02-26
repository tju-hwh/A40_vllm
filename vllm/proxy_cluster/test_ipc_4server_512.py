from __future__ import annotations

from vllm.proxy_cluster.ipc_weight_share_poc import run_ipc_cluster


def main() -> int:
    report = run_ipc_cluster(
        num_requests=512,
        decode_steps=8,
        in_features=64,
        out_features=64,
        dtype_name="float16",
        seed=42,
    )
    print("Test: IPC 4-server + 512 requests")
    print(f"all_handles_same={report['all_handles_same']}")
    print(f"per_server_handled={report['per_server_handled']}")
    print(f"throughput_req_s={report['throughput_req_s']:.2f}")
    if not report["all_handles_same"]:
        raise SystemExit(1)
    if sum(report["per_server_handled"].values()) != 512:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

