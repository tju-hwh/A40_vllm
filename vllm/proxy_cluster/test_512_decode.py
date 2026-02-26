from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class ReqResult:
    ok: bool
    latency_s: float
    status_code: int
    error: str = ""


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    req_id: int,
    max_tokens: int,
    prompt_template: str,
) -> ReqResult:
    payload = {
        "model": model,
        "prompt": f"[request-{req_id}] {prompt_template}",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload)
        dt = time.perf_counter() - t0
        if resp.status_code != 200:
            return ReqResult(False, dt, resp.status_code, resp.text[:300])
        data = resp.json()
        if "choices" not in data:
            return ReqResult(False, dt, resp.status_code, "missing choices")
        return ReqResult(True, dt, resp.status_code)
    except Exception as exc:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return ReqResult(False, dt, -1, str(exc))


async def _run(args: argparse.Namespace) -> int:
    sem = asyncio.Semaphore(args.concurrency)
    if args.base_urls:
        base_urls = [u.strip().rstrip("/") for u in args.base_urls.split(",") if u.strip()]
        if not base_urls:
            raise ValueError("base_urls is empty after parsing")
    else:
        base_urls = [args.base_url.rstrip("/")]

    timeout = httpx.Timeout(args.timeout_s, connect=args.connect_timeout_s)
    limits = httpx.Limits(max_keepalive_connections=1024, max_connections=2048)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def _wrapped(i: int) -> ReqResult:
            async with sem:
                url = random.choice(base_urls) + "/v1/completions"
                return await _one_request(
                    client,
                    url,
                    args.model,
                    i,
                    args.max_tokens,
                    args.prompt_template,
                )

        t0 = time.perf_counter()
        results = await asyncio.gather(*[_wrapped(i) for i in range(args.num_requests)])
        total_s = time.perf_counter() - t0

    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]
    latencies = [r.latency_s for r in ok_results]

    print(f"Total requests: {len(results)}")
    print(f"Success: {len(ok_results)}")
    print(f"Failed: {len(fail_results)}")
    print(f"Wall time: {total_s:.3f}s")
    if latencies:
        print(f"Latency p50: {statistics.median(latencies):.3f}s")
        print(f"Latency p95: {statistics.quantiles(latencies, n=100)[94]:.3f}s")
        print(f"Latency max: {max(latencies):.3f}s")
    print(f"Throughput: {len(ok_results) / total_s:.2f} req/s")

    if fail_results:
        print("\nFirst 5 failures:")
        for item in fail_results[:5]:
            print(f"- status={item.status_code} latency={item.latency_s:.3f}s err={item.error}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Send 512 concurrent decode requests to proxy cluster ingress.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8102", help="Ingress server URL (server-2).")
    parser.add_argument(
        "--base-urls",
        default="",
        help="Optional comma-separated server URLs. If set, each request is sent to a random server.",
    )
    parser.add_argument("--model", required=True, help="Model id exposed by vLLM OpenAI server.")
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-template",
        default="Write a story of over 5000 words introducing an NBA basketball player.",
        help="Prompt text used for each request.",
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
