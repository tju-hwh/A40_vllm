#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def normalize_answer(s: str) -> str:
    x = (s or "").strip()
    x = x.replace("\\left", "").replace("\\right", "")
    x = x.replace("$", "")
    x = re.sub(r"\s+", "", x)
    return x


def extract_boxed_answer(text: str) -> str:
    m = BOXED_RE.findall(text or "")
    if not m:
        return ""
    return m[-1].strip()


def make_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and len(prompt) >= 2:
        system = str(prompt[0].get("content", "")).strip()
        user = str(prompt[-1].get("content", "")).strip()
        if system and user:
            return f"{system}\n\n{user}"
    q = str(row.get("question", "")).strip()
    return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{q}"


@dataclass
class EvalResult:
    request_id: str
    ok: bool
    status_code: int
    latency_s: float
    text_len: int
    exact_match: bool
    pred_answer: str
    gt_answer: str
    question: str
    output_text: str
    error: str = ""


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    request_id: str,
) -> EvalResult:
    prompt = make_prompt(row)
    gt = str(row.get("final_answer", ""))
    payload = {
        "model": model,
        "request_id": request_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stop": [],
    }
    t0 = time.time()
    try:
        r = await client.post(f"{base_url}/v1/completions", json=payload)
        latency = time.time() - t0
        if r.status_code != 200:
            return EvalResult(
                request_id=request_id,
                ok=False,
                status_code=r.status_code,
                latency_s=latency,
                text_len=0,
                exact_match=False,
                pred_answer="",
                gt_answer=gt,
                question=str(row.get("question", "")),
                output_text="",
                error=r.text[:800],
            )
        obj = r.json()
        text = obj.get("choices", [{}])[0].get("text", "")
        pred = extract_boxed_answer(text)
        exact = bool(pred) and normalize_answer(pred) == normalize_answer(gt)
        return EvalResult(
            request_id=request_id,
            ok=True,
            status_code=200,
            latency_s=latency,
            text_len=len(text),
            exact_match=exact,
            pred_answer=pred,
            gt_answer=gt,
            question=str(row.get("question", "")),
            output_text=text,
        )
    except Exception as e:
        return EvalResult(
            request_id=request_id,
            ok=False,
            status_code=0,
            latency_s=time.time() - t0,
            text_len=0,
            exact_match=False,
            pred_answer="",
            gt_answer=gt,
            question=str(row.get("question", "")),
            output_text="",
            error=repr(e),
        )


def load_rows(parquet_path: str, n: int, seed: int) -> list[dict[str, Any]]:
    table = pq.read_table(
        parquet_path,
        columns=["question", "final_answer", "prompt"],
    )
    rows = table.to_pylist()
    if n >= len(rows):
        return rows
    rnd = random.Random(seed)
    idx = rnd.sample(range(len(rows)), n)
    return [rows[i] for i in idx]


async def run_eval(args: argparse.Namespace) -> int:
    rows = load_rows(args.dataset, args.num_requests, args.seed)
    timeout = httpx.Timeout(args.timeout_s, connect=min(30.0, args.timeout_s))
    limits = httpx.Limits(max_connections=max(100, args.concurrency * 2))
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def wrapped(i: int, row: dict[str, Any]) -> EvalResult:
            async with sem:
                rid = f"{args.request_prefix}-{args.concurrency}-{i}-{int(time.time())}"
                return await one_request(
                    client=client,
                    base_url=args.base_url.rstrip("/"),
                    model=args.model,
                    row=row,
                    max_tokens=args.max_tokens,
                    request_id=rid,
                )

        tasks = [wrapped(i, row) for i, row in enumerate(rows)]
        results = await asyncio.gather(*tasks)

    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    exact = [r for r in ok if r.exact_match]
    avg_latency = sum(r.latency_s for r in results) / max(len(results), 1)
    p95_latency = sorted(r.latency_s for r in results)[int(0.95 * (len(results) - 1))]

    report = {
        "dataset": args.dataset,
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "ok_requests": len(ok),
        "failed_requests": len(fail),
        "exact_match": len(exact),
        "exact_match_rate": len(exact) / max(len(ok), 1),
        "avg_latency_s": avg_latency,
        "p95_latency_s": p95_latency,
        "sample_failures": [
            {"request_id": r.request_id, "status_code": r.status_code, "error": r.error}
            for r in fail[:5]
        ],
        "sample_predictions": [
            {
                "request_id": r.request_id,
                "pred_answer": r.pred_answer,
                "gt_answer": r.gt_answer,
                "text_len": r.text_len,
            }
            for r in ok[:5]
        ],
        "sample_outputs_first5": [
            {
                "request_id": r.request_id,
                "question": r.question,
                "pred_answer": r.pred_answer,
                "gt_answer": r.gt_answer,
                "text_len": r.text_len,
                "output_text": r.output_text[:args.sample_text_max_chars],
            }
            for r in results[:5]
        ],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if len(fail) == 0 else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8200")
    ap.add_argument("--model", default="/root/model/Qwen2-7B-Instruct")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--num-requests", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=20260303)
    ap.add_argument("--request-prefix", default="deepmath")
    ap.add_argument("--sample-text-max-chars", type=int, default=4000)
    ap.add_argument("--out-json", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
