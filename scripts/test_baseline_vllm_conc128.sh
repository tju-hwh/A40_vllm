#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /root/vllm

MODEL=${MODEL:-"/root/model/Qwen2-7B-Instruct"}
TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-"8210"}
TP_SIZE=${TP_SIZE:-"2"}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-"0.6"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
CONCURRENCY=${CONCURRENCY:-"128"}
NUM_REQUESTS=${NUM_REQUESTS:-"128"}
MAX_TOKENS=${MAX_TOKENS:-"4096"}
TIMEOUT_S=${TIMEOUT_S:-"900"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"$CONCURRENCY"}
SEED=${SEED:-"20260305"}

LOG_FILE=${LOG_FILE:-"/tmp/baseline_vllm_conc128.log"}
REPORT_FILE=${REPORT_FILE:-"/tmp/baseline_vllm_conc128.json"}

cleanup() {
  for p in $(ps -eo pid,cmd | awk '/vllm.entrypoints.openai.api_server/ && !/awk/ {print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader >/tmp/_baseline128_gpu_pids.txt 2>/dev/null; then
    cat /tmp/_baseline128_gpu_pids.txt | tr -d ' ' | sed '/^$/d' \
      | while read -r p; do
          [[ "$p" =~ ^[0-9]+$ ]] && kill -9 "$p" 2>/dev/null || true
        done
  fi
}

echo "[1/5] cleanup"
cleanup
rm -f "$LOG_FILE" "$REPORT_FILE"

echo "[2/5] start baseline vLLM (tp=${TP_SIZE})"
CUDA_VISIBLE_DEVICES=0,1 \
nohup /root/anaconda3/envs/verl/bin/python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager \
  >"$LOG_FILE" 2>&1 &

echo "[3/5] wait readiness"
READY=0
for i in $(seq 1 360); do
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  echo "baseline server not ready"
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi
echo "READY_BASELINE"

echo "[4/5] run concurrency eval"
BASE_URL="http://${HOST}:${PORT}" \
MODEL="$MODEL" \
TEST_FILE="$TEST_FILE" \
CONCURRENCY="$CONCURRENCY" \
NUM_REQUESTS="$NUM_REQUESTS" \
MAX_TOKENS="$MAX_TOKENS" \
TIMEOUT_S="$TIMEOUT_S" \
SEED="$SEED" \
REPORT_FILE="$REPORT_FILE" \
/root/anaconda3/envs/verl/bin/python - <<'PY'
import asyncio
import json
import os
import random
import time

import httpx
import pyarrow.parquet as pq


def make_prompt(row):
    prompt = row.get("prompt")
    if isinstance(prompt, list) and len(prompt) >= 2:
        try:
            system = str(prompt[0].get("content", "")).strip()
            user = str(prompt[-1].get("content", "")).strip()
            if system and user:
                return f"{system}\n\n{user}"
        except Exception:
            pass
    q = str(row.get("question", "")).strip()
    return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{q}"


def load_rows(path, n, seed):
    table = pq.read_table(path, columns=["question", "prompt", "final_answer"])
    rows = table.to_pylist()
    if n >= len(rows):
        return rows
    rnd = random.Random(seed)
    idx = rnd.sample(range(len(rows)), n)
    return [rows[i] for i in idx]


async def one_request(client, base_url, model, row, max_tokens, request_id):
    payload = {
        "model": model,
        "request_id": request_id,
        "prompt": make_prompt(row),
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
            return {"ok": False, "status": r.status_code, "latency": latency, "completion_tokens": 0}
        obj = r.json()
        usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
        c = int(usage.get("completion_tokens", 0) or 0)
        p = int(usage.get("prompt_tokens", 0) or 0)
        return {
            "ok": True,
            "status": 200,
            "latency": latency,
            "completion_tokens": c,
            "prompt_tokens": p,
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    except Exception:
        return {"ok": False, "status": 0, "latency": time.time() - t0, "completion_tokens": 0}


async def main():
    base_url = os.environ["BASE_URL"]
    model = os.environ["MODEL"]
    test_file = os.environ["TEST_FILE"]
    concurrency = int(os.environ["CONCURRENCY"])
    num_requests = int(os.environ["NUM_REQUESTS"])
    max_tokens = int(os.environ["MAX_TOKENS"])
    timeout_s = float(os.environ["TIMEOUT_S"])
    seed = int(os.environ["SEED"])
    report_file = os.environ["REPORT_FILE"]

    rows = load_rows(test_file, num_requests, seed)
    timeout = httpx.Timeout(timeout_s, connect=min(30.0, timeout_s))
    limits = httpx.Limits(max_connections=max(100, concurrency * 2))
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def wrapped(i, row):
            async with sem:
                rid = f"baseline-{concurrency}-{i}-{int(time.time())}"
                return await one_request(client, base_url, model, row, max_tokens, rid)
        results = await asyncio.gather(*[wrapped(i, row) for i, row in enumerate(rows)])
    elapsed = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    avg_latency = sum(r["latency"] for r in results) / max(len(results), 1)
    p95_latency = sorted(r["latency"] for r in results)[int(0.95 * (len(results) - 1))]
    sum_completion = sum(r.get("completion_tokens", 0) for r in ok)
    sum_prompt = sum(r.get("prompt_tokens", 0) for r in ok)
    report = {
        "base_url": base_url,
        "model": model,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "ok_requests": len(ok),
        "failed_requests": len(fail),
        "avg_latency_s": avg_latency,
        "p95_latency_s": p95_latency,
        "run_elapsed_s": elapsed,
        "sum_prompt_tokens": sum_prompt,
        "sum_completion_tokens": sum_completion,
        "decode_tok_per_s": (sum_completion / elapsed) if elapsed > 0 else 0.0,
        "sample_failures": fail[:5],
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


asyncio.run(main())
PY

echo "[5/5] done"
echo "report: $REPORT_FILE"
echo "log: $LOG_FILE"
