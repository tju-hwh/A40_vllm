#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /root/vllm

MODEL=${MODEL:-"/root/model/Qwen2-7B-Instruct"}
TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-"8210"}

# Keep these aligned with handoff test script knobs.
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-"0.6"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
CONCURRENCY=${CONCURRENCY:-"128"}
NUM_REQUESTS=${NUM_REQUESTS:-"128"}
MAX_TOKENS=${MAX_TOKENS:-"4096"}
TIMEOUT_S=${TIMEOUT_S:-"900"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"$CONCURRENCY"}

LOG_FILE=${LOG_FILE:-"/tmp/baseline_vllm_conc128.log"}
REPORT_FILE=${REPORT_FILE:-"/tmp/baseline_vllm_conc128.json"}

cleanup() {
  for p in $(ps -eo pid,cmd | awk '/vllm.entrypoints.openai.api_server|launch_four_server_ipc_vllm|launch_sequential_decode_router|kv_owner_state_server|proxy_server:create_app/ && !/awk/ {print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader >/tmp/_baseline128_gpu_pids.txt 2>/dev/null; then
    cat /tmp/_baseline128_gpu_pids.txt | tr -d ' ' | sed '/^$/d' \
      | while read -r p; do
          [[ "$p" =~ ^[0-9]+$ ]] && kill -9 "$p" 2>/dev/null || true
        done
  fi
}

echo "[1/6] cleanup"
cleanup
rm -f "$LOG_FILE" "$REPORT_FILE"

echo "[2/6] start baseline vLLM server (tp=2)"
CUDA_VISIBLE_DEVICES=0,1 \
nohup /root/anaconda3/envs/verl/bin/python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager \
  >"$LOG_FILE" 2>&1 &

echo "[3/6] wait readiness"
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

echo "[4/6] run deepmath eval (concurrency=${CONCURRENCY}, num_requests=${NUM_REQUESTS}, max_tokens=${MAX_TOKENS})"
/root/anaconda3/envs/verl/bin/python /root/vllm/scripts/deepmath_concurrency_eval.py \
  --dataset "$TEST_FILE" \
  --base-url "http://${HOST}:${PORT}" \
  --model "$MODEL" \
  --concurrency "$CONCURRENCY" \
  --num-requests "$NUM_REQUESTS" \
  --max-tokens "$MAX_TOKENS" \
  --timeout-s "$TIMEOUT_S" \
  --sample-text-max-chars 8000 \
  --out-json "$REPORT_FILE"

echo "[5/6] summarize baseline metrics"
python - <<'PY'
import json
import re

report = json.load(open("/tmp/baseline_vllm_conc128.json", "r", encoding="utf-8"))
print("[report]")
print(f"  ok_requests={report.get('ok_requests')}/{report.get('num_requests')}")
print(f"  avg_latency_s={float(report.get('avg_latency_s', 0.0)):.3f}")
print(f"  p95_latency_s={float(report.get('p95_latency_s', 0.0)):.3f}")
print(f"  run_elapsed_s={float(report.get('run_elapsed_s', 0.0)):.3f}")
print(f"  exact_match_rate={float(report.get('exact_match_rate', 0.0)):.4f}")
print(f"  sum_completion_tokens={int(report.get('sum_completion_tokens', 0))}")
print(f"  decode_tok_per_s={float(report.get('decode_tok_per_s', 0.0)):.3f}")

vals = []
for line in open("/tmp/baseline_vllm_conc128.log", "r", encoding="utf-8", errors="ignore"):
    m = re.search(r"Avg generation throughput:\s*([0-9.]+)\s*tokens/s", line)
    if m:
        vals.append(float(m.group(1)))
if vals:
    tail = vals[-20:]
    print(f"  vllm_gen_tput_last={tail[-1]:.2f} tok/s")
    print(f"  vllm_gen_tput_tail_avg={sum(tail)/len(tail):.2f} tok/s")
else:
    print("  vllm_gen_tput=not_found_in_log")
PY

echo "[6/6] done"
echo "  report: $REPORT_FILE"
echo "  log: $LOG_FILE"

