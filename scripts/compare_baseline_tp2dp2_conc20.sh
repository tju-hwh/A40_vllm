#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /root/vllm

MODEL=${MODEL:-"/root/model/Qwen2-7B-Instruct"}
TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}
PORT=${PORT:-8210}
HOST=${HOST:-127.0.0.1}

LOG_FILE=/tmp/baseline_tp2dp2_vllm.log
REPORT_FILE=/tmp/baseline_tp2dp2_conc20.json
THROUGHPUT_FILE=/tmp/baseline_tp2dp2_throughput_tail.log
THROUGHPUT_SUMMARY=/tmp/baseline_tp2dp2_throughput_summary.json

cleanup_port_proc() {
  local pids
  pids=$(ss -lptn "sport = :${PORT}" 2>/dev/null | awk -F 'pid=' 'NR>1{print $2}' | awk -F',' '{print $1}' | tr -d ' ' || true)
  for p in ${pids}; do
    [[ -n "${p}" ]] && kill -9 "${p}" 2>/dev/null || true
  done
}

clean_all_gpu_procs() {
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits >/tmp/_baseline_gpu_pids.txt 2>/dev/null; then
    cat /tmp/_baseline_gpu_pids.txt | tr -d ' ' | sed '/^$/d' \
      | while read -r p; do
          [[ "${p}" =~ ^[0-9]+$ ]] && kill -9 "${p}" 2>/dev/null || true
        done
  fi
}

echo "[1/6] cleanup old baseline server on port ${PORT} + all GPU processes"
cleanup_port_proc
clean_all_gpu_procs
rm -f "${LOG_FILE}" "${REPORT_FILE}" "${THROUGHPUT_FILE}" "${THROUGHPUT_SUMMARY}"

echo "[2/6] start baseline vLLM (tp=2, dp=2)"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup /root/anaconda3/envs/verl/bin/python -m vllm.entrypoints.openai.api_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL}" \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-backend mp \
  --gpu-memory-utilization 0.60 \
  --max-model-len 3072 \
  --max-num-seqs 8 \
  --enforce-eager \
  >"${LOG_FILE}" 2>&1 &

echo "[3/6] wait readiness"
READY=0
for i in $(seq 1 360); do
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "${READY}" != "1" ]]; then
  echo "baseline server not ready"
  tail -n 120 "${LOG_FILE}" || true
  exit 1
fi
echo "READY_BASELINE"

echo "[4/6] run deepmath concurrency=20"
/root/anaconda3/envs/verl/bin/python /root/vllm/scripts/deepmath_concurrency_eval.py \
  --dataset "${TEST_FILE}" \
  --base-url "http://${HOST}:${PORT}" \
  --model "${MODEL}" \
  --concurrency 20 \
  --num-requests 20 \
  --max-tokens 2048 \
  --timeout-s 3600 \
  --sample-text-max-chars 8000 \
  --out-json "${REPORT_FILE}"

echo "[5/6] extract decode throughput from log"
if command -v rg >/dev/null 2>&1; then
  rg "Avg generation throughput" "${LOG_FILE}" | tail -n 200 > "${THROUGHPUT_FILE}" || true
else
  grep "Avg generation throughput" "${LOG_FILE}" | tail -n 200 > "${THROUGHPUT_FILE}" || true
fi

python - <<'PY'
import json
import re
from pathlib import Path

pth = Path("/tmp/baseline_tp2dp2_throughput_tail.log")
vals = []
if pth.exists():
    for line in pth.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"Avg generation throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s", line)
        if m:
            vals.append(float(m.group(1)))
summary = {
    "count": len(vals),
    "min_tokens_per_s": min(vals) if vals else None,
    "max_tokens_per_s": max(vals) if vals else None,
    "avg_tokens_per_s": (sum(vals) / len(vals)) if vals else None,
}
Path("/tmp/baseline_tp2dp2_throughput_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[6/6] done"
echo "report: ${REPORT_FILE}"
echo "throughput lines: ${THROUGHPUT_FILE}"
echo "throughput summary: ${THROUGHPUT_SUMMARY}"
echo "server log: ${LOG_FILE}"
