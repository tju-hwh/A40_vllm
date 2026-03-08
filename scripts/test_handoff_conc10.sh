#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /root/vllm

TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}
MODEL=${MODEL:-"/root/model/Qwen2-7B-Instruct"}
OUT_JSON=${OUT_JSON:-"/tmp/deepmath_handoff_conc10.json"}

# Keep defaults aligned with your current stable single-request script.
OWNER_GPU_MEM_UTIL=${OWNER_GPU_MEM_UTIL:-"0.6"}
CONSUMER_GPU_MEM_UTIL=${CONSUMER_GPU_MEM_UTIL:-"0.6"}
CUTOVERS=${CUTOVERS:-"512,512,512"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"3072"}

CONCURRENCY=${CONCURRENCY:-"128"}
NUM_REQUESTS=${NUM_REQUESTS:-"128"}
MAX_TOKENS=${MAX_TOKENS:-"4096"}
TIMEOUT_S=${TIMEOUT_S:-"900"}
OWNER_MAX_NUM_SEQS=${OWNER_MAX_NUM_SEQS:-"$CONCURRENCY"}
CONSUMER_MAX_NUM_SEQS=${CONSUMER_MAX_NUM_SEQS:-"$CONCURRENCY"}
ENABLE_CUDA_MPS=${ENABLE_CUDA_MPS:-"1"}
CUDA_MPS_ACTIVE_THREAD_PERCENTAGES=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGES:-"100,40,40,40"}

# Per-server CUDA graph capture configs (override by env if needed).
OWNER_COMPILATION_CONFIG=${OWNER_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1]}'}
SERVER2_COMPILATION_CONFIG=${SERVER2_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[32,24,16,8,4,2,1]}'}
SERVER3_COMPILATION_CONFIG=${SERVER3_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[8,4,2,1]}'}
SERVER4_COMPILATION_CONFIG=${SERVER4_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[2,1]}'}

# OWNER_COMPILATION_CONFIG=${OWNER_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1]}'}
# # SERVER2_COMPILATION_CONFIG=${SERVER2_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1]}'}
# # SERVER3_COMPILATION_CONFIG=${SERVER3_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[32,24,16,8,4,2,1]}'}
# SERVER2_COMPILATION_CONFIG='{"level":3,"use_inductor":true,"use_cudagraph":true}'
# SERVER3_COMPILATION_CONFIG='{"level":3,"use_inductor":true,"use_cudagraph":true}'
# SERVER4_COMPILATION_CONFIG=${SERVER4_COMPILATION_CONFIG:-'{"level":3,"use_inductor":true,"use_cudagraph":true,"cudagraph_capture_sizes":[1]}'}
# # SERVER4_COMPILATION_CONFIG='{"level":0,"use_inductor":false,"use_cudagraph":false}'

if command -v rg >/dev/null 2>&1; then
  MATCH_BIN="rg"
else
  MATCH_BIN="grep"
fi

cleanup() {
  for p in $(ps -eo pid,cmd | awk '/launch_four_server_ipc_vllm|launch_sequential_decode_router|vllm.entrypoints.openai.api_server|kv_owner_state_server|proxy_server:create_app/ && !/awk/ {print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader >/tmp/_handoff_conc10_gpu_pids.txt 2>/dev/null; then
    cat /tmp/_handoff_conc10_gpu_pids.txt | tr -d ' ' | sed '/^$/d' \
      | while read -r p; do
          [[ "$p" =~ ^[0-9]+$ ]] && kill -9 "$p" 2>/dev/null || true
        done
  fi
}

echo "[1/7] cleanup"
cleanup
rm -f "$OUT_JSON"
rm -f /tmp/kv_owner_state_conc10.log /tmp/launch4_conc10.log /tmp/router_conc10.log
rm -f /tmp/vllm_shared_kv_pool.pkl /tmp/vllm_shared_kv_pool.pkl.cuda0 /tmp/vllm_shared_kv_pool.pkl.cuda1
rm -f /tmp/vllm_shared_kv_pool.pkl.alloc.*.pkl /tmp/vllm_shared_kv_pool.pkl.alloc.*.lock
rm -rf /tmp/vllm_kv_ipc && mkdir -p /tmp/vllm_kv_ipc

echo "[2/7] start owner-state"
nohup /root/anaconda3/envs/verl/bin/uvicorn \
  vllm.proxy_cluster.kv_owner_state_server:create_app \
  --factory --host 127.0.0.1 --port 8300 \
  >/tmp/kv_owner_state_conc10.log 2>&1 &

echo "[3/7] start 4 servers (tp=2)"
MPS_ARGS=()
if [[ "$ENABLE_CUDA_MPS" == "1" ]]; then
  MPS_ARGS+=(
    --enable-cuda-mps
    --mps-active-thread-percentages "$CUDA_MPS_ACTIVE_THREAD_PERCENTAGES"
  )
fi

nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_four_server_ipc_vllm \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --server1-port 8101 --server2-port 8102 --server3-port 8103 --server4-port 8104 \
  --owner-gpu-memory-utilization "$OWNER_GPU_MEM_UTIL" \
  --consumer-gpu-memory-utilization "$CONSUMER_GPU_MEM_UTIL" \
  --owner-max-num-seqs "$OWNER_MAX_NUM_SEQS" --consumer-max-num-seqs "$CONSUMER_MAX_NUM_SEQS" \
  --owner-max-model-len "$MAX_MODEL_LEN" --consumer-max-model-len "$MAX_MODEL_LEN" \
  --owner-startup-delay-s 2 --owner-ready-timeout-s 600 \
  --owner-cuda-visible-devices 0,1 \
  --consumer-cuda-visible-devices-all 0,1 \
  --owner-tensor-parallel-size 2 \
  --consumer-tensor-parallel-size 2 \
  --owner-compilation-config "$OWNER_COMPILATION_CONFIG" \
  --no-consumer-enforce-eager \
  --server2-compilation-config "$SERVER2_COMPILATION_CONFIG" \
  --server3-compilation-config "$SERVER3_COMPILATION_CONFIG" \
  --server4-compilation-config "$SERVER4_COMPILATION_CONFIG" \
  --consumer-attention-backend TORCH_SDPA \
  --kv-owner-state-url http://127.0.0.1:8300 \
  --kv-transfer-config-template '{"kv_connector":"CudaIpcConnector","kv_role":"kv_both","kv_rank":0,"kv_parallel_size":1}' \
  --shared-kv-pool-enable \
  --shared-kv-pool-meta-path /tmp/vllm_shared_kv_pool.pkl \
  "${MPS_ARGS[@]}" \
  >/tmp/launch4_conc10.log 2>&1 &

echo "[4/7] start router"
nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_sequential_decode_router \
  --host 127.0.0.1 --port 8200 \
  --server1-url http://127.0.0.1:8101 \
  --server2-url http://127.0.0.1:8102 \
  --server3-url http://127.0.0.1:8103 \
  --server4-url http://127.0.0.1:8104 \
  --routing-mode sequential_handoff \
  --decode-cutovers "$CUTOVERS" \
  --upstream-max-model-len "$MAX_MODEL_LEN" \
  --max-response-length 4096 \
  --request-timeout-s 3600 \
  --connect-timeout-s 60 \
  --kv-owner-state-url http://127.0.0.1:8300 \
  --skip-wait-upstreams-ready \
  >/tmp/router_conc10.log 2>&1 &

echo "[5/7] wait readiness"
for i in $(seq 1 240); do
  hs=$(curl -s --max-time 2 http://127.0.0.1:8300/healthz || true)
  hr=$(curl -s --max-time 2 http://127.0.0.1:8200/healthz || true)
  ok=1
  for port in 8101 8102 8103 8104; do
    if ! curl -s --max-time 2 "http://127.0.0.1:${port}/v1/models" | "$MATCH_BIN" -q '"data"'; then
      ok=0
      break
    fi
  done
  if echo "$hs" | "$MATCH_BIN" -q ok && echo "$hr" | "$MATCH_BIN" -q ok && [[ "$ok" == "1" ]]; then
    echo "READY_ALL"
    break
  fi
  sleep 2
done

echo "[6/7] run deepmath concurrency=${CONCURRENCY} (num_requests=${NUM_REQUESTS}, max_tokens=${MAX_TOKENS})"
/root/anaconda3/envs/verl/bin/python /root/vllm/scripts/deepmath_concurrency_eval.py \
  --dataset "$TEST_FILE" \
  --base-url http://127.0.0.1:8200 \
  --model "$MODEL" \
  --concurrency "$CONCURRENCY" \
  --num-requests "$NUM_REQUESTS" \
  --max-tokens "$MAX_TOKENS" \
  --timeout-s "$TIMEOUT_S" \
  --out-json "$OUT_JSON"

echo "[7/7] summarize"
python - <<'PY'
import json
import os
import re

report_path = os.environ.get("OUT_JSON", "/tmp/deepmath_handoff_conc10.json")
o = json.load(open(report_path))
ok = int(o.get("ok_requests", 0))
num = int(o.get("num_requests", 0))
avg_lat = float(o.get("avg_latency_s", 0.0))
req_tput = (ok / avg_lat) if avg_lat > 0 else 0.0
print("[report]")
print(f"  ok_requests={ok}/{num}")
print(f"  avg_latency_s={avg_lat:.3f}")
print(f"  approx_req_throughput={req_tput:.3f} req/s")
print(f"  exact_match_rate={float(o.get('exact_match_rate', 0.0)):.4f}")

vals = []
for line in open("/tmp/launch4_conc10.log", "r", encoding="utf-8", errors="ignore"):
    m = re.search(r"Avg generation throughput:\\s*([0-9.]+)\\s*tokens/s", line)
    if m:
        vals.append(float(m.group(1)))

if vals:
    tail = vals[-16:]
    mean_tail = sum(tail) / len(tail)
    print(f"  vllm_gen_tput_last={tail[-1]:.2f} tok/s")
    print(f"  vllm_gen_tput_tail_avg={mean_tail:.2f} tok/s")
else:
    print("  vllm_gen_tput=not_found_in_log")
PY

echo "Done."
echo "  report: $OUT_JSON"
echo "  logs: /tmp/kv_owner_state_conc10.log /tmp/launch4_conc10.log /tmp/router_conc10.log"
