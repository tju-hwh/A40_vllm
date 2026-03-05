#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /root/vllm

TRAIN_FILE=${TRAIN_FILE:-"/root/data/deepmath/train.parquet"}
TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}
MODEL=${MODEL:-"/root/model/Qwen2-7B-Instruct"}
OWNER_GPU_MEM_UTIL=${OWNER_GPU_MEM_UTIL:-"0.60"}
CONSUMER_GPU_MEM_UTIL=${CONSUMER_GPU_MEM_UTIL:-"0.22"}

cleanup() {
  for p in $(ps -eo pid,cmd | awk '/launch_four_server_ipc_vllm|launch_sequential_decode_router|vllm.entrypoints.openai.api_server|kv_owner_state_server/ && !/awk/ {print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader >/tmp/_deepmath_gpu_pids.txt 2>/dev/null; then
    cat /tmp/_deepmath_gpu_pids.txt | tr -d ' ' | sed '/^$/d' \
      | while read -r p; do
          [[ "$p" =~ ^[0-9]+$ ]] && kill -9 "$p" 2>/dev/null || true
        done
  fi
}

echo "[1/7] cleanup old processes"
cleanup
rm -f /tmp/deepmath_conc10.json /tmp/deepmath_conc20.json
rm -f /tmp/kv_owner_state_deepmath.log /tmp/launch4_deepmath.log /tmp/router_deepmath.log
rm -f /tmp/vllm_shared_kv_pool.pkl /tmp/vllm_shared_kv_pool.pkl.cuda0 /tmp/vllm_shared_kv_pool.pkl.cuda1
rm -f /tmp/vllm_shared_kv_pool.pkl.alloc.*.pkl /tmp/vllm_shared_kv_pool.pkl.alloc.*.lock
rm -rf /tmp/vllm_kv_ipc && mkdir -p /tmp/vllm_kv_ipc

echo "[2/7] start owner-state"
nohup /root/anaconda3/envs/verl/bin/uvicorn \
  vllm.proxy_cluster.kv_owner_state_server:create_app \
  --factory --host 127.0.0.1 --port 8300 \
  >/tmp/kv_owner_state_deepmath.log 2>&1 &

echo "[3/7] start 4 servers (tp=2, max_tokens=2048 profile)"
nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_four_server_ipc_vllm \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --server1-port 8101 --server2-port 8102 --server3-port 8103 --server4-port 8104 \
  --owner-gpu-memory-utilization "${OWNER_GPU_MEM_UTIL}" \
  --consumer-gpu-memory-utilization "${CONSUMER_GPU_MEM_UTIL}" \
  --owner-max-num-seqs 2 --consumer-max-num-seqs 2 \
  --owner-max-model-len 3072 --consumer-max-model-len 3072 \
  --owner-startup-delay-s 2 --owner-ready-timeout-s 600 \
  --owner-cuda-visible-devices 0,1 \
  --consumer-cuda-visible-devices-all 0,1 \
  --owner-tensor-parallel-size 2 \
  --consumer-tensor-parallel-size 2 \
  --consumer-attention-backend TORCH_SDPA \
  --kv-owner-state-url http://127.0.0.1:8300 \
  --kv-transfer-config-template '{"kv_connector":"CudaIpcConnector","kv_role":"kv_both","kv_rank":0,"kv_parallel_size":1}' \
  --shared-kv-pool-enable \
  --shared-kv-pool-meta-path /tmp/vllm_shared_kv_pool.pkl \
  >/tmp/launch4_deepmath.log 2>&1 &

echo "[4/7] start handoff router"
nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_sequential_decode_router \
  --host 127.0.0.1 --port 8200 \
  --server1-url http://127.0.0.1:8101 \
  --server2-url http://127.0.0.1:8102 \
  --server3-url http://127.0.0.1:8103 \
  --server4-url http://127.0.0.1:8104 \
  --routing-mode sequential_handoff \
  --decode-cutovers 512,1024,1536 \
  --max-response-length 4096 \
  --request-timeout-s 3600 \
  --connect-timeout-s 60 \
  --kv-owner-state-url http://127.0.0.1:8300 \
  --skip-wait-upstreams-ready \
  >/tmp/router_deepmath.log 2>&1 &

echo "[5/7] wait readiness"
for i in $(seq 1 240); do
  hs=$(curl -s --max-time 2 http://127.0.0.1:8300/healthz || true)
  hr=$(curl -s --max-time 2 http://127.0.0.1:8200/healthz || true)
  ok=1
  for port in 8101 8102 8103 8104; do
    if ! curl -s --max-time 2 "http://127.0.0.1:${port}/v1/models" | grep -q '"data"'; then
      ok=0
      break
    fi
  done
  if echo "$hs" | grep -q ok && echo "$hr" | grep -q ok && [[ "$ok" == "1" ]]; then
    echo "READY_ALL"
    break
  fi
  sleep 2
done

echo "[6/7] run deepmath concurrency=10 (num_requests=10, max_tokens=2048)"
/root/anaconda3/envs/verl/bin/python /root/vllm/scripts/deepmath_concurrency_eval.py \
  --dataset "$TEST_FILE" \
  --base-url http://127.0.0.1:8200 \
  --model "$MODEL" \
  --concurrency 10 \
  --num-requests 10 \
  --max-tokens 2048 \
  --timeout-s 900 \
  --out-json /tmp/deepmath_conc10.json

echo "[7/7] run deepmath concurrency=20 (num_requests=20, max_tokens=2048)"
/root/anaconda3/envs/verl/bin/python /root/vllm/scripts/deepmath_concurrency_eval.py \
  --dataset "$TEST_FILE" \
  --base-url http://127.0.0.1:8200 \
  --model "$MODEL" \
  --concurrency 20 \
  --num-requests 20 \
  --max-tokens 2048 \
  --timeout-s 900 \
  --out-json /tmp/deepmath_conc20.json

echo "Done. Reports:"
echo "  /tmp/deepmath_conc10.json"
echo "  /tmp/deepmath_conc20.json"
echo "Logs:"
echo "  /tmp/kv_owner_state_deepmath.log"
echo "  /tmp/launch4_deepmath.log"
echo "  /tmp/router_deepmath.log"
