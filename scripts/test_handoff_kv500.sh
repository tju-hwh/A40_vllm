#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /root/vllm

MODEL=${MODEL:-"/root/model/Qwen2-7B-Instruct"}
OWNER_GPU_MEM_UTIL=${OWNER_GPU_MEM_UTIL:-"0.6"}
CONSUMER_GPU_MEM_UTIL=${CONSUMER_GPU_MEM_UTIL:-"0.6"}
# CUTOVERS=${CUTOVERS:-"256,512,768"}
CUTOVERS=${CUTOVERS:-"512,512,512"}

cleanup() {
  for p in $(ps -eo pid,cmd | awk '/launch_four_server_ipc_vllm|launch_sequential_decode_router|vllm.entrypoints.openai.api_server|kv_owner_state_server|proxy_server:create_app/ && !/awk/ {print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader >/tmp/_kv500_gpu_pids.txt 2>/dev/null; then
    cat /tmp/_kv500_gpu_pids.txt | tr -d ' ' | sed '/^$/d' \
      | while read -r p; do
          [[ "$p" =~ ^[0-9]+$ ]] && kill -9 "$p" 2>/dev/null || true
        done
  fi
}

echo "[1/7] cleanup"
cleanup
rm -f /tmp/kv500_owner.log /tmp/kv500_launch4.log /tmp/kv500_router.log /tmp/handoff_tp2_long_ok.json
rm -f /tmp/vllm_shared_kv_pool.pkl /tmp/vllm_shared_kv_pool.pkl.cuda0 /tmp/vllm_shared_kv_pool.pkl.cuda1
rm -f /tmp/vllm_shared_kv_pool.pkl.alloc.*.pkl /tmp/vllm_shared_kv_pool.pkl.alloc.*.lock
rm -rf /tmp/vllm_kv_ipc && mkdir -p /tmp/vllm_kv_ipc

echo "[2/7] start owner-state"
nohup /root/anaconda3/envs/verl/bin/uvicorn \
  vllm.proxy_cluster.kv_owner_state_server:create_app \
  --factory --host 127.0.0.1 --port 8300 \
  >/tmp/kv500_owner.log 2>&1 &

echo "[3/7] start 4 servers (tp=2)"
nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_four_server_ipc_vllm \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --server1-port 8101 --server2-port 8102 --server3-port 8103 --server4-port 8104 \
  --owner-gpu-memory-utilization "$OWNER_GPU_MEM_UTIL" \
  --consumer-gpu-memory-utilization "$CONSUMER_GPU_MEM_UTIL" \
  --owner-max-num-seqs 1 --consumer-max-num-seqs 1 \
  --owner-max-model-len 3072 --consumer-max-model-len 3072 \
  --owner-startup-delay-s 2 --owner-ready-timeout-s 480 \
  --owner-cuda-visible-devices 0,1 \
  --consumer-cuda-visible-devices-all 0,1 \
  --owner-tensor-parallel-size 2 \
  --consumer-tensor-parallel-size 2 \
  --consumer-attention-backend TORCH_SDPA \
  --kv-owner-state-url http://127.0.0.1:8300 \
  --kv-transfer-config-template '{"kv_connector":"CudaIpcConnector","kv_role":"kv_both","kv_rank":0,"kv_parallel_size":1}' \
  --shared-kv-pool-enable \
  --shared-kv-pool-meta-path /tmp/vllm_shared_kv_pool.pkl \
  >/tmp/kv500_launch4.log 2>&1 &

echo "[4/7] start router"
nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_sequential_decode_router \
  --host 127.0.0.1 --port 8200 \
  --server1-url http://127.0.0.1:8101 \
  --server2-url http://127.0.0.1:8102 \
  --server3-url http://127.0.0.1:8103 \
  --server4-url http://127.0.0.1:8104 \
  --routing-mode sequential_handoff \
  --decode-cutovers "$CUTOVERS" \
  --verbose-log \
  --kv-owner-state-url http://127.0.0.1:8300 \
  --skip-wait-upstreams-ready \
  >/tmp/kv500_router.log 2>&1 &

echo "[5/7] wait readiness"
for i in $(seq 1 220); do
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

echo "[6/7] run target request (max_tokens=500)"
RID=${RID:-"tp2-stable-long"}
curl -sS --max-time 420 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"/root/model/Qwen2-7B-Instruct\",\"request_id\":\"${RID}\",\"prompt\":\"Introduce the method of braised meat\",\"max_tokens\":500,\"temperature\":0,\"top_p\":1,\"stop\":[]}" \
  >/tmp/handoff_tp2_long_ok.json

echo "[7/7] summarize"
python - <<'PY'
import json
obj=json.load(open('/tmp/handoff_tp2_long_ok.json'))
text=obj['choices'][0].get('text','')
print('finish_reason=', obj['choices'][0].get('finish_reason'))
print('len=', len(text))
print('head=', text[:220].replace('\n', ' '))
print('tail=', text[-220:].replace('\n', ' '))
PY

echo "--- hop logs ---"
grep -n "handoff decode_idx\\|tp2-stable-long" /tmp/kv500_router.log | tail -n 80 || true

echo "--- kv warnings ---"
grep -c "kv_cache mismatch" /tmp/kv500_launch4.log || true
grep -n "kv_cache mismatch\\|load degraded\\|missing tensor meta\\|owner lookup miss" /tmp/kv500_launch4.log | tail -n 80 || true

echo "Done."
echo "  output: /tmp/handoff_tp2_long_ok.json"
echo "  logs: /tmp/kv500_owner.log /tmp/kv500_launch4.log /tmp/kv500_router.log"
