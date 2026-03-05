#!/usr/bin/env bash
set -euo pipefail

# Final validated TP=2 handoff test script
# - 4 vLLM servers (owner+3 consumers)
# - tensor_parallel_size=2
# - shared_kv_pool + CUDA IPC + owner-state

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl
cd /root/vllm

if command -v rg >/dev/null 2>&1; then
  MATCH_BIN="rg"
else
  MATCH_BIN="grep"
fi

cleanup() {
  for p in $(ps -eo pid,cmd | awk '/launch_four_server_ipc_vllm|launch_sequential_decode_router|vllm.entrypoints.openai.api_server|kv_owner_state_server/ && !/awk/ {print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader >/tmp/_tp2_gpu_pids.txt 2>/dev/null; then
    cat /tmp/_tp2_gpu_pids.txt | tr -d ' ' | sed '/^$/d' \
      | while read -r p; do
          [[ "$p" =~ ^[0-9]+$ ]] && kill -9 "$p" 2>/dev/null || true
        done
  fi
}

echo "[1/6] cleanup old processes"
cleanup
rm -f /tmp/kv_owner_state_tp2.log /tmp/launch4_tp2.log /tmp/router_tp2.log
rm -f /tmp/handoff_tp2_short_ok.json /tmp/handoff_tp2_long_ok.json
rm -f /tmp/vllm_shared_kv_pool.pkl /tmp/vllm_shared_kv_pool.pkl.cuda0 /tmp/vllm_shared_kv_pool.pkl.cuda1
rm -f /tmp/vllm_shared_kv_pool.pkl.alloc.*.pkl /tmp/vllm_shared_kv_pool.pkl.alloc.*.lock
rm -rf /tmp/vllm_kv_ipc && mkdir -p /tmp/vllm_kv_ipc

echo "[2/6] start owner-state"
nohup /root/anaconda3/envs/verl/bin/uvicorn \
  vllm.proxy_cluster.kv_owner_state_server:create_app \
  --factory --host 127.0.0.1 --port 8300 \
  >/tmp/kv_owner_state_tp2.log 2>&1 &

echo "[3/6] start 4 servers (tp=2)"
nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_four_server_ipc_vllm \
  --model /root/model/Qwen2-7B-Instruct \
  --host 127.0.0.1 \
  --server1-port 8101 --server2-port 8102 --server3-port 8103 --server4-port 8104 \
  --owner-gpu-memory-utilization 0.18 \
  --consumer-gpu-memory-utilization 0.06 \
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
  >/tmp/launch4_tp2.log 2>&1 &

echo "[4/6] start handoff router"
nohup /root/anaconda3/envs/verl/bin/python -m vllm.proxy_cluster.launch_sequential_decode_router \
  --host 127.0.0.1 --port 8200 \
  --server1-url http://127.0.0.1:8101 \
  --server2-url http://127.0.0.1:8102 \
  --server3-url http://127.0.0.1:8103 \
  --server4-url http://127.0.0.1:8104 \
  --routing-mode sequential_handoff \
  --decode-cutovers 256,512,768 \
  --kv-owner-state-url http://127.0.0.1:8300 \
  --skip-wait-upstreams-ready \
  >/tmp/router_tp2.log 2>&1 &

echo "[5/6] wait readiness"
for i in $(seq 1 200); do
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

echo "[6/6] run short + long handoff tests"
curl -sS --max-time 180 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/root/model/Qwen2-7B-Instruct","request_id":"tp2-stable-short","prompt":"Introduce the method of braised meat","max_tokens":64,"temperature":0,"top_p":1,"stop":[]}' \
  >/tmp/handoff_tp2_short_ok.json

curl -sS --max-time 420 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/root/model/Qwen2-7B-Instruct","request_id":"tp2-stable-long","prompt":"Introduce the method of braised meat","max_tokens":500,"temperature":0,"top_p":1,"stop":[]}' \
  >/tmp/handoff_tp2_long_ok.json

RID="tp2-$(date +%s)-$RANDOM-$i"
echo "/tmp/${RID}.json"
curl -sS --max-time 420 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/root/model/Qwen2-7B-Instruct","request_id":"$RID","prompt":"详细介绍北京的历史和吃喝玩乐","max_tokens":3510,"temperature":0,"top_p":1,"stop":[]}' \
  >/tmp/${RID}.json

# 单请求脚本： /root/vllm/scripts/test_handoff_kv500.sh

RID="tp2-$(date +%s)-$RANDOM"
echo "/tmp/${RID}.json"
curl -sS --max-time 420 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"/root/model/Qwen2-7B-Instruct\",\"request_id\":\"${RID}\",\"prompt\":\"分布详细介绍北京、上海、洛阳的历史和吃喝玩乐,写18000字\",\"max_tokens\":4096,\"temperature\":0,\"top_p\":1,\"stop\":[]}" \
  > /tmp/${RID}.json
# 打印结果
curl -s http://127.0.0.1:8200/__proxy_state

python - <<'PY'
import json
for p in ("/tmp/handoff_tp2_short_ok.json", "/tmp/handoff_tp2_long_ok.json"):
    obj = json.load(open(p))
    if "choices" in obj:
        text = obj["choices"][0].get("text", "")
        print(p, "OK", "len=", len(text), "tail=", repr(text[-100:]))
    else:
        print(p, "ERR", obj)
PY

echo "Done. Logs:"
echo "  /tmp/kv_owner_state_tp2.log"
echo "  /tmp/launch4_tp2.log"
echo "  /tmp/router_tp2.log"
