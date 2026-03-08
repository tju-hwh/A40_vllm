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
ENABLE_CUDA_MPS=${ENABLE_CUDA_MPS:-"0"}
CUDA_MPS_ACTIVE_THREAD_PERCENTAGES=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGES:-"100,100,100,100"}

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