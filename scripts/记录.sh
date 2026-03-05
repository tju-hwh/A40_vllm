

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
  -d "{\"model\":\"/root/model/Qwen2-7B-Instruct\",\"request_id\":\"${RID}\",\"prompt\":\"分别详细介绍北京、上海、洛阳、西安的历史和吃喝玩乐,写38000字\",\"max_tokens\":4096,\"temperature\":1,\"top_p\":1,\"stop\":[]}" \
  > /tmp/${RID}.json

RID="tp2-$(date +%s)-$RANDOM"
echo "/tmp/${RID}.json"
curl -sS --max-time 420 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"/root/model/Qwen2-7B-Instruct\",\"request_id\":\"${RID}\",\"prompt\":\"写一个杜兰特在总决赛前四节艰难维持比分但是在加时赛三分绝杀詹姆斯的剧本10000字\",\"max_tokens\":4096,\"temperature\":1,\"top_p\":1,\"stop\":[]}" \
  > /tmp/${RID}.json
# 打印结果
curl -s http://127.0.0.1:8200/__proxy_state


并发10的命令:
CONCURRENCY=128 NUM_REQUESTS=128 MAX_TOKENS=4096 MAX_MODEL_LEN=4096 CUTOVERS=1024,1024,1024 bash /root/vllm/scripts/test_handoff_conc10.sh