

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
  -d "{\"model\":\"/root/model/Qwen2-7B-Instruct\",\"request_id\":\"${RID}\",\"prompt\":\"逐个城市介绍，分别详细介绍北京、上海、洛阳、西安的历史和吃喝玩乐,每个城市介绍至少38000字，当你全部介绍完后，求解这道数学题设有一个函数，它等于自然对数函数 ln作用在1 加上 sin x上的结果，再乘以指数函数 e 的 x 的平方次幂。现在要求研究这个函数在 x 等于 0 附近的性质。请将该函数在 x 等于 0 处进行六阶泰勒展开，也就是说，把它展开成一个关于 x 的多项式形式，一直到 x 的六次幂为止，并且写出每一项的系数。同时需要说明当 x 趋近于 0 时，高于六次幂的项统一用小 o 的 x 的六次方来表示。 在完成展开之后，再利用得到的展开式计算下面的极限,当 x 趋近于 0 时，用该函数减去 x，再除以 x 的平方，求这个极限的值。请给出完整的推导过程\",\"max_tokens\":4096,\"temperature\":0.6,\"top_p\":1,\"stop\":[]}" \
  > /tmp/${RID}.json

RID="tp2-$(date +%s)-$RANDOM"
echo "/tmp/${RID}.json"
curl -sS --max-time 420 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"/root/model/Qwen2-7B-Instruct\",\"request_id\":\"${RID}\",\"prompt\":\"写一个50000字的故事，详细介绍杜兰特在总决赛前四节艰难维持比分但是在加时赛三分绝杀詹姆斯的剧本10000字\",\"max_tokens\":4096,\"temperature\":1,\"top_p\":1,\"stop\":[]}" \
  > /tmp/${RID}.json


RID="tp2-$(date +%s)-$RANDOM"
echo "/tmp/${RID}.json"
curl -sS --max-time 420 http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"/root/model/Qwen2-7B-Instruct\",\"request_id\":\"${RID}\",\"prompt\":\"设有一个函数，它等于自然对数函数 ln作用在1 加上 sin x上的结果，再乘以指数函数 e 的 x 的平方次幂。现在要求研究这个函数在 x 等于 0 附近的性质。请将该函数在 x 等于 0 处进行六阶泰勒展开，也就是说，把它展开成一个关于 x 的多项式形式，一直到 x 的六次幂为止，并且写出每一项的系数。同时需要说明当 x 趋近于 0 时，高于六次幂的项统一用小 o 的 x 的六次方来表示。 在完成展开之后，再利用得到的展开式计算下面的极限,当 x 趋近于 0 时，用该函数减去 x，再除以 x 的平方，求这个极限的值。请给出完整的推导过程\",\"max_tokens\":4096,\"temperature\":0.5,\"top_p\":1,\"stop\":[]}" \
  > /tmp/${RID}.json


# 打印结果
curl -s http://127.0.0.1:8200/__proxy_state


并发10的命令:
CONCURRENCY=128 NUM_REQUESTS=128 MAX_TOKENS=4096 MAX_MODEL_LEN=4096 CUTOVERS=1024,1024,1024 bash /root/vllm/scripts/test_handoff_conc10.sh
并发10的命令:
CONCURRENCY=128 NUM_REQUESTS=128 MAX_TOKENS=4096 MAX_MODEL_LEN=4096 CUTOVERS=2048,2048,2048 bash /root/vllm/scripts/test_handoff_conc10.sh

OWNER_MAX_NUM_SEQS=128 CONSUMER_MAX_NUM_SEQS=128 CONCURRENCY=128 NUM_REQUESTS=128 MAX_TOKENS=4096 MAX_MODEL_LEN=4096 CUTOVERS=512,512,1024 bash /root/vllm/scripts/test_handoff_conc10.sh

CONCURRENCY=128 \
NUM_REQUESTS=128 \
MAX_TOKENS=4096 \
MAX_MODEL_LEN=4096 \
CUTOVERS=1024,1024,1024 \
KV_HANDOFF_SERIAL_BARRIER=1 \
KV_HANDOFF_MIN_LAYERS=56 \
KV_HANDOFF_SOFT_MIN_LAYERS=8 \
KV_HANDOFF_WAIT_TIMEOUT_S=10 \
KV_HANDOFF_STABLE_POLLS=1 \
bash /root/vllm/scripts/test_handoff_conc10.sh