#!/bin/bash
# usage: run_v2.sh TAG MODEL CATS NP PORT TEMP GRAMMAR THINKING SAMPLING_JSON SEED THREADS
TAG=$1; MODEL=$2; CATS=$3; NP=$4; PORT=$5; TEMP=$6; ROOT=/content/work/bfcl/v2/$TAG
mkdir -p $ROOT
export V2_GRAMMAR=$7 V2_THINKING=$8 V2_SAMPLING="$9"
[ "${10}" != "none" ] && export V2_SEED=${10}
THREADS=${11}
CTX=$(( 8192 * NP ))
echo "{\"tag\": \"$TAG\", \"model\": \"$MODEL\", \"cats\": \"$CATS\", \"np\": $NP, \"temperature\": $TEMP, \"grammar\": \"$7\", \"thinking\": $8, \"sampling\": $9, \"seed\": \"${10}\", \"threads\": $THREADS, \"cache_prompt\": false}" > $ROOT/protocol.json
/content/work/llama.cpp/build-cuda/bin/llama-server -m $MODEL -ngl 99 -c $CTX -np $NP --host 127.0.0.1 --port $PORT --seed 0 --no-webui > $ROOT/server.log 2>&1 &
SPID=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
export BFCL_PROJECT_ROOT=$ROOT LOCAL_SERVER_ENDPOINT=127.0.0.1 LOCAL_SERVER_PORT=$PORT
date -Is > $ROOT/gen_start.txt
/content/bfcl-env/bin/python /content/work/v2/bfcl_v2.py generate --model Qwen/Qwen3-0.6B-FC --test-category $CATS --temperature $TEMP --skip-server-setup --local-model-path /content/work/Qwen3-0.6B --num-threads $THREADS --include-input-log > $ROOT/generate.log 2>&1
echo gen_exit=$? >> $ROOT/status.txt; date -Is > $ROOT/gen_end.txt
kill $SPID; wait $SPID
/content/bfcl-env/bin/python /content/work/v2/bfcl_v2.py evaluate --model Qwen/Qwen3-0.6B-FC --test-category $CATS > $ROOT/evaluate.log 2>&1
echo eval_exit=$? >> $ROOT/status.txt
