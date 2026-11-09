#!/usr/bin/env bash
# FR-E17 parity on the cluster: fixed prompts per model, then BF16 and Q4_0 on CUDA (P100) and CPU.
#   source scripts/hpc_env.sh && bash scripts/parity_submit.sh            # all three Qwen3 models
#   MODELS="qwen3-0.6b" bash scripts/parity_submit.sh
# Afterwards (once every job finished):
#   bash scripts/parity_submit.sh report
# Prompts come from the Oct 28 damage runs (BF16, seed 1); only Qwen3 has a prompt handler in the testbed.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${PARITY_OUT:-$TESTBED_OUT/parity}
MODELS=${MODELS:-"qwen3-0.6b qwen3-1.7b qwen3-4b"}
declare -A RUN=([qwen3-0.6b]=damage_0p6b/61417027a6 [qwen3-1.7b]=damage_1p7b/3520cf30d3 [qwen3-4b]=damage_4b/f1f84f5993)
declare -A FILE=([qwen3-0.6b]=Qwen3-0.6B [qwen3-1.7b]=Qwen3-1.7B [qwen3-4b]=Qwen3-4B)
declare -A VAR=([bf16]=BF16 [q4_0]=Q4_0-pure)
mkdir -p "$OUT"

if [ "${1:-}" = report ]; then
  for m in $MODELS; do
    args=()
    for v in bf16 q4_0; do args+=(--pair "$v" "$OUT/$m.$v.cuda.jsonl" "$OUT/$m.$v.cpu.jsonl"); done
    .venv/bin/python scripts/parity_report.py --model "$m" "${args[@]}" --out "results/m2/parity_$m.json"
  done
  exit 0
fi

for m in $MODELS; do
  P=$OUT/prompts_$m.jsonl
  if [ ! -f "$P" ]; then     # the replay runs bfcl's executor: a short CPU job, not the login node
    srun -p defq -A coe -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/parity_prompts.py \
      --run-dir "$TESTBED_OUT/${RUN[$m]}" --variant bf16 --seed 1 --n 200 --out "$P"
  fi
  for v in bf16 q4_0; do
    model=$TESTBED_MODELS/${FILE[$m]}-${VAR[$v]}.gguf
    sbatch -J "par-$m-$v-cuda" -p gpuqs --gres=gpu:p100:1 --exclude=cs001,cs002 \
      --export=ALL,MODEL="$model",BACKEND=cuda,PROMPTS="$P",OUT="$OUT/$m.$v.cuda.jsonl" slurm/parity.sbatch
    # CPU is slow: one 0.6B Q4_0 episode step took ~80 s on 4 cores (prompts are 3-5k tokens, cache off)
    sbatch -J "par-$m-$v-cpu" -p cpuqs -c 32 --mem=48G -t 72:00:00 \
      --export=ALL,MODEL="$model",BACKEND=cpu,PROMPTS="$P",OUT="$OUT/$m.$v.cpu.jsonl" slurm/parity.sbatch
  done
done
