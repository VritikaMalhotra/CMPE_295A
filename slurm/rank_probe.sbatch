#!/bin/bash
# D-1 timing probe for the R2b ranking: on one GPU, BF16 base logits and a few one-tensor-at-Q4_0 scores at two
# chunk counts, so RANK_CHUNKS can be set to the largest value at which one model's ranking takes <= 6 h on
# 2 GPUs. Uses RANK_TEXT (the truncated ranking text) and writes to $TESTBED_OUT/rank_probe/<model>/c<chunks>.
#   sbatch --gres=gpu:a100:1 --exclude=cs001,cs002 --export=ALL,RANK_TEXT=<file> slurm/rank_probe.sbatch Qwen3-0.6B
#SBATCH -J rank-probe
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
MODEL=${1:?usage: sbatch slurm/rank_probe.sbatch <ModelName>}
: "${RANK_TEXT:?set RANK_TEXT}"
.venv/bin/python scripts/rank_tensors.py list --model "$MODEL"
N=$(.venv/bin/python -c "import json,sys; print(sum(t['candidate'] for t in json.load(open(sys.argv[1]))['tensors']))" \
    "$TESTBED_OUT/rank/$MODEL/tensors.json")
for C in ${PROBE_CHUNKS:-20 200}; do
  export RANK_CHUNKS=$C RANK_OUT=$TESTBED_OUT/rank_probe/$MODEL/c$C
  mkdir -p "$RANK_OUT" && cp "$TESTBED_OUT/rank/$MODEL/tensors.json" "$RANK_OUT/"
  t0=$(date +%s); .venv/bin/python scripts/rank_tensors.py base --model "$MODEL"; t1=$(date +%s)
  .venv/bin/python scripts/rank_tensors.py run --model "$MODEL" --task 0 --ntasks 1 --limit ${PROBE_TENSORS:-4}
  t2=$(date +%s)
  echo "PROBE chunks=$C base_s=$((t1 - t0)) score_s_total=$((t2 - t1)) tensors=${PROBE_TENSORS:-4} candidates=$N"
done
