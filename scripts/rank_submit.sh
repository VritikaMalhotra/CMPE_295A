#!/usr/bin/env bash
# R2b for one model as one chain of Slurm jobs (runbook steps 4-5):
#   1. BF16 base logits on the ranking text            (GPU)
#   2. ranking array, one tensor at a time at Q4_0       (GPU x RANK_TASKS)
#   3. select at 5.5 bits per weight + build + FR-E10    (CPU; fails if the file misses 5.5 +/- 0.1)
#   4. KL of the R2b file against BF16 on CUDA and on CPU
#   source scripts/hpc_env.sh
#   RANK_TEXT=<file> RANK_CHUNKS=<n> [RANK_TASKS=2] [RANK_GRES=gpu:1] [RANK_NICE=50] bash scripts/rank_submit.sh <ModelName>
# Every score file records ctx, chunks and the text's SHA-256; rank_tensors.py refuses to mix settings.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:?usage: rank_submit.sh <ModelName>}
: "${RANK_TEXT:?set RANK_TEXT}" "${RANK_CHUNKS:?set RANK_CHUNKS}"
T=${RANK_TASKS:-2}; G=${RANK_GRES:-gpu:1}; N=${RANK_NICE:-50}
export RANK_TEXT RANK_CHUNKS
.venv/bin/python scripts/rank_tensors.py list --model "$M"
common=(--parsable --nice="$N" --gres="$G" --exclude=cs001,cs002)
B=$(sbatch "${common[@]}" -t 02:00:00 --mem=32G slurm/rank_tensors.sbatch "$M" base)
A=$(sbatch "${common[@]}" -d afterok:"$B" --array=0-$((T - 1)) -t 08:00:00 --mem=32G slurm/rank_tensors.sbatch "$M" run)
A=${A%%;*}
S=$(sbatch --parsable --nice="$N" -d afterok:"$A" -A coe -p cpuqs -c 8 --mem=32G -t 02:00:00 -J r2b-build \
      -o "$CMPE295_GROUP/logs/r2b-build_%j.out" \
      --wrap "source scripts/hpc_env.sh; log_gpu; export LLAMA_QUANTIZE=\$LLAMA_DIR/build-cpu/bin/llama-quantize; \
              .venv/bin/python scripts/rank_tensors.py select --model $M --bpw 5.5 && bash scripts/make_layer_selective.sh $M")
K1=$(sbatch --parsable --nice="$N" -d afterok:"$S" --export=ALL,KL_OUT="$TESTBED_OUT/kl_m3/$M-R2b" slurm/kl_check.sbatch "$M" R2b-LS5.5)
K2=$(sbatch --parsable --nice="$N" -d afterok:"$S" --export=ALL,KL_OUT="$TESTBED_OUT/kl_cpu_m3/$M-R2b" slurm/kl_check_cpu.sbatch "$M" R2b-LS5.5)
echo "$M: base $B, ranking array $A (${T} tasks), select+build $S, KL cuda $K1, KL cpu $K2 (RANK_CHUNKS=$RANK_CHUNKS)"
echo "$M base=$B array=$A build=$S kl_cuda=$K1 kl_cpu=$K2 chunks=$RANK_CHUNKS text=$RANK_TEXT $(date -Iseconds)" \
  >> "$TESTBED_OUT/rank/submissions.txt"
