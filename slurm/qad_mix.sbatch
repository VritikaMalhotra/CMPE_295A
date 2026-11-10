#!/bin/bash
# QAD training mix + R2b calibration conversations (train/build_qad_mix.py build; Workbook 2 package K, FR-E5).
# Before: python train/build_qad_mix.py fetch --out $CMPE295_GROUP/data/qad_mix   (login node, qad-data env)
#         .venv/bin/python scripts/benchmark_ngrams.py dump --out $CMPE295_GROUP/data/calib/benchmark_text.txt
#   sbatch slurm/qad_mix.sbatch
# CPU only: normalise, 13-gram filter, dedupe, count tokens with the Qwen3-0.6B tokenizer, write train.jsonl,
# rank_conversations.jsonl, rank_text.txt and manifest.json under $CMPE295_GROUP/data/qad_mix.
#SBATCH -J qad-mix
#SBATCH -A coe
#SBATCH -p cpuqs
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu                                   # host and job id
OUT=${QAD_MIX_OUT:-$CMPE295_GROUP/data/qad_mix}
"${QAD_PY:-$CMPE295_GROUP/envs/qad-data/bin/python}" train/build_qad_mix.py build --out "$OUT" \
  --bench "$CMPE295_GROUP/data/calib/benchmark_text.txt" --tokenizer "$MODEL_SRC_DIR/Qwen3-0.6B"
