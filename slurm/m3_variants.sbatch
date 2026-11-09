#!/bin/bash
# M3 / R2a variants of one model: importance matrix + IQ4_XS, and Q5_0-pure, Q6_K-pure (scripts/make_m3_variants.sh).
# Needs the model's BF16 file and the FR-E5-checked calibration text (docs/m3_quant.md, step 1).
#   sbatch slurm/m3_variants.sbatch Qwen3-0.6B [IQ4_XS Q5_0-pure Q6_K-pure]
# The imatrix pass runs the BF16 model over 200 x 512 tokens; a P100 (12 GB) holds every model in the grid at
# BF16 with a 512-token context (as in the M2 KL jobs). A100 instead: sbatch --gres=gpu:a100:1 --exclude=cs001,cs002 ...
#SBATCH -J m3-variants
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
bash scripts/make_m3_variants.sh "$@"
