#!/bin/bash
# KL check of one model's quantized files against its BF16 file (scripts/kl_check.sh). Needs the model files
# (slurm/make_models.sbatch), llama-perplexity (slurm/build_llama.sbatch) and the eval text
# (scripts/get_eval_text.sh on the login node).
#   sbatch slurm/kl_check.sbatch Qwen3-0.6B [Q4_0-pure Q4_K_M Q8_0]
# A P100 (12 GB) holds every model in the grid at BF16 with a 512-token context.
#SBATCH -J kl-check
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
bash scripts/kl_check.sh "$@"
