#!/bin/bash
# The cluster's own GGUF files for the skeleton (BF16, Q4_K_M, pure Q4_0, Q8_0) from the Hugging Face
# checkpoint, with the pinned converter. Download on the login node first (compute nodes have no network):
#   source scripts/hpc_env.sh && DOWNLOAD_ONLY=1 bash scripts/make_probe_models.sh Qwen/Qwen3-0.6B $TESTBED_MODELS
#   sbatch slurm/make_models.sbatch [Qwen/Qwen3-0.6B]
# The SHA-256 of each file is printed; the skeleton run reports whether it equals the Oct 8 pilot's.
#SBATCH -J make-models
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
export HF_HUB_OFFLINE=1
VARIANTS=${VARIANTS:-"BF16 Q4_K_M Q4_0-pure Q8_0"} bash scripts/make_probe_models.sh "${1:-Qwen/Qwen3-0.6B}" "$TESTBED_MODELS"
