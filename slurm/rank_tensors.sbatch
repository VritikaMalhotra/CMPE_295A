#!/bin/bash
# R2b tensor-sensitivity ranking: each array task quantizes its share of the linear weights one at a time to
# Q4_0 (rest BF16) and scores the KL divergence from BF16 (scripts/rank_tensors.py run). Resumable: finished
# tensors have a score file and are skipped, so a requeued or resubmitted task continues where it stopped.
# Before the array:  python scripts/rank_tensors.py list --model M   (login node)
#                    sbatch slurm/rank_tensors.sbatch M base         (one GPU job: BF16 logits)
# Then:              sbatch --array=0-7 slurm/rank_tensors.sbatch M run
# After:             python scripts/rank_tensors.py select --model M ; bash scripts/make_layer_selective.sh M
# Each task writes one BF16-sized temporary file at a time (1.5 GB for 0.6B, 8 GB for 4B) and deletes it.
#SBATCH -J rank-tensors
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
MODEL=${1:?usage: sbatch [--array=0-7] slurm/rank_tensors.sbatch <ModelName> base|run}
STEP=${2:-run}
.venv/bin/python scripts/rank_tensors.py "$STEP" --model "$MODEL"
