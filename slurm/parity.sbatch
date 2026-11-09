#!/bin/bash
# FR-E17 parity: one model file on one backend over the fixed prompts (scripts/parity_run.py, resumable).
# Submitted by scripts/parity_submit.sh, which picks the partition: CUDA on a P100 (gpuqs), CPU on cpuqs.
#   sbatch -p gpuqs --gres=gpu:p100:1 --export=ALL,MODEL=...,BACKEND=cuda,PROMPTS=...,OUT=... slurm/parity.sbatch
#   sbatch -p cpuqs -c 16             --export=ALL,MODEL=...,BACKEND=cpu,PROMPTS=...,OUT=...  slurm/parity.sbatch
#SBATCH -J parity
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -A coe
#SBATCH --requeue
#SBATCH -o parity_%j.out
set -euo pipefail
source scripts/hpc_env.sh
log_gpu
source .venv/bin/activate
python scripts/parity_run.py --prompts "$PROMPTS" --model "$MODEL" --backend "$BACKEND" --out "$OUT"
