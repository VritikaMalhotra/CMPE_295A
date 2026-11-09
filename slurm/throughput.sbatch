#!/bin/bash
# NFR-5 throughput pilot: one array task = one shard of the throughput configuration at concurrency N.
# Submitted by scripts/throughput_submit.sh with -p gpuqs --gres=gpu:<type>:1 --array=0-(N-1) and
# LABEL, N, PER in the environment.
#SBATCH -J throughput
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -A coe
#SBATCH -o throughput_%A_%a.out
set -euo pipefail
source scripts/hpc_env.sh
log_gpu
source .venv/bin/activate
python -m testbed.run +experiment=throughput experiment.name="throughput_$LABEL" local=true skip_hash=true \
  shard="$SLURM_ARRAY_TASK_ID/$N" limit="$PER"
