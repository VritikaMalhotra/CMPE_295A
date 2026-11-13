#!/bin/bash
# C4 simulated user for one experiment (decision 4): llama-server with the experiment's simulator model on one
# A100/H100, announced in $ENDPOINT for the agent jobs. Submitted by scripts/tau2_submit.sh with EXP and
# ENDPOINT in the environment; stopped by that script's cleanup job once every agent job has ended.
#SBATCH -J simuser
#SBATCH -p gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs001,cs002
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2-01:00:00
#SBATCH -A coe
#SBATCH -o simuser_%j.out
set -euo pipefail
source scripts/hpc_env.sh
log_gpu
source .venv/bin/activate
export LITELLM_LOCAL_MODEL_COST_MAP=True
exec python -m testbed.run +experiment="$EXP" simuser_serve=true simuser_endpoint="$ENDPOINT" skip_hash=true
