#!/bin/bash
# A1 smoke: one BFCL and one tau2 episode per new prompt handler (SmolLM3-3B, LFM2.5-1.2B, LFM2.5-2.6B), BF16,
# greedy, on one GPU, using an already running simulated-user job (its endpoint file):
#   sbatch --export=ALL,ENDPOINT=$TESTBED_OUT/simuser/<name>.endpoint.json slurm/a1_smoke.sbatch
#SBATCH -J a1-smoke
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs001,cs002
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/cmpe295-bertin/logs/a1-smoke_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/hpc_env.sh
log_gpu
source .venv/bin/activate
export LITELLM_LOCAL_MODEL_COST_MAP=True TESTBED_SIMUSER_ENDPOINT=${ENDPOINT:?set ENDPOINT}
for exp in ${SMOKE_EXPS:-a1_smoke_smollm3_3b a1_smoke_lfm2p5_1p2b a1_smoke_lfm2p5_2p6b}; do
  echo "=== $exp"
  python -m testbed.run +experiment=$exp local=true simuser_endpoint="$ENDPOINT" || echo "FAILED $exp"
done
