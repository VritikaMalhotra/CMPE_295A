#!/bin/bash
# Onboarding step 4 / spike S1 (Workbook Tables 35, 37b): one tool call through the testbed backend on a GPU.
#   sbatch slurm/smoke_s1.sbatch                       # P100, Qwen3-0.6B Q4_K_M
#   sbatch --gres=gpu:a100:1 --exclude=cs002 slurm/smoke_s1.sbatch   # same on an A100 (cs002 is broken; or gpu:h100:1)
# Result: $TESTBED_OUT/smoke/s1_<job>.json (pass, timings, GPU memory, CUDA_VISIBLE_DEVICES).
#SBATCH -J smoke-s1
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
MODEL=${MODEL:-$TESTBED_MODELS/Qwen3-0.6B-Q4_K_M.gguf}
.venv/bin/python scripts/smoke_toolcall.py --model "$MODEL" --binary "$LLAMA_CUDA_BIN" \
    --out "$TESTBED_OUT/smoke/s1_${SLURM_JOB_ID}.json"
