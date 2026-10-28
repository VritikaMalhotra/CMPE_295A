#!/bin/bash
# Onboarding step 2 (Workbook Table 37b): build the pinned llama.cpp (CPU + CUDA for P100/A100/H100) inside a
# job, into the group folder ($LLAMA_DIR). Fetch the source on the login node first:
#   source scripts/hpc_env.sh && bash scripts/setup_env.sh --llama --cuda      # login node: fetches only
#   sbatch slurm/build_llama.sbatch                                            # from the repo root
#SBATCH -J llama-build
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:p100:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p'); flags avx2=$(grep -c -m1 avx2 /proc/cpuinfo)"
command -v nvcc && nvcc --version | tail -1
bash scripts/setup_env.sh --llama --cuda --build-only
for b in "$LLAMA_CUDA_BIN" "$LLAMA_CPU_BIN"; do echo "== $b"; "$b" --version 2>&1 | tail -2; done
