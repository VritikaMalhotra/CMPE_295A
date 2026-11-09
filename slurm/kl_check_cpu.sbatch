#!/bin/bash
# KL check on the CPU backend (M3 exit: every 4-bit variant KL-checked against BF16 on both backends).
# Same script and settings as slurm/kl_check.sbatch, but llama.cpp's CPU build, no GPU layers, and its own
# output folder ($TESTBED_OUT/kl_cpu/<model>), so the CUDA results are not overwritten.
#   sbatch slurm/kl_check_cpu.sbatch Qwen3-0.6B [Q4_0-pure Q4_K_M IQ4_XS Q5_0-pure Q6_K-pure Q8_0]
# CPU speed (docs/m2_decisions.md): a 0.6B BF16 pass over 50 x 512 tokens takes minutes on 32 cores; 4B is
# several times slower, hence the long limit.
#SBATCH -J kl-check-cpu
#SBATCH -A coe
#SBATCH -p cpuqs
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu                                   # host and job id (no GPU on cpuqs)
NAME=${1:?usage: sbatch slurm/kl_check_cpu.sbatch <ModelName> [variant ...]}
shift
export LLAMA_PERPLEXITY=$LLAMA_DIR/build-cpu/bin/llama-perplexity
export KL_NGL=0
export KL_OUT=$TESTBED_OUT/kl_cpu/$NAME
export TESTBED_CPU_THREADS=${SLURM_CPUS_PER_TASK:-32}
bash scripts/kl_check.sh "$NAME" ${*:-Q4_0-pure Q4_K_M IQ4_XS Q5_0-pure Q6_K-pure Q8_0}
