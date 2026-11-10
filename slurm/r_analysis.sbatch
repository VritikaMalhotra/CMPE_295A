#!/bin/bash
# C10 analysis in R on a CPU node: the planted-effect checks, or one real-data mixed-effects run.
#   sbatch slurm/r_analysis.sbatch planted  results/m2/planted_effect.json
#   sbatch slurm/r_analysis.sbatch real     results/m2/mixed_effects_<model>.json <ref_run_dir> <cell_run_dir>
# R 4.4.1 + lme4 come from the gnu14 module and the group library. scripts/hpc_env.sh is sourced for the paths,
# then every module is purged and LD_LIBRARY_PATH reset: its NVIDIA HPC SDK module blocks gnu14, and lme4's
# Rcpp then fails to load (testbed/mixed_effects.py).
#SBATCH -J r-analysis
#SBATCH -A coe
#SBATCH -p cpuqs
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/r-analysis_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/hpc_env.sh
log_gpu
module purge
unset LD_LIBRARY_PATH CUDAToolkit_ROOT CUDACXX
module load gnu14 R/4.4.1
export R_LIBS_USER=${R_LIBS_USER:-$CMPE295_GROUP/envs/R-4.4.1-lib}
echo "$(Rscript --version 2>&1 | head -1)"
MODE=$1; OUT=$2
case $MODE in
  planted) .venv/bin/python scripts/planted_effect.py --out "$OUT" ;;
  real)    .venv/bin/python scripts/mixed_effects_real.py --ref "$3" --cell "$4" --out "$OUT" ;;
  *) echo "usage: r_analysis.sbatch planted|real OUT [REF CELL]"; exit 2 ;;
esac
