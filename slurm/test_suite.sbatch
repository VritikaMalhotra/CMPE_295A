#!/bin/bash
# The full test suite on a CPU node, with the tests that need cluster tools switched on: llama.cpp's
# test-gbnf-validator (R1 property tests) and R 4.4.1 + lme4 (mixed-effects tests). Stands in for CI, which
# runs only on pushes to main/m2 and on pull requests. Writes the pytest summary and junit XML next to the log.
#   sbatch slurm/test_suite.sbatch            (from the checkout to test)
#SBATCH -J test-suite
#SBATCH -A coe
#SBATCH -p cpuqs
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
echo "commit $(git rev-parse HEAD)"
export LLAMA_GBNF_VALIDATOR=${LLAMA_GBNF_VALIDATOR:-$CMPE295_GROUP/src/build-tests-d812350/bin/test-gbnf-validator}
module purge
unset LD_LIBRARY_PATH CUDAToolkit_ROOT CUDACXX
module load gnu14 R/4.4.1
export R_LIBS_USER=${R_LIBS_USER:-$CMPE295_GROUP/envs/R-4.4.1-lib}
echo "validator: $LLAMA_GBNF_VALIDATOR ($([ -x "$LLAMA_GBNF_VALIDATOR" ] && echo found || echo missing)); $(Rscript --version 2>&1 | head -1)"
.venv/bin/python -m pytest -q -rs --junitxml "$CMPE295_GROUP/logs/test-suite_${SLURM_JOB_ID}.xml" tests
