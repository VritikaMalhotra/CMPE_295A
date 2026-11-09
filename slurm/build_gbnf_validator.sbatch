#!/bin/bash
# Build llama.cpp's GBNF validator (tests/test-gbnf-validator.cpp) for the R1 property test.
#   sbatch slurm/build_gbnf_validator.sbatch
# Out-of-tree CPU build at the pinned llama.cpp commit; the shared source tree and its build-cpu/build-cuda
# directories are not touched. Afterwards:
#   export LLAMA_GBNF_VALIDATOR=$CMPE295_GROUP/src/build-tests-d812350/bin/test-gbnf-validator
#   pytest tests/test_r1_property.py
#SBATCH -J gbnf-validator
#SBATCH -p cpuqs
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -A coe
#SBATCH -o gbnf_validator_%j.out
set -euo pipefail
source scripts/hpc_env.sh
log_gpu
OUT=${GBNF_BUILD_DIR:-$CMPE295_GROUP/src/build-tests-d812350}
cmake -S "$LLAMA_DIR" -B "$OUT" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Release $LLAMA_CPU_FLAGS
cmake --build "$OUT" --target test-gbnf-validator -j "${SLURM_CPUS_PER_TASK:-8}"
"$OUT/bin/test-gbnf-validator" 2>&1 | head -1 || true      # prints its usage line
ls -la "$OUT/bin/test-gbnf-validator"
