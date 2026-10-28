# Source this on the SJSU CoE HPC (login node or inside a Slurm job):  source scripts/hpc_env.sh
# Shared group paths, caches off the (full) home file system, modules, and the llama.cpp builds.
# Facts behind these choices (2026-10-26): docs/cluster.md.
export CMPE295_GROUP=${CMPE295_GROUP:-/scratch/cmpe295-bertin}
export TESTBED_OUT=${TESTBED_OUT:-$CMPE295_GROUP/runs}
export TESTBED_MODELS=${TESTBED_MODELS:-$CMPE295_GROUP/models}
export MODEL_SRC_DIR=${MODEL_SRC_DIR:-$CMPE295_GROUP/models/hf}        # Hugging Face checkpoints (download on the login node)
export HF_HOME=${HF_HOME:-$CMPE295_GROUP/cache/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CMPE295_GROUP/cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$CMPE295_GROUP/envs/uv-python}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$CMPE295_GROUP/cache/pip}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CMPE295_GROUP/cache/xdg}
# one shared llama.cpp tree per pinned commit; builds target P100 (6.0), A100 (8.0) and H100 (9.0)
export LLAMA_DIR=${LLAMA_DIR:-$CMPE295_GROUP/src/llama.cpp-d812350}
export LLAMA_CUDA_BIN=${LLAMA_CUDA_BIN:-$LLAMA_DIR/build-cuda/bin/llama-server}
export LLAMA_CPU_BIN=${LLAMA_CPU_BIN:-$LLAMA_DIR/build-cpu/bin/llama-server}
export CUDA_ARCHS=${CUDA_ARCHS:-"60;80;90"}
# Portable CPU code: a binary built on one node type must run on every other (no -march=native).
export LLAMA_CPU_FLAGS=${LLAMA_CPU_FLAGS:-"-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"}
export TESTBED_SITE=coe-hpc
# Slurm does not pin CPUs on these nodes either: llama.cpp would use every core of the node (28 on g2)
[ -n "${SLURM_CPUS_PER_TASK:-}" ] && export TESTBED_CPU_THREADS=$SLURM_CPUS_PER_TASK
umask 0007                                    # files in the group folder stay group-writable

if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /opt/ohpc/admin/lmod/lmod/init/bash; do [ -f "$f" ] && . "$f" && break; done
fi
if command -v module >/dev/null 2>&1; then
  module load cmake/3.24.2 >/dev/null 2>&1 || true
  # CUDA 12.6 comes with the NVIDIA HPC SDK module, which also points CC/CXX at nvc/nvc++: keep gcc for
  # C/C++ (llama.cpp and the Python wheels expect it) and take only nvcc from the module.
  module load nvhpc-hpcx-cuda12/24.11 >/dev/null 2>&1 || true
  unset CC CXX FC F77 F90
  if [ -n "${NVHPC_ROOT:-}" ]; then          # SDK layout: CUDA in cuda/12.6, cuBLAS in math_libs/12.6
    export CUDAToolkit_ROOT=$NVHPC_ROOT/cuda/12.6
    export CUDACXX=$CUDAToolkit_ROOT/bin/nvcc
    export LD_LIBRARY_PATH=$CUDAToolkit_ROOT/lib64:$NVHPC_ROOT/math_libs/12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  fi
fi
mkdir -p "$TESTBED_OUT" "$TESTBED_MODELS" "$MODEL_SRC_DIR" "$HF_HOME" "$UV_CACHE_DIR" "$CMPE295_GROUP/logs" 2>/dev/null || true

on_login_node() { [ -z "${SLURM_JOB_ID:-}" ] && [[ "$(hostname)" == coe-hpc* ]]; }
log_gpu() {   # every job logs what it was given; GPUs are not fenced on these nodes (docs/cluster.md)
  echo "job ${SLURM_JOB_ID:-none} on $(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
}
