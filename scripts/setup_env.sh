#!/usr/bin/env bash
# One-time environment setup (Colab, laptop, or the CoE HPC).
#   scripts/setup_env.sh [--llama] [--cuda] [--tau2] [--cache DIR] [--fetch-only | --build-only]
# --llama   also build llama.cpp at the pilot's pinned commit (CPU build; add --cuda for the CUDA build)
# --tau2    also install tau2-bench-verified (Amazon AGI) at the pinned commit, editable, into ext/
# --cache   directory to keep/restore the llama.cpp builds between sessions (e.g. a Drive folder on Colab)
# --fetch-only  download everything (Python env, tau2, llama.cpp source) but compile nothing
# --build-only  compile llama.cpp only, no network (CoE HPC compute nodes have no outbound access)
# On the CoE HPC (source scripts/hpc_env.sh first) the login node does --fetch-only automatically, and
# slurm/build_llama.sbatch does --build-only inside a job: heavy work never runs on the login node.
# LLAMA_DIR, CUDA_ARCHS and LLAMA_CPU_FLAGS override the llama.cpp location, GPU targets and CPU flags.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_TAG=v0.6.0                                   # pilot: d81235049384534c167caea52b85a694f6103d14
LLAMA_COMMIT=d81235049384534c167caea52b85a694f6103d14
BFCL=bfcl-eval==2026.3.23
TAU2_REPO=https://github.com/amazon-agi/tau2-bench-verified
TAU2_COMMIT=864350a8971a8f8ee9e7b8472e2edc380a806b0c     # = configs/experiment/skeleton.yaml tau2.commit
WANT_LLAMA=0; WANT_CUDA=0; WANT_TAU2=0; CACHE=""; FETCH_ONLY=0; BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in --llama) WANT_LLAMA=1;; --cuda) WANT_CUDA=1;; --tau2) WANT_TAU2=1;; --cache) CACHE=$2; shift;;
    --fetch-only) FETCH_ONLY=1;; --build-only) BUILD_ONLY=1;;
    *) echo "unknown $1"; exit 2;; esac
  shift
done
if [ "${TESTBED_SITE:-}" = coe-hpc ] && [ -z "${SLURM_JOB_ID:-}" ] && [ $BUILD_ONLY = 0 ]; then
  FETCH_ONLY=1      # login node: fetch here, compile in a job (sbatch slurm/build_llama.sbatch)
fi
[ $BUILD_ONLY = 1 ] && [ -n "${TESTBED_SITE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ] && {
  echo "--build-only compiles; on the cluster run it in a job: sbatch slurm/build_llama.sbatch"; exit 2; }
JOBS=${SLURM_CPUS_PER_TASK:-$(nproc)}

cd "$ROOT"
if [ $BUILD_ONLY = 0 ]; then
command -v uv >/dev/null || pip install -q uv || pip install -q --user uv
export PATH=$HOME/.local/bin:$PATH
# Colab exports UV_SYSTEM_PYTHON=true, which makes `uv pip` install into the system Python even with a
# venv active; always name the venv's interpreter explicitly.
unset UV_SYSTEM_PYTHON
[ -d .venv ] || uv venv -p 3.12 .venv
. .venv/bin/activate
PY=$ROOT/.venv/bin/python
uv pip install -q --python "$PY" -e ".[dev]"
# bfcl-eval pins heavy provider SDKs it never uses for local checking; the checker needs only numpy + mpmath
uv pip install -q --python "$PY" --no-deps "$BFCL"
python -c "import bfcl_eval, hydra, testbed; print('python ok:', bfcl_eval.__name__, hydra.__version__)"

if [ $WANT_TAU2 = 1 ]; then
  T=$ROOT/ext/tau2-bench-verified
  if [ ! -d "$T/.git" ]; then
    mkdir -p "$ROOT/ext"
    git clone --filter=blob:none "$TAU2_REPO" "$T"
  fi
  git -C "$T" fetch -q origin "$TAU2_COMMIT" 2>/dev/null || true
  git -C "$T" checkout -q "$TAU2_COMMIT"
  # editable, so the task data under data/ is found without TAU2_DATA_DIR (README)
  uv pip install -q --python "$PY" -e "$T"
  LITELLM_LOCAL_MODEL_COST_MAP=True python -c "from tau2.run import load_tasks; t = load_tasks('airline', 'base'); print('tau2 ok:', len(t), 'airline base tasks')"
fi
fi   # BUILD_ONLY = 0

if [ $WANT_LLAMA = 1 ]; then
  SRC=${LLAMA_DIR:-$ROOT/llama.cpp}
  restore() { [ -n "$CACHE" ] && [ -f "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" ] && [ "$SRC" = "$ROOT/llama.cpp" ] && tar -xf "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" -C "$ROOT"; }
  if [ ! -x "$SRC/build-cpu/bin/llama-server" ] || { [ $WANT_CUDA = 1 ] && [ ! -x "$SRC/build-cuda/bin/llama-server" ]; }; then
    restore || true
  fi
  if [ ! -d "$SRC/.git" ] && [ ! -d "$SRC/src" ]; then
    [ $BUILD_ONLY = 1 ] && { echo "no llama.cpp source at $SRC; run setup_env.sh --llama on the login node first"; exit 2; }
    git clone --depth 1 --branch $LLAMA_TAG https://github.com/ggml-org/llama.cpp "$SRC"
  fi
  got=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)
  [ "$got" = "$LLAMA_COMMIT" ] || echo "WARNING: llama.cpp at $got, pilot used $LLAMA_COMMIT"
  if [ $FETCH_ONLY = 1 ]; then
    echo "fetched llama.cpp source to $SRC (not compiled here)."
    [ -n "${TESTBED_SITE:-}" ] && echo "next: sbatch slurm/build_llama.sbatch"
    exit 0
  fi
  if [ ! -x "$SRC/build-cpu/bin/llama-server" ]; then
    cmake -S "$SRC" -B "$SRC/build-cpu" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release ${LLAMA_CPU_FLAGS:-}
    cmake --build "$SRC/build-cpu" -j "$JOBS" --target llama-server llama-quantize llama-bench
  fi
  if [ $WANT_CUDA = 1 ] && [ ! -x "$SRC/build-cuda/bin/llama-server" ]; then
    # CUDA_ARCHS (the cluster: "60;80;90") or this machine's GPU; a build only runs on the GPUs it targets
    ARCH=${CUDA_ARCHS:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}
    cmake -S "$SRC" -B "$SRC/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-80}" \
          -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release ${LLAMA_CPU_FLAGS:-}
    cmake --build "$SRC/build-cuda" -j "$JOBS" --target llama-server llama-quantize llama-bench
  fi
  # CPU llama-bench (C9 speed tables) was added after the first cached build. The cache keeps binaries but not
  # object files, so this recompiles the CPU libraries (~10-15 min on 2 cores). The CUDA build is not touched:
  # GPU speed comes from the per-step timings of the episodes.
  if [ $WANT_CUDA = 0 ] && [ ! -x "$SRC/build-cpu/bin/llama-bench" ]; then   # CPU machines only
    cmake -S "$SRC" -B "$SRC/build-cpu" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release ${LLAMA_CPU_FLAGS:-} >/dev/null
    cmake --build "$SRC/build-cpu" -j "$JOBS" --target llama-bench
  fi
  if [ -n "$CACHE" ] && [ ! -f "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" ] && [ "$SRC" = "$ROOT/llama.cpp" ]; then
    mkdir -p "$CACHE"
    tar -cf "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" -C "$ROOT" --exclude='llama.cpp/build-*/CMakeFiles' llama.cpp
  fi
  "$SRC/build-cpu/bin/llama-server" --version 2>&1 | head -2
fi
