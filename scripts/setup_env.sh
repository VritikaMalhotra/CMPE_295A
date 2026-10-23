#!/usr/bin/env bash
# One-time environment setup (Colab, laptop, or the CoE HPC login node).
#   scripts/setup_env.sh [--llama] [--cuda] [--cache DIR]
# --llama   also build llama.cpp at the pilot's pinned commit (CPU build; add --cuda for the CUDA build)
# --cache   directory to keep/restore the llama.cpp builds between sessions (e.g. a Drive folder on Colab)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_TAG=v0.6.0                                   # pilot: d81235049384534c167caea52b85a694f6103d14
LLAMA_COMMIT=d81235049384534c167caea52b85a694f6103d14
BFCL=bfcl-eval==2026.3.23
WANT_LLAMA=0; WANT_CUDA=0; CACHE=""
while [ $# -gt 0 ]; do
  case $1 in --llama) WANT_LLAMA=1;; --cuda) WANT_CUDA=1;; --cache) CACHE=$2; shift;; *) echo "unknown $1"; exit 2;; esac
  shift
done

command -v uv >/dev/null || pip install -q uv
cd "$ROOT"
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

if [ $WANT_LLAMA = 1 ]; then
  SRC=$ROOT/llama.cpp
  restore() { [ -n "$CACHE" ] && [ -f "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" ] && tar -xf "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" -C "$ROOT"; }
  if [ ! -x "$SRC/build-cpu/bin/llama-server" ] || { [ $WANT_CUDA = 1 ] && [ ! -x "$SRC/build-cuda/bin/llama-server" ]; }; then
    restore || true
  fi
  if [ ! -d "$SRC/.git" ] && [ ! -d "$SRC/src" ]; then
    git clone --depth 1 --branch $LLAMA_TAG https://github.com/ggml-org/llama.cpp "$SRC"
  fi
  got=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)
  [ "$got" = "$LLAMA_COMMIT" ] || echo "WARNING: llama.cpp at $got, pilot used $LLAMA_COMMIT"
  if [ ! -x "$SRC/build-cpu/bin/llama-server" ]; then
    cmake -S "$SRC" -B "$SRC/build-cpu" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SRC/build-cpu" -j "$(nproc)" --target llama-server llama-quantize
  fi
  if [ $WANT_CUDA = 1 ] && [ ! -x "$SRC/build-cuda/bin/llama-server" ]; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
    cmake -S "$SRC" -B "$SRC/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-80}" \
          -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SRC/build-cuda" -j "$(nproc)" --target llama-server llama-quantize
  fi
  if [ -n "$CACHE" ] && [ ! -f "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" ]; then
    mkdir -p "$CACHE"
    tar -cf "$CACHE/llama.cpp-$LLAMA_COMMIT.tar" -C "$ROOT" --exclude='llama.cpp/build-*/CMakeFiles' llama.cpp
  fi
  "$SRC/build-cpu/bin/llama-server" --version 2>&1 | head -2
fi
