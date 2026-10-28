#!/usr/bin/env bash
# Build GGUF files for a Qwen3 model with the pinned llama.cpp converter (M2 model probe, clean-clone rerun,
# and the cluster's own copies of the skeleton files).
#   scripts/make_probe_models.sh Qwen/Qwen3-1.7B <models_dir> [cache_dir]
#   VARIANTS="BF16 Q4_0-pure Q4_K_M Q8_0" scripts/make_probe_models.sh Qwen/Qwen3-0.6B <models_dir>
#   DOWNLOAD_ONLY=1 scripts/make_probe_models.sh Qwen/Qwen3-0.6B <models_dir>   # CoE HPC login node
# Same recipe as the pilot's 0.6B files: convert_hf_to_gguf --outtype bf16, then llama-quantize (--pure for
# Q4_0-pure; llama.cpp's default recipe for Q4_K_M and Q8_0). The Hugging Face checkpoint goes to
# $MODEL_SRC_DIR/<name> (default /content/work); a complete local copy is never downloaded again, so the
# conversion can run on a compute node without network access.
set -euo pipefail
REPO=$1; mkdir -p "$2"; M=$(realpath "$2"); CACHE=${3:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${LLAMA_DIR:-$ROOT/llama.cpp}
QUANT=${LLAMA_QUANTIZE:-$LLAMA/build-cuda/bin/llama-quantize}
[ -x "$QUANT" ] || QUANT=$LLAMA/build-cpu/bin/llama-quantize
VARIANTS=${VARIANTS:-"BF16 Q4_0-pure"}
NAME=$(basename "$REPO")                     # e.g. Qwen3-1.7B
SRC=${MODEL_SRC_DIR:-/content/work}/$NAME
mkdir -p "$M"
for v in $VARIANTS; do
  if [ -n "$CACHE" ] && [ -f "$CACHE/$NAME-$v.gguf" ] && [ ! -f "$M/$NAME-$v.gguf" ]; then cp "$CACHE/$NAME-$v.gguf" "$M/"; fi
done
have_src() { [ -f "$SRC/config.json" ] && ls "$SRC"/*.safetensors >/dev/null 2>&1; }
if [ ! -f "$M/$NAME-BF16.gguf" ] || [ "${DOWNLOAD_ONLY:-0}" = 1 ]; then
  if ! have_src; then
    "$ROOT/.venv/bin/hf" download "$REPO" --local-dir "$SRC" >/dev/null
  fi
  git -C "$SRC" rev-parse HEAD 2>/dev/null || cat "$SRC/.cache/huggingface/download/"*.metadata 2>/dev/null | head -1 || true
fi
[ "${DOWNLOAD_ONLY:-0}" = 1 ] && { echo "downloaded $REPO to $SRC"; exit 0; }
if [ ! -f "$M/$NAME-BF16.gguf" ]; then
  have_src || { echo "no checkpoint at $SRC (download it first: DOWNLOAD_ONLY=1 $0 $REPO $2)"; exit 2; }
  "$ROOT/.venv/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --outfile "$M/$NAME-BF16.gguf"
fi
for v in $VARIANTS; do
  out="$M/$NAME-$v.gguf"
  [ -f "$out" ] && continue
  case $v in
    BF16) ;;
    Q4_0-pure) "$QUANT" --pure "$M/$NAME-BF16.gguf" "$out" Q4_0 >/dev/null ;;
    *)         "$QUANT" "$M/$NAME-BF16.gguf" "$out" "$v" >/dev/null ;;
  esac
done
for v in $VARIANTS; do
  if [ -n "$CACHE" ]; then cp -n "$M/$NAME-$v.gguf" "$CACHE/" 2>/dev/null || true; fi
  sha256sum "$M/$NAME-$v.gguf" | tee "$M/$NAME-$v.gguf.sha256"
done
