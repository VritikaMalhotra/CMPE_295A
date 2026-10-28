#!/usr/bin/env bash
# Build BF16 and pure-Q4_0 GGUFs for a larger Qwen3 model (M2 model probe), with the pinned llama.cpp converter.
#   scripts/make_probe_models.sh Qwen/Qwen3-1.7B <models_dir> [cache_dir]
# Same recipe as the pilot's 0.6B files: convert_hf_to_gguf --outtype bf16, then llama-quantize --pure Q4_0.
set -euo pipefail
REPO=$1; M=$(realpath "$2"); CACHE=${3:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=$(basename "$REPO")                     # e.g. Qwen3-1.7B
mkdir -p "$M"
for v in BF16 Q4_0-pure; do
  if [ -n "$CACHE" ] && [ -f "$CACHE/$NAME-$v.gguf" ] && [ ! -f "$M/$NAME-$v.gguf" ]; then cp "$CACHE/$NAME-$v.gguf" "$M/"; fi
done
if [ ! -f "$M/$NAME-BF16.gguf" ]; then
  "$ROOT/.venv/bin/hf" download "$REPO" --local-dir "/content/work/$NAME" >/dev/null
  "$ROOT/.venv/bin/python" "$ROOT/llama.cpp/convert_hf_to_gguf.py" "/content/work/$NAME" --outtype bf16 \
      --outfile "$M/$NAME-BF16.gguf"
fi
if [ ! -f "$M/$NAME-Q4_0-pure.gguf" ]; then
  "$ROOT/llama.cpp/build-cuda/bin/llama-quantize" --pure "$M/$NAME-BF16.gguf" "$M/$NAME-Q4_0-pure.gguf" Q4_0 >/dev/null
fi
if [ -n "$CACHE" ]; then cp -n "$M/$NAME-BF16.gguf" "$M/$NAME-Q4_0-pure.gguf" "$CACHE/" 2>/dev/null || true; fi
sha256sum "$M/$NAME-BF16.gguf" "$M/$NAME-Q4_0-pure.gguf"
