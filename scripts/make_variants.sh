#!/usr/bin/env bash
# Build the GGUF variants the skeleton needs that the Oct 8 pilot did not produce (R2: Q8_0), from the
# pilot's BF16 file, and verify the pilot files against progress/quant_files.csv.
# Owner: Yashashav (R2).   Usage: scripts/make_variants.sh <models_dir> <llama-quantize>
set -euo pipefail
M=${1:-${TESTBED_MODELS:-models}}
Q=${2:-llama.cpp/build-cuda/bin/llama-quantize}
Q=$(realpath "$Q")          # resolve before cd: a relative path would break below
cd "$M"

declare -A PILOT=(
  [Qwen3-0.6B-BF16.gguf]=9a07b327d6114b23cd022cd0b188629aa0d02f88c863becaa34af5fc374281a2
  [Qwen3-0.6B-Q4_K_M.gguf]=40b0bbe7c787ac694f2f635ead2bcfebf19a7d6f1130d63155155fc0b83bdc6c
  [Qwen3-0.6B-Q4_0-pure.gguf]=90d6e31f1034a14e7d382c0a5de3e0c2bafa1e1b6b2d1869ebea5b8e2c4482ab
)
for f in "${!PILOT[@]}"; do
  [ -f "$f" ] || { echo "missing $f (copy it from Drive: CMPE295A_pilot_2026-10-08/models/)"; exit 1; }
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "${PILOT[$f]}" ] && echo "ok  $f" || { echo "SHA-256 MISMATCH $f: $got"; exit 1; }
  echo "$got  $f" > "$f.sha256"
done

# R2: uniform 8-bit from the same BF16 source, default (non-pure) recipe as llama.cpp ships it
if [ ! -f Qwen3-0.6B-Q8_0.gguf ]; then
  "$Q" Qwen3-0.6B-BF16.gguf Qwen3-0.6B-Q8_0.gguf Q8_0
fi
sha256sum Qwen3-0.6B-Q8_0.gguf | tee Qwen3-0.6B-Q8_0.gguf.sha256
