#!/usr/bin/env bash
# KL check of a model's quantized GGUF files against its BF16 file, with llama.cpp's perplexity tool
# (Workbook 2, Sprint 5: "BF16 GGUF conversions of all six models with KL checks"; Ch. 6 real-4-bit audit:
# mean KL divergence, top-1 agreement and perplexity). Run it inside a GPU job: slurm/kl_check.sbatch.
#   bash scripts/kl_check.sh <ModelName> [variant ...]        e.g. Qwen3-0.6B   (default: Q4_0-pure Q4_K_M Q8_0)
# Inputs: $TESTBED_MODELS/<ModelName>-<variant>.gguf (scripts/make_probe_models.sh) and the eval text
# (scripts/get_eval_text.sh). Output in $TESTBED_OUT/kl/<ModelName>/: BF16.log, <variant>.log and
# kl_<ModelName>.json (scripts/kl_report.py). The base log-probability file is several GB and is deleted at
# the end unless KL_KEEP_BASE=1.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:?usage: kl_check.sh <ModelName> [variant ...]}
shift
VARIANTS=${*:-Q4_0-pure Q4_K_M Q8_0}
M=${TESTBED_MODELS:?source scripts/hpc_env.sh first}
PPL=${LLAMA_PERPLEXITY:-${LLAMA_DIR:?}/build-cuda/bin/llama-perplexity}
TEXT=${KL_TEXT:-${CMPE295_GROUP:?}/data/eval/wikitext-2-raw/wiki.test.raw}
CTX=${KL_CTX:-512}          # both passes must use the same context; the base file records it
CHUNKS=${KL_CHUNKS:-50}     # 50 chunks x 255 scored tokens per chunk
NGL=${KL_NGL:-99}
OUT=${KL_OUT:-$TESTBED_OUT/kl/$NAME}
PY=${PY:-$ROOT/.venv/bin/python}

[ -x "$PPL" ] || { echo "no llama-perplexity at $PPL: sbatch slurm/build_llama.sbatch"; exit 2; }
[ -f "$TEXT" ] || { echo "no eval text at $TEXT: run scripts/get_eval_text.sh on the login node"; exit 2; }
[ -f "$M/$NAME-BF16.gguf" ] || { echo "no $M/$NAME-BF16.gguf: sbatch slurm/make_models.sbatch <hf_repo>"; exit 2; }
mkdir -p "$OUT"
BASE=$OUT/BF16.kld
COMMON=(-c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" -t "${TESTBED_CPU_THREADS:-4}")

echo "kl_check $NAME: variants [$VARIANTS], ctx $CTX, chunks $CHUNKS, text $TEXT"
"$PPL" -m "$M/$NAME-BF16.gguf" -f "$TEXT" "${COMMON[@]}" --kl-divergence-base "$BASE" > "$OUT/BF16.log" 2>&1 \
  || { tail -20 "$OUT/BF16.log"; exit 1; }
grep -E "Final estimate" "$OUT/BF16.log" | sed "s/^/  BF16: /" || true
for v in $VARIANTS; do
  f=$M/$NAME-$v.gguf
  if [ ! -f "$f" ]; then echo "  skip $v: no $f"; continue; fi
  "$PPL" -m "$f" "${COMMON[@]}" --kl-divergence-base "$BASE" --kl-divergence > "$OUT/$v.log" 2>&1 \
    || { tail -20 "$OUT/$v.log"; exit 1; }
  grep -E "Mean +KLD|Same top p" "$OUT/$v.log" | sed "s/^/  $v: /" || true
done
"$PY" "$ROOT/scripts/kl_report.py" --model "$NAME" --dir "$OUT" --models-dir "$M" --text "$TEXT" \
  --ctx "$CTX" --chunks "$CHUNKS" --out "$OUT/kl_$NAME.json"
[ "${KL_KEEP_BASE:-0}" = 1 ] || rm -f "$BASE"
