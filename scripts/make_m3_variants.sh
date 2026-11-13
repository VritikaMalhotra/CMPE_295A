#!/usr/bin/env bash
# M3 and R2a variants of one model from its BF16 file (Workbook 2 FR-E5, FR-E10; Ch. 8 "M3 damage map v1").
#   bash scripts/make_m3_variants.sh <ModelName> [variant ...]     e.g. Qwen3-0.6B
# Default variants: IQ4_XS (with an importance matrix), Q5_0-pure and Q6_K-pure (R2a uniform bits).
# Owner: Yashashav (model conversion, R2). Run inside a GPU job: slurm/m3_variants.sbatch.
#
# Inputs: $TESTBED_MODELS/<ModelName>-BF16.gguf (scripts/make_probe_models.sh) and the calibration text
# $M3_CALIB (default $CMPE295_GROUP/data/calib/calib.txt), which must pass the FR-E5 13-gram check first
# (scripts/benchmark_ngrams.py; the report's "pass" must be true).
# Outputs next to the other files: <ModelName>-<variant>.gguf + .sha256, and <ModelName>.imatrix.gguf (kept,
# it is part of how IQ4_XS was built). Existing files are never rebuilt.
#
# Choices to confirm (see docs/m3_quant.md): Q5_0 and Q6_K are built with --pure, like the Q4_0 control, so
# their bits per weight are the nominal 5.5 and 6.56 that Workbook 2 Table 19 lists; IQ4_XS uses llama.cpp's
# default recipe with the imatrix (as llama.cpp ships IQ4_XS), not --pure.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:?usage: make_m3_variants.sh <ModelName> [variant ...]}
shift
VARIANTS=${*:-IQ4_XS Q5_0-pure Q6_K-pure}
M=${TESTBED_MODELS:?source scripts/hpc_env.sh first}
LLAMA=${LLAMA_DIR:?}
QUANT=${LLAMA_QUANTIZE:-$LLAMA/build-cuda/bin/llama-quantize}
IMAT=${LLAMA_IMATRIX:-$LLAMA/build-cuda/bin/llama-imatrix}
CALIB=${M3_CALIB:-${CMPE295_GROUP:?}/data/calib/calib.txt}
CALIB_REPORT=${M3_CALIB_REPORT:-$ROOT/results/m3/calib_ngram_check.json}
IM_CHUNKS=${M3_IMATRIX_CHUNKS:-200}      # 200 x 512 tokens of calibration text
IM_CTX=${M3_IMATRIX_CTX:-512}
NGL=${M3_NGL:-99}
BF16=$M/$NAME-BF16.gguf
IMATRIX_FILE=$M/$NAME.imatrix.gguf

[ -f "$BF16" ] || { echo "no $BF16: sbatch slurm/make_models.sbatch <hf_repo>"; exit 2; }
[ -x "$QUANT" ] || { echo "no llama-quantize at $QUANT"; exit 2; }

needs_imatrix=0
for v in $VARIANTS; do case $v in IQ*) needs_imatrix=1 ;; esac; done
if [ "$needs_imatrix" = 1 ] && [ ! -f "$IMATRIX_FILE" ]; then
  [ -x "$IMAT" ] || { echo "no llama-imatrix at $IMAT: sbatch slurm/build_llama.sbatch"; exit 2; }
  [ -f "$CALIB" ] || { echo "no calibration text at $CALIB: see docs/m3_quant.md step 1"; exit 2; }
  python3 - "$CALIB_REPORT" "$CALIB" <<'EOF' || { echo "calibration text has not passed the FR-E5 13-gram check"; exit 2; }
import hashlib, json, sys
rep = json.load(open(sys.argv[1]))
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert rep.get("pass") is True and rep.get("calib_out_sha256") == sha, (rep.get("pass"), rep.get("calib_out_sha256"), sha)
EOF
  echo "imatrix $NAME: $IM_CHUNKS chunks x $IM_CTX tokens of $CALIB"
  "$IMAT" -m "$BF16" -f "$CALIB" -o "$IMATRIX_FILE" -c "$IM_CTX" --chunks "$IM_CHUNKS" -ngl "$NGL" \
    -t "${TESTBED_CPU_THREADS:-4}" > "$M/$NAME.imatrix.log" 2>&1 || { tail -20 "$M/$NAME.imatrix.log"; exit 1; }
  sha256sum "$IMATRIX_FILE" | tee "$IMATRIX_FILE.sha256"
fi

for v in $VARIANTS; do
  out=$M/$NAME-$v.gguf
  if [ -f "$out" ]; then echo "exists: $out"; continue; fi
  case $v in
    *-pure) "$QUANT" --pure "$BF16" "$out" "${v%-pure}" > "$out.log" 2>&1 ;;
    IQ*)    "$QUANT" --imatrix "$IMATRIX_FILE" "$BF16" "$out" "$v" > "$out.log" 2>&1 ;;
    *)      "$QUANT" "$BF16" "$out" "$v" > "$out.log" 2>&1 ;;
  esac || { tail -20 "$out.log"; exit 1; }
  grep -E "model size|quant size" "$out.log" | sed "s/^/  $v: /" || true
  sha256sum "$out" | tee "$out.sha256"
done
