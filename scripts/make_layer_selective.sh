#!/usr/bin/env bash
# Build the R2b layer-selective variant from the ranking (scripts/rank_tensors.py select): Q4_0 everywhere
# (--pure, like the Q4_0 control) except the chosen tensors at Q8_0, then check FR-E10: whole-file bits per
# weight within 5.5 +/- 0.1. Writes $TESTBED_MODELS/<ModelName>-R2b-LS5.5.gguf + .sha256 + .check.json.
#   bash scripts/make_layer_selective.sh <ModelName>          (inside a job, or on a CPU node: no GPU needed)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=${1:?usage: make_layer_selective.sh <ModelName>}
M=${TESTBED_MODELS:?source scripts/hpc_env.sh first}
QUANT=${LLAMA_QUANTIZE:-${LLAMA_DIR:?}/build-cuda/bin/llama-quantize}
R=${RANK_OUT:-$TESTBED_OUT/rank/$NAME}
TT=$R/tensor_types.txt
OUT=$M/$NAME-R2b-LS5.5.gguf
[ -f "$TT" ] || { echo "no $TT: python scripts/rank_tensors.py select --model $NAME"; exit 2; }
[ -f "$OUT" ] || "$QUANT" --pure --tensor-type-file "$TT" "$M/$NAME-BF16.gguf" "$OUT" Q4_0 > "$OUT.log" 2>&1 \
  || { tail -20 "$OUT.log"; exit 1; }
sha256sum "$OUT" | tee "$OUT.sha256"
"${PY:-$ROOT/.venv/bin/python}" - "$OUT" "${LLAMA_DIR}/gguf-py" "$R/selection.json" <<'PY'
import json, os, sys
sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
f = sys.argv[1]
r = GGUFReader(f)
n = sum(int(t.n_elements) for t in r.tensors)
bpw = os.path.getsize(f) * 8 / n
types = {}
for t in r.tensors:
    types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + int(t.n_elements)
sel = json.load(open(sys.argv[3]))
rep = {"file": f, "bpw_file": bpw, "target": 5.5, "within_0.1": abs(bpw - 5.5) <= 0.1,
       "predicted_bpw": sel["predicted_bpw"], "n_q8_0_tensors": sel["n_chosen"],
       "share_of_linear_weights_q8_0": sel["share_of_linear_weights"], "weights_by_type": types}
json.dump(rep, open(f + ".check.json", "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("bpw_file", "within_0.1", "predicted_bpw", "n_q8_0_tensors")}))
sys.exit(0 if rep["within_0.1"] else 1)
PY
