#!/usr/bin/env bash
# C9 speed tables with llama-bench (pp512 prompt processing, tg128 generation, 3 repetitions) for every GGUF
# in a models folder, on one backend. Output: one JSON per model + a CSV summary, all under <out_dir>.
#   scripts/bench.sh cpu  <models_dir> <out_dir> [threads]
#   scripts/bench.sh cuda <models_dir> <out_dir>
# Development data on a shared Colab machine; the cost study uses the exclusive-node protocol (M6).
set -euo pipefail
BACKEND=$1; M=$(realpath "$2"); OUT=$3; T=${4:-$(nproc)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/llama.cpp/build-$BACKEND/bin/llama-bench
[ -x "$B" ] || { echo "missing $B (build target llama-bench)"; exit 1; }
mkdir -p "$OUT"
HOST=$(hostname)
for f in "$M"/*.gguf; do
  name=$(basename "$f" .gguf)
  if [ "$BACKEND" = cuda ]; then args=(-ngl 99); else args=(-ngl 0 -t "$T"); fi
  echo "== $name ($BACKEND)"
  "$B" -m "$f" -p 512 -n 128 -r 3 "${args[@]}" -o json > "$OUT/$name.$BACKEND.$HOST.json"
done
python3 - "$OUT" "$BACKEND" <<'P'
import glob, json, sys, csv, os
out, backend = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(f"{out}/*.{backend}.*.json")):
    for r in json.load(open(p)):
        test = "pp512" if r.get("n_prompt") else "tg128"
        rows.append({"model": os.path.basename(p).split(f".{backend}.")[0], "backend": backend, "test": test,
                     "tokens_per_s": round(r["avg_ts"], 1), "stdev": round(r["stddev_ts"], 1),
                     "threads": r.get("n_threads"), "cpu": r.get("cpu_info", ""), "gpu": r.get("gpu_info", ""),
                     "build": r.get("build_commit", "")})
with open(f"{out}/bench_{backend}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['model']:<36} {r['test']:<6} {r['tokens_per_s']:>9} ± {r['stdev']}")
P
