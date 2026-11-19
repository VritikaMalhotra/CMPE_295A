#!/bin/bash
# Damage map v1 (M3, Dec 4): per-model H1 damage previews for all six models, then the cross-model map.
# For each model: analysis/damage_table.py (step-level amplification, paired cluster bootstrap), which_tasks_fail.py
# (success and failure rates, mean ± 1 SD over seeds 1-4) and worked_examples.py (BF16 vs Q4_0-pure); then
# analysis/damage_map.py over all of them. Development data, preview (the pre-registration is not filed).
#   sbatch -A coe -p cpuqs -c 4 --mem=32G -t 08:00:00 -o $CMPE295_GROUP/logs/damage_previews_%j.out scripts/damage_previews.sh
set -euo pipefail
cd "${REPO:-$SLURM_SUBMIT_DIR}"
source scripts/hpc_env.sh
source .venv/bin/activate
echo "repo $(pwd) at $(git rev-parse --short HEAD)"
mkdir -p results/m3 docs/figs docs/damage_previews

# the run directory of an experiment: the hash dir with specs.jsonl and the most finished episodes
rundir() {
  local best="" n=-1 d c
  for d in "$TESTBED_OUT/$1"/*/; do
    [ -f "$d/specs.jsonl" ] || continue
    c=$(find "$d/episodes" -name '*.json' 2>/dev/null | wc -l)
    [ "$c" -gt "$n" ] && { n=$c; best=${d%/}; }
  done
  [ -n "$best" ] || { echo "no run dir for $1" >&2; exit 1; }
  echo "$best"
}

MODELS="qwen3-0.6b:qwen3_0p6b qwen3-1.7b:qwen3_1p7b qwen3-4b:qwen3_4b smollm3-3b:smollm3_3b lfm2.5-1.2b:lfm2p5_1p2b lfm2.5-2.6b:lfm2p5_2p6b"
for m in $MODELS; do
  name=${m%%:*}; key=${m##*:}
  ref=$(rundir ref_$key); q40=$(rundir h1_${key}_q4_0); qkm=$(rundir h1_${key}_q4_k_m); iq4=$(rundir h1_${key}_iq4_xs)
  cells=(--cell Q4_0-pure=$q40 --cell Q4_K_M=$qkm --cell IQ4_XS=$iq4)
  echo "== $name $(date -Iseconds): ref $ref"
  python analysis/damage_table.py --model "$name" --ref BF16="$ref" "${cells[@]}" \
    --json results/m3/damage_preview_$name.json --md docs/damage_previews/damage_preview_$name.md
  python analysis/which_tasks_fail.py --model "$name" --ref BF16="$ref" "${cells[@]}" \
    --json results/m3/which_tasks_fail_$name.json --md docs/damage_previews/which_tasks_fail_$name.md \
    --fig docs/figs/which_tasks_fail_$name.png
  python analysis/worked_examples.py --model "$name" --ref BF16="$ref" --cell Q4_0-pure="$q40" \
    --out docs/damage_previews/worked_examples_$name.md --json results/m3/worked_examples_$name.json
done
python analysis/damage_map.py --out-json results/m3/damage_map_v1.json --md docs/m3_damage_map_v1.md \
  --fig docs/figs/damage_map_v1.png
echo "done $(date -Iseconds)"
