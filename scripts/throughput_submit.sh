#!/usr/bin/env bash
# NFR-5 throughput pilot on the cluster. Each level is "<gpu type>:<N>": an array of N single-GPU jobs of that
# type, PER episodes each; levels run one after the other (each waits for the previous one).
#   source scripts/hpc_env.sh && bash scripts/throughput_submit.sh
#   DEP=<job ids, colon-separated> bash scripts/throughput_submit.sh   # start after other GPU jobs of yours
#   bash scripts/throughput_submit.sh report        # once all levels finished -> results/m2/throughput.json
# Why these levels (measured 2026-11-08): every GPU partition has QoS gpu3nodes = at most 2 nodes per user.
# P100 nodes have 1 GPU, so one account runs at most 2 P100 jobs; A100 nodes have 4 GPUs (cs001 and cs002 are
# bad), so up to 8 single-GPU jobs fit on two A100 nodes.
set -euo pipefail
cd "$(dirname "$0")/.."
LEVELS=${LEVELS:-"p100:1 p100:4 a100:4 a100:8"}
PER=${PER:-24}
JOBS=$TESTBED_OUT/throughput_jobs.txt

if [ "${1:-}" = report ]; then
  .venv/bin/python scripts/throughput_report.py --jobs "$JOBS" --out-root "$TESTBED_OUT" --out results/m2/throughput.json
  exit 0
fi

: > "$JOBS"
dep=${DEP:-}
for lv in $LEVELS; do
  gpu=${lv%%:*}; n=${lv##*:}; label="${gpu}_n${n}"
  # create the experiment directory once here, so N jobs never race to write specs.jsonl
  .venv/bin/python -m testbed.run +experiment=throughput experiment.name="throughput_$label" dry_run=true \
    skip_hash=true | grep -E "specs|dir:"
  jid=$(sbatch --parsable ${dep:+--dependency=afterany:$dep} -J "tp-$label" -p gpuqs --gres="gpu:$gpu:1" \
          --exclude=cs001,cs002 --array=0-$((n - 1)) --export=ALL,LABEL="$label",N="$n",PER="$PER" \
          slurm/throughput.sbatch)
  echo "$label $n $jid" >> "$JOBS"
  echo "level $label: array $jid${dep:+ (after $dep)}"
  dep=$jid
done
