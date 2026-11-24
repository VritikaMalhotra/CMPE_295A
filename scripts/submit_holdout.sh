#!/usr/bin/env bash
# Submit the confirmatory holdout stages (pre-registration https://osf.io/xwq5y, docs/preregistration_osf.md §3):
#   C1a  hold_ref_<model>   BF16 references, 6 models                       (first: the inclusion set comes from them)
#   C1b  hold_h1_<model>    Q4_0 / Q4_K_M / IQ4_XS cells                    (after every C1a array has ended)
#   C2   hold_cpu_<model>   CPU subset, Qwen3-0.6B and 1.7B, greedy         (after C1, on cpuqs)
#   C3   hold_c3_<model>    9 repairs, 3 core models                        (after C1, alongside C2)
# Each experiment: python -m testbed.run writes array.sbatch + report.sbatch (no submit), then this script submits the
# array (GPU arrays on one A100 each, cs002 excluded) and its report job. BFCL only, no simulated user.
#   source scripts/hpc_env.sh; bash scripts/submit_holdout.sh            # GPU type: HOLD_GRES (default gpu:a100:1)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TESTBED_OUT:?source scripts/hpc_env.sh first}"
GRES=${HOLD_GRES:-gpu:a100:1}
LOG=$TESTBED_OUT/holdout_submissions.txt
REF="qwen3_0p6b qwen3_1p7b qwen3_4b smollm3_3b lfm2p5_1p2b lfm2p5_2p6b"
CPU="qwen3_0p6b qwen3_1p7b"
CORE="qwen3_0p6b qwen3_1p7b lfm2p5_1p2b"

submit() {   # submit <experiment> [<dependency job ids, colon-joined>]
  local e=$1 dep=${2:-} out dir arr rep
  out=$(.venv/bin/python -m testbed.run +experiment="$e" 2>&1 | grep -v WARNING)
  dir=$(echo "$out" | sed -n 's/^ *dir: //p' | tail -1)
  [ -f "$dir/array.sbatch" ] || { echo "$e: no array.sbatch (nothing pending?)" >&2; echo "$out" | tail -3 >&2; return 1; }
  local opts=(--parsable)
  [ -n "$dep" ] && opts+=(--dependency=afterany:"$dep")
  grep -q -- "--gres" "$dir/array.sbatch" && opts+=(--gres="$GRES" --exclude=cs002)
  arr=$(sbatch "${opts[@]}" "$dir/array.sbatch"); arr=${arr%%;*}
  rep=$(sbatch --parsable --dependency=afterany:"$arr" "$dir/report.sbatch")
  echo "$(date -Iseconds) $e $(basename "$dir") array=$arr report=$rep dep=${dep:-none}" | tee -a "$LOG" >&2
  echo "$arr"
}
join() { local IFS=:; echo "$*"; }

a=(); for m in $REF; do a+=("$(submit hold_ref_$m)"); done
c1a=$(join "${a[@]}")
b=(); for m in $REF; do b+=("$(submit hold_h1_$m "$c1a")"); done
c1=$(join "${a[@]}" "${b[@]}")
for m in $CPU; do submit hold_cpu_$m "$c1" >/dev/null; done
for m in $CORE; do submit hold_c3_$m "$c1" >/dev/null; done
echo "submitted; log: $LOG" >&2
