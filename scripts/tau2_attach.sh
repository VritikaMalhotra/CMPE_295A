#!/usr/bin/env bash
# Add experiments to a simulated-user job that is already running (scripts/tau2_submit.sh started it), so they
# share one simulator instead of taking another GPU node under the 2-node QoS. The experiments must use the same
# simulator settings (each agent task checks the endpoint file names that simulator).
#   source scripts/hpc_env.sh && bash scripts/tau2_attach.sh <endpoint.json> <simulator job id> <experiment> [...]
# Submits each experiment's agent array (after the simulator has started) and report. It does not touch the
# simulator's stop job: afterwards replace it with one that waits for every array that uses the simulator, e.g.
#   scancel <old stop job>; sbatch --dependency=afterany:<a1>:<a2>:... -A coe -p defq -c 1 -t 00:05:00 \
#       -J simuser-stop --wrap "scancel <simulator job id>"
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 3 ] || { echo "usage: tau2_attach.sh <endpoint.json> <sim job id> <experiment> [...]"; exit 2; }
ENDPOINT=$1; SIM=$2; shift 2
AGENT_GRES=${AGENT_GRES:-gpu:a100:1}
AGENT_MEM=${AGENT_MEM:-24G}
export LITELLM_LOCAL_MODEL_COST_MAP=True
squeue -h -j "$SIM" -o %T | grep -q RUNNING || { echo "simulator job $SIM is not running"; exit 1; }
[ -f "$ENDPOINT" ] || { echo "no endpoint file $ENDPOINT"; exit 1; }
for EXP in "$@"; do
  out=$(.venv/bin/python -m testbed.run +experiment="$EXP" skip_hash=true simuser_endpoint="$ENDPOINT")
  echo "$out" | grep -E "specs|dir:|wrote|WARNING" || true
  dir=$(echo "$out" | sed -n 's/^ *dir: //p')
  [ -f "$dir/array.sbatch" ] || { echo "no array.sbatch in $dir (nothing pending?)"; exit 1; }
  arr=$(sbatch --parsable --dependency=after:"$SIM" --gres="$AGENT_GRES" --mem="$AGENT_MEM" --exclude=cs001,cs002 \
          --export=ALL,TESTBED_SIMUSER_ENDPOINT="$ENDPOINT",LITELLM_LOCAL_MODEL_COST_MAP=True "$dir/array.sbatch")
  arr=${arr%%;*}
  rep=$(sbatch --parsable --dependency=afterany:"$arr" "$dir/report.sbatch")
  echo "$EXP: agents $arr ($AGENT_GRES, $AGENT_MEM), report $rep (simulator $SIM)"
  echo "$EXP sim=$SIM array=$arr report=$rep attached $(date -Iseconds)" >> "$TESTBED_OUT/simuser/submissions.txt"
done
