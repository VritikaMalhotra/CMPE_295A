#!/usr/bin/env bash
# Run an experiment that has tau2 benchmarks on the cluster (decision 4: the simulated user is its own job).
#   source scripts/hpc_env.sh && bash scripts/tau2_submit.sh <experiment> [<experiment> ...]
# Submits, in order:
#   1. slurm/simuser.sbatch: one simulator for all the experiments given (they must use the same simulator
#      settings; each agent job checks it); A100 by default, SIM_GRES=gpu:h100:1 for an H100;
#   2. each experiment's own array job (python -m testbed.run writes it), starting once the simulator job has
#      started; every array task reads the simulator's endpoint file and keeps its own C4 store;
#   3. each experiment's report job after its array, and one job that stops the simulator after all arrays.
# GPU QoS: at most 2 GPU nodes per user. The simulator holds one A100 GPU, so agent tasks default to A100s too
# (AGENT_GRES): 3 next to the simulator plus 4 on a second A100 node. On P100s only one agent task would run.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: tau2_submit.sh <experiment> [<experiment> ...]"; exit 2; }
SIM_GRES=${SIM_GRES:-gpu:a100:1}
AGENT_GRES=${AGENT_GRES:-gpu:a100:1}
NAME=$(IFS=+; echo "$*")
ENDPOINT=$TESTBED_OUT/simuser/$NAME.endpoint.json
mkdir -p "$TESTBED_OUT/simuser"
export LITELLM_LOCAL_MODEL_COST_MAP=True

# each run writes specs.jsonl, array.sbatch and report.sbatch in its experiment directory (and submits nothing)
dirs=()
for EXP in "$@"; do
  out=$(.venv/bin/python -m testbed.run +experiment="$EXP" skip_hash=true simuser_endpoint="$ENDPOINT")
  echo "$out" | grep -E "specs|dir:|wrote|WARNING" || true
  dir=$(echo "$out" | sed -n 's/^ *dir: //p')
  [ -f "$dir/array.sbatch" ] || { echo "no array.sbatch in $dir (nothing pending?)"; exit 1; }
  dirs+=("$dir")
done
rm -f "$ENDPOINT"                       # a stale file from an earlier simulator would point nowhere

sim=$(sbatch --parsable --gres="$SIM_GRES" -o "$TESTBED_OUT/simuser/%j.out" \
        --export=ALL,EXP="$1",ENDPOINT="$ENDPOINT" slurm/simuser.sbatch)
echo "simulator $sim ($SIM_GRES, endpoint $ENDPOINT)"
arrs=()
for i in "${!dirs[@]}"; do
  arr=$(sbatch --parsable --dependency=after:"$sim" --gres="$AGENT_GRES" --exclude=cs001,cs002 \
          --export=ALL,TESTBED_SIMUSER_ENDPOINT="$ENDPOINT",LITELLM_LOCAL_MODEL_COST_MAP=True "${dirs[$i]}/array.sbatch")
  arr=${arr%%;*}
  rep=$(sbatch --parsable --dependency=afterany:"$arr" "${dirs[$i]}/report.sbatch")
  arrs+=("$arr")
  echo "${@:$((i + 1)):1}: agents $arr ($AGENT_GRES, after the simulator starts), report $rep"
  echo "${@:$((i + 1)):1} sim=$sim array=$arr report=$rep $(date -Iseconds)" >> "$TESTBED_OUT/simuser/submissions.txt"
done
stop=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${arrs[*]}") -A coe -p defq -c 1 -t 00:05:00 \
         -J simuser-stop -o "$TESTBED_OUT/simuser/stop_%j.out" --wrap "scancel $sim")
echo "simulator stop $stop (after all agents)"
