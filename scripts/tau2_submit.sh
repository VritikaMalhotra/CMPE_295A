#!/usr/bin/env bash
# Run an experiment that has tau2 benchmarks on the cluster (decision 4: the simulated user is its own job).
#   source scripts/hpc_env.sh && bash scripts/tau2_submit.sh <experiment>
# Submits, in order:
#   1. slurm/simuser.sbatch: the simulator (A100 by default; SIM_GRES=gpu:h100:1 for an H100);
#   2. the experiment's own array job (python -m testbed.run writes it), starting once the simulator job has
#      started; every array task reads the simulator's endpoint file and keeps its own C4 store;
#   3. the report job after the array, and a cleanup job that stops the simulator after the array.
# GPU QoS: at most 2 GPU nodes per user. The simulator holds one A100 GPU, so agent tasks default to A100s too
# (AGENT_GRES): 3 next to the simulator plus 4 on a second A100 node. On P100s only one agent task would run.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=${1:?usage: tau2_submit.sh <experiment>}
SIM_GRES=${SIM_GRES:-gpu:a100:1}
AGENT_GRES=${AGENT_GRES:-gpu:a100:1}
ENDPOINT=$TESTBED_OUT/simuser/$EXP.endpoint.json
mkdir -p "$TESTBED_OUT/simuser"
export LITELLM_LOCAL_MODEL_COST_MAP=True

# writes specs.jsonl, array.sbatch and report.sbatch in the experiment directory (and submits nothing)
out=$(.venv/bin/python -m testbed.run +experiment="$EXP" skip_hash=true simuser_endpoint="$ENDPOINT")
echo "$out" | grep -E "specs|dir:|wrote|WARNING" || true
dir=$(echo "$out" | sed -n 's/^ *dir: //p')
[ -f "$dir/array.sbatch" ] || { echo "no array.sbatch in $dir (nothing pending?)"; exit 1; }
rm -f "$ENDPOINT"                       # a stale file from an earlier simulator would point nowhere

sim=$(sbatch --parsable --gres="$SIM_GRES" -o "$TESTBED_OUT/simuser/%j.out" \
        --export=ALL,EXP="$EXP",ENDPOINT="$ENDPOINT" slurm/simuser.sbatch)
arr=$(sbatch --parsable --dependency=after:"$sim" --gres="$AGENT_GRES" --exclude=cs001,cs002 \
        --export=ALL,TESTBED_SIMUSER_ENDPOINT="$ENDPOINT",LITELLM_LOCAL_MODEL_COST_MAP=True "$dir/array.sbatch")
arr=${arr%%;*}
rep=$(sbatch --parsable --dependency=afterany:"$arr" "$dir/report.sbatch")
stop=$(sbatch --parsable --dependency=afterany:"$arr" -A coe -p defq -c 1 -t 00:05:00 -J simuser-stop \
         -o "$TESTBED_OUT/simuser/stop_%j.out" --wrap "scancel $sim")
echo "simulator $sim ($SIM_GRES, endpoint $ENDPOINT)"
echo "agents    $arr ($AGENT_GRES, start after the simulator starts)"
echo "report    $rep, simulator stop $stop (after the agents)"
echo "$EXP sim=$sim array=$arr report=$rep stop=$stop $(date -Iseconds)" >> "$TESTBED_OUT/simuser/submissions.txt"
