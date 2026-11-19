#!/bin/bash
# C4 simulated user for one experiment (decision 4): llama-server with the experiment's simulator model on one
# A100/H100, announced in $ENDPOINT for the agent jobs. Submitted by scripts/tau2_submit.sh with EXP and
# ENDPOINT in the environment; stopped by that script's cleanup job once every agent job has ended.
#SBATCH -J simuser
#SBATCH -p gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs002
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2-01:00:00
#SBATCH -A coe
#SBATCH -o simuser_%j.out
#
# Restart: gpt-oss's llama-server segfaulted twice after 10-13 h (jobs 120510 and 130139, 2026-11-17). When the
# service exits with an error it is started again on a new port and rewrites $ENDPOINT; agent jobs wait for the
# new endpoint (testbed.simuser_service.reconnect) and re-run the interrupted episode. At most
# SIMUSER_MAX_RESTARTS restarts (default 200). scancel (the cleanup job) stops the loop.
# The crashes follow load, not one node: 150003 on cs003 crashed 21 times in 9 h (2026-11-18) under ~7 agents.
set -euo pipefail
source scripts/hpc_env.sh
log_gpu
source .venv/bin/activate
export LITELLM_LOCAL_MODEL_COST_MAP=True
MAX_RESTARTS=${SIMUSER_MAX_RESTARTS:-200}
# SIMUSER_OVERRIDES: extra Hydra overrides (word-split), e.g. 'simuser_parallel=1 simuser_extra_args=[--no-cache-prompt]'
SIMUSER_OVERRIDES=${SIMUSER_OVERRIDES:-}
stop=""; child=""
trap 'stop=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM INT
set +e
for ((i = 0; i <= MAX_RESTARTS; i++)); do
  python -m testbed.run +experiment="$EXP" simuser_serve=true simuser_endpoint="$ENDPOINT" skip_hash=true $SIMUSER_OVERRIDES &
  child=$!
  wait "$child"; rc=$?
  [ -n "$stop" ] && { wait "$child" 2>/dev/null; echo "simuser: stopped"; exit 0; }
  [ "$rc" -eq 0 ] && exit 0
  echo "simuser: service exited with $rc at $(date -Iseconds) (start $((i + 1))); restart $((i + 1)) of $MAX_RESTARTS in 15 s" >&2
  sleep 15
done
echo "simuser: gave up after $MAX_RESTARTS restarts" >&2
exit 1
