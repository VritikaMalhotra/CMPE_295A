#!/bin/bash
# Several configurations (or shards of one) at once on ONE GPU: each worker runs `python -m testbed.run ...
# local=true only_config=<key> [shard=i/n]` with its own llama-server on the same GPU. Under QoS gpu3nodes a user
# gets 2 GPU nodes, so a small model's array runs one episode stream per node; packing K streams into one GPU
# (Qwen3-0.6B at ctx 40,960 needs about 6 GB) multiplies that without another node. The per-configuration
# (per-shard) locks in testbed/run.py keep two processes from ever running the same specs, so several pack jobs,
# and the experiment's own array tasks, can share one list safely; finished episodes are never rerun.
#   sbatch --gres=gpu:a100:1 --export=ALL,PACK_LIST=<file>,PACK_K=4,REPO=<checkout>,TESTBED_SIMUSER_ENDPOINT=<file> \
#          slurm/pack.sbatch
# PACK_LIST: one item per line, "<experiment> <config_key> [<i>/<n>]", worked through in order.
#SBATCH -J tb-pack
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs001,cs002
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 1-12:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "${REPO:-$SLURM_SUBMIT_DIR}"
source scripts/hpc_env.sh
log_gpu
source .venv/bin/activate
export LITELLM_LOCAL_MODEL_COST_MAP=True
: "${PACK_LIST:?set PACK_LIST}" "${TESTBED_SIMUSER_ENDPOINT:?set TESTBED_SIMUSER_ENDPOINT}"
export TESTBED_CPU_THREADS=2
echo "repo $(pwd) at $(git rev-parse --short HEAD); list $PACK_LIST; K=${PACK_K:-4}"
run_one() {
  set -- $1
  local shard=""
  [ -n "${3:-}" ] && shard="shard=$3"
  echo "start $1 $2 $shard $(date -Iseconds)"
  python -m testbed.run +experiment="$1" local=true only_config="'$2'" $shard simuser_endpoint="$TESTBED_SIMUSER_ENDPOINT" \
    > "$CMPE295_GROUP/logs/pack_${SLURM_JOB_ID}_$1_$2${3:+_${3/\//of}}.log" 2>&1 || echo "FAILED $1 $2 $shard"
  echo "end $1 $2 $shard $(date -Iseconds)"
}
export -f run_one
grep -v '^\s*#' "$PACK_LIST" | grep -v '^\s*$' | xargs -d '\n' -P "${PACK_K:-4}" -I{} bash -c 'run_one "$@"' _ {}
echo "pack done $(date -Iseconds)"
