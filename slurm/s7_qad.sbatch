#!/bin/bash
# Spike S7 on the cluster (Workbook 2 Table 35; onboarding step 6): the toy QAD run, once uninterrupted (A) and
# once cut by a 10-minute Slurm limit (B). B catches USR1 60 s before the limit, stops train/qad_toy.py, requeues
# itself and resumes from the last checkpoint (every 5 steps). Pass: B's per-step losses equal A's.
#   RUN=A S7_TAG=s7 S7_STEPS=50 sbatch -t 01:00:00 slurm/s7_qad.sbatch
#   RUN=B S7_TAG=s7 S7_STEPS=50 sbatch -t 00:10:00 slurm/s7_qad.sbatch
#   .venv/bin/python train/compare_loss.py $TESTBED_OUT/s7/s7/A $TESTBED_OUT/s7/s7/B --b-log <B's .out> --out ...
# Choosing S7_STEPS: run A with 50 steps first and read the seconds per step from loss.jsonl. If 50 steps end
# well inside 9 minutes, raise S7_STEPS (about 1.5 x the steps that fit before the signal) so the limit cuts B
# mid-run, and build matching data first: train/make_qad_data.py --n $((4 * S7_STEPS + 16)) (login node).
# A and B run on the same GPU type (A100); cs002 is excluded (docs/cluster.md).
#SBATCH -J s7-qad
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cs002
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
RUN=${RUN:?set RUN=A or RUN=B}
TAG=${S7_TAG:-s7}
STEPS=${S7_STEPS:-50}
DATA=${S7_DATA:-$CMPE295_GROUP/data/qad/seqs.pt}
OUT=$TESTBED_OUT/s7/$TAG/$RUN
MAX_REQUEUE=${S7_MAX_REQUEUE:-3}
LIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null || echo unknown)
echo "S7 run $RUN tag $TAG steps $STEPS limit $LIMIT restart ${SLURM_RESTART_COUNT:-0} start $(date +%s)"
requeue() {
  echo "walltime near at $(date +%s): stopping qad_toy and requeueing $SLURM_JOB_ID"
  kill -TERM "$PID" 2>/dev/null || true
  wait "$PID" || true
  if [ "${SLURM_RESTART_COUNT:-0}" -lt "$MAX_REQUEUE" ]; then scontrol requeue "$SLURM_JOB_ID"; else echo "requeue limit reached"; fi
  exit 0
}
trap requeue USR1
.venv/bin/python train/qad_toy.py --model "$MODEL_SRC_DIR/Qwen3-0.6B" --data "$DATA" --out "$OUT" \
  --steps "$STEPS" --ckpt-every 5 &
PID=$!
wait "$PID"
echo "S7 run $RUN finished at $(date +%s)"
