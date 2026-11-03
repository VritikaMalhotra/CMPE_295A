#!/bin/bash
# Two-minute check that a job can requeue itself on this cluster (S7 and S10 depend on it). Slurm sends USR1
# 60 s before the 2-minute limit; the trap requeues once. Pass: the log shows "restart 1" after "restart 0".
#   sbatch slurm/requeue_probe.sbatch && scontrol show config | grep -i requeue
#SBATCH -J requeue-probe
#SBATCH -A coe
#SBATCH -p defq
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:02:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
echo "probe job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} at $(date +%s)"
if [ "${SLURM_RESTART_COUNT:-0}" -ge 1 ]; then echo "PASS: requeued and restarted"; exit 0; fi
requeue() { echo "USR1 at $(date +%s): scontrol requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap requeue USR1
sleep 300 &
wait $!
echo "FAIL: no USR1 before the limit"
