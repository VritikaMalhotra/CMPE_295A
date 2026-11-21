#!/bin/bash
# R3 training, one arm of one model (train/qad.py train; pre-registration §2, Workbook 2 FR-E11).
#   sbatch --export=ALL,QAD_ARGS="--max-steps 100" slurm/qad_train.sbatch <ModelName> <kl|klce|control>
# Data: $CMPE295_GROUP/data/qad_pack/<ModelName> (train/qad.py pack). Output: $TESTBED_OUT/r3/<ModelName>/<arm>
# (+ QAD_TAG, e.g. pilot). Slurm sends USR1 300 s before the walltime; the trainer checkpoints and stops, and this
# script requeues itself (at most QAD_MAX_REQUEUE times); the next start resumes from ckpt.pt with identical losses
# (tests/test_qad.py). Python: the group training env (torch, transformers), not the testbed .venv.
#SBATCH -J qad-train
#SBATCH -A coe
#SBATCH -p gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu
M=${1:?usage: sbatch slurm/qad_train.sbatch <ModelName> <arm>}
ARM=${2:?arm: kl, klce or control}
PY=${QAD_PY:-/scratch/cmpe295-bertin/envs/qad-data/bin/python}
OUT=$TESTBED_OUT/r3/$M/${QAD_TAG:+$QAD_TAG/}$ARM
MAX_REQUEUE=${QAD_MAX_REQUEUE:-5}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=${QAD_EXTRA_LIB:-/scratch/cmpe295-bertin/envs/pytest-lib}${PYTHONPATH:+:$PYTHONPATH}   # sentencepiece, pytest
echo "qad $M $ARM -> $OUT, restart ${SLURM_RESTART_COUNT:-0}, $(git rev-parse --short HEAD), $(date -Iseconds)"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA in the training env'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
stopped=""
trap 'stopped=1; kill -USR1 "$PID" 2>/dev/null || true' USR1
"$PY" train/qad.py train --model-dir "$TESTBED_MODELS/hf/$M" --pack "$CMPE295_GROUP/data/qad_pack/$M" \
  --arm "$ARM" --out "$OUT" ${QAD_ARGS:-} &
PID=$!
set +e
wait "$PID"; rc=$?
[ -n "$stopped" ] && { wait "$PID"; rc=$?; }   # the first wait returns early when the trap fires
set -e
if [ -n "$stopped" ] && [ ! -d "$OUT/final" ]; then
  if [ "${SLURM_RESTART_COUNT:-0}" -lt "$MAX_REQUEUE" ]; then
    echo "walltime near: checkpointed, requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"
  else
    echo "requeue limit reached"; exit 1
  fi
fi
exit "$rc"
