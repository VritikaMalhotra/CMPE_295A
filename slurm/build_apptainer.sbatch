#!/bin/bash
# Stage B of the container: compile llama.cpp into the image, offline (apptainer/testbed.def).
# Stage A first, on the login node:  apptainer build --fakeroot $CMPE295_GROUP/containers/testbed-base.sif apptainer/base.def
#   sbatch slurm/build_apptainer.sbatch
# Writes $CMPE295_GROUP/containers/testbed.sif and its .sha256. Needs --fakeroot on compute nodes; check first with
# `apptainer buildcfg | grep -i fakeroot` (docs/m3_quant.md, step 0).
#SBATCH -J build-apptainer
#SBATCH -A coe
#SBATCH -p cpuqs
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /scratch/cmpe295-bertin/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc_env.sh
log_gpu                                   # host and job id (no GPU on cpuqs)
C=$CMPE295_GROUP/containers
mkdir -p "$C"
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$CMPE295_GROUP/cache/apptainer-tmp}
mkdir -p "$APPTAINER_TMPDIR"
BUILD_JOBS=${SLURM_CPUS_PER_TASK:-16} apptainer build --fakeroot \
  --build-arg BASE_SIF="$C/testbed-base.sif" --build-arg LLAMA_SRC="$LLAMA_DIR" \
  "$C/testbed.sif" apptainer/testbed.def
sha256sum "$C/testbed.sif" | tee "$C/testbed.sif.sha256"
apptainer test "$C/testbed.sif"
