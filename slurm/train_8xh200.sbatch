#!/bin/bash
# Pretraining on 1 node x 8 H200 (lab partition). Submitted by launch.py with RUN_ID set.
# Auto-resume: train.py saves runs/$RUN_ID/resume.pt and writes REQUEUE ~10 min before the time limit;
# this script then requeues the same job id (dependants waiting on afterok/afterany keep waiting).
# Crashes are retried by requeue up to 3 times (runs/$RUN_ID/FAILURES). Exit 0 only when the run is completed,
# so the eval job (afterok) starts only for completed runs; diverged runs exit 3.
#SBATCH --partition=h200-hp
#SBATCH --account=henaolab_h200
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --open-mode=append
set -o pipefail
source /work/wac20/scaling_gd_for_lm/slurm/env.sh
R=runs/$RUN_ID
if [[ ! -f $R/DONE ]]; then
  torchrun --standalone --nproc_per_node=8 train.py --run-id "$RUN_ID"
  rc=$?
  if [[ -f $R/REQUEUE ]]; then
    rm -f $R/REQUEUE; echo "requeue $SLURM_JOB_ID at $(date)"; scontrol requeue "$SLURM_JOB_ID"; exit 0
  fi
  if [[ $rc -ne 0 ]]; then
    echo "$(date) job $SLURM_JOB_ID exit $rc" >> $R/FAILURES
    if [[ $(wc -l < $R/FAILURES) -lt 3 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
    exit $rc
  fi
fi
python -c "import json,sys; sys.exit(0 if json.load(open('$R/status.json'))['status']=='completed' else 3)"
