#!/bin/bash
# Phase 0 GPU checks on scavenge H200s: GPU-only tests (flex/bf16 exactness) + short throughput/memory pilots.
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -o pipefail
source /work/wac20/scaling_gd_for_lm/slurm/env.sh
eval "$CMD"
