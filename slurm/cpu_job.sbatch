#!/bin/bash
# Generic CPU-only job (manifests, checksums, aggregation, plotting). Usage: sbatch --export=ALL,CMD="..." slurm/cpu_job.sbatch
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/work/wac20/scaling_gd_for_lm/logs/cpu/%x-%j.out
set -eo pipefail
source /work/wac20/scaling_gd_for_lm/slurm/env.sh
echo "CMD: $CMD"
eval "$CMD"
