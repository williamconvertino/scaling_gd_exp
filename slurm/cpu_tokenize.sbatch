#!/bin/bash
# Llama-3.2 tokenization of FineWeb-Edu on CPU partitions only (DETAILS.md §4.3). Array task i handles parquet
# files i, i+N, ...; finished files have a .done marker with a sha256 and are skipped on requeue/resubmit.
#SBATCH --partition=scavenger
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/work/wac20/scaling_gd_for_lm/logs/tokenize/%x-%A_%a.out
set -eo pipefail
source /work/wac20/scaling_gd_for_lm/slurm/env.sh
export TOKENIZERS_PARALLELISM=true
python tokenize_llama3.py shard --worker-id "$SLURM_ARRAY_TASK_ID" --num-workers "${SLURM_ARRAY_TASK_COUNT:-32}"
