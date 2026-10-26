#!/bin/bash
# Evaluation of one selected checkpoint on 1 x H200 (scavenge, preemptible, requeued). Idempotent: every
# piece (test loss + per-token NLL, 7 benchmarks, samples) is written atomically and skipped when present.
# Afterwards refreshes selection / aggregation / figures. Fallback partition: launch.py --eval-fallback.
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -eo pipefail
source /work/wac20/scaling_gd_for_lm/slurm/env.sh
python eval_loss.py --run-id "$RUN_ID"
python eval_benchmarks.py --run-id "$RUN_ID"
python sample.py --run-id "$RUN_ID"
python select_runs.py && python aggregate.py && python render.py || echo "artifact refresh failed (non-fatal)"
