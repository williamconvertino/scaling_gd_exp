# Sourced by every job. Paths live here only.
export REPO=/work/wac20/scaling_gd_for_lm
# Env in $HOME: /work purges files with old timestamps (broke a conda env on 2026-10-15, MIGRATION_REPORT.md §3).
# venv on top of the old gdlm conda env (torch 2.14.1+cu130), with lm_eval==0.4.9 / transformers 4.57.1 pinned.
export VENV=/hpc/home/wac20/envs/scaling_gd
source /work/wac20/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/home/wac20/envs/gdlm
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME=/work/wac20/.cache/huggingface            # benchmark datasets + tokenizers (old jobs' cache)
# HF token from the repo's .env (gitignored, chmod 600); key may be `hf_token` or `HF_TOKEN`
if [[ -f "$REPO/.env" ]]; then
  _tok=$(grep -i -m1 '^hf_token=' "$REPO/.env" | cut -d= -f2- | tr -d '[:space:]"')
  [[ -n "$_tok" ]] && export HF_TOKEN="$_tok"; unset _tok
fi
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L 2>/dev/null || true
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} restarts=${SLURM_RESTART_COUNT:-0} $(date)"
