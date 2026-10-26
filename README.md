# GDLM experiments: Scaling In-Context Functional Gradient Descent for Language Modeling

The specs are in `local_docs/` (`DETAILS.md` > `EXPERIMENT_ORDER.md` > `experiments.md` > `figures.md` > `main.tex`).
The audit of the old codebase is `MIGRATION_REPORT.md`. Live state and open questions are in `STATUS.md`.

## Environment
`source slurm/env.sh` activates `/hpc/home/wac20/envs/scaling_gd`. It is a venv on top of the old `gdlm` conda env
(torch 2.14.1+cu130), with lm-eval 0.4.9 / transformers 4.57.1 / datasets 3.6 pinned. It lives in `$HOME`
because `/work` purges files with old timestamps.

## Files (flat)
| File | Role |
|---|---|
| `common.py` | paths, tokenizer groups, JSON/hash helpers |
| `data.py` | tokenization rules, split plan, packed windows, train/eval streams, manifests (`python data.py manifest`) |
| `tokenize_llama3.py` | Llama-3.2 tokenization (CPU array jobs), same documents/splits/order as Llama-2 |
| `models.py` | `gdlm`, `wang`, `transformer` (one file) |
| `count_params.py` | builds and freezes the 12 configs (`configs/frozen/`), writes `results/param_matching.md` |
| `train.py` | pretraining (DDP, bf16, auto-resume, interval checkpoints, early stopping) |
| `eval_loss.py` | checkpoint selection (val loss), test loss, per-token NLL dump |
| `eval_benchmarks.py` | 7 zero-shot tasks, lm-eval 0.4.9 |
| `sample.py` / `triviaqa.py` | copied from the old repo (glue changes only) |
| `eval_recurrence.py` | Figure 3 analysis from NLL dumps |
| `select_runs.py` → `aggregate.py` → `render.py` | selection, normalized results, figures/tables |
| `launch.py` | `configs/experiments.yaml` → sbatch (train chain + afterok evals); `--dry-run`, `--pilot` |
| `slurm/` | `train_8xh200`, `eval_1xh200_scavenge`, `cpu_tokenize`, `cpu_job`, `pilot_gpu` |
| `tests/` | `python -m pytest -q tests/` (exactness, causality, tying, counts, smoke) |

## Typical commands
```bash
python -m pytest -q tests/
python count_params.py --check
python launch.py --dry-run && python launch.py      # after setting phases_enabled in configs/experiments.yaml
python select_runs.py && python aggregate.py && python render.py
```
