# MIGRATION_REPORT.md: audit of `../tmlr_paper` (2026-10-20)

This is the Phase 0 audit required by `DETAILS.md` §2.1. Everything here was checked against the files
on disk on 2026-10-20. Items that need a user decision are marked **[ASK]** and also listed at the top of
`STATUS.md`.

## 1. Pretokenized FineWeb-Edu (Llama-2)

| Item | Finding |
|---|---|
| Location (before move) | `../tmlr_paper/outputs/data/fineweb_edu_llama2/` (packed splits), `../tmlr_paper/outputs/data/tok/` (per-parquet tokenized shards, 140 × `{bin,lens.npy,done}`) |
| Location (after move) | `data/llama2/` and `data/llama2_tok/` in this repo. Same filesystem (`/work`, NFS4 vast), so the move was a rename and no data was copied. |
| Raw text source | `HuggingFaceFW/fineweb-edu`, **`sample/100BT`** (140 parquet files, 97.27M docs). Local copy owned by another project: `/work/wac20/ctrl_tokens/text_conditioned/data/fineweb_edu_raw/sample/100BT`. Symlinked as `data/raw/fineweb_edu` and **not moved**, because it belongs to `ctrl_tokens`. |
| Tokenizer | `hf-internal-testing/llama-tokenizer`. It is ungated and claimed to be identical to Llama-2 (vocab 32,000, BOS=1, EOS=2, UNK=0). The check against the gated `meta-llama/Llama-2-7b-hf` could not run because the token lacks access (`tokenizer_check.json`: `"reference": "unavailable: OSError"`). |
| dtype | `uint16` flat token files, plus `<split>.docs.npy` (`int64` document start offsets, length `n_docs+1`) |
| BOS/EOS policy | Each document is `BOS(1) + tokens`, with **no EOS**. Special-token ids 0/1/2 inside a document are stripped (`split_special_tokens=True`). |
| Packing | Documents are concatenated in shuffled order. Window `i` of a split is `tokens[i*2048 : i*2048+2049]` (inputs = first 2048, targets = shifted). Document boundaries are kept with **document-masked attention**: segments start at every BOS and at window start. The loss mask drops (a) window position 0, (b) positions whose input is BOS, and (c) targets that are BOS. |
| Split procedure | **Document-level.** Global doc index = (sorted parquet stem, row order). Permutation with `np.random.default_rng(1234)`. In shuffled order: `val_small` (first 5.0M tokens), `val` (next 50.0M), `analysis` (next 20.0M), `train` = rest. Code: `gdlm/data/build_splits.py`. |
| Split sizes | `train` 97,206,647 docs / **114,782,422,963 tokens**. `val_small` 4,353 docs / 5,003,098. `val` 42,704 docs / 50,000,667. `analysis` 16,982 docs / 20,000,834. |
| ≥ 100B training tokens? | **Yes**: 114.78B ≥ 100B + one validation pass. 100B tokens = 190,735 steps × 524,288 tokens = 99.9999B, i.e. 48.83M windows of 2048, out of 56.05M available. |
| Held-out identity | **Document-based**, so it can be reconstructed exactly. The `id` column (`<urn:uuid:…>`) of the raw parquet gives stable document ids. Exported to `data/manifests/heldout_docs.json` (§4.2). |

**Old held-out roles vs. the new spec.** The old code used `val_small` for validation/early stopping and
`val` as the test split. `DETAILS.md` §4.1 says "formerly 10,000 validation and 25,000 test sequences".
`analysis` holds 9,766 windows of 2048 (≈ 10,000) and `val` holds 24,414 (≈ 25,000). The new pipeline uses:

- **validation** = old `analysis` (20.0M tokens, 16,982 docs)
- **test** = old `val` (50.0M tokens, 42,704 docs)
- old `val_small` (5.0M tokens) = reserved, unused. It is still held out and never trained on.

`data/llama2/{validation,test}.bin` are symlinks with clear names that point to the original files.

## 2. Code locations in the old repo

| Component | Old path | New handling |
|---|---|---|
| Raw download / link | `gdlm/data/download.py` | not needed (raw is symlinked) |
| Tokenization | `gdlm/data/tokenize.py` | reused in `data.py` (same `encode_batch` rules), made dtype-aware for Llama-3.2 (`tokenize_llama3.py`) |
| Split building | `gdlm/data/build_splits.py` | split plan reused in `data.py` (`split_plan`, same seed and rule) to export held-out doc ids and drive Llama-3.2 packing |
| Dataloader | `gdlm/data/loader.py` | reused in `data.py` (same windowing/step indexing), dtype-aware |
| Model code | `gdlm/model/{layers,attention,gd,llama,lm}.py` | **rewritten** in `models.py` (deviations from spec, see §6) |
| Training loop | `gdlm/train.py`, `gdlm/dist.py` | DDP / bf16 / atomic-checkpoint / requeue logic reused, loop rewritten per `DETAILS.md` §6 |
| LM-eval wrapper | `gdlm/eval/lm_eval_wrapper.py` | adapted in `eval_benchmarks.py` (7 tasks only, harness pinned at 0.4.9, strict metric keys) |
| TriviaQA | `gdlm/eval/triviaqa.py` (the lab protocol from `triviaqa_example/`, top-k branch decoding) | **copied directly** → `triviaqa.py`, glue only. There is also an older EM/F1 pipeline (`gdlm/eval/finetune.py` + `downstream.py`) matching `experiments.md` §7; it was not copied (see §7) |
| Sampling | `gdlm/eval/generate.py` (+ `pad_batch` from `eval/common.py`) | **copied directly** → `sample.py`, glue only |
| Recurrence analysis | `gdlm/eval/token_buckets.py` | rewritten (`eval_recurrence.py`) to use the saved per-token NLL dumps |

## 3. SLURM and environment

| Item | Value (from old scripts, verified with `sacctmgr`/`sinfo`) |
|---|---|
| 8×H200 training | `-p h200-hp -A henaolab_h200 --gres=gpu:h200:8 --cpus-per-task=32 --mem=256G`. QoS `henaolab_h200`. Max 8 GPUs across all of our jobs. Partition limit 45 days. |
| **Lab quota** | QoS `henaolab_h200` has a weekly `GrpTRESMins` billing limit of **48,336 GPU-min (≈ 806 GPU-h)**, **shared by the whole lab**. On 2026-10-20 the lab had used 24,813, leaving 23,523 GPU-min (392 GPU-h). Slurm starts a job only if its *time limit × GPUs* fits in the free quota. Old jobs held in `QOSGrpBillingMinutes` until the weekly reset. The old launcher ran 8-GPU jobs in 8 h segments for this reason. |
| 1×H200 eval (scavenge) | `-p scavenger-h200 -A scavenger-h200 --gres=gpu:h200:1 --cpus-per-task=8 --mem=80G`. Max 2 GPUs per user at once. **Max 24 h per job.** Preemptible. |
| CPU | `-p scavenger` (no account), 30-day limit. Old tokenizer: 32-task array, 8 CPUs, 64 GB, 24 h, `--requeue`. |
| `--requeue` | used everywhere, with `--open-mode=append`. Training checkpoints 12 min before the limit, writes `REQUEUE`, and the job script calls `scontrol requeue`. |
| Conda env | **Not** `llm-training` under `/work/wac20/miniconda3` (that `envs/` dir is empty). The old repo uses `/hpc/home/wac20/envs/gdlm` (torch 2.14.1+cu130, transformers 5.18, lm-eval **0.4.13**). It lives in `$HOME` because `/work` is purged of files with old timestamps, which broke an env in `/work` on 2026-10-15. Activation: `source /work/wac20/miniconda3/etc/profile.d/conda.sh; conda activate <env>`. |
| New env | `/hpc/home/wac20/envs/scaling_gd`: a venv with `--system-site-packages` on top of `gdlm`, and `lm_eval==0.4.9` installed `--no-deps` so the harness version matches the spec. |
| Env vars | `HF_HOME=/work/wac20/.cache/huggingface` in the old jobs (it holds the benchmark datasets, the Llama-2 tokenizer, and the Llama-3.2-3B tokenizer). The login-shell profile sets `HF_HOME=/work/wac20/.huggingface`, which holds the HF token. `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. W&B was optional (`WANDB_DISABLED`). No `.env` file exists. |
| HF token | `/work/wac20/.huggingface/token` exists, but it **does not have access to `meta-llama/Llama-3.2-1B`** (HTTP 403, gated) or Llama-2. **[ASK]** |

## 4. Historical relaxed-model assets (paper §6)

`figures/GDLM+.png`, `tables/0shot_pretraining.tex`, `tables/5shot_pretraining.tex` and their raw sources
are **not on disk**. I searched `../tmlr_paper` (including `archive/`) and the other project dirs under
`/work/wac20` and `/hpc/home/wac20` (depth ≤ 7). The old runs `gdaug-*` are 10k-step FineWeb screens, not
the SlimPajama 1.1B/1.3B runs. `paper_assets/historical/` is created but empty. **[ASK]** where these
files live.

## 5. Learning-rate history (`DETAILS.md` §6.2)

The old repo has stable GD runs at comparable sizes with **higher** peak LRs than the new defaults:

- 10k-step pilot at ~100M (`configs/decisions.yaml`), batch **128 × 2048**:
  - GDLM: `2e-3` best (3.019 vs 3.033 at `1e-3`). `4e-3` unstable at every size.
  - Transformer: `2e-3` best (2.785 vs 2.807 at `1e-3`).
- Long runs, all stable so far:
  - `gd_M_100B_lr2e-3` (107M): 13.8B tokens, val 2.943.
  - `gd_L_100B_lr1.4e-3` (347M, batch 256): 35B tokens, val 2.553.
  - `gdpos_*` at `2e-3`/`1.4e-3`: stable.

These used a different GD architecture (learned sink key, learned logit temperature, gained covariate
norms; see §6) and, at S/M, half the batch size. Per `DETAILS.md` §6.2 the defaults (`1e-3` / `6e-4` /
`3e-4`) are kept. **[ASK]** whether to raise them toward the old optima.

## 6. Risks found in the old code

1. **Learned sink key** in every GD head (`cfg.sink=True`): an extra learned key with zero value joins
   every kernel normalization. The kernel is then not normalized over `i < t` only, which conflicts with
   eq. (alpha). Not migrated.
2. **Learned logit temperature** `exp(s)` after the gainless RMS norm. This is a learned (scalar) gain,
   which conflicts with "no learned gain". Not migrated.
3. **Per-GD-layer gained RMSNorm** on the covariates (`pred_norms`). That gives each GD layer its own
   normalized covariates. The new GDLM uses one final generator norm, so one `phi` is shared by all GD
   layers.
4. **Readout used the uncentered table.** Only the *values* were mean-centered; the readout used `E`
   (softmax-equivalent, but it violates the stated zero-mean readout). The new model uses the centered
   table everywhere.
5. **Wang control** used **per-layer** positional tables. The new one uses one positional-covariate table
   shared by all GD layers, which mirrors GDLM's shared `phi`.
6. Hard-coded paths: `/work/wac20/tmlr_paper` in `slurm/env.sh` and the templates. A conda env in `/work`
   is at risk of purge.
7. **Data source is `sample-100BT`, not `sample-350BT`** as `experiments.md`/`main.tex` state. This is a
   paper-text discrepancy only. The data is used as-is (`DETAILS.md` §4.1, no re-tokenization). **[ASK]**
   whether the paper text should say 100BT.
8. **Llama-3.2 token budget.** On a 3,000-document sample, Llama-3.2 gives 0.849× the Llama-2 token
   count, so `sample-100BT` yields an estimated **~97.4B** Llama-3.2 tokens in total. That is below the
   ≥ 100B target for the future 1.2B runs. More raw documents (for example from `sample-350BT`, excluding
   the existing ones) would be needed. **[ASK]**
9. No off-by-one was found in the old causal masks (shifted slots were correct). No untied heads in the
   strict model (`tied_lowrank` with `W_o = W_v^T`). Single tokenizer per dataset (no mixing).
10. `lm-eval` 0.4.13 in the old env vs the spec's pinned 0.4.9. This is fixed in the new venv.

## 7. Not migrated (left in place)

Old runs (`outputs/runs`, 156 GB), smoke outputs, logs, bucket features, the allocation, augmentation and
multi-step screens, the analysis/figure code, `triviaqa_example/` (the lab's original TriviaQA example,
already integrated into `gdlm/eval/triviaqa.py`), and the EM/F1 TriviaQA pipeline
(`gdlm/eval/finetune.py`, `downstream.py`). The old repo itself is untouched apart from the data moves
(see `data/manifests/MOVES.md`).

## 8. GDLM vs. the paper: verified line by line (2026-10-20)

`models.py` was checked against main.tex §3–4 and App. B. Each row is also covered by a test in
`tests/test_models.py`, run in fp64/fp32 on CPU and with flex + bf16 on H200.

| Paper | Implementation | Test |
|---|---|---|
| eq. (covariates)/(layer-covariates): `φ_i = h^{(L)}_{i-1}` from a causal Llama generator, one φ shared by all GD layers | `GDModel.covariates`: generator blocks, then `final_norm`. φ_k pairs with label `y_k = z_{k+1}`, which is the paper's `i = k+1`. The same φ feeds every GD layer/head. | `test_gd_matches_direct_prop1`, `test_gdlm_kernel_weights_depend_on_content` |
| eq. (prop1): `f_t = Σ_{l,h} P_lh Σ_{i<t} α_ti e_{y_i}`, single step, blocks summed and not fed back | `GDPredictor.forward` sums all (l, h) outputs. No output is ever an input to a kernel. | direct loop implementation of eq. (prop1), fp64 rel. err < 1e-10, bf16 flex < 3e-2 |
| eq. (preconditioner): `P = W^{V⊤} W^V`, `W^O = W^{V⊤}` tied | Value map `W_v`, output map `W_v.t()` (same tensor) | `test_tied_weights_are_shared_parameters`, `test_preconditioner_is_psd` |
| eq. (kernel): `exp(⟨W φ_i, W φ_t⟩/√d_h)`, tied Q = K, RoPE on projected covariates | `u = RoPE(W_qk φ)`; `attend(u, u, v)` with scale `1/√d_h` | direct implementation incl. RoPE |
| eq. (alpha): normalized over examples `i < t` only, `η = n_t/Z` | Strict mask `k < j` (same document). **No sink.** Rows with no examples give f = 0. | `test_prediction_at_t_independent_of_its_target`, `test_future_tokens…`, `test_first_position_is_uniform…` |
| Zero-mean embeddings: subtract the vocab mean every forward pass | `centered_embedding()` used for generator input, labels and readout | `test_zero_mean_embedding_and_gainless_readout` |
| App. C output normalization: scale-only RMS, no learned gain, then tied readout `softmax(E^⊤ f̂)` | `readout`: `f / sqrt(mean f² + ε)`, `ε = 1e-20` (guards only exactly-zero rows), `@ E_bar^T`. **No temperature / logit scale.** | scale invariance `readout(c f) = readout(f)`; no extra `[V, d]` tensor |
| §4 multi-layer: L GD layers × H heads over the same covariates | `gd_layers = n_layers` of the generator, `gd_heads = d/64`, `d_h = 64` | parameter registry |

**Implicit choice (documented, not a deviation):** the generator ends with a gained RMSNorm (standard Llama
final norm), so φ = RMSNorm(h^{(L)}). The gain is a coordinate scaling that every `W_lh` can absorb. It changes
neither the kernel family nor the PSD/exactness argument.

**Differences from the old codebase (all toward the paper):** the learned sink key, the learned logit
temperature `exp(s)` and the per-GD-layer covariate norms were removed, and the readout now uses the centered
table (§6).
