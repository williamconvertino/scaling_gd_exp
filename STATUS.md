# STATUS.md: live run matrix

_Last updated: 2026-10-26 12:15. Phase 1 done: all four 50M/100M GDLM + Wang runs trained to 100B and evaluated (GDLM − Wang = −2.72 nats at 50M, ≈ −2.88 at 100M). `henaolab_h200` usage was reset after hitting its hard cap (58,752 billing-min, NoDecay) on 2026-10-23; cap unchanged. No Transformer comparison for now. Queue (train jobs only, no dependencies, no eval jobs): 400M GDLM running at micro-batch 32 as job 60013716 (26 h limit, swapped in at 13:25 from 59246994 so no further requeue; ~900k tok/s, up from ~770k at mb 16; ETA ~2026-10-27 09:15); 400M Wang and 1.2B GDLM (now mb 8) pending on the one-8-GPU-job association limit. Evals must be submitted by hand: `python launch.py --eval RUN_ID`. Time limits set to expected runtime + buffer (association MaxWall is 7 days): 400M Wang 4 days, 1.2B GDLM 4.5 days; 400M GDLM 26 h._

## Decisions (user, 2026-10-20)

1. **HF token:** loaded from `.env` (chmod 600) by `slurm/env.sh`. Gated `meta-llama/Llama-3.2-1B` access works.
   Its tokenizer is byte-identical to the cached 3.2-3B tokenizer (serialized sha256 prefix `b47fcb70…`).
2. **Llama-3.2 budget:** ~97.4B training tokens from `sample-100BT` is accepted.
3. **Paper text** (100BT vs 350BT): handled by the user.
4. **LRs:** the `DETAILS.md` defaults are kept.
5. **Historical GDLM+ assets:** these live in Overleaf; nothing to migrate.
6. **Lab quota:** not a constraint for now. The user will help if it runs out.

7. **1.2B GDLM (2026-10-21):** train it, GDLM only, queued behind everything else. Budget = all Llama-3.2 train windows: 185,766 steps = 97.39B tokens (LR schedule over that budget). Wang/Transformer 1.2B remain untrained.

8. **Scope cut (2026-10-25/26):** no Transformer comparison for now; Transformer train and eval jobs cancelled (50M Transformer keeps `resume.pt` = ckpt_050B, so it can resume later). 400M Wang and 1.2B GDLM queued without a dependency so they can be pushed through individually.

Open: the remaining cap (~890 GPU-h on 2026-10-26) is below the ~1,300 GPU-h the three runs need.

## Phase 0 checklist

| Step | Status |
|---|---|
| 1. Audit → `MIGRATION_REPORT.md` | done |
| 2. Move Llama-2 data, `data/manifests/llama2.json`, ≥ 100B tokens | done. 114.78B train tokens. Rename verified by inode/size/mtime; SHA-256 recorded. Held-out doc ids in `heldout_docs.json` (plan reproduces the split files exactly). |
| 3. Codebase | done (see `README.md`) |
| 4. 12 configs + matching report | done. `results/param_matching.md`: all within ±0.8% |
| 5. Tests + GPU pilots | CPU tests 47/47 pass, GPU flex/bf16 tests pass. All 9 pilots done; auto-resume verified (step 200→220); eval chain smoke-tested end to end (test loss + NLL dump, 7 benchmarks, samples, TriviaQA functions). **Gate passed.** |
| 6. `experiments.yaml` + `STATUS.md` | done; `phases_enabled: [1]` |
| 0b. Llama-3.2 tokenization | 140/140 files tokenized (doc counts match Llama-2). Packed (16/16); finalize checks passed (DONE: SHORT 97.40B, accepted); `data/manifests/llama3.json` written (manifest_sha256 `2db67bb8…`). **Phase 0b done.**. Plan: same held-out docs (4,353 / 42,704 / 16,982), train 97.40B tokens (accepted). |

## Pilots (1×H200, global batch 256, measured)

| Config | micro-batch | tok/s per GPU | MFU | peak mem (GB) | est. 100B on 8 GPUs |
|---|---|---|---|---|---|
| 50M gdlm | 32 | 505k | 24% | 37.6 | ~6.9 h |
| 50M wang | 32 | 400k | 32% | 36.6 | ~8.7 h |
| 50M transformer | 32 | 688k | 26% | 36.5 | ~5.0 h |
| 100M gdlm | 32 | 342k | 31% | 45.3 | ~10 h |
| 100M wang | 16 | 195k | 29% | 23.1 | ~18 h (now mb 32) |
| 100M transformer | 32 | 462k | 34% | 43.7 | ~7.5 h |
| 400M gdlm | 16 | 125k | 40% | 49.8 | ~28 h |
| 400M wang | 8 | 61k | 35% | 68.7 | ~57 h (feasible) |
| 400M transformer | 16 | 154k | 42% | 48.1 | ~23 h |
| 1.2B gdlm (Llama-3.2) | 4 | 48.8k | 43% | 82.9 | ~80 h for 97.4B (≈7 segments) |

1.2B GDLM pilot note: step-0 val loss is 15.6, above uniform (ln 128256 = 11.76); smaller GDLMs start at 9.4–9.7, below uniform. The data decodes correctly; the likely cause is the gainless readout, whose largest logit at init grows ∝ 0.02·d (≈41 at d = 2048), so wrong context-copying predictions start very confident. Loss falls to 12.1 by step 60 (LR still ~3e-6). The 600-step pilot (59247614) shows normal descent: 10.19 at step 100, 7.82 at step 200, 5.67 at step 660 (400M GDLM: 8.29 at step 100). Resolved; no change.

## Run matrix

| Scale | Family | Run id | Status | Jobs (train / eval) | Tokens | Best val loss | Selected ckpt | Eval |
|---|---|---|---|---|---|---|---|---|
| 50M | gdlm | 50M_gdlm_lr1e-3_s0 | done | 59069673 / 59069674 | 100.0B | 2.821 (100B) | ckpt_100B | test 2.826 ± 0.004; 7-bench mean 39.5 |
| 50M | wang | 50M_wang_lr1e-3_s0 | done | 59069675 / 59069676 | 100.0B | 5.542 (100B) | ckpt_100B | test 5.549 ± 0.003; 7-bench mean 36.2; GDLM − Wang = −2.723 [−2.727, −2.719] |
| 100M | gdlm | 100M_gdlm_lr6e-4_s0 | done | 59069677 / 59069678 | 100.0B | 2.639 (100B) | ckpt_100B | test 2.644 ± 0.004; 7-bench mean 42.5 |
| 100M | wang | 100M_wang_lr6e-4_s0 | done | 59069679 / 59069682 | 100.0B | 5.514 (100B) | ckpt_100B | test 5.521 ± 0.003; 7-bench mean 36.8; GDLM − Wang ≈ −2.877 (paired CI not yet computed) |
| 50M | transformer | 50M_transformer_lr1e-3_s0 | cancelled (scope cut); `resume.pt` = ckpt_050B | 59246990 (killed by quota), 59882903 / 59882904 (cancelled) | 50.0B | 2.750 (50B) | | |
| 100M | transformer | 100M_transformer_lr6e-4_s0 | cancelled (scope cut) | 59246992 / 59246993 | | | | |
| 400M | gdlm | 400M_gdlm_lr3e-4_s0 | training (mb 32; job 60013716, 26 h limit) | 60013716 (earlier 59246994) / (eval not queued) | 35.9B | 2.477 (29B) | | |
| 400M | wang | 400M_wang_lr3e-4_s0 | queued (no dependency) | 59246996 / (eval not queued) | | | | |
| 400M | transformer | 400M_transformer_lr3e-4_s0 | cancelled (scope cut) | 59246998 / 59246999 | | | | |
| 1.2B | gdlm | 1.2B_gdlm_lr2e-4_s0 | queued (no dependency; mb 8; 97.39B-token budget) | 59247000 / (eval not queued) | | | | |
| 1.2B | wang, transformer | n/a | configs only (not trained) | | | | | |
