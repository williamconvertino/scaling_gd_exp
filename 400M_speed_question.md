# 400M GDLM: continue as-is or restart with a larger global batch?

Goal: finish a fixed 100B-token pretraining run ASAP. Currently at ~30.4B tokens (2026-10-26 11:30).
Question: continue as-is, or restart from scratch with a larger global batch?

## Setup

- **Model:** 400M-param decoder LM, custom architecture ("GDLM": 24 transformer-style layers + 24
  "gradient-descent" attention-like layers), d=1024, 16 heads x 64 dim, FFN 2880, vocab 32k, seq len 2048.
  3.13 GFLOP/token for training (6N-style count, excludes norms/softmax).
- **Hardware:** 1 node, 8x H200 (141 GB each, 989 TFLOP/s bf16 dense peak).
  Lab account limited to one 8-GPU job at a time.
- **Parallelism:** plain PyTorch DDP, bf16 autocast, fp32 master weights, torch.compile.
- **Batch:** global 256 seqs x 2048 = 524k tokens/step; micro-batch 16 per GPU, 2 grad-accumulation passes
  (all-reduce once per step). Switching to micro-batch 32 (no accumulation) at the next automatic requeue;
  global batch unchanged.
- **Optimizer:** AdamW lr 3e-4, betas (0.9, 0.95), wd 0.1, clip 1.0, warmup 2B tokens, decay to 10% of peak
  over 190,735 steps (100B tokens).

## Measured

- 8 GPUs, micro-batch 16: 770k tok/s total (96k/GPU), MFU 30.4%, GPU busy ~80%, ~58 GB/GPU memory.
- Same model, 1 GPU, micro-batch 16 (no inter-GPU comm): 125k tok/s, MFU 40%, 49.8 GB peak.
- Related 100M model: micro-batch 16 (1 GPU) 195k tok/s/GPU vs micro-batch 32 (8 GPUs) 193k tok/s/GPU,
  i.e. no visible gain from doubling.

## Remaining work and constraints

- ~70B tokens left, about 25 h at the current rate.
- Global batch 256 matches the 50M/100M runs in the same scaling study. Changing it requires restarting from
  step 0 (data order is indexed by step x batch) and re-tuning the LR.

## Reasoning for continuing as-is

- Total compute is fixed: 100B tokens x 3.13 GFLOP/token ~ 3.1e20 FLOPs at any batch size. A larger batch can
  only raise achieved FLOP/s (MFU), not reduce the work.
- Memory use is not a speed measure. Each GPU already processes 16 x 2048 ~ 33k tokens per pass, which keeps
  the matmuls busy.
- The single-GPU run (no inter-GPU sync) reached 40% MFU, roughly the per-GPU ceiling for this model and code.
  So the most any batching change can recover is 40/30 ~ 1.33x, and in practice only part of that.
- Restarting from scratch only wins with a speedup above 100/70 ~ 1.43x.

| Option | Speed | Time to finish |
|---|---|---|
| Continue current run | 770k tok/s | ~25 h |
| Restart, global batch 512 (realistic +10-15%) | ~870k tok/s | ~32 h |
| Restart, optimistic +30% | ~1.0M tok/s | ~28 h |
