"""Batched greedy decoding and beam search (no KV cache; works for both architectures).

Sequences are left-padded with a separate padding segment, so padding never interacts with the
prompt and RoPE (relative) is unaffected.

Copied from ../tmlr_paper/gdlm/eval/generate.py (+ pad_batch / segments_with_padding from
../tmlr_paper/gdlm/eval/common.py). Changes are glue only: imports, a BOS-id module setting (Llama-3.2 BOS is
128000), and the CLI at the bottom that writes fixed-prompt samples for a run:

  python sample.py --run-id 50M_gdlm_lr1e-3_s0      -> runs/<run_id>/eval/samples.json
"""

import numpy as np
import torch

BOS_ID = 1          # glue: set by main() from the run's tokenizer


def pad_batch(seqs, device, pad_id=0, left=False):
    """List of 1-D int arrays -> (z [B,L], seg [B,L], valid [B,L]). Padding gets its own segment."""
    L = max(len(s) for s in seqs)
    z = np.full((len(seqs), L), pad_id, dtype=np.int64)
    valid = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        if left:
            z[i, L - len(s):] = s
            valid[i, L - len(s):] = True
        else:
            z[i, :len(s)] = s
            valid[i, :len(s)] = True
    z = torch.from_numpy(z).to(device)
    valid = torch.from_numpy(valid).to(device)
    seg = segments_with_padding(z, valid)
    return z, seg, valid


def segments_with_padding(z, valid):
    new = (z == BOS_ID) & valid
    new = new.clone()
    new[:, 0] = True
    seg = torch.cumsum(new.int(), dim=1)
    return torch.where(valid, seg, torch.zeros_like(seg))      # padding = segment 0 (never BOS-started)


@torch.no_grad()
def last_logprobs(model, seqs, device, max_len=2048):
    seqs = [s[-max_len:] for s in seqs]
    z, seg, _ = pad_batch(seqs, device, left=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
        logits = model.logits(z, seg)[:, -1]
    return torch.log_softmax(logits.float(), -1)


@torch.no_grad()
def greedy(model, prompts, device, max_new=256, stop_ids=(2,), stop_fn=None, batch=32):
    """prompts: list of int arrays (already with BOS). Returns list of generated id lists."""
    outs = [None] * len(prompts)
    order = np.argsort([len(p) for p in prompts])
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        cur = [list(prompts[i]) for i in idx]
        gen = [[] for _ in idx]
        done = [False] * len(idx)
        for _ in range(max_new):
            act = [k for k in range(len(idx)) if not done[k]]
            if not act:
                break
            lp = last_logprobs(model, [cur[k] for k in act], device)
            nxt = lp.argmax(-1).tolist()
            for k, t in zip(act, nxt):
                if t in stop_ids:
                    done[k] = True
                    continue
                gen[k].append(t)
                cur[k].append(t)
                if stop_fn is not None and stop_fn(gen[k]):
                    done[k] = True
        for k, i in enumerate(idx):
            outs[i] = gen[k]
    return outs


@torch.no_grad()
def beam_search(model, prompts, device, k=10, max_new=24, stop_ids=(2, 13), batch_questions=16):
    """Returns, per prompt, up to k (generated ids, total logprob) sorted best-first.
    A hypothesis finishes when it emits a stop id (not included in the output)."""
    results = []
    for b in range(0, len(prompts), batch_questions):
        P = prompts[b:b + batch_questions]
        hyps = [[(list(p), [], 0.0, False)] for p in P]          # (seq, gen, score, done)
        for _ in range(max_new):
            flat = [(qi, hi) for qi, hs in enumerate(hyps) for hi, h in enumerate(hs) if not h[3]]
            if not flat:
                break
            lp = last_logprobs(model, [hyps[qi][hi][0] for qi, hi in flat], device)
            top_v, top_i = lp.topk(k, dim=-1)
            cand = [[h for h in hs if h[3]] for hs in hyps]
            for row, (qi, hi) in enumerate(flat):
                seq, gen, sc, _ = hyps[qi][hi]
                for v, t in zip(top_v[row].tolist(), top_i[row].tolist()):
                    if t in stop_ids:
                        cand[qi].append((seq, gen, sc + v, True))
                    else:
                        cand[qi].append((seq + [t], gen + [t], sc + v, False))
            hyps = [sorted(c, key=lambda h: -h[2])[:k] for c in cand]
        for hs in hyps:
            results.append([(h[1], h[2]) for h in sorted(hs, key=lambda h: -h[2])])
    return results


# ----------------------------------------------------------------------------- glue: fixed-prompt samples
PROMPTS = ["The history of the printing press",
           "Photosynthesis is the process by which",
           "In mathematics, a prime number is",
           "The main causes of the French Revolution were",
           "To bake a simple loaf of bread, you need",
           "Water boils at a lower temperature at high altitude because"]


def main():
    import argparse
    import time

    from common import RUNS, TOKENIZERS, read_json, write_json
    from data import get_tokenizer
    from eval_loss import load_model, select_checkpoint
    global BOS_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()
    run_dir = RUNS / a.run_id
    out = run_dir / "eval" / "samples.json"
    if out.exists():
        print(f"{a.run_id}: samples exist, skipping")
        return
    torch.manual_seed(0)
    sel_p = run_dir / "eval" / "selected_checkpoint.json"
    best = read_json(sel_p)["selected"] if sel_p.exists() else select_checkpoint(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, rc = load_model(run_dir, best["name"], device, attn_impl="sdpa", compile=False)
    tok_name = rc["frozen"]["tokenizer"]
    info = TOKENIZERS[tok_name]
    BOS_ID = info["bos"]
    tok = get_tokenizer(tok_name)
    prompts = [[BOS_ID] + tok(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    gen = greedy(model, prompts, device, max_new=a.max_new, stop_ids=(info["eos"],))
    write_json(out, {"run_id": a.run_id, "checkpoint": best["name"], "checkpoint_tokens": best["tokens"],
                     "decoding": "greedy (sample.greedy), fixed prompts, seed 0", "max_new_tokens": a.max_new,
                     "samples": [{"prompt": p, "continuation": tok.decode(g)} for p, g in zip(PROMPTS, gen)],
                     "time": time.ctime()})
    print(f"{a.run_id}: wrote {out}")


if __name__ == "__main__":
    main()
