"""Closed-book TriviaQA factual recall (Table 8), following the lab's TriviaQA protocol (triviaqa_example/):

prepare   mandarjoshi/trivia_qa rc.nocontext, deduplicated by question_id; the original train split is split
          90/10 (seed 42) into train / validation, and the original validation split (which has answers)
          becomes the test split.  -> outputs/data/eval/triviaqa_v2/{train,validation,test}.jsonl
finetune  for each learning rate of the sweep (1e-5 ... 5e-4): fine-tune all weights of the pretrained best
          checkpoint on "Question: <q>\\nAnswer: <normalised answer><eos>" with the loss on the answer tokens
          only (sequences of <= 64 tokens, batch 8, AdamW wd 0.01, 10% warmup + linear decay over 6 epochs),
          validation loss every 500 steps with early stopping (patience 3) and the best state kept. The
          best state is scored on the validation split (top-1) and on the test split (top-1/5/10).
          The reported learning rate is the one with the best validation top-1 accuracy.
Scoring   (evaluation_utils.py of the protocol): the first answer token branches into the top-k tokens and each
          branch is continued greedily (10 new tokens in total, stop at EOS); a question counts as correct at k
          if the normalised gold answer is a substring of one of the first k normalised predictions.

  python triviaqa.py prepare
  python triviaqa.py finetune --run <run_id> [--lrs 1e-5 3e-5 ...]
Writes runs/<run_id>/triviaqa/lr<lr>.json (one per learning rate) and runs/<run_id>/eval/triviaqa.json
(the selected learning rate, its validation and test metrics, and sample predictions).

Copied directly from ../tmlr_paper/gdlm/eval/triviaqa.py (DETAILS.md §2.2). Glue changes only: imports and
paths, model loading (the run's selected checkpoint, SDPA attention), the run's tokenizer, and the old result
registry call removed. Full runs are "Not now" (EXPERIMENT_ORDER.md); only smoke-tested on a tiny model.
"""

import argparse
import json
import math
import random
import re
import time

import numpy as np
import torch

from common import DATA as DATA_DIR, RUNS as RUNS_DIR, write_json
from sample import pad_batch

BOS_ID = 1      # glue: Llama-2 runs (the only tokenizer trained so far)

EOS_ID = 2
DATA = DATA_DIR / "eval" / "triviaqa_v2"
LRS = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4]
MAX_LEN, BATCH, EPOCHS, EVAL_EVERY, PATIENCE, WARMUP, WD = 64, 8, 6, 500, 3, 0.1, 0.01
TOP_K = (1, 5, 10)
MAX_NEW = 10

# ------------------------------------------------------------------ protocol (triviaqa_example/evaluation_utils.py)
COMMON_WORDS = ["the", "a", "an", "sir", "lady", "lord", "dr", "doctor", "professor", "prof", "mr", "mrs", "ms",
                "miss", "king", "queen", "prince", "princess", "saint", "st", "mount", "mt"]


def normalize_answer(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    if len(words) == 1:
        return " ".join(words)
    kept = [w for w in words if w not in COMMON_WORDS]
    return " ".join(kept or words)


def is_correct(gold, pred):
    return gold in pred


def format_prompt(question, answer=None):
    return f"Question: {question}\nAnswer:" if answer is None else f"Question: {question}\nAnswer: {answer}"


# ------------------------------------------------------------------ data
def prepare():
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    def dedup(split):
        seen, keep = set(), []
        for i, q in enumerate(split["question_id"]):
            if q not in seen:
                seen.add(q)
                keep.append(i)
        return split.select(keep)

    train, test = dedup(ds["train"]), dedup(ds["validation"])
    assert not set(train["question_id"]) & set(test["question_id"])
    tv = train.train_test_split(test_size=0.1, seed=42, shuffle=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", tv["train"]), ("validation", tv["test"]), ("test", test)):
        with open(DATA / f"{name}.jsonl", "w") as f:
            for ex in split:
                f.write(json.dumps({"question_id": ex["question_id"], "question": ex["question"],
                                    "answer": ex["answer"]["value"],
                                    "normalized_answer": normalize_answer(ex["answer"]["value"])}) + "\n")
        print(f"{name}: {len(split)}")


def load_split(name):
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl")]


def encode(rows, tok):
    """(ids, first answer index) per row: [BOS] prompt ' ' normalised answer [EOS], truncated to MAX_LEN tokens."""
    prompts = tok([format_prompt(r["question"]) for r in rows], add_special_tokens=False)["input_ids"]
    fulls = tok([format_prompt(r["question"], r["normalized_answer"]) for r in rows],
                add_special_tokens=False)["input_ids"]
    out = []
    for p, f in zip(prompts, fulls):
        seq = ([BOS_ID] + f + [EOS_ID])[:MAX_LEN]
        out.append((np.asarray(seq, dtype=np.int64), min(len(p) + 1, len(seq))))
    return out


def batch_loss(model, items, device, reduce=True):
    z, seg, _ = pad_batch([s for s, _ in items], device)
    m = torch.zeros(z.shape[0], z.shape[1] - 1, dtype=torch.bool, device=device)
    for i, (s, a0) in enumerate(items):
        m[i, a0 - 1:len(s) - 1] = True          # predict the answer tokens and EOS
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model.logits(z[:, :-1], seg[:, :-1])
    lp = torch.log_softmax(logits.float(), -1).gather(-1, z[:, 1:, None]).squeeze(-1)
    nll = -(lp * m).sum()
    return (nll / m.sum()) if reduce else (nll, m.sum())


@torch.no_grad()
def eval_loss(model, items, device, batch=64):
    model.eval()
    tot, n = 0.0, 0
    for b in range(0, len(items), batch):
        s, c = batch_loss(model, items[b:b + batch], device, reduce=False)
        tot, n = tot + s.item(), n + c.item()
    model.train()
    return tot / n


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def predictions(model, tok, rows, device, k=10, batch=32):
    """Per question, k decoded strings: branch on the top-k first tokens, then greedy (protocol)."""
    from sample import last_logprobs
    model.eval()
    prompts = [[BOS_ID] + p for p in tok([format_prompt(r["question"]) for r in rows],
                                         add_special_tokens=False)["input_ids"]]
    out = []
    for b in range(0, len(prompts), batch):
        P = prompts[b:b + batch]
        first = last_logprobs(model, P, device).topk(k, -1).indices.tolist()
        seqs = [p + [t] for p, fs in zip(P, first) for t in fs]
        gen = [[t] for fs in first for t in fs]
        done = [t == EOS_ID for fs in first for t in fs]
        for _ in range(MAX_NEW - 1):
            act = [i for i, d in enumerate(done) if not d]
            if not act:
                break
            nxt = last_logprobs(model, [seqs[i] for i in act], device).argmax(-1).tolist()
            for i, t in zip(act, nxt):
                seqs[i].append(t)
                gen[i].append(t)
                done[i] = t == EOS_ID
        for q in range(len(P)):
            branch = []
            for g in gen[q * k:(q + 1) * k]:
                g = g[:g.index(EOS_ID)] if EOS_ID in g else g
                branch.append(tok.decode(g, skip_special_tokens=True).strip())
            out.append(branch)
    model.train()
    return out


def score(model, tok, rows, device, top_k=TOP_K):
    preds = predictions(model, tok, rows, device, k=max(top_k))
    ranks = []
    for r, ps in zip(rows, preds):
        norm = [normalize_answer(p) for p in ps]
        ranks.append(next((i + 1 for i, p in enumerate(norm) if is_correct(r["normalized_answer"], p)), None))
    n = len(rows)
    res = {"n": n}
    for k in top_k:
        acc = sum(1 for x in ranks if x is not None and x <= k) / n
        res[f"top_{k}_accuracy"] = acc
        res[f"top_{k}_se"] = math.sqrt(acc * (1 - acc) / n)
    res["samples"] = [{"q": r["question"], "gold": r["normalized_answer"], "preds": p[:3], "rank": x}
                      for r, p, x in list(zip(rows, preds, ranks))[:25]]
    return res


# ------------------------------------------------------------------ fine-tuning
def finetune_lr(run, lr, data, device, tok):
    from eval_loss import load_model, select_checkpoint
    rd = RUNS_DIR / run
    sel = rd / "eval" / "selected_checkpoint.json"
    ckpt = (json.loads(sel.read_text())["selected"] if sel.exists() else select_checkpoint(rd))["name"]
    model, rc = load_model(rd, ckpt, device, attn_impl="sdpa", compile=False)
    model.train()
    torch.manual_seed(42)
    opt = torch.optim.AdamW(model.param_groups(WD), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            fused=device.type == "cuda")
    train, val = data["train_items"], data["val_items"]
    steps_per_epoch = len(train) // BATCH
    total = EPOCHS * steps_per_epoch
    warm = int(WARMUP * total)
    rng = random.Random(42)
    best, since, best_state, best_step, log = float("inf"), 0, None, 0, []
    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        order = list(range(len(train)))
        rng.shuffle(order)
        for b in range(steps_per_epoch):
            f = lr * (step + 1) / warm if step < warm else lr * max(0.0, (total - step) / max(1, total - warm))
            for g in opt.param_groups:
                g["lr"] = f
            loss = batch_loss(model, [train[i] for i in order[b * BATCH:(b + 1) * BATCH]], device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % EVAL_EVERY == 0:
                vl = eval_loss(model, val, device)
                log.append({"step": step, "epoch": ep, "train_loss": loss.item(), "val_loss": vl, "lr": f,
                            "sec": time.time() - t0})
                print(json.dumps(log[-1]), flush=True)
                if vl < best:
                    best, since, best_step = vl, 0, step
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                else:
                    since += 1
                if since >= PATIENCE:
                    break
        if since >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_res = score(model, tok, data["val"], device, top_k=(1,))
    test_res = score(model, tok, data["test"], device)
    del model, opt, best_state
    torch.cuda.empty_cache()
    return {"lr": lr, "ckpt": ckpt, "best_step": best_step, "steps_run": step, "best_val_loss": best,
            "val": val_res, "test": test_res, "log": log, "sec": time.time() - t0}


def finetune(run, lrs):
    from data import get_tokenizer
    device = torch.device("cuda")
    tok = get_tokenizer(json.loads((RUNS_DIR / run / "run_config.json").read_text())["frozen"]["tokenizer"])
    data = {"train": load_split("train"), "val": load_split("validation"), "test": load_split("test")}
    data["train_items"], data["val_items"] = encode(data["train"], tok), encode(data["val"], tok)
    out = RUNS_DIR / run / "triviaqa"
    results = []
    for lr in lrs:
        p = out / f"lr{lr:.0e}.json"
        if not p.exists():
            write_json(p, finetune_lr(run, lr, data, device, tok))
        r = json.loads(p.read_text())
        results.append(r)
        print(f"lr {lr:.0e}: val top-1 {r['val']['top_1_accuracy']:.4f} test top-1 {r['test']['top_1_accuracy']:.4f}",
              flush=True)
    best = max(results, key=lambda r: r["val"]["top_1_accuracy"])
    summary = {"selected_lr": best["lr"], "val": best["val"], "test": best["test"], "best_step": best["best_step"],
               "sweep": [{"lr": r["lr"], "val_top_1": r["val"]["top_1_accuracy"],
                          "test_top_1": r["test"]["top_1_accuracy"], "best_step": r["best_step"]} for r in results],
               "protocol": {"max_len": MAX_LEN, "batch": BATCH, "epochs": EPOCHS, "eval_every": EVAL_EVERY,
                            "patience": PATIENCE, "warmup": WARMUP, "weight_decay": WD, "max_new_tokens": MAX_NEW}}
    write_json(RUNS_DIR / run / "eval" / "triviaqa.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "finetune"])
    ap.add_argument("--run")
    ap.add_argument("--lrs", nargs="+", type=float, default=LRS)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    else:
        finetune(a.run, a.lrs)


if __name__ == "__main__":
    main()
