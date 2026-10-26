"""Held-out test loss + per-token NLL dump for the selected checkpoint of a run (DETAILS.md §7.1).

  python eval_loss.py --run-id 50M_gdlm_lr1e-3_s0

Checkpoint selection (DETAILS.md §6.5): lowest validation loss among the saved interval/final checkpoints
(never test loss). Writes into runs/<run_id>/eval/:
  selected_checkpoint.json   chosen checkpoint, its step/tokens/validation loss, and all alternatives
  test_nll.npy               float32 [n_windows, 2048] per-token NLL of the test split (0 where masked)
  test_loss.json             mean NLL (nats/token) over masked targets + document-cluster bootstrap SE
and once per tokenizer results/raw/nll_index_<tok>_test.npz: window starts, loss mask, document index and
position-in-document of every target (identical for all runs of a tokenizer; used for paired analyses).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import RESULTS, RUNS, SEQ_LEN, TOKENIZERS, read_json, write_json
from data import eval_batch, eval_windows
from models import ModelConfig, build_model, compile_model


def read_jsonl(path):
    """Last record per step (see train.read_jsonl_upto)."""
    by = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            by[r.get("step", len(by))] = r
    return [by[k] for k in sorted(by)]


def select_checkpoint(run_dir):
    """Saved checkpoint with the lowest validation loss at its step."""
    run_dir = Path(run_dir)
    vals = {v["step"]: v for v in read_jsonl(run_dir / "val.jsonl")}
    cks = [c for c in read_jsonl(run_dir / "checkpoints.jsonl") if (run_dir / c["name"]).exists()]
    cands = [{**c, "val_loss": vals[c["step"]]["val_loss"]} for c in cks if c["step"] in vals]
    if not cands:
        raise SystemExit(f"{run_dir.name}: no saved checkpoint with a validation loss")
    best = min(cands, key=lambda c: c["val_loss"])
    sel = {"run_id": run_dir.name, "selected": best, "alternatives": cands,
           "rule": "lowest validation loss among saved interval/final checkpoints", "time": time.ctime()}
    write_json(run_dir / "eval" / "selected_checkpoint.json", sel)
    return best


def load_model(run_dir, ckpt_name, device, attn_impl="flex", compile=True):
    """Model of a run at a checkpoint (glue shared by the evaluation scripts)."""
    run_dir = Path(run_dir)
    rc = read_json(run_dir / "run_config.json")
    cfg = ModelConfig.from_dict({**rc["frozen"]["arch"], "attn_impl": attn_impl})
    model = build_model(cfg)
    sd = torch.load(run_dir / ckpt_name, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.to(device).eval()
    if compile and attn_impl == "flex" and device.type == "cuda":
        compile_model(model)
    return model, rc


def nll_index(tok, split="test"):
    """Model-independent index of the per-token dump: loss mask, document id and position of each target."""
    out = RESULTS / "raw" / f"nll_index_{tok}_{split}.npz"
    if out.exists():
        return out
    info = TOKENIZERS[tok]
    f, starts = eval_windows(info["data_dir"], split)
    n = len(f)
    tgt = starts[:, None] + 1 + np.arange(SEQ_LEN)[None]           # absolute offset of each target token
    valid = tgt < n
    ds = f.doc_starts
    doc = np.searchsorted(ds, np.minimum(tgt, n - 1), side="right") - 1
    pos = np.minimum(tgt, n - 1) - ds[doc]
    mask = np.zeros_like(valid)
    for i in range(0, len(starts), 4096):                           # same mask as training (data.loss_mask_for)
        _, _, _, m = eval_batch(f, starts[i:i + 4096], torch.device("cpu"), info["bos"])
        mask[i:i + 4096] = m.numpy()
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, starts=starts, mask=mask, doc=doc.astype(np.int32), pos=pos.astype(np.int32),
             n_tokens=n, split=split, tokenizer=tok)
    tmp.rename(out)
    return out


def cluster_bootstrap_se(nll, mask, doc, n_boot=1000, seed=0):
    """SE of the token-mean NLL, resampling documents (cluster bootstrap)."""
    d = doc[mask]
    s = np.bincount(d, weights=nll[mask].astype(np.float64))
    c = np.bincount(d).astype(np.float64)
    keep = c > 0
    s, c = s[keep], c[keep]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return float(est.std(ddof=1)), int(len(s))


@torch.no_grad()
def per_token_nll(model, tok, split, device, micro_batch=32):
    info = TOKENIZERS[tok]
    f, starts = eval_windows(info["data_dir"], split)
    out = np.zeros((len(starts), SEQ_LEN), dtype=np.float32)
    for i in range(0, len(starts), micro_batch):
        z, y, seg, m = eval_batch(f, starts[i:i + micro_batch], device, info["bos"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, nll = model(z, seg, y, m, per_token=True)
        out[i:i + len(z)] = (nll * m).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--micro-batch", type=int, default=32)
    a = ap.parse_args()
    run_dir = RUNS / a.run_id
    ev = run_dir / "eval"
    if (ev / "test_loss.json").exists():
        print(f"{a.run_id}: test loss exists, skipping")
        return
    best = select_checkpoint(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, rc = load_model(run_dir, best["name"], device)
    tok = rc["frozen"]["tokenizer"]
    idx = np.load(nll_index(tok))
    t0 = time.time()
    nll = per_token_nll(model, tok, "test", device, a.micro_batch)
    mask = idx["mask"]
    tmp = ev / "test_nll.tmp.npy"
    np.save(tmp, nll)
    tmp.rename(ev / "test_nll.npy")
    loss = float(nll[mask].astype(np.float64).mean())
    se, n_docs = cluster_bootstrap_se(nll, mask, idx["doc"])
    write_json(ev / "test_loss.json", {
        "run_id": a.run_id, "checkpoint": best["name"], "checkpoint_step": best["step"],
        "checkpoint_tokens": best["tokens"], "checkpoint_val_loss": best["val_loss"], "split": "test",
        "test_loss_nats_per_token": loss, "bootstrap_se": se, "bootstrap": "document cluster, 1000 resamples",
        "n_targets": int(mask.sum()), "n_documents": n_docs, "n_windows": int(len(nll)),
        "nll_dump": "eval/test_nll.npy", "nll_index": str(nll_index(tok).relative_to(RESULTS.parent)),
        "seconds": time.time() - t0, "time": time.ctime()})
    print(f"{a.run_id}: test loss {loss:.4f} ± {se:.4f} at {best['name']}")


if __name__ == "__main__":
    main()
