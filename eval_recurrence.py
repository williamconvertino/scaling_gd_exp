"""In-context recurrence analysis (Figure 3, experiments.md §6) from saved per-token test NLL dumps.

  python eval_recurrence.py --scale 100M

Features of every test target y_j = z_{j+1} (model-independent, cached once per tokenizer in
results/raw/recurrence_features_<tok>_test.npz). The visible context is z_s..z_j: the same window and
document as the target (attention never crosses documents or windows):
  prior_count      occurrences of the target token id in the visible context: 0 / 1 / 2+
  last_distance    (j + 1) - k for the latest such occurrence at position k; bins 1-16, 17-64, 65-256,
                   257-1024, >1024 (only for recurring targets)
  bigram_seen      (z_j, y_j) occurred as (z_{k-1}, z_k) with k <= j in the visible context
  trigram_seen     (z_{j-1}, z_j, y_j) occurred as (z_{k-2}, z_{k-1}, z_k) with k <= j
Per pair of selected runs at the scale: paired per-token NLL differences L(a) - L(b), averaged within
each bin, with 95% document-cluster bootstrap CIs (the same resampled documents for both models).
Writes results/normalized/recurrence_<scale>.csv. Bins with < 1000 targets or < 50 documents are flagged
`supported=False` and are not plotted.
"""

import argparse

import numpy as np
import pandas as pd

from common import RESULTS, RUNS, SEQ_LEN, TOKENIZERS, read_json
from data import eval_windows, segments_from_bos

DIST_BINS = [(1, 16, "1–16"), (17, 64, "17–64"), (65, 256, "65–256"), (257, 1024, "257–1024"),
             (1025, 10 ** 9, ">1024")]
MIN_TARGETS, MIN_DOCS = 1000, 50


def _windows(tok):
    import torch
    info = TOKENIZERS[tok]
    f, starts = eval_windows(info["data_dir"], "test")
    n = len(f)
    z = np.zeros((len(starts), SEQ_LEN + 1), dtype=np.int64)
    for i, s in enumerate(starts):
        e = min(int(s) + SEQ_LEN + 1, n)
        z[i, :e - s] = f.tokens[s:e]
    seg = segments_from_bos(torch.from_numpy(z[:, :-1]), info["bos"]).numpy()
    return z, seg


def _seen_before(keys_ctx, keys_q, pos_ctx, pos_q, group_ctx, group_q, need_last=False):
    """For each query (group, key, position j): count of context events with the same group/key and
    position <= j, and (optionally) the latest such position. Vectorized with one lexsort."""
    nC, nQ = len(keys_ctx), len(keys_q)
    g = np.concatenate([group_ctx, group_q])
    k = np.concatenate([keys_ctx, keys_q])
    p = np.concatenate([pos_ctx, pos_q])
    typ = np.concatenate([np.zeros(nC, np.int8), np.ones(nQ, np.int8)])     # context before query at equal pos
    order = np.lexsort((typ, p, k, g))
    g, k, p, typ = g[order], k[order], p[order], typ[order]
    new_grp = np.ones(len(g), bool)
    new_grp[1:] = (g[1:] != g[:-1]) | (k[1:] != k[:-1])
    is_ctx = (typ == 0).astype(np.int64)
    cum = np.cumsum(is_ctx)
    start_idx = np.maximum.accumulate(np.where(new_grp, np.arange(len(g)), 0))
    base = np.where(start_idx > 0, cum[start_idx - 1], 0)
    count = cum - base
    out_count = np.empty(nQ, np.int64)
    qmask = typ == 1
    qidx = order[qmask] - nC
    out_count[qidx] = count[qmask]
    if not need_last:
        return out_count, None
    lastpos = np.where(is_ctx == 1, p, -1)
    # running max of context positions within the group (positions increase within a group)
    seg_id = np.cumsum(new_grp)
    lp = pd.Series(lastpos).groupby(seg_id).cummax().to_numpy()
    out_last = np.empty(nQ, np.int64)
    out_last[qidx] = lp[qmask]
    return out_count, out_last


def features(tok):
    out = RESULTS / "raw" / f"recurrence_features_{tok}_test.npz"
    if out.exists():
        return dict(np.load(out))
    z, seg = _windows(tok)
    res = compute_features(z, seg, TOKENIZERS[tok]["vocab"])
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, **res)
    tmp.rename(out)
    return res


def compute_features(z, seg, vocab):
    """z: [W, T+1] window tokens, seg: [W, T] segment ids of the inputs. Returns per-target features [W, T]."""
    W, T = seg.shape
    x, y = z[:, :-1], z[:, 1:]
    j = np.broadcast_to(np.arange(T), (W, T))
    grp = (np.arange(W)[:, None] * (T + 1) + seg).astype(np.int64)          # (window, document) group
    flat = lambda a: np.ascontiguousarray(a).reshape(-1)
    # unigram: context tokens z_k (k <= j) vs target y_j
    cnt, last = _seen_before(flat(x), flat(y), flat(j), flat(j), flat(grp), flat(grp), need_last=True)
    dist = np.where(cnt > 0, flat(j) + 1 - last, 0)
    # bigram: context pairs (z_{k-1}, z_k) ending at k (both in the group) vs (z_j, y_j)
    V = np.int64(vocab)
    same_prev = np.zeros_like(seg, dtype=bool)
    same_prev[:, 1:] = seg[:, 1:] == seg[:, :-1]
    bc = (x[:, :-1] * V + x[:, 1:])
    bc_ok = same_prev[:, 1:]
    bq = x * V + y
    b_cnt, _ = _seen_before(flat(bc)[flat(bc_ok)], flat(bq), flat(j[:, 1:])[flat(bc_ok)], flat(j),
                            flat(grp[:, 1:])[flat(bc_ok)], flat(grp))
    # trigram: (z_{k-2}, z_{k-1}, z_k) vs (z_{j-1}, z_j, y_j); key = bigram key * V + third
    tc = (x[:, :-2] * V + x[:, 1:-1]) * V + x[:, 2:]
    tc_ok = same_prev[:, 1:-1] & same_prev[:, 2:]
    tq = np.zeros_like(x)
    tq[:, 1:] = (x[:, :-1] * V + x[:, 1:]) * V + y[:, 1:]
    tq_ok = same_prev.copy()                                                 # z_{j-1} in the same document
    t_cnt, _ = _seen_before(flat(tc)[flat(tc_ok)], flat(tq), flat(j[:, 2:])[flat(tc_ok)], flat(j),
                            flat(grp[:, 2:])[flat(tc_ok)], flat(grp))
    t_cnt = np.where(flat(tq_ok), t_cnt, 0)
    res = {"prior_count": cnt.reshape(W, T).astype(np.int32), "last_distance": dist.reshape(W, T).astype(np.int32),
           "bigram_seen": (b_cnt > 0).reshape(W, T), "trigram_seen": (t_cnt > 0).reshape(W, T),
           "context_length": (j - np.maximum.accumulate(np.where(~same_prev, j, 0), axis=1) + 1).astype(np.int32)}
    return res


def bins(feat):
    pc = feat["prior_count"]
    yield "prior_count", {"Never": pc == 0, "Once": pc == 1, "2+": pc >= 2}
    d = feat["last_distance"]
    yield "last_distance", {lab: (pc > 0) & (d >= lo) & (d <= hi) for lo, hi, lab in DIST_BINS}
    yield "bigram_seen", {"No": ~feat["bigram_seen"], "Yes": feat["bigram_seen"]}
    yield "trigram_seen", {"No": ~feat["trigram_seen"], "Yes": feat["trigram_seen"]}


def selected_runs(scale):
    p = RESULTS / "normalized" / "selection.csv"
    sel = pd.read_csv(p)
    sel = sel[(sel["scale"] == scale) & (sel["selected"] == True)]                    # noqa: E712
    return {r["family"]: r["run_id"] for _, r in sel.iterrows()
            if (RUNS / r["run_id"] / "eval" / "test_nll.npy").exists()}


def analyze(scale, n_boot=1000, seed=0):
    runs = selected_runs(scale)
    pairs = [(a, b) for a, b in (("gdlm", "transformer"), ("gdlm", "wang"), ("transformer", "wang"))
             if a in runs and b in runs]
    if not pairs:
        print(f"{scale}: no pair of selected runs with NLL dumps")
        return None
    t = read_json(RUNS / runs[pairs[0][0]] / "eval" / "test_loss.json")
    tok = t["nll_index"].split("nll_index_")[1].split("_test")[0]
    for fam in runs:
        assert read_json(RUNS / runs[fam] / "eval" / "test_loss.json")["nll_index"] == t["nll_index"], \
            "runs at one scale must share tokenizer and test split"
    idx = np.load(RESULTS.parent / t["nll_index"])
    mask, doc = idx["mask"], idx["doc"]
    feat = features(tok)
    nll = {f: np.load(RUNS / r / "eval" / "test_nll.npy") for f, r in runs.items()}
    rng = np.random.default_rng(seed)
    udocs = np.unique(doc[mask])
    remap = np.full(doc.max() + 1, -1, np.int64)
    remap[udocs] = np.arange(len(udocs))
    boot_w = rng.poisson(1.0, size=(n_boot, len(udocs))).astype(np.float64)   # Poisson bootstrap over documents
    rows = []
    total = int(mask.sum())
    for a, b in pairs:
        diff = nll[a].astype(np.float64) - nll[b]
        for feature, bmap in bins(feat):
            for lab, sel in bmap.items():
                m = sel & mask
                n = int(m.sum())
                d = remap[doc[m]]
                ndoc = len(np.unique(d))
                s = np.bincount(d, weights=diff[m], minlength=len(udocs))
                c = np.bincount(d, minlength=len(udocs)).astype(np.float64)
                bs = (boot_w @ s) / np.maximum(boot_w @ c, 1)
                rows.append({"scale": scale, "tokenizer": tok, "comparison": f"{a} - {b}", "model_a": a, "model_b": b,
                             "run_a": runs[a], "run_b": runs[b], "feature": feature, "bin": lab, "n_targets": n,
                             "frac_targets": n / total, "n_documents": ndoc,
                             "mean_diff": float(diff[m].mean()) if n else np.nan,
                             "ci_lo": float(np.quantile(bs, 0.025)) if n else np.nan,
                             "ci_hi": float(np.quantile(bs, 0.975)) if n else np.nan,
                             "mean_nll_a": float(nll[a][m].mean()) if n else np.nan,
                             "mean_nll_b": float(nll[b][m].mean()) if n else np.nan,
                             "supported": n >= MIN_TARGETS and ndoc >= MIN_DOCS,
                             "ci": "95% document-cluster Poisson bootstrap, 1000 resamples"})
    df = pd.DataFrame(rows)
    out = RESULTS / "normalized" / f"recurrence_{scale}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="100M")
    ap.add_argument("--features-only", default=None, help="tokenizer: only build the feature cache")
    a = ap.parse_args()
    if a.features_only:
        features(a.features_only)
    else:
        analyze(a.scale)
