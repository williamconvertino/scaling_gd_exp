"""Normalize run records and evaluation outputs into long-format tables (results/normalized/*.csv + .parquet).

  python aggregate.py

  runs           one row per (non-pilot) run: experiments.md §9 schema + selection + test loss
  val_curves     validation loss vs tokens for every run (last record per step)
  benchmarks     seven-task results of selected runs (long format, percent)
  test_gaps      paired test-loss differences L(model) - L(Wang) within each scale, paired document bootstrap
Nothing is filled in for missing runs; raw files are never modified. Comparisons across tokenizers are
rejected (gaps are only computed within a scale, whose three families share tokenizer and test split).
"""

import json

import numpy as np
import pandas as pd

from common import RESULTS, RUNS, read_json
from eval_loss import read_jsonl

NORM = RESULTS / "normalized"


def save(df, name):
    NORM.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORM / f"{name}.csv", index=False)
    try:
        df.to_parquet(NORM / f"{name}.parquet", index=False)
    except Exception as e:          # parquet is a convenience copy
        print(f"parquet {name}: {e}")


def selection():
    p = NORM / "selection.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["run_id", "selected"])


def runs_table():
    sel = selection()
    selected = set(sel.loc[sel["selected"] == True, "run_id"]) if len(sel) else set()      # noqa: E712
    rows = []
    for p in sorted(RUNS.glob("*/record.json")):
        d = p.parent
        if read_json(d / "run_config.json").get("pilot"):
            continue
        r = read_json(p)
        tl = d / "eval" / "test_loss.json"
        if tl.exists():
            t = read_json(tl)
            r.update(test_loss_nats_per_token=t["test_loss_nats_per_token"], test_loss_se=t["bootstrap_se"],
                     selected_checkpoint_tokens=t["checkpoint_tokens"], checkpoint_path=str(d / t["checkpoint"]))
        r["selected_run"] = r["run_id"] in selected
        r["checkpoints"] = json.dumps(r.get("checkpoints"))
        if r.get("training_flops") is not None:     # ~1e20 overflows int64 in parquet
            r["training_flops"] = float(r["training_flops"])
        rows.append(r)
    return pd.DataFrame(rows)


def val_curves(runs):
    rows = []
    for _, r in runs.iterrows():
        for v in read_jsonl(RUNS / r["run_id"] / "val.jsonl"):
            rows.append({"run_id": r["run_id"], "scale": r["scale_label"], "family": r["family"],
                         "tokenizer_id": r["tokenizer_id"], "step": v["step"], "tokens": v["tokens"],
                         "val_loss": v["val_loss"], "selected_run": r["selected_run"]})
    return pd.DataFrame(rows)


def benchmarks(runs):
    frames = []
    for _, r in runs[runs["selected_run"]].iterrows():
        p = RUNS / r["run_id"] / "eval" / "benchmarks.csv"
        if p.exists():
            b = pd.read_csv(p)
            b["scale"], b["family"] = r["scale_label"], r["family"]
            frames.append(b)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["run_id", "task", "display", "metric", "accuracy_pct", "stderr_pct", "n", "scale", "family"])
    bad = df[(df["accuracy_pct"] < 0) | (df["accuracy_pct"] > 100)]
    assert bad.empty, f"accuracy outside [0, 100]: {bad}"
    return df


def paired_gap(run_a, run_b, n_boot=2000, seed=0):
    """Mean over test targets of NLL_a - NLL_b, with a paired document-cluster bootstrap (same documents
    resampled for both models). Requires both runs to share the tokenizer/test split (same NLL index)."""
    ta, tb = read_json(RUNS / run_a / "eval" / "test_loss.json"), read_json(RUNS / run_b / "eval" / "test_loss.json")
    if ta["nll_index"] != tb["nll_index"]:
        raise ValueError(f"{run_a} vs {run_b}: different tokenizer/test split, refusing to compare")
    idx = np.load(RESULTS.parent / ta["nll_index"])
    m, doc = idx["mask"], idx["doc"]
    a = np.load(RUNS / run_a / "eval" / "test_nll.npy")
    b = np.load(RUNS / run_b / "eval" / "test_nll.npy")
    diff = (a.astype(np.float64) - b)[m]
    d = doc[m]
    s, c = np.bincount(d, weights=diff), np.bincount(d).astype(np.float64)
    keep = c > 0
    s, c = s[keep], c[keep]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(0, n_boot, 100):
        ix = rng.integers(0, len(s), size=(min(100, n_boot - i), len(s)))
        boots[i:i + len(ix)] = s[ix].sum(1) / c[ix].sum(1)
    return {"delta": float(diff.mean()), "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975)), "se": float(boots.std(ddof=1)), "n_targets": int(m.sum()),
            "n_documents": int(keep.sum())}


def test_gaps(runs):
    rows = []
    sel = runs[runs["selected_run"] & runs.get("test_loss_nats_per_token", pd.Series(dtype=float)).notna()] \
        if "test_loss_nats_per_token" in runs else runs.iloc[0:0]
    for scale, g in sel.groupby("scale_label"):
        w = g[g["family"] == "wang"]
        if w.empty:
            continue                      # no baseline -> gaps unavailable, not zero
        for _, r in g[g["family"] != "wang"].iterrows():
            res = paired_gap(r["run_id"], w.iloc[0]["run_id"])
            rows.append({"scale": scale, "family": r["family"], "run_id": r["run_id"],
                         "baseline_run_id": w.iloc[0]["run_id"], "tokenizer_id": r["tokenizer_id"],
                         "definition": "L(model) - L(Wang), nats/token, test split; negative favors the model",
                         "ci": "95% paired document-cluster bootstrap (2000 resamples)", **res})
    return pd.DataFrame(rows)


def main():
    runs = runs_table()
    if runs.empty:
        print("no runs yet")
        return
    save(runs, "runs")
    save(val_curves(runs), "val_curves")
    save(benchmarks(runs), "benchmarks")
    save(test_gaps(runs), "test_gaps")
    print(f"aggregated {len(runs)} runs -> {NORM}")


if __name__ == "__main__":
    main()
