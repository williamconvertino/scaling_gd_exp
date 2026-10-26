"""Sweep selection (DETAILS.md §6.6): for each (scale, family) pick the sweep member with the lowest best
validation loss (never test loss or benchmarks). Writes results/normalized/selection.csv with every
candidate. Only completed runs can be selected; running ones are listed as provisional alternatives.

  python select_runs.py
"""

import csv

from common import RESULTS, RUNS, read_json


def run_records():
    for p in sorted(RUNS.glob("*/record.json")):
        rc = read_json(p.parent / "run_config.json")
        if rc.get("pilot"):
            continue
        yield read_json(p)


def main():
    rows = []
    by = {}
    for r in run_records():
        by.setdefault((r["scale_label"], r["family"]), []).append(r)
    for (scale, fam), rs in sorted(by.items()):
        done = [r for r in rs if r["status"] == "completed" and r["best_valid_loss_nats_per_token"] is not None]
        sel = min(done, key=lambda r: r["best_valid_loss_nats_per_token"]) if done else None
        for r in rs:
            rows.append({"scale": scale, "family": fam, "run_id": r["run_id"], "lr": r["lr"], "seed": r["seed"],
                         "status": r["status"], "best_valid_loss": r["best_valid_loss_nats_per_token"],
                         "best_valid_tokens": r["best_valid_tokens"], "training_tokens": r["training_tokens"],
                         "selected": sel is not None and r["run_id"] == sel["run_id"],
                         "rule": "lowest best validation loss among completed sweep members"})
    out = RESULTS / "normalized" / "selection.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scale", "family", "run_id", "lr", "seed", "status", "best_valid_loss", "best_valid_tokens",
            "training_tokens", "selected", "rule"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"selection: {sum(r['selected'] for r in rows)} selected of {len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
