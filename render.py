"""Figures and tables from results/normalized (figures.md). Missing runs stay visibly missing.

  python render.py

  figures/fig2_pretraining.{pdf,png}        (a) validation curves by scale, (b) paired test-loss gaps vs Wang
  figures/appendix_pretrain_<scale>.{pdf,png}
  figures/fig3_recurrence.{pdf,png}         if results/normalized/recurrence_100M.csv exists
  figures/working_params_vs_performance.{pdf,png}   quick-look working plot (not a paper figure)
  tables/table1_seven_benchmarks.tex
  results/figure_manifests/<name>.json      inputs, run ids, counts, missing entries
Style: seaborn + matplotlib, fixed family palette/markers/line styles, no log axes, no offset notation.
Width: TMLR text width 6.5 in (letter, 1 in margins).
"""

import json
import time

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import seaborn as sns                    # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator   # noqa: E402

from common import FIGURES, RESULTS, SCALES, TABLES, FROZEN, read_json, write_json   # noqa: E402

TEXT_W = 6.5
FAMILIES = ["gdlm", "wang", "transformer"]
STYLE = {
    "gdlm": {"label": "GDLM", "color": "#3B6EA8", "marker": "o", "linestyle": "-"},
    "wang": {"label": "Wang et al.", "color": "#C27B31", "marker": "s", "linestyle": "--"},
    "transformer": {"label": "Transformer", "color": "#3F8B78", "marker": "^", "linestyle": "-."},
}
TOKENIZER_NOTE = {"50M": "Llama-2", "100M": "Llama-2", "400M": "Llama-2", "1.2B": "Llama-3.2"}
NORM = RESULTS / "normalized"
BENCH = [("WinoGrande", "winogrande"), ("PIQA", "piqa"), ("OpenBookQA", "openbookqa"), ("HellaSwag", "hellaswag"),
         ("BoolQ", "boolq"), ("ARC-E", "arc_easy"), ("ARC-C", "arc_challenge")]

sns.set_theme(context="paper", style="ticks", rc={
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "lines.linewidth": 1.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "axes.formatter.useoffset": False})


def load(name):
    p = NORM / f"{name}.csv"
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()


def frozen_totals():
    out = {}
    for p in FROZEN.glob("*.json"):
        r = read_json(p)
        out[(r["scale"], r["family"])] = r["actual_total_params"]
    return out


def fmt_params(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def save(fig, name, manifest):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", pad_inches=0.04, dpi=400)
    plt.close(fig)
    write_json(RESULTS / "figure_manifests" / f"{name}.json", {**manifest, "rendered": time.ctime()})


def padded(lo, hi, frac=0.05):
    s = (hi - lo) or abs(hi) or 1.0
    return lo - frac * s, hi + frac * s


def curves_panel(ax, cur, scale, sel_tokens):
    present = []
    for fam in FAMILIES:
        c = cur[(cur["scale"] == scale) & (cur["family"] == fam) & (cur["selected_run"])].sort_values("tokens")
        if c.empty:
            continue
        assert c["tokens"].is_monotonic_increasing
        st = STYLE[fam]
        c = c[c["tokens"] > 0]                      # step-0 loss (~ln V) would flatten the axis
        ax.plot(c["tokens"] / 1e9, c["val_loss"], color=st["color"], ls=st["linestyle"], lw=1.6, label=st["label"])
        t = sel_tokens.get((scale, fam))
        if t is not None and (c["tokens"] == t).any():
            v = c.loc[c["tokens"] == t, "val_loss"].iloc[0]
            ax.plot([t / 1e9], [v], marker=st["marker"], ms=5, color=st["color"], mec="white", mew=0.8, ls="none")
        present.append(fam)
    return present


def fig2(runs, cur, gaps, totals):
    sel_tokens = {}
    if not runs.empty and "selected_checkpoint_tokens" in runs:
        for _, r in runs[runs["selected_run"] & runs["selected_checkpoint_tokens"].notna()].iterrows():
            sel_tokens[(r["scale_label"], r["family"])] = r["selected_checkpoint_tokens"]
    fig = plt.figure(figsize=(TEXT_W, 3.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.62, wspace=0.38)
    missing = {}
    for i, scale in enumerate(SCALES):
        ax = fig.add_subplot(gs[0, i])
        present = curves_panel(ax, cur, scale, sel_tokens) if not cur.empty else []
        missing[scale] = [f for f in FAMILIES if f not in present]
        ref = totals.get((scale, "transformer"))
        ax.set_title(f"{scale} · {TOKENIZER_NOTE[scale]}", pad=3)
        if not present:
            ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes, color="0.45")
            ax.set_yticks([])
        else:
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlim(0, 100 * 1.04)
        ax.set_xticks([0, 25, 50, 75, 100])
        if i == 0:
            ax.set_ylabel("Validation loss\n(nats/token)")
        ax.grid(axis="y", color="0.92")
        sns.despine(ax=ax)
    fig.text(0.005, 0.97, "(a)", fontweight="bold")
    fig.text(0.5, 0.49, "Training tokens (billions)", ha="center", fontsize=8)
    # (b) paired gaps
    ax = fig.add_subplot(gs[1, 1:3])
    ax.axhline(0, color="0.35", lw=0.8)
    xs = np.arange(len(SCALES))
    lo_hi = [0.0]
    for k, fam in enumerate(["gdlm", "transformer"]):
        g = gaps[gaps["family"] == fam] if not gaps.empty else gaps
        for _, r in g.iterrows():
            x = SCALES.index(r["scale"]) + (-0.09 if fam == "gdlm" else 0.09)
            st = STYLE[fam]
            ax.errorbar([x], [r["delta"]], yerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]],
                        fmt=st["marker"], color=st["color"], ms=5, capsize=2.5, lw=1.2, mec="white", mew=0.6)
            lo_hi += [r["ci_lo"], r["ci_hi"]]
    m = max(abs(min(lo_hi)), abs(max(lo_hi))) or 0.05
    ax.set_ylim(-m * 1.15, m * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xticks(xs)
    ax.set_xticklabels(SCALES)
    ax.set_xlim(-0.5, len(SCALES) - 0.5)
    ax.set_xlabel("Model group (total parameters)")
    ax.set_ylabel("Test loss − Wang et al.\n(nats/token; < 0 better)")
    ax.grid(axis="y", color="0.92")
    sns.despine(ax=ax)
    fig.text(0.24, 0.45, "(b)", fontweight="bold")
    handles = [mpl.lines.Line2D([], [], color=STYLE[f]["color"], ls=STYLE[f]["linestyle"], marker=STYLE[f]["marker"],
                                ms=4, label=STYLE[f]["label"]) for f in FAMILIES]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    save(fig, "fig2_pretraining", {
        "inputs": ["results/normalized/val_curves.csv", "results/normalized/test_gaps.csv", "results/normalized/runs.csv"],
        "runs": sorted(cur[cur["selected_run"]]["run_id"].unique().tolist()) if not cur.empty else [],
        "missing_families_by_scale": missing, "selected_checkpoint_tokens": {f"{k[0]}/{k[1]}": v for k, v in sel_tokens.items()},
        "gap_definition": "test_loss(model) - test_loss(Wang) within scale; 95% paired document bootstrap",
        "tokenizers": TOKENIZER_NOTE, "seeds": 1, "x_axis": "linear, billions of training tokens",
        "note": "step-0 validation point omitted from the curves (near ln V); it is in val_curves.csv"})


def appendix_curves(cur, runs):
    for scale in SCALES:
        fig, ax = plt.subplots(figsize=(TEXT_W / 2, 2.3))
        present = curves_panel(ax, cur, scale, {}) if not cur.empty else []
        if not present:
            ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes, color="0.45")
        ax.set_xlim(0, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Training tokens (billions)")
        ax.set_ylabel("Validation loss (nats/token)")
        ax.set_title(f"{scale} group ({TOKENIZER_NOTE[scale]} tokenizer)")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if present:
            ax.legend(frameon=False)
        sns.despine(ax=ax)
        save(fig, f"appendix_pretrain_{scale.replace('.', 'p')}", {"scale": scale, "families": present})


def table1(bench, totals):
    lines = [r"\begin{tabular}{ll" + "r" * 8 + "}", r"\toprule",
             "Size & Model & " + " & ".join(b for b, _ in BENCH) + r" & Mean \\", r"\midrule"]
    have = {}
    for _, r in bench.iterrows():
        have[(r["scale"], r["family"], r["task"])] = r["accuracy_pct"]
    missing = []
    for si, scale in enumerate(SCALES):
        vals = {f: [have.get((scale, f, t)) for _, t in BENCH] for f in FAMILIES}
        best = [max((v[i] for v in vals.values() if v[i] is not None), default=None) for i in range(len(BENCH))]
        means = {f: (np.mean(v) if all(x is not None for x in v) else None) for f, v in vals.items()}
        bmean = max((m for m in means.values() if m is not None), default=None)
        for k, fam in enumerate(FAMILIES):
            cells = []
            for i, x in enumerate(vals[fam]):
                if x is None:
                    cells.append(r"\textcolor{gray}{--}")
                    continue
                s = f"{x:.1f}"
                cells.append(rf"\textbf{{{s}}}" if (x == best[i] and sum(v[i] is not None for v in vals.values()) > 1) else s)
            mm = means[fam]
            cells.append(r"\textcolor{gray}{--}" if mm is None else
                         (rf"\textbf{{{mm:.1f}}}" if mm == bmean and sum(m is not None for m in means.values()) > 1 else f"{mm:.1f}"))
            if all(x is None for x in vals[fam]):
                missing.append(f"{scale}/{fam}")
            size = scale if k == 0 else ""
            lines.append(f"{size} & {STYLE[fam]['label']} & " + " & ".join(cells) + r" \\")
        if si < len(SCALES) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    head = ("% Generated by render.py from results/normalized/benchmarks.csv. Zero-shot, LM-eval-harness 0.4.9.\n"
            "% acc: WinoGrande, BoolQ; acc_norm: PIQA, OpenBookQA, HellaSwag, ARC-E, ARC-C. Mean = unweighted mean\n"
            "% of the seven percentages. -- = not run. Bold = best within size. Requires \\usepackage{booktabs,xcolor}.\n")
    (TABLES / "table1_seven_benchmarks.tex").write_text(head + "\n".join(lines) + "\n")
    write_json(RESULTS / "figure_manifests" / "table1_seven_benchmarks.json",
               {"inputs": ["results/normalized/benchmarks.csv"], "missing_rows": missing,
                "runs": sorted(bench["run_id"].unique().tolist()) if not bench.empty else [], "rendered": time.ctime()})


def working_plot(runs, bench):
    """Quick look: total params (linear x, per tokenizer group) vs test loss and mean benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.3))
    sel = runs[runs["selected_run"]] if not runs.empty else runs
    for fam in FAMILIES:
        st = STYLE[fam]
        r = sel[(sel["family"] == fam) & sel["tokenizer_id"].str.contains("llama-tokenizer")] if not sel.empty else sel
        if not r.empty and "test_loss_nats_per_token" in r:
            rr = r[r["test_loss_nats_per_token"].notna()].sort_values("actual_total_params")
            axes[0].plot(rr["actual_total_params"] / 1e6, rr["test_loss_nats_per_token"], color=st["color"],
                         marker=st["marker"], ls=st["linestyle"], label=st["label"])
        if not bench.empty:
            b = bench[(bench["family"] == fam) & bench["scale"].isin(["50M", "100M", "400M"])]
            mean = b.groupby("run_id")["accuracy_pct"].agg(["mean", "count"])
            mean = mean[mean["count"] == 7]
            if not mean.empty:
                p = sel.set_index("run_id").loc[mean.index, "actual_total_params"] / 1e6
                o = np.argsort(p.values)
                axes[1].plot(p.values[o], mean["mean"].values[o], color=st["color"], marker=st["marker"],
                             ls=st["linestyle"], label=st["label"])
    axes[0].set_ylabel("Test loss (nats/token)")
    axes[1].set_ylabel("Mean of 7 benchmarks (%)")
    for ax in axes:
        ax.set_xlabel("Total parameters (millions)")
        ax.set_xlim(0, 450)
        sns.despine(ax=ax)
    fig.subplots_adjust(wspace=0.35)
    axes[0].legend(frameon=False)
    fig.suptitle("Working plot (not a paper figure); Llama-2 tokenizer groups, linear x", fontsize=7, color="0.4")
    save(fig, "working_params_vs_performance", {"note": "quick-look; Llama-2 groups only"})


def fig3(scale="100M"):
    p = NORM / f"recurrence_{scale}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["supported"]]
    comps = [c for c in ("gdlm - transformer", "gdlm - wang") if c in set(df["comparison"])]
    if not comps:
        return
    cstyle = {"gdlm - transformer": ("GDLM − Transformer", "#3B6EA8", "o"), "gdlm - wang": ("GDLM − Wang et al.", "#6E6E6E", "s")}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.4), gridspec_kw={"width_ratios": [3, 5], "wspace": 0.3})
    lims = [0.0]
    for ax, feat, order, xl in ((axes[0], "prior_count", ["Never", "Once", "2+"], "Previous occurrences in context"),
                                (axes[1], "last_distance", ["1–16", "17–64", "65–256", "257–1024", ">1024"],
                                 "Distance to last occurrence (tokens)")):
        ax.axhline(0, color="0.35", lw=0.8)
        for k, c in enumerate(comps):
            lab, col, mk = cstyle[c]
            d = df[(df["comparison"] == c) & (df["feature"] == feat)].set_index("bin")
            xs = [i + (k - (len(comps) - 1) / 2) * 0.18 for i, b in enumerate(order) if b in d.index]
            bs = [b for b in order if b in d.index]
            y = d.loc[bs, "mean_diff"].values
            ax.errorbar(xs, y, yerr=[y - d.loc[bs, "ci_lo"].values, d.loc[bs, "ci_hi"].values - y], fmt=mk,
                        color=col, ms=4.5, capsize=2.5, lw=1.1, label=lab)
            lims += list(d.loc[bs, "ci_lo"]) + list(d.loc[bs, "ci_hi"])
        n = df[(df["comparison"] == comps[0]) & (df["feature"] == feat)].set_index("bin")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{b}\nn={int(n.loc[b, 'n_targets']) / 1e6:.1f}M" if b in n.index else f"{b}\n(no data)"
                            for b in order])
        ax.set_xlabel(xl)
        sns.despine(ax=ax)
    m = max(abs(min(lims)), abs(max(lims))) * 1.15 or 0.05
    for ax in axes:
        ax.set_ylim(-m, m)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].set_ylabel("Δ token loss (nats/token)\n(< 0 favors GDLM)")
    axes[1].legend(frameon=False, loc="upper right")
    save(fig, "fig3_recurrence", {"inputs": [str(p.relative_to(RESULTS.parent))], "scale": scale,
                                  "ci": "95% document-cluster bootstrap", "comparisons": comps})


def main():
    runs, cur, gaps, bench = load("runs"), load("val_curves"), load("test_gaps"), load("benchmarks")
    totals = frozen_totals()
    if not runs.empty:
        runs["selected_run"] = runs["selected_run"].astype(bool)
    if not cur.empty:
        cur["selected_run"] = cur["selected_run"].astype(bool)
    fig2(runs, cur, gaps, totals)
    appendix_curves(cur, runs)
    table1(bench, totals)
    if not runs.empty:
        working_plot(runs, bench)
    fig3("100M")
    print("rendered figures/ and tables/")


if __name__ == "__main__":
    main()
