"""Dry-run configuration builder and parameter matcher (DETAILS.md §5.2, experiments.md §2).

For each scale in configs/model_groups.yaml: instantiate the Transformer reference on the meta device,
solve the free knob of gdlm (generator FFN width) and wang (number of GD layers) for the closest total
(unique trainable scalars, tied tensors once), and freeze all 12 configs to configs/frozen/<scale>_<family>.json
with config_sha256. Writes the matching report results/param_matching.{md,csv}.

  python count_params.py            # build/verify; refuses to change an existing frozen config
  python count_params.py --check    # only verify frozen configs against fresh instantiation
"""

import argparse
import csv
import json

import torch
import yaml

from common import CONFIGS, FROZEN, RESULTS, SEQ_LEN, TOKENIZERS, read_json, sha256_json, write_json
from models import ModelConfig, build_model


def load_groups():
    return yaml.safe_load((CONFIGS / "model_groups.yaml").read_text())


def count(cfg):
    with torch.device("meta"):
        m = build_model(cfg)
    c = m.param_counts()
    c["by_part"] = parts(m)
    return c


def parts(m):
    """Unique parameter counts by component (embedding / generator / GD predictor / positional / other)."""
    out = {}
    for n, p in m.unique_params():
        k = ("embedding" if n == "embed.weight" else "generator" if n.startswith(("blocks.", "final_norm"))
             else "gd" if n.startswith("gd.") else "positional" if n == "pos" else "other")
        out[k] = out.get(k, 0) + p.numel()
    return out


def flops_per_token(cfg, T=SEQ_LEN):
    """Training FLOPs per token as implemented: 3 x forward; forward = 2 x (matmul weights, the tied V/O
    map counted twice, readout V x d once) + causal attention 2 x 2 x (T/2) x dh per head-layer.
    Embedding lookups, norms and softmax are excluded."""
    d, V = cfg.d, cfg.vocab
    mm, heads = V * d, 0
    if cfg.family in ("transformer", "gdlm"):
        H, Hkv, dh = cfg.n_heads, cfg.n_kv_heads or cfg.n_heads, cfg.head_dim
        mm += cfg.n_layers * (d * H * dh * 2 + d * Hkv * dh * 2 + 3 * d * cfg.ffn)
        heads += cfg.n_layers * H * dh
    if cfg.family in ("gdlm", "wang"):
        p = cfg.d if cfg.family == "gdlm" else cfg.cov_dim
        gd = cfg.gd_layers * cfg.gd_heads * cfg.gd_head_dim
        mm += gd * p + 2 * gd * d
        heads += gd
    fwd = 2 * mm + 2 * T * heads
    return 3 * fwd


def act_bytes_per_token(cfg):
    """Rough analytic activation memory per token for backward (bf16 activations, no checkpointing,
    fp32 logits + their grad). The GPU pilots' measured peak memory is authoritative."""
    d, V, b = cfg.d, cfg.vocab, 2
    tot = V * (b + 4 + 4)
    if cfg.family in ("transformer", "gdlm"):
        H, Hkv, dh = cfg.n_heads, cfg.n_kv_heads or cfg.n_heads, cfg.head_dim
        tot += cfg.n_layers * b * (4 * d + 2 * H * dh + 2 * Hkv * dh + 3 * cfg.ffn)
    if cfg.family in ("gdlm", "wang"):
        tot += b * (3 * cfg.gd_layers * cfg.gd_heads * cfg.gd_head_dim + 2 * d)
    return tot


def make_cfg(scale, family, g, **over):
    s = g["scales"][scale]
    t = dict(s["transformer"])
    base = dict(family=family, vocab=s["vocab"], d=t["d"], max_seq_len=g["seq_len"],
                rope_theta=t["rope_theta"], rope_scaling=t.get("rope_scaling", {}))
    if family == "transformer":
        base.update({k: t[k] for k in ("n_layers", "n_heads", "n_kv_heads", "head_dim", "ffn")})
    elif family == "gdlm":
        base.update({k: t[k] for k in ("n_layers", "n_heads", "n_kv_heads", "head_dim", "ffn")})
        base.update(gd_layers=t["n_layers"], gd_heads=t["d"] // 64, gd_head_dim=64)
    else:
        base.update(gd_layers=1, gd_heads=t["d"] // 64, gd_head_dim=64, cov_dim=s["wang"]["cov_dim"],
                    n_layers=0, n_heads=0, n_kv_heads=0, ffn=0)
    base.update(over)
    return ModelConfig(**base)


def solve(scale, family, g, target):
    """Closest total to the reference along the family's free knob (counted analytically per step, then
    the chosen config is re-counted by instantiation)."""
    spec = g["scales"][scale][family]
    if spec["solve"] == "ffn":
        m = spec.get("ffn_multiple", 64)
        ref = g["scales"][scale]["transformer"]["ffn"]
        lo, hi = count(make_cfg(scale, family, g, ffn=m))["total"], count(make_cfg(scale, family, g, ffn=2 * m))["total"]
        per = hi - lo                       # params per `m` FFN units (linear in ffn)
        best = min(range(m, ref + 1, m), key=lambda f: abs(lo + (f // m - 1) * per - target))
        return make_cfg(scale, family, g, ffn=best)
    if spec["solve"] == "gd_layers":
        lo, hi = count(make_cfg(scale, family, g, gd_layers=1))["total"], count(make_cfg(scale, family, g, gd_layers=2))["total"]
        per = hi - lo
        L = max(1, round((target - lo) / per) + 1)
        return make_cfg(scale, family, g, gd_layers=L)
    raise ValueError(spec)


def analytic_counts(a):
    """Unique trainable parameters computed from the config alone (no instantiation)."""
    d, V = a["d"], a["vocab"]
    out = {"embedding": V * d, "generator": 0, "gd": 0, "positional": 0}
    if a["family"] in ("transformer", "gdlm"):
        H, Hkv, dh, f = a["n_heads"], a["n_kv_heads"] or a["n_heads"], a["head_dim"], a["ffn"]
        attn = d * H * dh * 2 + d * Hkv * dh * 2          # wq, wo, wk, wv
        mlp = 3 * d * f                                    # SwiGLU w1, w3, w2
        out["generator"] = a["n_layers"] * (attn + mlp + 2 * d) + d    # 2 norms per block + final norm
    if a["family"] in ("gdlm", "wang"):
        p = d if a["family"] == "gdlm" else a["cov_dim"]
        out["gd"] = a["gd_layers"] * a["gd_heads"] * a["gd_head_dim"] * (p + d)   # W_qk + W_v (W_o = W_v^T)
    if a["family"] == "wang":
        out["positional"] = a["max_seq_len"] * a["cov_dim"]
    out["total"] = sum(out.values())
    return out


def data_section(tdef):
    """Samples (2048-token packed sequences) of each pretraining split, from data/<tok>/meta.json."""
    import math
    from common import TOKENIZERS
    from data import SPLIT_ROLES
    tps = tdef["batch_seqs"] * SEQ_LEN
    steps = round(tdef["budget_tokens"] / tps)
    used = steps * tdef["batch_seqs"]
    L = ["", "## Pretraining data: samples per split", "",
         "FineWeb-Edu `sample-100BT`, document-level split (the same raw documents for both tokenizers; "
         "`data/manifests/heldout_docs.json`). A **sample** is one packed window of 2048 input tokens (+1 target). "
         "Documents are `BOS + text` with no EOS; attention and loss never cross documents. Training reads windows "
         "in a fixed shuffled order (`data_seed` 1234). Validation and test are evaluated in full; the last partial "
         "window is padded and its padding masked. *Loss targets* exclude window starts, positions whose input is BOS "
         "and BOS targets.", "",
         "| Tokenizer (models) | Split | Documents | Tokens | Samples available (full windows) | Samples used | "
         "Loss targets per evaluation |", "|---|---|---:|---:|---:|---:|---:|"]
    targets = {("llama2", "validation"): 19_957_116, ("llama2", "test"): 49_890_884}   # measured (val.jsonl, test_loss.json)
    tr2 = None
    for tok, models in (("llama2", "50M / 100M / 400M"), ("llama3", "1.2B, not trained")):
        meta_p = TOKENIZERS[tok]["data_dir"] / "meta.json"
        if not meta_p.exists():
            continue
        sp = read_json(meta_p)["splits"]
        for role in ("train", "validation", "test", "reserved"):
            m = sp[SPLIT_ROLES[role]]
            full = (m["tokens"] - 1) // SEQ_LEN
            if role == "train":
                tr2 = tr2 or full
                u = f"{used:,} ({steps:,} steps × {tdef['batch_seqs']})" + (" **> available**" if used > full else "")
            elif role == "reserved":
                u = "0 (held out, unused)"
            else:
                u = f"{math.ceil((m['tokens'] - 1) / SEQ_LEN):,} (all, incl. 1 partial)"
            t = targets.get((tok, role))
            label = {"train": "train", "validation": "validation", "test": "test",
                     "reserved": "reserved (old `val_small`)"}[role]
            tname = "Llama-2" if tok == "llama2" else "Llama-3.2"
            tcell = f"{t:,}" if t else ("–" if role in ("train", "reserved") else "pending (no eval yet)")
            L.append(f"| {tname} ({models}) | {label} | {m['docs']:,} | {m['tokens']:,} | {full:,} | {u} | {tcell} |")
    L += ["", f"Training uses {used:,} samples = {used * SEQ_LEN / 1e9:.2f}B input tokens "
          f"({100 * used / tr2:.1f}% of the Llama-2 training windows), the same samples in the same order for all "
          "three families at a scale. With Llama-3.2, the 100B budget exceeds the ~97.4B available training tokens "
          "(accepted); a future 1.2B run would need a ~97B budget or a partial second pass, to be decided before any "
          "1.2B training."]
    return L


def models_md(path=None):
    """models.md: exact parameter counts (analytic, from configs/frozen) and settings of every model."""
    from common import REPO
    import yaml
    exps = {e["name"]: e for e in yaml.safe_load((CONFIGS / "experiments.yaml").read_text())["experiments"]}
    tdef = yaml.safe_load((CONFIGS / "experiments.yaml").read_text())["train_defaults"]
    recs = [read_json(FROZEN / f"{s}_{f}.json") for s in ("50M", "100M", "400M", "1.2B")
            for f in ("gdlm", "wang", "transformer")]
    name = {"gdlm": "GDLM", "wang": "Wang et al. (position-only GD, parameter-matched)", "transformer": "Transformer"}
    L = ["# Models", "",
         "Generated by `python count_params.py --models-md` from `configs/frozen/*.json`. Counts are computed "
         "**analytically from the configs** (`count_params.analytic_counts`): unique trainable parameters, tied tensors "
         "counted once (input/output embedding; GD W_O = W_V^T). Each total was cross-checked against the "
         "instantiated count in the frozen registry (all equal). Reference = Transformer total of the scale; "
         "matching tolerance ±2%.", "",
         "## Parameter counts", "",
         "| Scale | Model | Tokenizer (vocab) | Total | Δ vs. reference | Embedding (tied) | Non-embedding | "
         "Generator | GD predictor | Positional covariates |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in recs:
        a = r["arch"]
        c = analytic_counts(a)
        assert c["total"] == r["actual_total_params"], (r["scale"], r["family"], c["total"], r["actual_total_params"])
        tok = "Llama-2 (32,000)" if a["vocab"] == 32000 else "Llama-3.2 (128,256)"
        dash = lambda x: f"{x:,}" if x else "–"
        L.append(f"| {r['scale']} | {name[r['family']]} | {tok} | **{c['total']:,}** | "
                 f"{100 * (c['total'] - r['target_total_params']) / r['target_total_params']:+.2f}% | {c['embedding']:,} | "
                 f"{c['total'] - c['embedding']:,} | {dash(c['generator'])} | {dash(c['gd'])} | {dash(c['positional'])} |")
    L += ["", "## Architecture settings", "",
          "| Scale | Model | d | Layers | Q / KV heads | Head dim | FFN (SwiGLU) | GD layers × heads (= kernels) | "
          "GD head dim | Covariate (dim) | RoPE θ | Train GFLOPs/token |",
          "|---|---|---:|---:|---|---:|---:|---|---:|---|---|---:|"]
    for r in recs:
        a = r["arch"]
        fam = r["family"]
        gen = fam != "wang"
        rope = f"{a['rope_theta']:,.0f}" + (" (+llama3 scaling)" if a["rope_scaling"] else "")
        cov = {"gdlm": f"generator output φ (d = {a['d']})", "wang": f"learned position table ({a['cov_dim']})",
               "transformer": "–"}[fam]
        gd = f"{a['gd_layers']} × {a['gd_heads']} (= {a['gd_layers'] * a['gd_heads']:,})" if fam != "transformer" else "–"
        L.append(f"| {r['scale']} | {name[fam].split(' (')[0]} | {a['d']} | {a['n_layers'] if gen else '–'} | "
                 f"{str(a['n_heads']) + ' / ' + str(a['n_kv_heads']) if gen else '–'} | {a['head_dim'] if gen else '–'} | "
                 f"{a['ffn'] if gen else '–'} | {gd} | {a['gd_head_dim'] if fam != 'transformer' else '–'} | {cov} | "
                 f"{rope if fam != 'wang' else '– (no RoPE)'} | {r['flops_per_token_train'] / 1e9:.2f} |")
    L += ["", "## Training settings (per run)", "",
          "| Scale | Model | Run id | Status | Peak LR | Micro-batch / GPU | Checkpoint every | config_sha256 |",
          "|---|---|---|---|---:|---:|---:|---|"]
    for r in recs:
        e = exps[f"{r['scale']}_{r['family']}"]
        lr = e["sweep"]["lr"][0]
        rid = f"{r['scale']}_{r['family']}_lr{lr:.0e}_s0".replace("e-0", "e-")
        st = "configs only (not trained)" if e["phase"] == "never" else f"phase {e['phase']}"
        L.append(f"| {r['scale']} | {name[r['family']].split(' (')[0]} | `{rid}` | {st} | {lr:g} | {e['micro_batch']} | "
                 f"{e['ckpt_every_tokens'] / 1e9:g}B | `{r['config_sha256'][:12]}` |")
    L += ["", "Shared by every run: context 2048; global batch 256 sequences (524,288 tokens/step); "
          f"budget {tdef['budget_tokens'] / 1e9:g}B tokens (190,735 steps); AdamW β = (0.9, 0.95), weight decay 0.1 on "
          "matrices only (not embeddings, positional covariates or norm gains), grad clip 1.0; linear warmup over "
          "2B tokens then cosine to 10% of peak; bf16 autocast with fp32 master weights; validation every 1B tokens; "
          "init N(0, 0.02) (residual output projections 0.02/√(2L); Wang positional table N(0, 1)); "
          "RMSNorm ε = 1e-5; seed 0; data_seed 1234."]
    L += data_section(tdef)
    L += ["", "## Parameter formulas", "",
          "- Embedding (tied, counted once): `V·d`.",
          "- Llama block (Transformer, GDLM generator): `d·H·dh·2 + d·H_kv·dh·2` (attention) `+ 3·d·FFN` (SwiGLU) "
          "`+ 2d` (norm gains); plus one final norm `d`.",
          "- GD predictor (GDLM, Wang): `L_gd · H_gd · dh_gd · (p + d)`. The tied query/key map is `dh×p` per head "
          "and the tied value map is `dh×d` per head (output = W_Vᵀ, no extra parameters); `p = d` for GDLM and "
          "`p = cov_dim` for Wang. The readout has no parameters (gainless RMS norm + tied centered embedding).",
          "- Wang positional covariates: `2048 · cov_dim`.", "",
          "Notes: GDLM uses one GD layer per generator layer with d/64 heads of 64; its generator FFN width was "
          "reduced to absorb the GD parameters. Wang uses d/64 heads per GD layer and the number of GD layers was "
          "chosen to match the total. The 1.2B generator keeps Llama-3.2 GQA (32 Q / 8 KV); its GD kernels are "
          "independent heads (no GQA)."]
    (path or REPO / "models.md").write_text("\n".join(L) + "\n")
    print(f"wrote {path or REPO / 'models.md'}")


def frozen_record(scale, family, cfg, g, target):
    c = count(cfg)
    tol = g["tolerance"]
    mism = (c["total"] - target) / target
    arch = cfg.arch()
    rec = {"scale": scale, "family": family, "tokenizer": g["scales"][scale]["tokenizer"],
           "tokenizer_id": TOKENIZERS[g["scales"][scale]["tokenizer"]]["hf_id"],
           "data_seed": g["scales"][scale]["data_seed"], "arch": arch, "config_sha256": sha256_json(arch),
           "target_total_params": target, "actual_total_params": c["total"], "embedding_params": c["embedding"],
           "nonembedding_params": c["nonembedding"], "params_by_part": c["by_part"], "mismatch_frac": mism,
           "within_tolerance": abs(mism) <= tol, "flops_per_token_train": flops_per_token(cfg),
           "flops_method": " ".join(flops_per_token.__doc__.split()),
           "act_bytes_per_token_est": act_bytes_per_token(cfg), "trainable": scale != "1.2B"}
    if family == "wang":
        rec["label"] = "Wang et al. (position-only GD, parameter-matched)"
        rec["deviations_from_wang_et_al"] = [
            "parameter-matched: depth (number of GD layers) chosen to match the Transformer total",
            "one learned positional-covariate table shared by all GD layers (width cov_dim), unit-RMS init",
            "tied QK and tied V/O (W_O = W_V^T) per head, softmax kernel normalized over examples i < t",
            "zero-mean tied embeddings and gainless RMS readout, as in gdlm",
            "Llama-2 / Llama-3.2 tokenizers, FineWeb-Edu, 2048 context, document-masked kernels"]
    return rec


def build(write=True):
    g = load_groups()
    recs = []
    for scale in g["scales"]:
        ref_cfg = make_cfg(scale, "transformer", g)
        target = count(ref_cfg)["total"]
        for family in ("gdlm", "wang", "transformer"):
            cfg = ref_cfg if family == "transformer" else solve(scale, family, g, target)
            recs.append(frozen_record(scale, family, cfg, g, target))
    if write:
        for r in recs:
            path = FROZEN / f"{r['scale']}_{r['family']}.json"
            if path.exists():
                old = read_json(path)
                if old["config_sha256"] != r["config_sha256"]:
                    raise SystemExit(f"{path} exists with a different config_sha256; a changed config needs a "
                                     f"new frozen name and run_id")
            write_json(path, r)
        report(recs)
    return recs


def check():
    """Every frozen config re-instantiates to its recorded counts."""
    bad = []
    for p in sorted(FROZEN.glob("*.json")):
        r = read_json(p)
        c = count(ModelConfig.from_dict(r["arch"]))
        if c["total"] != r["actual_total_params"] or sha256_json(r["arch"]) != r["config_sha256"]:
            bad.append(p.name)
    print("frozen configs OK" if not bad else f"MISMATCH: {bad}")
    return not bad


def report(recs):
    RESULTS.mkdir(parents=True, exist_ok=True)
    cols = ["scale", "family", "tokenizer", "target_total_params", "actual_total_params", "mismatch_pct",
            "embedding_params", "nonembedding_params", "generator_params", "gd_params", "positional_params",
            "d", "n_layers", "n_heads", "n_kv_heads", "ffn", "gd_layers", "gd_heads", "gd_head_dim", "cov_dim",
            "kernels", "gflops_per_token_train", "act_kb_per_token_est", "within_tolerance", "config_sha256"]
    rows = []
    for r in recs:
        a = r["arch"]
        rows.append({"scale": r["scale"], "family": r["family"], "tokenizer": r["tokenizer"],
                     "target_total_params": r["target_total_params"], "actual_total_params": r["actual_total_params"],
                     "mismatch_pct": round(100 * r["mismatch_frac"], 3), "embedding_params": r["embedding_params"],
                     "nonembedding_params": r["nonembedding_params"],
                     "generator_params": r["params_by_part"].get("generator", 0),
                     "gd_params": r["params_by_part"].get("gd", 0),
                     "positional_params": r["params_by_part"].get("positional", 0),
                     **{k: a[k] for k in ("d", "n_layers", "n_heads", "n_kv_heads", "ffn", "gd_layers", "gd_heads",
                                          "gd_head_dim", "cov_dim")},
                     "kernels": a["gd_layers"] * a["gd_heads"],
                     "gflops_per_token_train": round(r["flops_per_token_train"] / 1e9, 3),
                     "act_kb_per_token_est": round(r["act_bytes_per_token_est"] / 1024, 1),
                     "within_tolerance": r["within_tolerance"], "config_sha256": r["config_sha256"][:12]})
    with open(RESULTS / "param_matching.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = ["# Parameter matching report", "",
             "Generated by `python count_params.py` from `configs/model_groups.yaml`. Counts are unique trainable "
             "scalars of models instantiated on the meta device (tied tensors counted once). Reference = Transformer "
             "total; tolerance ±2%. FLOPs: " + " ".join(flops_per_token.__doc__.split()) + " Activation "
             "memory: rough analytic estimate (see `act_bytes_per_token`); measured GPU peaks are in STATUS.md.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:,}" if isinstance(r[c], int) and not isinstance(r[c], bool)
                                       else str(r[c]) for c in cols) + " |")
    flagged = [f"{r['scale']} {r['family']}" for r in rows if not r["within_tolerance"]]
    lines += ["", "Outside tolerance: " + (", ".join(flagged) if flagged else "none")]
    (RESULTS / "param_matching.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--models-md", action="store_true", help="write models.md from the frozen configs")
    a = ap.parse_args()
    if a.models_md:
        models_md()
        raise SystemExit(0)
    if a.check:
        raise SystemExit(0 if check() else 1)
    build()
