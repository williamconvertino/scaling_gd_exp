"""Seven zero-shot benchmarks with LM-eval-harness 0.4.9 (DETAILS.md §7.2). Selected checkpoint only.

  python eval_benchmarks.py --run-id 50M_gdlm_lr1e-3_s0

Adapted from ../tmlr_paper/gdlm/eval/lm_eval_wrapper.py: BOS is prepended to every context; contexts are
scored in one forward pass (left-truncated to 2048 tokens). Each task is run and written separately
(runs/<run_id>/eval/benchmarks/<task>.json, atomic), so a preempted job resumes at the next task.
A missing metric key fails loudly. Normalized long-format rows: runs/<run_id>/eval/benchmarks.csv.
"""

import argparse
import csv
import json
import time

import torch

from common import RUNS, SEQ_LEN, TOKENIZERS, read_json, write_json
from data import get_tokenizer
from eval_loss import load_model, select_checkpoint
from sample import pad_batch

HARNESS_VERSION = "0.4.9"
# display order of the paper (figures.md §4) -> (harness task, metric key)
TASKS = [("WinoGrande", "winogrande", "acc"), ("PIQA", "piqa", "acc_norm"), ("OpenBookQA", "openbookqa", "acc_norm"),
         ("HellaSwag", "hellaswag", "acc_norm"), ("BoolQ", "boolq", "acc"), ("ARC-E", "arc_easy", "acc_norm"),
         ("ARC-C", "arc_challenge", "acc_norm")]


class HarnessLM:
    """lm_eval.api.model.LM adapter for all three families (scoring only; generation is not needed)."""

    def __new__(cls, *a, **k):
        from lm_eval.api.model import LM

        class _LM(LM):
            def __init__(self, model, tok, bos, device, batch_size):
                super().__init__()
                self.model, self.tok, self.bos, self.dev, self.bs = model, tok, bos, device, batch_size

            def enc(self, s):
                return self.tok(s, add_special_tokens=False)["input_ids"] if s else []

            def _encode_pair(self, context, continuation):
                n = len(context) - len(context.rstrip())
                if n > 0:
                    continuation = context[-n:] + continuation
                    context = context[:-n]
                whole = self.enc(context + continuation)
                ctx = self.enc(context)
                return ctx, whole[len(ctx):]

            @torch.no_grad()
            def _score(self, pairs):
                out = [None] * len(pairs)
                seqs = [(([self.bos] + c + x)[-(SEQ_LEN + 1):], len(x)) for c, x in pairs]
                order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][0]))
                for b in range(0, len(order), self.bs):
                    idx = order[b:b + self.bs]
                    z, seg, _ = pad_batch([seqs[i][0] for i in idx], self.dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                        logits = self.model.logits(z[:, :-1], seg[:, :-1])
                    lp = torch.log_softmax(logits.float(), -1)
                    for r, i in enumerate(idx):
                        s, nc = seqs[i]
                        L = len(s) - 1
                        sl = slice(L - nc, L)
                        tgt = z[r, 1:][sl]
                        out[i] = (lp[r, sl].gather(-1, tgt[:, None]).sum().item(),
                                  bool((lp[r, sl].argmax(-1) == tgt).all()))
                return out

            def loglikelihood(self, requests):
                return self._score([self._encode_pair(*r.args) for r in requests])

            def loglikelihood_rolling(self, requests):
                raise NotImplementedError("not used by the seven tasks")

            def generate_until(self, requests):
                raise NotImplementedError("not used by the seven tasks")

        return _LM(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="debug only; never for reported numbers")
    a = ap.parse_args()
    import lm_eval
    assert lm_eval.__version__ == HARNESS_VERSION, f"lm_eval {lm_eval.__version__} != pinned {HARNESS_VERSION}"
    run_dir = RUNS / a.run_id
    out_dir = run_dir / "eval" / ("benchmarks" if a.limit is None else f"benchmarks_limit{a.limit}")
    sel_p = run_dir / "eval" / "selected_checkpoint.json"
    best = read_json(sel_p)["selected"] if sel_p.exists() else select_checkpoint(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    for name, task, metric in TASKS:
        p = out_dir / f"{task}.json"
        if p.exists():
            continue
        if model is None:
            model, rc = load_model(run_dir, best["name"], device, attn_impl="sdpa", compile=False)
            tok_name = rc["frozen"]["tokenizer"]
            lm = HarnessLM(model, get_tokenizer(tok_name), TOKENIZERS[tok_name]["bos"], device, a.batch)
        t0 = time.time()
        res = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=0, limit=a.limit, bootstrap_iters=1000,
                                      random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                      fewshot_random_seed=0, log_samples=False)
        r = res["results"][task]
        key, se_key = f"{metric},none", f"{metric}_stderr,none"
        if key not in r or r[key] is None:
            raise SystemExit(f"metric {key} missing for {task}: {sorted(r)}")
        cfg = res.get("configs", {}).get(task, {})
        write_json(p, {"task": task, "display": name, "metric": metric, "value": r[key], "stderr": r.get(se_key),
                       "n_samples": res.get("n-samples", {}).get(task), "version": res.get("versions", {}).get(task),
                       "harness_version": lm_eval.__version__, "checkpoint": best["name"],
                       "checkpoint_tokens": best["tokens"], "bos_prepended": True, "num_fewshot": 0,
                       "limit": a.limit, "seconds": time.time() - t0,
                       "config": {k: cfg.get(k) for k in ("dataset_path", "dataset_name", "test_split",
                                                          "validation_split", "output_type", "target_delimiter")},
                       "raw_results": r})
        print(f"{a.run_id} {task}: {metric} = {100 * r[key]:.2f}", flush=True)
    if a.limit is None:
        rows = []
        for name, task, metric in TASKS:
            d = read_json(out_dir / f"{task}.json")
            rows.append({"run_id": a.run_id, "task": task, "display": name, "metric": metric,
                         "accuracy_pct": 100 * d["value"], "stderr_pct": 100 * d["stderr"] if d["stderr"] else None,
                         "n": (d["n_samples"] or {}).get("effective"), "checkpoint": d["checkpoint"],
                         "checkpoint_tokens": d["checkpoint_tokens"], "harness_version": d["harness_version"]})
        tmp = run_dir / "eval" / "benchmarks.csv.tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        tmp.rename(run_dir / "eval" / "benchmarks.csv")


if __name__ == "__main__":
    main()
