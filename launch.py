"""Turn configs/experiments.yaml into sbatch submissions (DETAILS.md §8).

  python launch.py --dry-run          # show what would be submitted
  python launch.py                    # submit planned runs of enabled phases, in priority order
  python launch.py --pilot 50M_gdlm   # Phase 0 throughput/memory pilot (scavenge H200, --max-steps)
  python launch.py --eval RUN_ID      # (re)submit the evaluation job of a finished run

Training runs are chained in priority order (each waits on the previous one with afterany; the lab account
allows one 8-GPU job at a time). Each run's eval job waits on its training job with afterok. The `runs:` section
at the bottom of experiments.yaml records run ids, status and job ids; the rest of the file is left untouched.
"""

import argparse
import itertools
import json
import math
import re
import subprocess
import time

import yaml

from common import CONFIGS, FROZEN, REPO, RUNS, TOKENIZERS, code_sha256, git_commit, read_json, write_json
from data import manifest_sha

EXP = CONFIGS / "experiments.yaml"
MARK = "# ---- runs (maintained by launch.py; everything below this line is rewritten) ----"
TRAIN_HOURS = 12            # fallback time limit when no throughput measurement exists (the job requeues itself)
MAX_HOURS = 7 * 24 - 1      # association MaxWall on henaolab_h200 is 7 days
TIME_BUFFER = 1.25          # time limit = estimated remaining runtime x this
MULTI_GPU_EFF = 0.85        # 8-GPU per-GPU throughput / 1-GPU pilot (measured 0.77-1.0 on 100M/400M runs)


def median_tok_per_s(path):
    """Median throughput over the second half of a train.jsonl (skips compile/warm-up), or None."""
    if not path.exists():
        return None
    v = [r["tok_per_s"] for r in map(json.loads, path.read_text().splitlines()) if "tok_per_s" in r]
    v = sorted(v[len(v) // 2:])
    return v[len(v) // 2] if v else None


def train_hours(run_id, exp, world=8):
    """Time limit for one 8-GPU job: remaining tokens / measured throughput, with a buffer.

    Throughput comes from the run's own log when it has trained before (micro-batch may have changed since, so
    this is a conservative or exact estimate), else from the newest 1-GPU pilot of the same experiment."""
    rc = read_json(RUNS / run_id / "run_config.json")
    tc = rc["train"]
    done = 0
    log = RUNS / run_id / "train.jsonl"
    tps = median_tok_per_s(log)
    if tps:
        done = max(r.get("tokens", 0) for r in map(json.loads, log.read_text().splitlines()))
    else:
        pilots = sorted(RUNS.glob(f"pilot_{exp['name']}_mb*"), key=lambda p: p.stat().st_mtime, reverse=True)
        per_gpu = next(filter(None, (median_tok_per_s(p / "train.jsonl") for p in pilots)), None)
        tps = per_gpu and per_gpu * world * MULTI_GPU_EFF
    if not tps:
        return TRAIN_HOURS
    hours = (tc["budget_tokens"] - done) / tps / 3600 * TIME_BUFFER
    return max(1, min(MAX_HOURS, math.ceil(hours)))


def load():
    return yaml.safe_load(EXP.read_text())


def save_runs(runs):
    head = EXP.read_text().split(MARK)[0]
    EXP.write_text(head + MARK + "\n" + yaml.safe_dump({"runs": runs}, sort_keys=False, width=120))


def fmt_lr(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def expand(exp):
    """Cartesian product of the sweep axes -> run ids."""
    keys = sorted(exp["sweep"])
    for vals in itertools.product(*(exp["sweep"][k] for k in keys)):
        p = dict(zip(keys, vals))
        yield f"{exp['scale']}_{exp['family']}_lr{fmt_lr(p['lr'])}_s{p['seed']}", p


def run_config(exp, params, defaults, run_id):
    fz = read_json(FROZEN / f"{exp['scale']}_{exp['family']}.json")
    tok = fz["tokenizer"]
    man = read_json(REPO / "data" / "manifests" / f"{tok}.json") if (REPO / "data" / "manifests" / f"{tok}.json").exists() else {}
    from common import SEQ_LEN
    defaults = {**defaults, **exp.get("train", {})}      # per-experiment overrides (e.g. 1.2B token budget)
    tps = defaults["batch_seqs"] * SEQ_LEN
    total = round(defaults["budget_tokens"] / tps)
    train = {**defaults, "lr": params["lr"], "seed": params["seed"], "data_seed": fz["data_seed"],
             "micro_batch": exp["micro_batch"], "ckpt_every_tokens": exp["ckpt_every_tokens"],
             "tokens_per_step": tps, "total_steps": total, "warmup_steps": round(defaults["warmup_tokens"] / tps)}
    return {"run_id": run_id, "experiment": exp["name"], "phase": exp["phase"], "frozen": fz, "train": train,
            "runtime": exp.get("runtime", {}),
            "data": {"tokenizer": tok, "data_dir": str(TOKENIZERS[tok]["data_dir"]),
                     "manifest_sha256": man.get("manifest_sha256"), "tokenizer_revision": man.get("tokenizer_revision")},
            "code_sha256": code_sha256(), "git_commit": git_commit(), "created": time.ctime()}


def write_run_config(rc):
    """Create runs/<id>/run_config.json. An existing run keeps its config; a different frozen config is refused."""
    d = RUNS / rc["run_id"]
    p = d / "run_config.json"
    if p.exists():
        old = read_json(p)
        if old["frozen"]["config_sha256"] != rc["frozen"]["config_sha256"]:
            raise SystemExit(f"{rc['run_id']}: frozen config changed; a changed config needs a new run_id")
        return old
    d.mkdir(parents=True, exist_ok=True)
    write_json(p, rc)
    return rc


def sbatch(args, env, dry):
    cmd = ["sbatch", "--parsable", f"--export=ALL,{env}", *args]
    if dry:
        print("DRY  " + " ".join(cmd))
        return None
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        raise SystemExit(f"sbatch failed: {out.stderr}")
    return out.stdout.strip().split(";")[0]


def submit_train(run_id, dep, dry, hours=TRAIN_HOURS):
    (REPO / "logs" / "train").mkdir(parents=True, exist_ok=True)
    args = [f"--job-name=tr_{run_id}", f"--time={hours}:00:00",
            f"--output={REPO}/logs/train/{run_id}-%j.out"]
    if dep:
        args.append(f"--dependency=afterany:{dep}")
    return sbatch(args + ["slurm/train_8xh200.sbatch"], f"RUN_ID={run_id}", dry)


def submit_eval(run_id, dep, dry, fallback=False):
    (REPO / "logs" / "eval").mkdir(parents=True, exist_ok=True)
    args = [f"--job-name=ev_{run_id}", f"--output={REPO}/logs/eval/{run_id}-%j.out"]
    if fallback:      # normal 1xH200 on the lab partition (only if scavenge is unavailable for long)
        args += ["--partition=h200-hp", "--account=henaolab_h200", "--time=12:00:00"]
    if dep:
        args.append(f"--dependency=afterok:{dep}")
    return sbatch(args + ["slurm/eval_1xh200_scavenge.sbatch"], f"RUN_ID={run_id}", dry)


def last_train_job(runs):
    """Most recent train job still queued/running (the chain continues after it)."""
    q = subprocess.run(["squeue", "--me", "-h", "-o", "%i"], capture_output=True, text=True).stdout.split()
    jobs = [r.get("train_job") for r in runs.values() if r.get("train_job") in q]
    return max(jobs, key=int) if jobs else None


def launch(dry):
    cfg = load()
    runs = cfg.get("runs") or {}
    enabled = set(cfg.get("phases_enabled") or [])
    todo = []
    for exp in sorted(cfg["experiments"], key=lambda e: e["priority"]):
        for run_id, params in expand(exp):
            st = runs.setdefault(run_id, {"experiment": exp["name"], "phase": exp["phase"], "status": "planned"})
            if exp["phase"] == "never" or exp["phase"] not in enabled or st["status"] != "planned":
                continue
            todo.append((exp, run_id, params))
    if not todo:
        print("nothing to submit (check phases_enabled / statuses)")
    dep = last_train_job(runs)
    for exp, run_id, params in todo:
        rc = write_run_config(run_config(exp, params, cfg["train_defaults"], run_id)) if not dry else None
        if not dry:
            assert rc["data"]["manifest_sha256"], "data manifest missing: run `python data.py manifest` first"
        hours = train_hours(run_id, exp) if not dry else TRAIN_HOURS
        tj = submit_train(run_id, dep, dry, hours)
        ej = submit_eval(run_id, tj, dry)
        print(f"{run_id}: train {tj} ({hours} h, after {dep}), eval {ej}")
        if not dry:
            runs[run_id].update(status="submitted", train_job=tj, eval_job=ej, submitted=time.ctime())
            dep = tj
    if not dry:
        save_runs(runs)


def pilot(name, steps, micro_batch, runtime, dry):
    """Short pilot of the real config (throughput / peak memory), separate run dir `pilot_<name>_mb<m>`."""
    cfg = load()
    exp = next(e for e in cfg["experiments"] if e["name"] == name)
    run_id, params = next(expand(exp))
    pid = f"pilot_{name}_mb{micro_batch}" + "".join(f"_{k}{v}" for k, v in sorted(runtime.items()))
    rc = run_config({**exp, "micro_batch": micro_batch, "runtime": {**exp.get("runtime", {}), **runtime}},
                    params, cfg["train_defaults"], pid)
    rc["pilot"] = True
    if not dry:
        write_run_config(rc)
    (REPO / "logs" / "pilot").mkdir(parents=True, exist_ok=True)
    cmd = f"torchrun --standalone --nproc_per_node=1 train.py --run-id {pid} --max-steps {steps}"
    return sbatch([f"--job-name={pid}", f"--output={REPO}/logs/pilot/{pid}-%j.out", "slurm/pilot_gpu.sbatch"],
                  f"CMD={cmd}", dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", default=None, help="experiment name, e.g. 50M_gdlm")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--runtime", default="", help="e.g. grad_ckpt=1,gd_chunk=16")
    ap.add_argument("--eval", default=None, help="run id: submit its eval job now")
    ap.add_argument("--eval-fallback", action="store_true")
    a = ap.parse_args()
    if a.pilot:
        rt = {}
        for kv in filter(None, a.runtime.split(",")):
            k, v = kv.split("=")
            rt[k] = int(v) if re.fullmatch(r"\d+", v) else v
        exp = next(e for e in load()["experiments"] if e["name"] == a.pilot)
        print(pilot(a.pilot, a.steps, a.micro_batch or exp["micro_batch"], rt, a.dry_run))
    elif a.eval:
        print(submit_eval(a.eval, None, a.dry_run, a.eval_fallback))
    else:
        launch(a.dry_run)


if __name__ == "__main__":
    main()
