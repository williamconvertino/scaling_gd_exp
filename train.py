"""Pretraining (DETAILS.md §6): DDP, bf16 autocast with fp32 master weights, resumable and requeue-safe.

  torchrun --standalone --nproc_per_node=8 train.py --run-id 50M_gdlm_lr1e-3_s0

Reads runs/<run_id>/run_config.json (written by launch.py) and writes into runs/<run_id>/:
  train.jsonl       training loss / lr / grad norm / throughput every `log_every` steps
  val.jsonl         validation loss (full validation split) at step 0, every 1B tokens and at the end
  ckpt_<NNN>B.pt    interval checkpoints (model + optimizer + counters + data position), never deleted;
                    listed with their step/tokens in checkpoints.jsonl
  resume.pt         rolling full-state checkpoint (overwritten every ~45 min and before the time limit)
  segments.jsonl    one line per job segment: wall time, GPUs, GPU-hours, peak memory, steps
  record.json       run record (experiments.md §9 schema + provenance), updated at every validation
  status.json       running | completed | failed (+ stop_reason); DONE when finished
  REQUEUE           written when the job stops near its time limit (the sbatch script requeues)
"""

import argparse
import contextlib
import json
import math
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from common import RUNS, SEQ_LEN, TOKENIZERS, code_sha256, git_commit, read_json, write_json
from data import TokenFile, TrainStream, eval_batch, eval_windows, manifest_sha, order_fingerprint, window_order
from models import ModelConfig, build_model, compile_model

H200_BF16_PEAK = 989e12


# ----------------------------------------------------------------------------- distributed helpers
def init_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"))
        return dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local}")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, dev


def is_dist():
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_dist():
        dist.barrier()


def all_sum(t):
    if is_dist():
        dist.all_reduce(t)
    return t


def bcast_flag(flag, device):
    t = torch.tensor([1 if flag else 0], device=device)
    if is_dist():
        dist.broadcast(t, 0)
    return bool(t.item())


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- schedule
def lr_at(step, tc):
    """Linear warmup over warmup_steps, cosine decay to min_lr_frac * peak at total_steps."""
    w, T = tc["warmup_steps"], tc["total_steps"]
    if step < w:
        return tc["lr"] * (step + 1) / w
    p = min(1.0, (step - w) / max(1, T - w))
    return tc["lr"] * (tc["min_lr_frac"] + (1 - tc["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * p)))


def crossed(step, tokens_per_step, every):
    """True at the first step whose cumulative token count reaches a new multiple of `every` tokens."""
    return int(step * tokens_per_step // every) > int((step - 1) * tokens_per_step // every)


def ckpt_name(nominal_tokens):
    b = nominal_tokens / 1e9
    return f"ckpt_{int(b):03d}B.pt" if abs(b - round(b)) < 1e-9 else f"ckpt_{b:05.1f}B.pt"


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, data_dir, split, micro_batch, rank, world, device, bos_id):
    """Mean next-token NLL (nats/token) over the full split, masked as in training."""
    model.eval()
    f, starts = eval_windows(data_dir, split)
    mine = starts[rank::world]
    tot = torch.zeros(2, device=device, dtype=torch.float64)
    for i in range(0, len(mine), micro_batch):
        z, y, seg, m = eval_batch(f, mine[i:i + micro_batch], device, bos_id)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, tok = model(z, seg, y, m, per_token=True)
        tot[0] += (tok.double() * m).sum()
        tot[1] += m.sum()
    all_sum(tot)
    model.train()
    return (tot[0] / tot[1]).item(), int(tot[1].item())


class Monitor:
    """Early stopping on convergence and divergence (DETAILS.md §6.2, §6.5), rebuilt from val.jsonl."""

    def __init__(self, tc, vals):
        self.tc, self.vals, self.spike_since = tc, list(vals), None   # vals: [(tokens, loss)]
        self.slack = tc["tokens_per_step"]       # validations sit at the first step past each 1B multiple

    @property
    def best(self):
        return min((v for _, v in self.vals), default=float("inf"))

    def update(self, tokens, loss):
        """Returns (stop_reason or None, failed: bool)."""
        tc = self.tc
        if not math.isfinite(loss):
            return "diverged: non-finite validation loss", True
        prev_best = self.best
        self.vals.append((tokens, loss))
        if loss > prev_best + tc["spike_nats"]:
            if self.spike_since is None:
                self.spike_since = tokens
            elif tokens - self.spike_since >= tc["spike_recover_tokens"]:
                return f"diverged: validation loss > best + {tc['spike_nats']} for >= " \
                       f"{tc['spike_recover_tokens'] / 1e9:g}B tokens", True
        else:
            self.spike_since = None
        if tokens >= tc["es_min_tokens"]:
            old = min((v for t, v in self.vals if t <= tokens - tc["es_window_tokens"] + self.slack), default=None)
            if old is not None and old - self.best < tc["es_min_delta"]:
                return f"converged: best validation loss improved < {tc['es_min_delta']} nats over the " \
                       f"trailing {tc['es_window_tokens'] / 1e9:g}B tokens", False
        return None, False


def read_jsonl_upto(path, step):
    """Records with step <= `step`, last one per step (after a crash, entries past the resume point are
    superseded by the rerun; the raw file is never rewritten)."""
    if not Path(path).exists():
        return []
    by = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["step"] <= step:
            by[r["step"]] = r
    return [by[k] for k in sorted(by)]


# ----------------------------------------------------------------------------- run record
def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def segments_summary(run_dir):
    p = run_dir / "segments.jsonl"
    segs = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    return {"wall_seconds": sum(s["seconds"] for s in segs), "gpu_hours": sum(s["gpu_hours"] for s in segs),
            "peak_mem_gb": max((s["peak_mem_gb"] for s in segs), default=None), "n_segments": len(segs)}


def write_record(run_dir, rc, model, step, extra):
    vals = read_jsonl_upto(run_dir / "val.jsonl", step)
    best = min(vals, key=lambda v: v["val_loss"]) if vals else None
    fz = rc["frozen"]
    tps = rc["train"]["tokens_per_step"]
    rec = {"run_id": rc["run_id"], "family": fz["family"], "scale_label": fz["scale"],
           "target_total_params": fz["target_total_params"], "actual_total_params": model.param_counts()["total"],
           "embedding_params": model.param_counts()["embedding"],
           "nonembedding_params": model.param_counts()["nonembedding"],
           "tokenizer_id": fz["tokenizer_id"], "tokenizer_revision": rc["data"]["tokenizer_revision"],
           "data_manifest_sha256": rc["data"]["manifest_sha256"], "config_sha256": fz["config_sha256"],
           "seed": rc["train"]["seed"], "data_seed": rc["train"]["data_seed"], "lr": rc["train"]["lr"],
           "status": extra.get("status", "running"), "stop_reason": extra.get("stop_reason"),
           "optimizer_steps": step, "training_tokens": step * tps,
           "schedule_total_steps": rc["train"]["total_steps"],
           "schedule_finished": step >= rc["train"]["total_steps"],
           "best_valid_loss_nats_per_token": best["val_loss"] if best else None,
           "best_valid_tokens": best["tokens"] if best else None,
           "selected_checkpoint_tokens": None, "test_loss_nats_per_token": None,
           "flops_per_token_train": fz["flops_per_token_train"], "flops_method": fz["flops_method"],
           "training_flops": fz["flops_per_token_train"] * step * tps,
           "checkpoints": sorted(p.name for p in run_dir.glob("ckpt_*.pt")),
           "gpu_type": gpu_name(), "world_size": extra.get("world"), **segments_summary(run_dir),
           "code_sha256": rc["code_sha256"], "git_commit": rc["git_commit"],
           "data_order_fingerprint": rc["data"].get("order_fingerprint"), "updated": time.ctime()}
    write_json(run_dir / "record.json", rec)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--max-steps", type=int, default=0, help="pilot: stop after N steps (no DONE)")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.runs_dir) / args.run_id
    rc = read_json(run_dir / "run_config.json")
    tc, fz = rc["train"], rc["frozen"]
    rank, world, device = init_dist()
    if (run_dir / "DONE").exists():
        if rank == 0:
            print(f"{args.run_id}: DONE already")
        return
    tok = TOKENIZERS[fz["tokenizer"]]
    data_dir, bos = tok["data_dir"], tok["bos"]

    torch.manual_seed(tc["seed"])
    random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mcfg = ModelConfig.from_dict({**fz["arch"], **rc["runtime"]})
    assert ModelConfig.from_dict(fz["arch"]).arch() == fz["arch"]
    with torch.device(device):
        model = build_model(mcfg)
    assert model.param_counts()["total"] == fz["actual_total_params"], "instantiated count != frozen registry"
    opt = torch.optim.AdamW(model.param_groups(tc["weight_decay"]), lr=tc["lr"], betas=tuple(tc["betas"]),
                            eps=tc["adam_eps"], fused=device.type == "cuda")

    start_step = 0
    resume = run_dir / "resume.pt"
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_step = ck["step"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if rank == 0:
            print(f"resumed from {resume} at step {start_step}", flush=True)
        del ck
    elif rank == 0 and (run_dir / "train.jsonl").exists():
        # no resume state but logs exist: a crash before the first resume checkpoint; restart cleanly
        for name in ("train.jsonl", "val.jsonl"):
            p = run_dir / name
            p.rename(run_dir / f"{name}.aborted_{int(time.time())}")
    barrier()

    if not args.no_compile and device.type == "cuda":
        compile_model(model)
    ddp = model
    if is_dist():
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)

    total = tc["total_steps"]
    stop_at = min(total, start_step + args.max_steps) if args.max_steps else total
    tps = tc["tokens_per_step"]
    stream = TrainStream(data_dir, tc["data_seed"], tc["batch_seqs"], tc["micro_batch"], rank, world, start_step,
                         total, device, bos)
    if rank == 0 and "order_fingerprint" not in rc["data"]:
        rc["data"]["order_fingerprint"] = order_fingerprint(stream.order)
        write_json(run_dir / "run_config.json", rc)
    accum = stream.accum

    vals = read_jsonl_upto(run_dir / "val.jsonl", start_step)
    vals = [(v["tokens"], v["val_loss"]) for v in vals]
    mon = Monitor(tc, vals)
    deadline = job_deadline(rank, device)
    term = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: term.__setitem__("flag", True))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seg_t0 = t_ckpt = t_log = time.time()
    tok_log = 0
    if rank == 0:
        write_json(run_dir / "status.json", {"status": "running", "step": start_step, "updated": time.ctime()})

    def log(path, rec):
        if rank == 0:
            with open(run_dir / path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def validate(step):
        vl, n = evaluate(model, data_dir, "validation", tc["eval_micro_batch"], rank, world, device, bos)
        rec = {"step": step, "tokens": step * tps, "val_loss": vl, "val_tokens": n, "time": time.time()}
        log("val.jsonl", rec)
        if rank == 0:
            print(json.dumps(rec), flush=True)
        return vl

    def save_full(path, step):
        if rank == 0:
            atomic_save({"model": model.state_dict(), "optim": opt.state_dict(), "step": step, "tokens": step * tps,
                         "data_position": {"step": step, "data_seed": tc["data_seed"],
                                           "windows_consumed": step * tc["batch_seqs"]},
                         "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                                 "python": random.getstate()},
                         "run_id": rc["run_id"], "config_sha256": fz["config_sha256"]}, path)
        barrier()

    def segment(step, how):
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        peak_t = torch.tensor([peak], device=device)
        if is_dist():
            dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
        sec = time.time() - seg_t0
        log("segments.jsonl", {"job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
                               "end": time.ctime(), "seconds": sec, "world": world, "gpu": gpu_name(),
                               "gpu_hours": sec * world / 3600, "peak_mem_gb": float(peak_t.item()),
                               "steps": [start_step, step], "how": how})

    def finish(step, status, reason):
        if rank == 0:
            write_record(run_dir, rc, model, step, {"status": status, "stop_reason": reason, "world": world})
            write_json(run_dir / "status.json", {"status": status, "stop_reason": reason, "step": step,
                                                 "tokens": step * tps, "updated": time.ctime()})
            (run_dir / "DONE").write_text(time.ctime())
            resume.unlink(missing_ok=True)
        barrier()

    if start_step == 0 and not vals:
        validate(0)
    model.train()
    step = start_step
    stop_reason, failed = None, False
    while step < stop_at:
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        for a, (z, y, seg, m) in enumerate(stream.next_step()):
            sync = a == accum - 1
            ctx = ddp.no_sync() if (is_dist() and not sync) else contextlib.nullcontext()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = ddp(z, seg, y, m) / accum
                loss.backward()
            loss_acc += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_log += tps

        if step % tc["log_every"] == 0 or step == stop_at:
            all_sum(loss_acc)
            dt = time.time() - t_log
            rec = {"step": step, "tokens": step * tps, "loss": loss_acc.item() / world, "lr": lr,
                   "grad_norm": float(gnorm), "tok_per_s": tok_log / dt,
                   "mfu": tok_log / dt * fz["flops_per_token_train"] / (world * H200_BF16_PEAK)}
            log("train.jsonl", rec)
            if rank == 0 and step % (tc["log_every"] * 20) == 0:
                print(json.dumps(rec), flush=True)
            t_log, tok_log = time.time(), 0
            if not math.isfinite(rec["loss"]):
                stop_reason, failed = "diverged: non-finite training loss", True

        is_val = crossed(step, tps, tc["val_every_tokens"]) or step == total
        if is_val and not failed:
            vl = validate(step)
            r, failed = mon.update(step * tps, vl)
            stop_reason = stop_reason or r
            if rank == 0:
                write_record(run_dir, rc, model, step, {"status": "running", "world": world})
            t_log, tok_log = time.time(), 0
        name = None
        if crossed(step, tps, tc["ckpt_every_tokens"]):      # interval checkpoint (also the 100B final one)
            name = ckpt_name(step * tps // tc["ckpt_every_tokens"] * tc["ckpt_every_tokens"])
        elif step == total:
            name = ckpt_name(tc["budget_tokens"])
        elif stop_reason and not failed:                      # converged early: stopping checkpoint
            name = ckpt_name(step * tps)
        if name:
            save_full(run_dir / name, step)
            log("checkpoints.jsonl", {"name": name, "step": step, "tokens": step * tps, "time": time.time()})
            t_log, tok_log = time.time(), 0
        if stop_reason:
            break

        near_limit = rank == 0 and deadline is not None and time.time() > deadline
        if bcast_flag(near_limit or term["flag"], device) and step < stop_at:
            save_full(resume, step)
            segment(step, "requeue")
            stream.close()
            if rank == 0:
                write_record(run_dir, rc, model, step, {"status": "running", "world": world})
                (run_dir / "REQUEUE").write_text(str(step))
                print(f"time limit / SIGTERM at step {step}: saved resume.pt, exiting for requeue", flush=True)
            barrier()
            return
        if bcast_flag(rank == 0 and time.time() - t_ckpt > tc["resume_every_min"] * 60, device):
            save_full(resume, step)
            t_ckpt = t_log = time.time()
            tok_log = 0

    stream.close()
    segment(step, "end")
    if failed:
        finish(step, "failed", stop_reason)
    elif stop_reason:
        finish(step, "completed", stop_reason)
    elif step >= total:
        finish(step, "completed", None)
    else:                                  # --max-steps pilot: keep a resume point (exercises auto-resume)
        save_full(resume, step)
        if rank == 0:
            write_record(run_dir, rc, model, step, {"status": "pilot", "world": world})
    if is_dist():
        dist.destroy_process_group()


def job_deadline(rank, device, margin_min=10):
    """Unix time to checkpoint and exit before the Slurm time limit (None outside Slurm)."""
    t = 0.0
    if rank == 0 and os.environ.get("SLURM_JOB_ID"):
        try:
            from datetime import datetime
            end = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%e"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            t = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S").timestamp() - margin_min * 60
        except Exception as e:
            print(f"could not read job end time: {e}", flush=True)
    tt = torch.tensor([t], dtype=torch.float64, device=device)
    if is_dist():
        dist.broadcast(tt, 0)
    return tt.item() or None


if __name__ == "__main__":
    main()
