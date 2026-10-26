"""FineWeb-Edu data: tokenization rules, document-level splits, packed windows, train/eval streams, manifests.

Migrated from ../tmlr_paper/gdlm/data/{tokenize,build_splits,loader}.py (same rules, so the moved Llama-2
data stays readable and the Llama-3.2 data is built the same way), extended to uint32 token ids.

Format of a packed split <dir>/<split>.bin: token ids (uint16 for Llama-2, uint32 for Llama-3.2), documents
concatenated in shuffled order, each document = BOS + tokens, no EOS. <split>.docs.npy holds int64 document
start offsets (length n_docs + 1). Window i of a split is tokens[i*T : i*T + T + 1].

Within a window, attention never crosses documents (segments start at each BOS and at the window start).
The loss mask drops window position 0, positions whose input is BOS, and targets that are BOS.

  python data.py manifest --tok llama2      # sha256 checksums + split definitions -> data/manifests/llama2.json
  python data.py heldout                    # held-out raw document ids -> data/manifests/heldout_docs.json
"""

import argparse
import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from common import DATA, MANIFESTS, SEQ_LEN, TOKENIZERS, read_json, sha256_file, write_json

RAW = DATA / "raw" / "fineweb_edu"
SPLIT_SEED = 1234                     # document shuffle of the old build_splits.py
# held-out splits in shuffled-document order (old names); roles in the new pipeline in SPLIT_ROLES
HELD_OUT = [("val_small", 5_000_000), ("val", 50_000_000), ("analysis", 20_000_000)]
SPLIT_ROLES = {"validation": "analysis", "test": "val", "reserved": "val_small", "train": "train"}


# ----------------------------------------------------------------------------- tokenization
def get_tokenizer(tok="llama2"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZERS[tok]["hf_id"])


def special_ids(tokenizer):
    return {i for i in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id,
                        tokenizer.pad_token_id) if i is not None}


def encode_batch(tokenizer, texts, drop=None):
    """Document token ids. Special-token strings in the text are not parsed as specials, and special ids
    never appear inside a document (old rule: ids 0/1/2 stripped for Llama-2)."""
    drop = np.array(sorted(drop if drop is not None else special_ids(tokenizer)), dtype=np.int64)
    try:
        ids = tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    except TypeError:
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    out = []
    for x in ids:
        a = np.asarray(x, dtype=np.int64)
        out.append(a[~np.isin(a, drop)])
    return out


# ----------------------------------------------------------------------------- split plan (old build_splits)
def load_index(tok_dir):
    """Global document index over tokenized parquet files (sorted stem order, then row order)."""
    stems = sorted(p.name[:-5] for p in Path(tok_dir).glob("*.done"))
    lens, file_id, rows = [], [], []
    for i, s in enumerate(stems):
        l = np.load(Path(tok_dir) / f"{s}.lens.npy").astype(np.int64)
        lens.append(l)
        file_id.append(np.full(len(l), i, dtype=np.int32))
        rows.append(np.arange(len(l), dtype=np.int64))
    return stems, np.concatenate(lens), np.concatenate(file_id), np.concatenate(rows)


def split_plan(lens, seed=SPLIT_SEED):
    """Same rule as the old build_splits.plan: one permutation of the global document index; held-out
    splits take the first documents until their token targets are reached; train is the rest.
    Returns {split: array of global document indices in packing order}."""
    perm = np.random.default_rng(seed).permutation(len(lens))
    cum = np.cumsum(lens[perm])
    splits, a = {}, 0
    for name, n_tok in HELD_OUT:
        b = int(np.searchsorted(cum, cum[a - 1] + n_tok if a > 0 else n_tok)) + 1
        splits[name] = perm[a:b]
        a = b
    splits["train"] = perm[a:]
    return splits


def export_heldout(tok_dir=DATA / "llama2_tok", out=MANIFESTS / "heldout_docs.json"):
    """Raw-document identity of the held-out splits: parquet stem, row and FineWeb `id` per document,
    in packing order. Checks the plan against the moved Llama-2 split files."""
    import pyarrow.parquet as pq
    stems, lens, file_id, rows = load_index(tok_dir)
    splits = split_plan(lens)
    d2 = TOKENIZERS["llama2"]["data_dir"]
    res = {"source": "HuggingFaceFW/fineweb-edu sample/100BT", "raw_dir": str(RAW.resolve()),
           "split_seed": SPLIT_SEED, "n_raw_docs": int(len(lens)), "roles": SPLIT_ROLES, "splits": {}}
    by_file = {}
    for name in ("val_small", "val", "analysis"):
        ids = splits[name]
        starts = np.load(d2 / f"{name}.docs.npy")
        assert len(starts) == len(ids) + 1 and starts[-1] == lens[ids].sum(), f"{name}: plan != moved split files"
        assert (np.diff(starts) == lens[ids]).all(), f"{name}: document lengths differ from the plan"
        for gi in ids:
            by_file.setdefault(int(file_id[gi]), []).append(int(rows[gi]))
    raw_ids = {}
    for f, rs in by_file.items():
        col = pq.read_table(RAW / f"{stems[f]}.parquet", columns=["id"]).column(0)
        for r in rs:
            raw_ids[(f, r)] = col[r].as_py()
    for name in ("val_small", "val", "analysis"):
        ids = splits[name]
        docs = [{"file": stems[file_id[g]], "row": int(rows[g]), "id": raw_ids[(int(file_id[g]), int(rows[g]))]}
                for g in ids]
        res["splits"][name] = {"role": [k for k, v in SPLIT_ROLES.items() if v == name][0], "n_docs": len(docs),
                               "llama2_tokens": int(lens[ids].sum()), "docs": docs}
    res["train"] = {"n_docs": int(len(splits["train"])), "llama2_tokens": int(lens[splits["train"]].sum()),
                    "rule": "all remaining documents, in split_plan order"}
    write_json(out, res, indent=None)
    print(f"wrote {out}: " + ", ".join(f"{k} {v['n_docs']} docs" for k, v in res["splits"].items()))


# ----------------------------------------------------------------------------- packed token files
def split_file(data_dir, split):
    """Logical split name (train/validation/test/reserved) -> file stem (old split names are kept on disk)."""
    return SPLIT_ROLES.get(split, split)


class TokenFile:
    def __init__(self, data_dir, split, dtype=None):
        data_dir = Path(data_dir)
        stem = split_file(data_dir, split)
        self.path = data_dir / f"{stem}.bin"
        dtype = dtype or dtype_of(data_dir)
        self.tokens = np.memmap(self.path, dtype=np.dtype(dtype), mode="r")
        docs = data_dir / f"{stem}.docs.npy"
        self.doc_starts = np.load(docs) if docs.exists() else None

    def __len__(self):
        return self.tokens.shape[0]

    def n_windows(self, T=SEQ_LEN):
        return (len(self) - 1) // T

    def window(self, i, T=SEQ_LEN):
        return np.asarray(self.tokens[i * T: i * T + T + 1], dtype=np.int64)

    def windows(self, idx, T=SEQ_LEN):
        return np.stack([self.window(int(i), T) for i in idx])


def dtype_of(data_dir):
    """Token dtype of a packed data dir: manifest > meta.json > uint16 (the Llama-2 data predates both)."""
    data_dir = Path(data_dir)
    for p in (data_dir / "meta.json",):
        if p.exists():
            d = read_json(p)
            if "dtype" in d:
                return d["dtype"]
    for tok, info in TOKENIZERS.items():
        if Path(info["data_dir"]).resolve() == data_dir.resolve():
            return info["dtype"]
    return "uint16"


def segments_from_bos(z, bos_id):
    """Segment ids for a packed window: a new segment starts at every BOS and at position 0."""
    new = (z == bos_id).clone()
    new[:, 0] = True
    return torch.cumsum(new.int(), dim=1)


def segment_positions(seg):
    """Position of each token inside its segment."""
    B, T = seg.shape
    idx = torch.arange(T, device=seg.device).expand(B, T)
    start = torch.ones_like(seg, dtype=torch.bool)
    start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    first = torch.where(start, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return idx - first


def loss_mask_for(z, y, bos_id):
    """Drop window position 0, positions whose input is BOS (no in-document example yet), BOS targets."""
    m = ((z != bos_id) & (y != bos_id)).clone()
    m[:, 0] = False
    return m


def to_batch(arr, device, bos_id, non_blocking=True):
    """[B, T+1] int64 numpy -> (z, y, seg, loss_mask) on device."""
    x = torch.from_numpy(arr)
    x = x.pin_memory().to(device, non_blocking=non_blocking) if device.type == "cuda" else x.to(device)
    z, y = x[:, :-1], x[:, 1:]
    return z, y, segments_from_bos(z, bos_id), loss_mask_for(z, y, bos_id)


# ----------------------------------------------------------------------------- training stream
def window_order(n_windows, data_seed):
    """Training window order: a fixed permutation of all windows of train.bin, seeded by the scale's
    data_seed (independent of the model seed). Identical for every family and seed within a scale."""
    return np.random.default_rng(data_seed).permutation(n_windows)


def order_fingerprint(order, n=1_000_000):
    return hashlib.sha256(np.ascontiguousarray(order[:n], dtype=np.int64).tobytes()).hexdigest()


class TrainStream:
    """Deterministic micro-batch iterator with a prefetch thread.

    Global step s uses order[s*B : (s+1)*B]; accumulation index a and rank r read the slice
    s*B + (a*world + r)*mb + [0, mb). Resuming at step s is exact and independent of the world size as
    long as B is divisible by mb*world."""

    def __init__(self, data_dir, data_seed, batch_seqs, micro_batch, rank, world, start_step, total_steps,
                 device, bos_id, T=SEQ_LEN, prefetch=6):
        self.f = TokenFile(data_dir, "train")
        self.B, self.mb, self.rank, self.world, self.T = batch_seqs, micro_batch, rank, world, T
        assert batch_seqs % (micro_batch * world) == 0, "batch_seqs must divide by micro_batch*world"
        self.accum = batch_seqs // (micro_batch * world)
        self.order = window_order(self.f.n_windows(T), data_seed)
        need = total_steps * batch_seqs
        if need > len(self.order):    # data availability is confirmed for the whole schedule up front
            raise RuntimeError(f"schedule needs {need} windows, train split has {len(self.order)}")
        self.device, self.bos_id = device, bos_id
        self.step = start_step
        self.q = queue.Queue(maxsize=prefetch)
        self._stop = False
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def _worker(self):
        s = self.step
        while not self._stop:
            for a in range(self.accum):
                base = s * self.B + (a * self.world + self.rank) * self.mb
                self.q.put(self.f.windows(np.sort(self.order[base:base + self.mb]), self.T))
            s += 1

    def next_step(self):
        """List of `accum` micro-batches for the next optimizer step."""
        out = [to_batch(self.q.get(), self.device, self.bos_id) for _ in range(self.accum)]
        self.step += 1
        return out

    def close(self):
        self._stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


# ----------------------------------------------------------------------------- evaluation windows
def eval_windows(data_dir, split, T=SEQ_LEN):
    """All non-overlapping windows of a held-out split. The last, partial window is padded."""
    f = TokenFile(data_dir, split)
    starts = np.arange(0, len(f) - 1, T)
    return f, starts


def eval_window_batch(f, starts, T=SEQ_LEN, pad_id=0):
    """Windows at the given token offsets: [B, T+1] tokens and a validity mask (padding is invalid)."""
    n = len(f)
    arr = np.full((len(starts), T + 1), pad_id, dtype=np.int64)
    valid = np.zeros((len(starts), T + 1), dtype=bool)
    for b, s in enumerate(starts):
        e = min(int(s) + T + 1, n)
        arr[b, :e - s] = f.tokens[s:e]
        valid[b, :e - s] = True
    return arr, valid


def eval_batch(f, starts, device, bos_id, T=SEQ_LEN):
    """(z, y, seg, loss_mask) for evaluation windows; padded targets are masked out."""
    arr, valid = eval_window_batch(f, starts, T)
    z, y, seg, m = to_batch(arr, device, bos_id)
    return z, y, seg, m & torch.from_numpy(valid[:, 1:]).to(device)


# ----------------------------------------------------------------------------- manifests
def build_manifest(tok, out=None):
    """Checksums, token counts and split definitions of a packed data dir."""
    info = TOKENIZERS[tok]
    d = Path(info["data_dir"])
    meta = read_json(d / "meta.json") if (d / "meta.json").exists() else {}
    files = {}
    for p in sorted(d.iterdir()):
        if p.is_symlink() or not p.is_file() or p.suffix not in (".bin", ".npy"):
            continue
        t0 = time.time()
        files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        print(f"{p.name}: {files[p.name]['sha256'][:16]}... ({time.time() - t0:.0f}s)", flush=True)
    splits = {}
    for role, stem in SPLIT_ROLES.items():
        if not (d / f"{stem}.bin").exists():
            continue
        n_tok = (d / f"{stem}.bin").stat().st_size // np.dtype(info["dtype"]).itemsize
        n_docs = len(np.load(d / f"{stem}.docs.npy", mmap_mode="r")) - 1
        splits[role] = {"file": f"{stem}.bin", "tokens": int(n_tok), "docs": int(n_docs),
                        "windows_2048": int((n_tok - 1) // SEQ_LEN)}
    tk = get_tokenizer(tok)
    man = {"tokenizer": tok, "tokenizer_id": info["hf_id"], "tokenizer_vocab": len(tk),
           "tokenizer_revision": tokenizer_revision(tk), "bos_id": info["bos"], "eos_inserted": False,
           "dtype": info["dtype"], "source": meta.get("source", "HuggingFaceFW/fineweb-edu sample/100BT"),
           "split_seed": SPLIT_SEED, "split_rule": "document-level, see data.split_plan",
           "packing": "BOS + document, no EOS, concatenated; windows of 2048+1; document-masked attention",
           "heldout_docs": "data/manifests/heldout_docs.json", "splits": splits, "files": files,
           "created": time.ctime()}
    man["manifest_sha256"] = hashlib.sha256(json.dumps(
        {k: man[k] for k in ("tokenizer_id", "dtype", "splits", "files")}, sort_keys=True).encode()).hexdigest()
    write_json(out or MANIFESTS / f"{tok}.json", man)
    print(f"wrote {out or MANIFESTS / f'{tok}.json'}")


def tokenizer_revision(tk):
    """sha256 of the tokenizer's serialized vocabulary/merges (stable across cache locations)."""
    try:
        return "tokenizer.json:sha256:" + hashlib.sha256(tk.backend_tokenizer.to_str().encode()).hexdigest()
    except Exception:
        return "vocab:sha256:" + hashlib.sha256(json.dumps(sorted(tk.get_vocab().items())).encode()).hexdigest()


def manifest_sha(tok):
    p = MANIFESTS / f"{tok}.json"
    return read_json(p)["manifest_sha256"] if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "heldout"])
    ap.add_argument("--tok", default="llama2", choices=list(TOKENIZERS))
    a = ap.parse_args()
    if a.cmd == "manifest":
        build_manifest(a.tok)
    else:
        export_heldout()


if __name__ == "__main__":
    main()
