"""Llama-3.2 tokenization of the same FineWeb-Edu documents (DETAILS.md §4.3). CPU partitions only.

Stage 1 (array job, slurm/cpu_tokenize.sbatch):
  python tokenize_llama3.py shard --worker-id i --num-workers n
    per raw parquet file -> data/llama3_tok/<stem>.{bin (uint32), lens.npy, done (json with sha256)}
    Same rules as Llama-2 (data.encode_batch): BOS + document, special ids stripped, no EOS.
Stage 2 (array job):
  python tokenize_llama3.py pack --worker-id i --num-workers n   then   python tokenize_llama3.py finalize
    packs documents into data/llama3/<split>.bin in the *Llama-2 split plan's* document order, so held-out
    sets are the same raw documents (data/manifests/heldout_docs.json) and train keeps the same document order.
    Then `python data.py manifest --tok llama3`.

Gated tokenizer: HF_TOKEN from the repo's .env (slurm/env.sh). sample-100BT yields ~97.4B Llama-3.2 tokens
(< 100B target); accepted by the user on 2026-10-20, so finalize reports it rather than failing.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import DATA, TOKENIZERS, sha256_file, write_json
from data import RAW, encode_batch, get_tokenizer, load_index, split_plan

TOK = DATA / "llama3_tok"
OUT = DATA / "llama3"
L2_TOK = DATA / "llama2_tok"
DT = np.uint32


def shard(worker, n, tokenizer_id=None):
    TOK.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id) if tokenizer_id else get_tokenizer("llama3")
    info = TOKENIZERS["llama3"]
    assert len(tok) == info["vocab"] and tok.bos_token_id == info["bos"], "unexpected Llama-3.2 tokenizer"
    import pyarrow.parquet as pq
    files = sorted(RAW.glob("*.parquet"))
    for i, path in enumerate(files):
        if i % n != worker or (TOK / f"{path.stem}.done").exists():
            continue
        t0, lens = time.time(), []
        tmp = TOK / f"{path.stem}.bin.tmp"
        with open(tmp, "wb") as f:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
                for a in encode_batch(tok, batch.column(0).to_pylist()):
                    doc = np.empty(len(a) + 1, dtype=DT)
                    doc[0] = info["bos"]
                    doc[1:] = a
                    f.write(doc.tobytes())
                    lens.append(len(doc))
        n_l2 = len(np.load(L2_TOK / f"{path.stem}.lens.npy", mmap_mode="r"))
        assert len(lens) == n_l2, f"{path.stem}: {len(lens)} docs != {n_l2} Llama-2 docs (document identity)"
        os.replace(tmp, TOK / f"{path.stem}.bin")
        np.save(TOK / f"{path.stem}.lens.npy", np.asarray(lens, dtype=np.int64))
        write_json(TOK / f"{path.stem}.done", {"docs": len(lens), "tokens": int(sum(lens)), "sec": time.time() - t0,
                                               "sha256": sha256_file(TOK / f"{path.stem}.bin"),
                                               "tokenizer": tokenizer_id or info["hf_id"]})
        print(f"{path.stem}: {len(lens)} docs, {sum(lens) / 1e6:.1f}M tokens, {time.time() - t0:.0f}s", flush=True)


def plan():
    """Llama-2 plan (document identity/order) with Llama-3.2 lengths."""
    stems2, lens2, fid, rows = load_index(L2_TOK)
    splits = split_plan(lens2)
    stems3 = sorted(p.name[:-5] for p in TOK.glob("*.done"))
    assert stems3 == stems2, "stage 1 incomplete"
    lens3 = np.concatenate([np.load(TOK / f"{s}.lens.npy") for s in stems3])
    starts3 = np.concatenate([np.concatenate([[0], np.cumsum(np.load(TOK / f"{s}.lens.npy"))[:-1]]) for s in stems3])
    return stems3, lens3, fid, starts3, splits


def write_docs(ids, stems, lens, fid, starts, out_arr, offset):
    dest = offset + np.concatenate([[0], np.cumsum(lens[ids])[:-1]])
    order = np.lexsort((starts[ids], fid[ids]))
    mm = {}
    for j in order:
        d = ids[j]
        f = int(fid[d])
        if f not in mm:
            mm = {f: np.memmap(TOK / f"{stems[f]}.bin", dtype=DT, mode="r")}
        s, n = int(starts[d]), int(lens[d])
        out_arr[dest[j]:dest[j] + n] = mm[f][s:s + n]


def pack(worker, n):
    OUT.mkdir(parents=True, exist_ok=True)
    stems, lens, fid, starts, splits = plan()
    if worker == 0:
        meta = {"source": "HuggingFaceFW/fineweb-edu sample/100BT", "tokenizer": TOKENIZERS["llama3"]["hf_id"],
                "dtype": "uint32", "split_plan": "Llama-2 plan (data.split_plan, seed 1234): same documents and order",
                "splits": {}}
        for name, ids in splits.items():
            ntok = int(lens[ids].sum())
            meta["splits"][name] = {"docs": int(len(ids)), "tokens": ntok}
            np.save(OUT / f"{name}.docs.npy", np.concatenate([[0], np.cumsum(lens[ids])]).astype(np.int64))
            if name != "train":
                arr = np.memmap(OUT / f"{name}.bin", dtype=DT, mode="w+", shape=(ntok,))
                write_docs(ids, stems, lens, fid, starts, arr, 0)
                arr.flush()
        write_json(OUT / "meta.json", meta)
    ids = splits["train"]
    ntr = int(lens[ids].sum())
    fd = os.open(OUT / "train.bin", os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != ntr * 4:
        os.ftruncate(fd, ntr * 4)
    os.close(fd)
    arr = np.memmap(OUT / "train.bin", dtype=DT, mode="r+", shape=(ntr,))
    b = np.linspace(0, len(ids), n + 1).astype(np.int64)
    lo, hi = b[worker], b[worker + 1]
    if (OUT / f"train.part{worker}.done").exists():
        return
    off = int(lens[ids[:lo]].sum())
    for c in range(lo, hi, 2_000_000):
        chunk = ids[c:min(c + 2_000_000, hi)]
        write_docs(chunk, stems, lens, fid, starts, arr, off)
        off += int(lens[chunk].sum())
    arr.flush()
    (OUT / f"train.part{worker}.done").write_text(time.ctime())


def finalize(n):
    missing = [i for i in range(n) if not (OUT / f"train.part{i}.done").exists()]
    assert not missing, f"missing parts {missing}"
    tr = np.memmap(OUT / "train.bin", dtype=DT, mode="r")
    docs = np.load(OUT / "train.docs.npy")
    assert docs[-1] == len(tr)
    bos = TOKENIZERS["llama3"]["bos"]
    for i in np.random.default_rng(0).integers(0, len(docs) - 1, 1000):
        assert tr[docs[i]] == bos
    assert int(tr[:10_000_000].max()) < TOKENIZERS["llama3"]["vocab"]
    ntr = len(tr)
    status = "OK" if ntr >= 100e9 else f"SHORT: {ntr / 1e9:.2f}B < 100B training tokens"
    (OUT / "DONE").write_text(f"{time.ctime()} {status}")
    print(status)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shard", "pack", "finalize"])
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--tokenizer-id", default=None, help="override (e.g. a verified local copy)")
    a = ap.parse_args()
    if a.cmd == "shard":
        shard(a.worker_id, a.num_workers, a.tokenizer_id)
    elif a.cmd == "pack":
        pack(a.worker_id, a.num_workers)
    else:
        finalize(a.num_workers)
