"""Brute-force checks of the recurrence features and the paired bootstrap bookkeeping."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import segments_from_bos                       # noqa: E402
from eval_recurrence import compute_features            # noqa: E402


def brute(z, seg):
    W, T = seg.shape
    out = {k: np.zeros((W, T), np.int64) for k in ("prior_count", "last_distance", "bigram_seen", "trigram_seen")}
    for w in range(W):
        for j in range(T):
            s = min(k for k in range(j + 1) if seg[w, k] == seg[w, j])
            ctx = list(z[w, s:j + 1])                     # visible context z_s..z_j
            y = z[w, j + 1]
            occ = [s + i for i, t in enumerate(ctx) if t == y]
            out["prior_count"][w, j] = len(occ)
            out["last_distance"][w, j] = (j + 1 - occ[-1]) if occ else 0
            out["bigram_seen"][w, j] = any(z[w, k - 1] == z[w, j] and z[w, k] == y for k in range(s + 1, j + 1))
            out["trigram_seen"][w, j] = j - 1 >= s and any(
                z[w, k - 2] == z[w, j - 1] and z[w, k - 1] == z[w, j] and z[w, k] == y for k in range(s + 2, j + 1))
    return out


def test_recurrence_features_match_brute_force():
    rng = np.random.default_rng(0)
    z = rng.integers(2, 6, size=(4, 41))                  # tiny vocab -> many repeats
    z[1, 10] = 1
    z[2, 0] = 1
    z[2, 25] = 1
    seg = segments_from_bos(torch.from_numpy(z[:, :-1]), 1).numpy()
    got = compute_features(z, seg, 8)
    ref = brute(z, seg)
    for k in ref:
        assert np.array_equal(got[k].astype(np.int64), ref[k]), k


def test_train_stream_resume_is_bit_identical():
    from data import TrainStream
    dev = torch.device("cpu")
    kw = dict(data_dir="data/llama2", data_seed=1234, batch_seqs=8, micro_batch=2, total_steps=10, device=dev, bos_id=1)
    full = TrainStream(rank=0, world=2, start_step=0, **kw)
    ref = [full.next_step() for _ in range(5)]
    full.close()
    res = TrainStream(rank=0, world=2, start_step=3, **kw)
    got = [res.next_step() for _ in range(2)]
    res.close()
    for a, b in zip(ref[3:], got):
        for (za, *_), (zb, *_) in zip(a, b):
            assert torch.equal(za, zb)
    # a different world size sees the same global batch (union over ranks)
    one = TrainStream(rank=0, world=1, start_step=3, **kw)
    s1 = torch.cat([z for z, *_ in one.next_step()])
    one.close()
    s2 = []
    for r in range(2):
        t = TrainStream(rank=r, world=2, start_step=3, **kw)
        s2 += [z for z, *_ in t.next_step()]
        t.close()
    key = lambda x: sorted(map(tuple, x.tolist()))
    assert key(s1) == key(torch.cat(s2))
