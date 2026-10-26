"""Required correctness tests (DETAILS.md §5.3, main.tex App. E). Run: python -m pytest -q tests/

GPU-only tests (flex/bf16 exactness) are skipped without CUDA."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import FROZEN, read_json                                          # noqa: E402
from count_params import count                                                # noqa: E402
from data import loss_mask_for, segments_from_bos                             # noqa: E402
from models import Masks, ModelConfig, build_model                            # noqa: E402

BOS = 1
FP32_TOL = 1e-5      # relative, fp32 eager vs fp64 direct
BF16_TOL = 3e-2      # relative, bf16 autocast flex (GPU) vs fp64 direct; documented in STATUS.md


def tiny(family, impl="eager", **kw):
    base = dict(family=family, vocab=50, d=32, max_seq_len=64, rope_theta=10000.0, attn_impl=impl)
    if family in ("transformer", "gdlm"):
        base.update(n_layers=2, n_heads=4, n_kv_heads=4, head_dim=8, ffn=48)
    if family in ("gdlm", "wang"):
        base.update(gd_layers=3, gd_heads=2, gd_head_dim=8)
    if family == "wang":
        base.update(cov_dim=16)
    base.update(kw)
    torch.manual_seed(0)
    return build_model(ModelConfig(**base))


def batch(B=3, T=24, V=50, seed=1, docs=True):
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(3, V, (B, T + 1), generator=g)
    if docs:                                   # document boundaries inside the window
        z[0, 9] = BOS
        z[1, 0] = BOS
        z[1, 15] = BOS
    zi, y = z[:, :-1], z[:, 1:]
    return zi, y, segments_from_bos(zi, BOS)


def direct_prediction(model, z, seg):
    """Eq. (prop1) with eq. (kernel)/(alpha), written as explicit loops over queries t and examples i < t
    in the same document (fp64). Example k carries label y_k = z_{k+1}; query j uses k < j."""
    m = model.double()
    cfg = m.cfg
    with torch.no_grad():
        E = m.embed.weight - m.embed.weight.mean(0, keepdim=True)
        if cfg.family == "gdlm":
            phi = m.covariates(z, seg, E, Masks(seg, "eager"))
        else:
            from data import segment_positions
            phi = m.pos[segment_positions(seg)]
        B, T = z.shape
        f = torch.zeros(B, T, cfg.d, dtype=torch.float64)
        dh = cfg.gd_head_dim
        for b in range(B):
            for j in range(T):
                ks = [k for k in range(j) if seg[b, k] == seg[b, j]]
                if not ks:
                    continue
                for l in range(cfg.gd_layers):
                    for h in range(cfg.gd_heads):
                        W = m.gd.W_qk[l, h * dh:(h + 1) * dh]
                        Wv = m.gd.W_v[l, h * dh:(h + 1) * dh]
                        P = Wv.t() @ Wv
                        def u(k):
                            x = (W @ phi[b, k])[None, None, None]
                            if m.gd.rotary is None:
                                return x[0, 0, 0]
                            cos, sin = m.gd.rotary.cos[k].double(), m.gd.rotary.sin[k].double()
                            x1, x2 = x[0, 0, 0].chunk(2)
                            return x[0, 0, 0] * cos + torch.cat([-x2, x1]) * sin
                        uj = u(j)
                        kap = torch.stack([torch.exp(uj @ u(k) / math.sqrt(dh)) for k in ks])
                        alpha = kap / kap.sum()
                        ey = torch.stack([E[z[b, k + 1]] if k + 1 < T else E[0] for k in ks])
                        f[b, j] += P @ (alpha[:, None] * ey).sum(0)
    m.float()
    return f


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# ----------------------------------------------------------------------------- exactness
@pytest.mark.parametrize("family", ["gdlm", "wang"])
def test_gd_matches_direct_prop1(family):
    model = tiny(family)
    z, y, seg = batch()
    ref = direct_prediction(model, z, seg)
    m64 = model.double()
    with torch.no_grad():
        f64, _ = m64.predict(z, seg)
    assert rel(f64, ref) < 1e-10
    m32 = model.float()
    with torch.no_grad():
        f32, _ = m32.predict(z, seg)
    assert rel(f32.double(), ref) < FP32_TOL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (flex attention, bf16)")
@pytest.mark.parametrize("family", ["gdlm", "wang"])
def test_gd_flex_bf16_matches_direct(family):
    dims = dict(head_dim=16, gd_head_dim=16)            # flex attention needs head dim >= 16
    model = tiny(family, **dims)
    z, y, seg = batch(T=64)
    ref = direct_prediction(model, z, seg)
    mc = tiny(family, impl="flex", **dims)
    mc.load_state_dict(model.float().state_dict())
    mc = mc.cuda()
    with torch.no_grad():
        f32, _ = mc.predict(z.cuda(), seg.cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fbf, _ = mc.predict(z.cuda(), seg.cuda())
    assert rel(f32.double().cpu(), ref) < FP32_TOL * 10
    assert rel(fbf.double().cpu(), ref) < BF16_TOL


@pytest.mark.parametrize("family", ["gdlm", "wang", "transformer"])
def test_sdpa_path_matches_eager(family):
    model = tiny(family).double()
    z, _, seg = batch()
    ms = tiny(family, impl="sdpa").double()
    ms.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.allclose(model.logits(z, seg), ms.logits(z, seg), atol=1e-10)


# ----------------------------------------------------------------------------- causality
@pytest.mark.parametrize("family", ["gdlm", "wang", "transformer"])
def test_future_tokens_do_not_change_earlier_logits(family):
    model = tiny(family).double()
    z, _, _ = batch(docs=False)
    for t in (1, 7, 15, 23):
        z2 = z.clone()
        z2[:, t:] = (z2[:, t:] + 5) % 47 + 3          # change inputs at positions >= t (keeps them non-BOS)
        with torch.no_grad():
            a = model.logits(z, segments_from_bos(z, BOS))
            b = model.logits(z2, segments_from_bos(z2, BOS))
        assert torch.allclose(a[:, :t], b[:, :t], atol=1e-12), f"logits before {t} changed"


@pytest.mark.parametrize("family", ["gdlm", "wang"])
def test_prediction_at_t_independent_of_its_target(family):
    """The logits at position j (predicting y_j = z_{j+1}) must not change when y_j changes."""
    model = tiny(family).double()
    z, _, _ = batch(docs=False)
    zz = torch.cat([z, torch.full((z.shape[0], 1), 7)], dim=1)     # window with explicit targets
    for j in (0, 3, 10, 23):
        z2 = zz.clone()
        z2[:, j + 1] = (z2[:, j + 1] + 11) % 47 + 3
        with torch.no_grad():
            a = model.logits(zz[:, :j + 1], segments_from_bos(zz[:, :j + 1], BOS))[:, j]
            b = model.logits(z2[:, :j + 1], segments_from_bos(z2[:, :j + 1], BOS))[:, j]
            # and within a longer window that contains y_j as a later input
            c = model.logits(zz, segments_from_bos(zz, BOS))[:, j]
            d = model.logits(z2, segments_from_bos(z2, BOS))[:, j]
        assert torch.allclose(a, b, atol=1e-12) and torch.allclose(c, d, atol=1e-12)
        assert torch.allclose(a, c, atol=1e-10)


@pytest.mark.parametrize("family", ["gdlm", "wang"])
def test_first_position_is_uniform_and_documents_do_not_leak(family):
    model = tiny(family).double()
    z, _, seg = batch()
    with torch.no_grad():
        f, _ = model.predict(z, seg)
        logits = model.logits(z, seg)
    assert f[:, 0].abs().max() == 0 and logits[:, 0].abs().max() == 0       # no examples -> uniform
    assert f[0, 9].abs().max() == 0 and f[1, 15].abs().max() == 0           # document starts (BOS input)
    # changing the previous document leaves the next document's predictions unchanged
    z2 = z.clone()
    z2[0, :9] = (z2[0, :9] + 3) % 47 + 3
    with torch.no_grad():
        l2 = model.logits(z2, segments_from_bos(z2, BOS))
    assert torch.allclose(logits[0, 9:], l2[0, 9:], atol=1e-12)


def test_loss_mask_policy():
    z, y, seg = batch()
    m = loss_mask_for(z, y, BOS)
    assert not m[:, 0].any()
    assert not m[z == BOS].any() and not m[y == BOS].any()


# ----------------------------------------------------------------------------- tying / normalization
@pytest.mark.parametrize("family", ["gdlm", "wang", "transformer"])
def test_tied_weights_are_shared_parameters(family):
    model = tiny(family)
    names = [n for n, _ in model.named_parameters()]
    V, d = model.cfg.vocab, model.cfg.d
    vd = [n for n, p in model.named_parameters() if p.shape in ((V, d), (d, V))]
    assert vd == ["embed.weight"], f"untied output head? {vd}"
    assert not any(k in n for n in names for k in ("lm_head", "W_o", "W_q.", "W_k.", "wo_gd", "output"))
    if family != "transformer":
        # output map = W_v^T: the GD output changes when (only) W_v changes, and there is no other map
        assert {n for n in names if n.startswith("gd.")} == {"gd.W_qk", "gd.W_v"}
    # readout uses the embedding table: perturbing a row changes only that logit column's computation
    z, _, seg = batch(docs=False)
    with torch.no_grad():
        a = model.logits(z, seg)
        model.embed.weight[5] += 1.0
        b = model.logits(z, seg)
    assert not torch.allclose(a[..., 5], b[..., 5])


@pytest.mark.parametrize("family", ["gdlm", "wang"])
def test_zero_mean_embedding_and_gainless_readout(family):
    model = tiny(family).double()
    with torch.no_grad():
        model.embed.weight += 0.3                          # a raw table with nonzero mean
    E = model.centered_embedding()
    assert E.mean(0).abs().max() < 1e-12
    assert not any("norm" in n and not n.startswith(("blocks.", "final_norm")) for n, _ in model.named_parameters())
    z, _, seg = batch()
    with torch.no_grad():
        f, E = model.predict(z, seg)
        assert torch.allclose(model.readout(f, E), model.readout(3.7 * f, E), atol=1e-9)   # scale-only norm
        lg = model.readout(f, E)
        has = f.abs().sum(-1) > 0
        fn = f / f.pow(2).mean(-1, keepdim=True).add(model.cfg.readout_eps).sqrt()
        assert torch.allclose(lg[has], (fn @ E.t())[has], atol=1e-10)


def test_wang_kernel_weights_depend_on_position_only():
    model = tiny("wang").double()
    z, _, seg = batch(docs=False)
    perm = z[:, torch.randperm(z.shape[1])]
    z2 = torch.where(perm == BOS, 3, perm)
    with torch.no_grad():
        from data import segment_positions
        a = model.gd.kernel_weights(model.pos[segment_positions(seg)], Masks(seg, "eager"))
        seg2 = segments_from_bos(z2, BOS)
        b = model.gd.kernel_weights(model.pos[segment_positions(seg2)], Masks(seg2, "eager"))
    assert torch.equal(a, b)


def test_gdlm_kernel_weights_depend_on_content():
    model = tiny("gdlm").double()
    z, _, seg = batch(docs=False)
    z2 = z.clone()
    z2[:, 2] = (z2[:, 2] + 9) % 47 + 3
    with torch.no_grad():
        E = model.centered_embedding()
        a = model.gd.kernel_weights(model.covariates(z, seg, E, Masks(seg, "eager")), Masks(seg, "eager"))
        b = model.gd.kernel_weights(model.covariates(z2, seg, E, Masks(seg, "eager")), Masks(seg, "eager"))
    assert not torch.allclose(a, b)


def test_preconditioner_is_psd():
    model = tiny("gdlm").double()
    for l in range(model.cfg.gd_layers):
        for h in range(model.cfg.gd_heads):
            P = model.gd.preconditioner(l, h)
            assert torch.allclose(P, P.t()) and torch.linalg.eigvalsh(P).min() > -1e-10


# ----------------------------------------------------------------------------- registry / smoke
FROZEN_FILES = sorted(FROZEN.glob("*.json"))


@pytest.mark.parametrize("path", FROZEN_FILES, ids=[p.stem for p in FROZEN_FILES])
def test_instantiated_counts_equal_registry(path):
    r = read_json(path)
    c = count(ModelConfig.from_dict(r["arch"]))
    assert c["total"] == r["actual_total_params"] and c["embedding"] == r["embedding_params"]
    assert r["within_tolerance"]


@pytest.mark.parametrize("path", FROZEN_FILES, ids=[p.stem for p in FROZEN_FILES])
def test_forward_backward_smoke_every_scale(path):
    r = read_json(path)
    cfg = ModelConfig.from_dict(r["arch"])
    cfg.attn_impl = "eager"
    if r["scale"] == "1.2B":                        # meta device only (not trained)
        with torch.device("meta"):
            m = build_model(cfg)
            z = torch.randint(0, cfg.vocab, (1, 32))
            seg = torch.ones_like(z)
            loss = m(z, seg, z, torch.ones_like(z, dtype=torch.bool))
        assert loss.shape == ()
        return
    torch.manual_seed(0)
    m = build_model(cfg)
    T = 32
    z = torch.randint(3, cfg.vocab, (1, T + 1))
    zi, y = z[:, :-1], z[:, 1:]
    seg = segments_from_bos(zi, BOS)
    loss = m(zi, seg, y, loss_mask_for(zi, y, BOS))
    loss.backward()
    # GD models start above ln V: the gainless readout aligns f with in-context embeddings (copy logits ~0.02 d)
    assert math.isfinite(loss.item()) and loss.item() < 2 * math.log(cfg.vocab)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in m.unique_params())
