"""Model families: `gdlm`, `wang` (position-only GD) and `transformer` (Llama decoder).

Notation follows main.tex §3-4 and App. B. A window has inputs z_0..z_{T-1}; position j predicts y_j = z_{j+1}.

GD prediction (eq. prop1 with eq. kernel/alpha). Example k pairs the covariate phi_k with the label
y_k = z_{k+1}; the query at position j uses only examples k < j (strict, so y_j is never seen):

    f_j = sum_{l,h} W_lh^T  sum_{k<j} alpha^{lh}_{jk} W_lh^V e_{y_k},
    alpha^{lh}_{jk} = softmax_{k<j}( <R_j W_lh phi_j, R_k W_lh phi_k> / sqrt(d_h) )

with tied Q = K = W_lh (plus RoPE R) and tied V/O (O = W_lh^{V T}, no separate parameter). Every (l, h)
uses the same covariates and the same label embeddings; block outputs are summed and never fed back.
Positions with no in-document example (window start, document start) get f = 0, i.e. uniform logits.
The readout is a gainless RMS normalization of f followed by the tied zero-mean embedding table:
E_bar = E - mean_v(E) is recomputed every forward pass and is used everywhere (inputs, labels, readout).

  gdlm:        phi_k = h^{(L)}_k, final (normed) output of a causal Llama generator over z_{<=k}
               (= phi_i = h^{(L)}_{i-1} in the paper's 1-based example index i = k + 1).
  wang:        phi_k = P[pos_k], a learned position-only covariate (position within the document);
               no content mixing, no generator, no RoPE.
  transformer: pre-norm Llama decoder (RoPE, RMSNorm, SwiGLU, bias-free, tied embeddings; GQA if
               n_kv_heads < n_heads).
"""

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from data import segment_positions

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _flex_compiled = torch.compile(flex_attention, dynamic=False)
except Exception:  # pragma: no cover
    flex_attention = create_block_mask = _flex_compiled = None


@dataclass
class ModelConfig:
    family: str                     # gdlm | wang | transformer
    vocab: int
    d: int                          # embedding width (= covariate width for gdlm)
    max_seq_len: int = 2048
    # Llama decoder (transformer) / covariate generator (gdlm)
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 64
    ffn: int = 0
    rope_theta: float = 10000.0
    rope_scaling: dict = field(default_factory=dict)   # Llama-3 frequency scaling (1.2B)
    norm_eps: float = 1e-5
    # GD predictor (gdlm, wang)
    gd_layers: int = 0
    gd_heads: int = 0
    gd_head_dim: int = 64
    cov_dim: int = 0                # positional-covariate width (wang); gdlm uses d
    readout_eps: float = 1e-20      # only guards exactly-zero rows; f is ~1e-4 at init
    init_std: float = 0.02
    # runtime (not part of the frozen architecture)
    attn_impl: str = "flex"         # flex (training, fixed T) | sdpa (variable-length eval) | eager (tests)
    grad_ckpt: bool = False
    gd_chunk: int = 0               # GD layers per attention call (0 = all)

    ARCH_KEYS = ("family", "vocab", "d", "max_seq_len", "n_layers", "n_heads", "n_kv_heads", "head_dim", "ffn",
                 "rope_theta", "rope_scaling", "norm_eps", "gd_layers", "gd_heads", "gd_head_dim", "cov_dim",
                 "readout_eps", "init_std")

    def arch(self):
        d = asdict(self)
        return {k: d[k] for k in self.ARCH_KEYS}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ----------------------------------------------------------------------------- attention + masks
class Masks:
    """Document masks for one batch of windows. seg: [B, T] segment ids.
      causal: query j sees key k iff k <= j and seg[k] == seg[j]   (Llama attention)
      strict: query j sees key k iff k <  j and seg[k] == seg[j]   (GD examples i < t)"""

    def __init__(self, seg, impl):
        self.seg, self.impl, self._c = seg, impl, {}

    def has_examples(self):
        """[B, T] True where the strict mask row is non-empty (previous position in the same document)."""
        if "has" not in self._c:
            h = torch.zeros_like(self.seg, dtype=torch.bool)
            h[:, 1:] = self.seg[:, 1:] == self.seg[:, :-1]
            self._c["has"] = h
        return self._c["has"]

    def dense(self, kind):
        if kind not in self._c:
            T = self.seg.shape[1]
            tri = torch.ones(T, T, dtype=torch.bool, device=self.seg.device).tril(0 if kind == "causal" else -1)
            self._c[kind] = (tri & (self.seg[:, :, None] == self.seg[:, None, :]))[:, None]
        return self._c[kind]

    def sdpa_mask(self, kind):
        """Dense mask for SDPA. Empty strict rows (no examples) get their diagonal so SDPA stays finite;
        GDPredictor zeroes those rows afterwards (has_examples)."""
        key = "sdpa_" + kind
        if key not in self._c:
            m = self.dense(kind)
            if kind == "strict":
                T = self.seg.shape[1]
                eye = torch.eye(T, dtype=torch.bool, device=self.seg.device)
                m = m | (eye & ~self.has_examples()[:, :, None])[:, None]
            self._c[key] = m
        return self._c[key]

    def block(self, kind):
        key = "block_" + kind
        if key not in self._c:
            B, T = self.seg.shape
            seg = self.seg
            if kind == "causal":
                def mask_mod(b, h, q, k):
                    return (k <= q) & (seg[b, q] == seg[b, k])
            else:
                def mask_mod(b, h, q, k):
                    return (k < q) & (seg[b, q] == seg[b, k])
            self._c[key] = create_block_mask(mask_mod, B, None, T, T, device=seg.device)
        return self._c[key]

    def build(self, kinds):
        """Create masks eagerly (outside compiled regions)."""
        for k in kinds:
            self.block(k) if self.impl == "flex" else self.sdpa_mask(k) if self.impl == "sdpa" else self.dense(k)
        return self


def attend(q, k, v, masks, kind, scale=None):
    """Softmax attention; q [B, Hq, T, dh], k/v [B, Hkv, T, dh]. Empty rows give 0."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    gqa = q.shape[1] != k.shape[1]
    if masks.impl == "flex":
        fa = _flex_compiled if q.is_cuda else flex_attention
        return fa(q, k, v, block_mask=masks.block(kind), scale=scale, enable_gqa=gqa)
    if masks.impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=masks.sdpa_mask(kind), scale=scale, enable_gqa=gqa)
    if gqa:
        r = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    s = torch.matmul(q.to(dt), k.to(dt).transpose(-1, -2)) * scale
    m = masks.dense(kind)
    s = s.masked_fill(~m, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.where(m.any(-1, keepdim=True), p, torch.zeros_like(p))
    return torch.matmul(p, v.to(dt)).to(v.dtype)


# ----------------------------------------------------------------------------- building blocks
def upcast(x):
    """fp32 for half precision inputs, unchanged otherwise (fp64 exactness tests)."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = upcast(x)
        return (self.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))).to(x.dtype)


def llama3_inv_freq(inv, sc):
    """Llama-3 RoPE frequency scaling (HF `rope_type: llama3`)."""
    factor, lo, hi = sc["factor"], sc["low_freq_factor"], sc["high_freq_factor"]
    old = sc["original_max_position_embeddings"]
    lo_wl, hi_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    out = torch.where(wl > lo_wl, inv / factor, inv)
    smooth = (old / wl - lo) / (hi - lo)
    mid = (1 - smooth) * out / factor + smooth * out
    is_mid = (wl >= hi_wl) & (wl <= lo_wl)
    return torch.where(is_mid, mid, out)


class Rotary(nn.Module):
    """RoPE, HF/Llama rotate-half convention, positions = window index (relative, so documents are fine)."""

    def __init__(self, dh, max_len, theta=10000.0, scaling=None):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh))
        if scaling:
            inv = llama3_inv_freq(inv, scaling)
        f = torch.outer(torch.arange(max_len, dtype=torch.float64), inv)
        emb = torch.cat([f, f], dim=-1)
        self.register_buffer("cos", emb.cos().float(), persistent=False)
        self.register_buffer("sin", emb.sin().float(), persistent=False)

    def forward(self, x):                  # [B, H, T, dh]
        T = x.shape[2]
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def split_heads(x, H):                     # [B, T, H*dh] -> [B, H, T, dh]
    B, T, D = x.shape
    return x.view(B, T, H, D // H).transpose(1, 2)


def merge_heads(x):                        # [B, H, T, dh] -> [B, T, H*dh]
    B, H, T, dh = x.shape
    return x.transpose(1, 2).reshape(B, T, H * dh)


class SelfAttention(nn.Module):
    def __init__(self, cfg, rotary):
        super().__init__()
        d, H, Hkv, dh = cfg.d, cfg.n_heads, cfg.n_kv_heads or cfg.n_heads, cfg.head_dim
        self.H, self.Hkv = H, Hkv
        self.wq = nn.Linear(d, H * dh, bias=False)
        self.wk = nn.Linear(d, Hkv * dh, bias=False)
        self.wv = nn.Linear(d, Hkv * dh, bias=False)
        self.wo = nn.Linear(H * dh, d, bias=False)
        self.wo.residual_out = True
        self.rotary = rotary

    def forward(self, x, masks):
        q = self.rotary(split_heads(self.wq(x), self.H))
        k = self.rotary(split_heads(self.wk(x), self.Hkv))
        v = split_heads(self.wv(x), self.Hkv)
        return self.wo(merge_heads(attend(q, k, v, masks, "causal")))


class SwiGLU(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.w1 = nn.Linear(d, ffn, bias=False)    # gate
        self.w3 = nn.Linear(d, ffn, bias=False)    # up
        self.w2 = nn.Linear(ffn, d, bias=False)    # down
        self.w2.residual_out = True

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaBlock(nn.Module):
    def __init__(self, cfg, rotary):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d, cfg.norm_eps)
        self.attn = SelfAttention(cfg, rotary)
        self.mlp_norm = RMSNorm(cfg.d, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d, cfg.ffn)

    def forward(self, h, masks):
        h = h + self.attn(self.attn_norm(h), masks)
        return h + self.mlp(self.mlp_norm(h))


class GDPredictor(nn.Module):
    """sum over gd_layers x gd_heads kernels of one functional GD step (see module docstring).

    W_qk[l]: [H*dh, p] tied query/key map of layer l (one row block per head).
    W_v[l]:  [H*dh, d] value map; the output map is W_v[l]^T (same tensor, never a separate parameter)."""

    def __init__(self, cfg, cov_dim, rotary):
        super().__init__()
        L, H, dh = cfg.gd_layers, cfg.gd_heads, cfg.gd_head_dim
        self.L, self.H, self.dh, self.p, self.d = L, H, dh, cov_dim, cfg.d
        self.W_qk = nn.Parameter(torch.empty(L, H * dh, cov_dim))
        self.W_v = nn.Parameter(torch.empty(L, H * dh, cfg.d))
        self.rotary = rotary
        self.chunk = cfg.gd_chunk or L
        self.grad_ckpt = cfg.grad_ckpt

    def _chunk(self, phi, vals, masks, l0, l1):
        nH = (l1 - l0) * self.H
        Wqk = self.W_qk[l0:l1].reshape(nH * self.dh, self.p)
        Wv = self.W_v[l0:l1].reshape(nH * self.dh, self.d)
        u = split_heads(F.linear(phi, Wqk), nH)
        if self.rotary is not None:
            u = self.rotary(u)
        if u.shape[0] != vals.shape[0]:
            u = u.expand(vals.shape[0], -1, -1, -1)
        v = split_heads(F.linear(vals, Wv), nH)
        o = attend(u, u, v, masks, "strict")                 # Q = K = u (tied, PSD kernel)
        return F.linear(merge_heads(o), Wv.t())              # O = W_v^T (tied, P = W_v^T W_v)

    def forward(self, phi, vals, masks):
        f = None
        for l0 in range(0, self.L, self.chunk):
            l1 = min(self.L, l0 + self.chunk)
            if self.grad_ckpt and self.training:
                o = checkpoint(self._chunk, phi, vals, masks, l0, l1, use_reentrant=False)
            else:
                o = self._chunk(phi, vals, masks, l0, l1)
            f = o if f is None else f + o
        return f * masks.has_examples()[..., None].to(f.dtype)

    @torch.no_grad()
    def kernel_weights(self, phi, masks):
        """Normalized kernel weights alpha [B, L*H, T, T] (dense; analysis/tests only)."""
        u = split_heads(F.linear(phi, self.W_qk.reshape(-1, self.p)), self.L * self.H)
        if self.rotary is not None:
            u = self.rotary(u)
        s = torch.matmul(u.double(), u.double().transpose(-1, -2)) / math.sqrt(self.dh)
        m = masks.dense("strict")
        p = torch.softmax(s.masked_fill(~m, float("-inf")), dim=-1)
        return torch.where(m.any(-1, keepdim=True), p, torch.zeros_like(p))

    def preconditioner(self, l, h):
        """P_lh = W_lh^{V T} W_lh^V as a dense [d, d] matrix (tests/analysis)."""
        W = self.W_v[l, h * self.dh:(h + 1) * self.dh]
        return W.t() @ W


# ----------------------------------------------------------------------------- models
class LM(nn.Module):
    """Shared interface: logits(z, seg) and a masked next-token loss."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, z, seg, y=None, loss_mask=None, per_token=False):
        logits = self.logits(z, seg)
        if y is None:
            return logits
        B, T, V = logits.shape
        tok = F.cross_entropy(logits.float().view(-1, V), y.reshape(-1), reduction="none").view(B, T)
        m = loss_mask.to(tok.dtype)
        loss = (tok * m).sum() / m.sum().clamp(min=1.0)
        return (loss, tok) if per_token else loss

    def unique_params(self):
        seen, out = set(), []
        for n, p in self.named_parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append((n, p))
        return out

    def param_counts(self):
        tot = sum(p.numel() for _, p in self.unique_params())
        emb = self.embed.weight.numel()
        return {"total": tot, "embedding": emb, "nonembedding": tot - emb}

    def param_groups(self, weight_decay):
        """Weight decay on matrices only; not on embeddings, positional covariates or norm gains."""
        decay, no_decay = [], []
        for n, p in self.unique_params():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 and n not in ("embed.weight", "pos") else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def _init(self, n_residual):
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                s = std / math.sqrt(2 * n_residual) if getattr(m, "residual_out", False) else std
                nn.init.normal_(m.weight, 0.0, s)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, std)
            elif isinstance(m, GDPredictor):
                nn.init.normal_(m.W_qk, 0.0, std)
                nn.init.normal_(m.W_v, 0.0, std)

    def masks(self, seg, kinds):
        return Masks(seg, self.cfg.attn_impl).build(kinds)

    def run_blocks(self, h, masks):
        for blk in self.blocks:
            h = checkpoint(blk, h, masks, use_reentrant=False) if self.cfg.grad_ckpt and self.training \
                else blk(h, masks)
        return h


class Transformer(LM):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.rotary = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling or None)
        self.blocks = nn.ModuleList([LlamaBlock(cfg, self.rotary) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d, cfg.norm_eps)
        self._init(cfg.n_layers)

    def logits(self, z, seg):
        h = self.run_blocks(self.embed(z), self.masks(seg, ("causal",)))
        return self.final_norm(h) @ self.embed.weight.t()           # tied output head


class GDModel(LM):
    """gdlm (generator covariates) and wang (position-only covariates)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        if cfg.family == "gdlm":
            self.rotary = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling or None)
            self.blocks = nn.ModuleList([LlamaBlock(cfg, self.rotary) for _ in range(cfg.n_layers)])
            self.final_norm = RMSNorm(cfg.d, cfg.norm_eps)             # last generator op: phi = norm(h^(L))
            gd_rot = self.rotary if cfg.gd_head_dim == cfg.head_dim else \
                Rotary(cfg.gd_head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling or None)
            self.gd = GDPredictor(cfg, cfg.d, gd_rot)
            self._init(max(cfg.n_layers, 1))
        elif cfg.family == "wang":
            self.pos = nn.Parameter(torch.empty(cfg.max_seq_len, cfg.cov_dim))
            self.gd = GDPredictor(cfg, cfg.cov_dim, None)
            self._init(1)
            # unit-RMS covariates, the scale of the normalized generator covariates in gdlm
            nn.init.normal_(self.pos, 0.0, 1.0)
        else:
            raise ValueError(cfg.family)

    def centered_embedding(self):
        E = self.embed.weight
        return E - E.mean(dim=0, keepdim=True)

    def covariates(self, z, seg, E, masks):
        if self.cfg.family == "wang":
            return self.pos[segment_positions(seg)]
        h = self.run_blocks(F.embedding(z, E), masks)
        return self.final_norm(h)

    def predict(self, z, seg):
        """Pre-normalization GD prediction f [B, T, d] and the centered embedding."""
        E = self.centered_embedding()
        kinds = ("strict",) if self.cfg.family == "wang" else ("causal", "strict")
        masks = self.masks(seg, kinds)
        phi = self.covariates(z, seg, E, masks)
        # labels of the examples: y_k = z_{k+1}; the last position's label is never visible to any query
        nxt = torch.cat([z[:, 1:], z[:, -1:]], dim=1)
        vals = F.embedding(nxt, E)
        return self.gd(phi, vals, masks), E

    def readout(self, f, E):
        ff = upcast(f)
        fn = ff * torch.rsqrt(ff.pow(2).mean(-1, keepdim=True) + self.cfg.readout_eps)   # no learned gain
        return fn.to(f.dtype) @ E.t()

    def logits(self, z, seg):
        f, E = self.predict(z, seg)
        return self.readout(f, E)


def build_model(cfg):
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    return Transformer(cfg) if cfg.family == "transformer" else GDModel(cfg)


def compile_model(model):
    """Compile blocks and the GD predictor in place (state-dict keys unchanged)."""
    for b in getattr(model, "blocks", []):
        b.compile()
    if hasattr(model, "gd"):
        model.gd.compile()
