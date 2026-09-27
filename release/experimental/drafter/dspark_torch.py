# SPDX-License-Identifier: AGPL-3.0-only
"""Differentiable PyTorch DSpark drafter for training and offline evaluation.

Ported from DeepSeek-V4.1-Flash `inference/model.py` (DSparkBlock/DSparkAttention,
Block hyper-connections, Gate/MoE, Markov and confidence heads) with the
tilelang kernels replaced by plain PyTorch ops of the same math:
  * sparse_attn  -> dense attention over [window context + own block] with the
                    per-head attention sink added to the softmax denominator;
  * hc_split_sinkhorn -> the same sigmoid/softmax/Sinkhorn arithmetic;
  * FP8 act_quant -> optional fake quantization (straight-through).

Training layout: every position c of a captured sequence is an anchor. Its
block holds [token c+1, noise x (B-1)] at positions c+1..c+B and attends to the
drafter context KV of positions max(0, c-127)..c plus its own block
(bidirectional within the block, as in inference). Block position j predicts
token c+2+j; the Markov head conditions on the previous token of the chain.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Args:
    dim: int = 5120
    vocab: int = 129280
    n_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_lora: int = 1280
    o_groups: int = 8
    o_lora: int = 1024
    window: int = 128
    hc: int = 4
    sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6
    experts: int = 128
    topk: int = 3
    inter: int = 2304
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    markov_rank: int = 256
    layers: int = 3
    n_targets: int = 3
    rope_theta: float = 10000.0
    noise_token: int = 128799
    max_pos: int = 1 << 20


E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def fp8_dequant(weight, scale, block=32):
    """E4M3 weight with one power-of-two (E8M0) scale per 32x32 block."""
    w = weight.float()
    s = scale.float().repeat_interleave(block, 0)[:w.shape[0]].repeat_interleave(block, 1)[:, :w.shape[1]]
    return w * s


def fp4_dequant(packed, scale, block=32):
    """MXFP4: two E2M1 codes per byte (low nibble first), one E8M0 scale per 32 along K."""
    b = packed.view(torch.uint8)
    codes = torch.stack([b & 15, b >> 4], dim=-1).flatten(-2).long()
    w = E2M1.to(packed.device)[codes]
    return w * scale.float().repeat_interleave(block, -1)[:, :w.shape[1]]


def fake_fp8(x, block=32):
    """Straight-through E4M3 fake quantization with power-of-two scales per 32 values."""
    shape = x.shape
    g = x.float().reshape(-1, block)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (g / scale).to(torch.float8_e4m3fn).float() * scale
    return (x + (q.reshape(shape).to(x.dtype) - x).detach())


class QLinear(nn.Linear):
    """Linear whose input may be fake-quantized to FP8 (E4M3, power-of-two scale per 32), as the
    reference/serving FP8 GEMMs do. Off by default (quant_act)."""
    quant_act = False

    def forward(self, x):
        return super().forward(fake_fp8(x) if self.quant_act else x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


def rope_table(dim, positions, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim))
    angles = positions.float()[..., None] * freqs
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, cis, inverse=False):
    """Rotate adjacent pairs of the last dim; cis broadcasts over x's leading dims."""
    shape = x.shape
    z = torch.view_as_complex(x.float().reshape(*shape[:-1], -1, 2))
    z = z * (cis.conj() if inverse else cis)
    return torch.view_as_real(z).reshape(shape).to(x.dtype)


def hc_split_sinkhorn(mixes, scale, base, hc, iters, eps):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class Attention(nn.Module):
    def __init__(self, a: Args, quant_kv: bool):
        super().__init__()
        self.a, self.quant_kv = a, quant_kv
        self.attn_sink = nn.Parameter(torch.zeros(a.n_heads))
        self.wq_a = QLinear(a.dim, a.q_lora, bias=False)
        self.q_norm = RMSNorm(a.q_lora, a.norm_eps)
        self.wq_b = QLinear(a.q_lora, a.n_heads * a.head_dim, bias=False)
        self.wkv = QLinear(a.dim, a.head_dim, bias=False)
        self.kv_norm = RMSNorm(a.head_dim, a.norm_eps)
        self.wo_a = nn.Linear(a.n_heads * a.head_dim // a.o_groups, a.o_groups * a.o_lora, bias=False)
        self.wo_b = QLinear(a.o_groups * a.o_lora, a.dim, bias=False)

    def kv(self, x, cis):
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat([kv[..., :-self.a.rope_dim], apply_rope(kv[..., -self.a.rope_dim:], cis)], -1)
        return fake_fp8(kv) if self.quant_kv else kv

    def forward(self, x, block_cis, ctx_kv, ctx_mask):
        """x: [N,B,dim]; block_cis: [N,B,rope/2]; ctx_kv: [N,W,hd] (rotated); ctx_mask: [N,W] bool."""
        a = self.a
        n, b, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(-1, (a.n_heads, a.head_dim))
        if getattr(self, 'q_head_norm', False):      # vLLM's fused qnorm: per-head RMS, no weight
            q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + a.norm_eps)).to(q.dtype)
        q = torch.cat([q[..., :-a.rope_dim], apply_rope(q[..., -a.rope_dim:], block_cis[:, :, None])], -1)
        kv = torch.cat([ctx_kv, self.kv(x, block_cis)], dim=1)                      # [N, W+B, hd]
        mask = torch.cat([ctx_mask, ctx_mask.new_ones(n, b)], dim=1)
        scores = torch.einsum('nqhd,nkd->nqhk', q.float(), kv.float()) * a.head_dim ** -0.5
        scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        if getattr(self, 'causal_block', False):     # diagnostic: block query j sees block keys <= j only
            w = ctx_kv.shape[1]
            future = torch.triu(torch.ones(b, b, dtype=torch.bool, device=x.device), 1)
            scores[..., w:] = scores[..., w:].masked_fill(future[None, :, None, :], float('-inf'))
        sink = self.attn_sink.float()[None, None, :, None].expand(n, b, a.n_heads, 1)
        probs = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]            # sink only in denominator
        o = torch.einsum('nqhk,nkd->nqhd', probs, kv.float()).to(x.dtype)
        o = torch.cat([o[..., :-a.rope_dim], apply_rope(o[..., -a.rope_dim:], block_cis[:, :, None], True)], -1)
        o = o.view(n, b, a.o_groups, -1)
        wo_a = self.wo_a.weight.view(a.o_groups, a.o_lora, -1)
        o = torch.einsum('nbgd,grd->nbgr', o, wo_a)
        return self.wo_b(o.flatten(2))


class Expert(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.limit = a.swiglu_limit
        self.w1 = QLinear(a.dim, a.inter, bias=False)
        self.w2 = QLinear(a.inter, a.dim, bias=False)
        self.w3 = QLinear(a.dim, a.inter, bias=False)

    def forward(self, x, weights=None):
        dtype = x.dtype
        gate = self.w1(x).float().clamp(max=self.limit)
        up = self.w3(x).float().clamp(-self.limit, self.limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


class FrozenExperts(nn.Module):
    """Routed experts as frozen stacked bf16 tensors [E, out, in] (no grads, input grads flow)."""

    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.register_buffer('w1', torch.empty(a.experts, a.inter, a.dim, dtype=torch.bfloat16), persistent=False)
        self.register_buffer('w3', torch.empty(a.experts, a.inter, a.dim, dtype=torch.bfloat16), persistent=False)
        self.register_buffer('w2', torch.empty(a.experts, a.dim, a.inter, dtype=torch.bfloat16), persistent=False)

    def forward(self, x, weights, indices):
        lim = self.a.swiglu_limit
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        flat = indices.flatten()
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=self.a.experts).tolist()
        rows = order // indices.shape[1]
        slots = order % indices.shape[1]
        start = 0
        for e, c in enumerate(counts):
            if not c:
                continue
            r, s = rows[start:start + c], slots[start:start + c]
            start += c
            h = x[r]
            gate = F.linear(h, self.w1[e]).float().clamp(max=lim)
            up = F.linear(h, self.w3[e]).float().clamp(-lim, lim)
            act = (F.silu(gate) * up * weights[r, s, None]).to(x.dtype)
            y.index_add_(0, r, F.linear(act, self.w2[e]).float())
        return y


class MoE(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.gate_weight = nn.Parameter(torch.empty(a.experts, a.dim))
        self.gate_bias = nn.Parameter(torch.zeros(a.experts), requires_grad=False)  # selection bias, frozen
        self.experts = FrozenExperts(a)
        self.shared = Expert(a)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.a.dim)
        scores = F.softplus(F.linear(x.float(), self.gate_weight.float())).sqrt()
        indices = (scores + self.gate_bias).topk(self.a.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * self.a.route_scale
        y = self.experts(x, weights, indices) + self.shared(x).float()
        return y.to(x.dtype).view(shape)


class Block(nn.Module):
    def __init__(self, a: Args, stage: int, quant_kv: bool):
        super().__init__()
        self.a = a
        self.attn = Attention(a, quant_kv)
        self.ffn = MoE(a)
        self.attn_norm = RMSNorm(a.dim, a.norm_eps)
        self.ffn_norm = RMSNorm(a.dim, a.norm_eps)
        mix = (2 + a.hc) * a.hc
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix, a.hc * a.dim))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix, a.hc * a.dim))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix))
        self.hc_attn_scale = nn.Parameter(torch.zeros(3))
        self.hc_ffn_scale = nn.Parameter(torch.zeros(3))
        if stage == 0:
            self.main_proj = QLinear(a.dim * a.n_targets, a.dim, bias=False)
            self.main_norm = RMSNorm(a.dim, a.norm_eps)
        if stage == a.layers - 1:
            self.norm = RMSNorm(a.dim, a.norm_eps)
            self.markov_embed = nn.Embedding(a.vocab, a.markov_rank)
            self.markov_head = nn.Linear(a.markov_rank, a.vocab, bias=False)
            self.confidence = nn.Linear(a.dim + a.markov_rank, 1, bias=False)

    def mixes(self, x, fn, scale, base):
        x = x.flatten(-2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.a.norm_eps)
        return hc_split_sinkhorn(F.linear(x, fn.float()) * rsqrt, scale.float(), base.float(),
                                 self.a.hc, self.a.sinkhorn_iters, self.a.hc_eps)

    @staticmethod
    def hc_pre(x, pre):
        return torch.sum(pre.unsqueeze(-1) * x.float(), dim=-2).to(x.dtype)

    @staticmethod
    def hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2).float() + torch.sum(comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=-3)
        return y.to(x.dtype)

    def forward(self, x, pre_mix, block_cis, ctx_kv, ctx_mask):
        residual = x
        attn_pre, attn_post, attn_comb = self.mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), block_cis, ctx_kv, ctx_mask)
        x = self.hc_post(h, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)))
        return self.hc_post(h, residual, ffn_post, ffn_comb), ffn_pre


class DSpark(nn.Module):
    def __init__(self, a: Args = Args(), quant_kv: bool = False):
        super().__init__()
        self.a = a
        self.layers = nn.ModuleList(Block(a, i, quant_kv) for i in range(a.layers))
        # Shared with the target, never trained.
        self.register_buffer('embed', torch.empty(a.vocab, a.dim, dtype=torch.bfloat16), persistent=False)
        self.register_buffer('head', torch.empty(a.vocab, a.dim, dtype=torch.bfloat16), persistent=False)

    def trainable(self):
        return [p for n, p in self.named_parameters() if p.requires_grad]

    def context_kv(self, aux, positions):
        """aux: [L, 3*dim] target features of one sequence -> per-layer rotated KV [L, hd]."""
        main_x = self.layers[0].main_norm(self.layers[0].main_proj(aux))
        cis = rope_table(self.a.rope_dim, positions, self.a.rope_theta)
        return [layer.attn.kv(main_x, cis) for layer in self.layers]

    def gather_window(self, kv, anchors):
        """kv: [L, hd]; anchors: [N] context ends -> [N, W, hd] window (positions c-W+1..c) and mask."""
        w = self.a.window
        idx = anchors[:, None] - (w - 1) + torch.arange(w, device=anchors.device)
        mask = idx >= 0
        return kv[idx.clamp_min(0)], mask

    def draft_hidden(self, ctx, anchors, anchor_tokens, anchor_positions, block):
        """Run the drafter for N anchors; returns final hidden [N, block, dim] and last pre_mix."""
        a = self.a
        n = anchors.shape[0]
        ids = anchor_tokens.new_full((n, block), a.noise_token)
        ids[:, 0] = anchor_tokens
        x = self.embed[ids].unsqueeze(2).repeat(1, 1, a.hc, 1)
        pre = x.new_zeros(n, block, a.hc, dtype=torch.float32)
        pre[..., 0] = 1
        pos = anchor_positions[:, None] + torch.arange(block, device=anchors.device)
        cis = rope_table(a.rope_dim, pos, a.rope_theta)
        for layer, kv in zip(self.layers, ctx):
            window, mask = self.gather_window(kv, anchors)
            x, pre = layer(x, pre, cis, window, mask)
        last = self.layers[-1]
        return last.norm(Block.hc_pre(x, pre))

    def logits(self, hidden, prev_tokens):
        """Base logits + Markov bias from the chain's previous token; also confidence logits."""
        last = self.layers[-1]
        m = last.markov_embed(prev_tokens)
        logits = F.linear(hidden.to(self.head.dtype), self.head).float() + last.markov_head(m).float()
        conf = last.confidence(torch.cat([hidden.float(), m.float()], -1)).squeeze(-1)
        return logits, conf


# ---------------------------------------------------------------- weight loading / export

def _safetensors_index(folder):
    import struct
    index = {}
    for f in sorted(Path(folder).glob('*.safetensors')):
        with open(f, 'rb') as s:
            n = struct.unpack('<Q', s.read(8))[0]
            header = json.loads(s.read(n))
        for k, v in header.items():
            if k != '__metadata__':
                index[k] = (f, 8 + n, v)
    return index


_DT = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F8_E4M3': torch.float8_e4m3fn,
       'F8_E8M0': torch.float8_e8m0fnu, 'I8': torch.int8, 'I16': torch.int16, 'I32': torch.int32,
       'F16': torch.float16}


def read_tensor(index, name):
    f, base, v = index[name]
    a, b = v['data_offsets']
    with open(f, 'rb') as s:
        s.seek(base + a)
        raw = bytearray(s.read(b - a))
    return torch.frombuffer(raw, dtype=_DT[v['dtype']]).reshape(v['shape'])


def exl3_expert_weights(index, prefix, device):
    """Full dequantized weights of one EXL3 (MUL1) routed expert, as served (w1/w3: [inter, dim], w2: [dim, inter])."""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    out = {}
    for name in ('w1', 'w2', 'w3'):
        stem = f'{prefix}.{name}.'
        packed = {k: read_tensor(index, stem + k).to(device) for k in ('trellis', 'suh', 'svh', 'mul1')}
        layer = LinearEXL3(config=None, in_features=packed['suh'].numel(), out_features=packed['svh'].numel(),
                           out_dtype=torch.float16, key=stem[:-1], **packed)
        w = layer.get_weight_tensor()                                   # [in, out]
        out[name] = w.t().contiguous() if w.shape[0] == packed['suh'].numel() else w
    return out


def load(model: DSpark, draft_dir, target_dir, device, experts='fp4', exl3_dir=None):
    """Load DeepSeek's DSpark weights (mtp.*) and the target embedding/head.
    experts='exl3' takes the routed experts from serving's EXL3 draft overlay instead."""
    d = _safetensors_index(draft_dir)
    t = _safetensors_index(target_dir)
    x = _safetensors_index(exl3_dir) if experts == 'exl3' else None

    def dense(name):
        if name + '.scale' in d:
            return fp8_dequant(read_tensor(d, name + '.weight').to(device), read_tensor(d, name + '.scale').to(device))
        return read_tensor(d, name + '.weight').to(device).float()

    with torch.no_grad():
        model.embed.copy_(read_tensor(t, 'embed.weight').to(device))
        model.head.copy_(read_tensor(t, 'head.weight').to(device))
        for i, layer in enumerate(model.layers):
            p = f'mtp.{i}.'
            at = layer.attn
            for n in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'):
                getattr(at, n).weight.copy_(dense(p + 'attn.' + n))
            at.attn_sink.copy_(read_tensor(d, p + 'attn.attn_sink'))
            at.q_norm.weight.copy_(read_tensor(d, p + 'attn.q_norm.weight').float())
            at.kv_norm.weight.copy_(read_tensor(d, p + 'attn.kv_norm.weight').float())
            layer.attn_norm.weight.copy_(read_tensor(d, p + 'attn_norm.weight').float())
            layer.ffn_norm.weight.copy_(read_tensor(d, p + 'ffn_norm.weight').float())
            for n in ('hc_attn_fn', 'hc_ffn_fn', 'hc_attn_base', 'hc_ffn_base', 'hc_attn_scale', 'hc_ffn_scale'):
                getattr(layer, n).copy_(read_tensor(d, p + n))
            layer.ffn.gate_weight.copy_(read_tensor(d, p + 'ffn.gate.weight').float())
            layer.ffn.gate_bias.copy_(read_tensor(d, p + 'ffn.gate.bias'))
            for n in ('w1', 'w2', 'w3'):
                getattr(layer.ffn.shared, n).weight.copy_(dense(p + 'ffn.shared_experts.' + n))
            for e in range(model.a.experts):
                if x is not None:
                    ws = exl3_expert_weights(x, p + f'ffn.experts.{e}', device)
                    for n in ('w1', 'w2', 'w3'):
                        getattr(layer.ffn.experts, n)[e].copy_(ws[n].to(torch.bfloat16))
                    continue
                for n in ('w1', 'w2', 'w3'):
                    q = p + f'ffn.experts.{e}.{n}'
                    w = fp4_dequant(read_tensor(d, q + '.weight').to(device), read_tensor(d, q + '.scale').to(device))
                    getattr(layer.ffn.experts, n)[e].copy_(w.to(torch.bfloat16))
            if i == 0:
                layer.main_proj.weight.copy_(dense(p + 'main_proj'))
                layer.main_norm.weight.copy_(read_tensor(d, p + 'main_norm.weight').float())
            if i == model.a.layers - 1:
                layer.norm.weight.copy_(read_tensor(d, p + 'norm.weight').float())
                layer.markov_embed.weight.copy_(read_tensor(d, p + 'markov_head.embed.weight').float())
                layer.markov_head.weight.copy_(read_tensor(d, p + 'markov_head.head.weight').float())
                layer.confidence.weight.copy_(read_tensor(d, p + 'confidence_head.proj.weight').float())
    return model
