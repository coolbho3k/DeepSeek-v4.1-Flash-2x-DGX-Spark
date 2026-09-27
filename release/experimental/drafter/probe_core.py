# SPDX-License-Identifier: AGPL-3.0-only
"""Rank-0 slices of layer-0 draft attention: q projection, output projection, and the attention core
(serving's own q -> our attention over our context -> wo_a groups 0-3) vs serving's wo_b input."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import dspark_torch as D
from train import load_sequence

K = 3
dev = torch.device('cuda')
with torch.device(dev):
    port = D.DSpark(D.Args())
D.load(port, '/draft', '/target', dev, experts='exl3', exl3_dir='/draft-exl3')
port.eval()
a = port.a
at = port.layers[0].attn
L0 = port.layers[0]
H = a.n_heads // 2          # rank-0 heads
res = {}
def add(k, x, y):
    res.setdefault(k, []).append(F.cosine_similarity(x.float().flatten(), y.float().flatten(), dim=0).item())
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for folder in sorted(Path('/data').glob('gate1-*')):
        dumps = sorted(Path('/debug').glob(f'*{folder.name}*/*.pt'))
        if not dumps:
            continue
        seq = load_sequence(folder)
        main_x = L0.main_norm(L0.main_proj(seq['aux'].to(dev)))
        pos_all = seq['positions'].to(dev)
        for dump in dumps:
            d = torch.load(dump, weights_only=True)
            q0 = int(dump.stem)
            # q projection on serving's input
            qin = d['a0.wq_b#in'].to(dev).to(torch.bfloat16)
            add('wq_b(rank0 rows)', torch.nn.functional.linear(qin, at.wq_b.weight[:H * a.head_dim]), d['a0.wq_b'].to(dev))
            add('q_norm(port) vs wq_b#in', at.q_norm(d['a0.fused_wqa_wkv'].to(dev)[..., :a.q_lora].to(torch.bfloat16)), qin)
            # output projection on serving's input (partial sum)
            oin = d['a0.wo_b#in'].to(dev).to(torch.bfloat16)
            oin_rw = d['a0.wo_b#in@rw'].to(dev).to(torch.bfloat16) if 'a0.wo_b#in@rw' in d else None
            add('wo_b(rank0 cols)', torch.nn.functional.linear(oin, at.wo_b.weight[:, :oin.shape[-1]]), d['a0.wo_b'].to(dev))
            # attention core from serving's q
            q = d['a0.wq_b'].to(dev).to(torch.bfloat16).unflatten(-1, (H, a.head_dim))[None]      # [1,K,H,hd]
            qpos = (q0 + torch.arange(K, device=dev))[None]
            cis = D.rope_table(a.rope_dim, qpos, a.rope_theta)
            q = torch.cat([q[..., :-a.rope_dim], D.apply_rope(q[..., -a.rope_dim:], cis[:, :, None])], -1)
            c = q0 - 1
            idx = torch.arange(c - a.window + 1, c + 1, device=dev)
            mask = (idx >= 0)[None]
            idx = idx.clamp_min(0)
            ctx = at.kv(main_x[idx], D.rope_table(a.rope_dim, pos_all[idx], a.rope_theta))[None]
            def remapped(fn):
                src = fn(idx).clamp(0, c)
                return at.kv(main_x[src], D.rope_table(a.rope_dim, pos_all[src], a.rope_theta))[None]
            maps = {
                'read cp2 of cp1 writes: 32*(p//64)+(p//2)%32': lambda p: 32 * (p // 64) + (p // 2) % 32,
                'p//2': lambda p: p // 2,
                'rank parity 0 only (p&~1)': lambda p: p - (p % 2),
                'rank parity 1 only (p|1)': lambda p: torch.minimum(p + 1 - (p % 2), torch.full_like(p, c)),
            }
            blk_in = d['a0.fused_wqa_wkv'].to(dev)[..., a.q_lora:].to(torch.bfloat16)          # serving's raw kv
            blk = at.kv_norm(blk_in)
            blk = torch.cat([blk[..., :-a.rope_dim], D.apply_rope(blk[..., -a.rope_dim:], cis[0])], -1)[None]
            kv = torch.cat([ctx, blk], 1)
            m = torch.cat([mask, mask.new_ones(1, K)], 1)
            s = torch.einsum('nqhd,nkd->nqhk', q.float(), kv.float()) * a.head_dim ** -0.5
            s = s.masked_fill(~m[:, None, None, :], float('-inf'))
            sink = at.attn_sink.float()[:H][None, None, :, None].expand(1, K, H, 1)
            p = torch.cat([s, sink], -1).softmax(-1)[..., :-1]
            o = torch.einsum('nqhk,nkd->nqhd', p, kv.float()).to(torch.bfloat16)
            o = torch.cat([o[..., :-a.rope_dim], D.apply_rope(o[..., -a.rope_dim:], cis[:, :, None], True)], -1)
            g = a.o_groups // 2
            q_raw = d['a0.wq_b'].to(dev).float().unflatten(-1, (H, a.head_dim))[None]
            def q_variant(kind):
                qq = q_raw
                if kind == 'headnorm':
                    qq = qq * torch.rsqrt(qq.square().mean(-1, keepdim=True) + 1e-6)
                if kind == 'headnorm_nope':
                    n_ = qq[..., :-a.rope_dim]
                    qq = torch.cat([n_ * torch.rsqrt(n_.square().mean(-1, keepdim=True) + 1e-6), qq[..., -a.rope_dim:]], -1)
                qq = qq.to(torch.bfloat16)
                return torch.cat([qq[..., :-a.rope_dim], D.apply_rope(qq[..., -a.rope_dim:], cis[:, :, None])], -1)
            def core_with(ctx_kv, scale=a.head_dim ** -0.5, v_nope_only=False, no_inverse=False, qk=None):
                kv2 = torch.cat([ctx_kv, blk], 1)
                qq = q if qk is None else q_variant(qk)
                s3 = torch.einsum('nqhd,nkd->nqhk', qq.float(), kv2.float()) * scale
                s3 = s3.masked_fill(~m[:, None, None, :], float('-inf'))
                p3 = torch.cat([s3, sink], -1).softmax(-1)[..., :-1]
                vv = kv2.float().clone()
                if v_nope_only:
                    vv[..., -a.rope_dim:] = 0
                o3 = torch.einsum('nqhk,nkd->nqhd', p3, vv).to(torch.bfloat16)
                if not no_inverse:
                    o3 = torch.cat([o3[..., :-a.rope_dim], D.apply_rope(o3[..., -a.rope_dim:], cis[:, :, None], True)], -1)
                return torch.einsum('nbgd,grd->nbgr', o3.view(1, K, g, -1), at.wo_a.weight.view(a.o_groups, a.o_lora, -1)[:g]).flatten(2)[0]
            hd = a.head_dim
            for vname, kw in {
                'scale 1/sqrt(512) (reference)': {},
                'scale 1/sqrt(448)': dict(scale=448 ** -0.5),
                'scale 1/sqrt(576)': dict(scale=576 ** -0.5),
                'scale 1/sqrt(128)': dict(scale=128 ** -0.5),
                'scale 1/512': dict(scale=1 / 512),
                'scale 1': dict(scale=1.0),
                'V nope only': dict(v_nope_only=True),
                'V nope only, no inverse': dict(v_nope_only=True, no_inverse=True),
                'no inverse rope': dict(no_inverse=True),
                'q per-head RMS norm': dict(qk='headnorm'),
                'q per-head RMS norm (nope part)': dict(qk='headnorm_nope'),
            }.items():
                add('kernel ' + vname, core_with(ctx, **kw), oin)
            core = o.view(1, K, g, -1)
            W = at.wo_a.weight
            layouts = {
                'wo_a groups 0-3 (reference)': W.view(a.o_groups, a.o_lora, -1)[:g],
                'wo_a groups 4-7': W.view(a.o_groups, a.o_lora, -1)[g:],
                'wo_a rows interleaved r-major': W.view(a.o_lora, a.o_groups, -1).transpose(0, 1)[:g],
                'wo_a rank0 rows 0:4096 as r-major': W[:g * a.o_lora].view(a.o_lora, g, -1).transpose(0, 1),
            }
            for lname, Wg in layouts.items():
                o = torch.einsum('nbgd,grd->nbgr', core, Wg.contiguous()).flatten(2)[0]
                add('core+' + lname, o, oin)
                if oin_rw is not None and lname.endswith('(reference)'):
                    add('core vs serving AFTER context rewrite', o, oin_rw)
                    add('serving before vs after rewrite', oin, oin_rw)
            # same core but with the context only (no block keys) and block only (no context)
            for name, mm in (('core ctx-only', torch.cat([mask, mask.new_zeros(1, K)], 1)), ('core block-only', torch.cat([torch.zeros_like(mask), mask.new_ones(1, K)], 1))):
                s2 = torch.einsum('nqhd,nkd->nqhk', q.float(), kv.float()) * a.head_dim ** -0.5
                s2 = s2.masked_fill(~mm[:, None, None, :], float('-inf'))
                p2 = torch.cat([s2, sink], -1).softmax(-1)[..., :-1]
                o2 = torch.einsum('nqhk,nkd->nqhd', p2, kv.float()).to(torch.bfloat16)
                o2 = torch.cat([o2[..., :-a.rope_dim], D.apply_rope(o2[..., -a.rope_dim:], cis[:, :, None], True)], -1)
                o2 = torch.einsum('nbgd,grd->nbgr', o2.view(1, K, g, -1), at.wo_a.weight.view(a.o_groups, a.o_lora, -1)[:g]).flatten(2)[0]
                add(name, o2, oin)
for k, v in res.items():
    print(json.dumps(dict(check=k, n=len(v), mean_cos=round(sum(v) / len(v), 5), min_cos=round(min(v), 4))))
