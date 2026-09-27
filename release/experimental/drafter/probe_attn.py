# SPDX-License-Identifier: AGPL-3.0-only
"""Which attention variant reproduces serving's draft layer-0 attention output?
Feeds serving's own recorded attention input (attn_in0) and compares with its attn_out0."""
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


PARTIAL = None


def kv_of(x, pos, rope=True, norm=True, double_norm=False, double_rope=False):
    kv = at.wkv(x)
    if norm:
        kv = at.kv_norm(kv)
    if double_norm:
        kv = at.kv_norm(kv)
    if rope:
        cis = D.rope_table(a.rope_dim, pos, a.rope_theta)
        r = D.apply_rope(kv[..., -a.rope_dim:], cis)
        if double_rope:
            r = D.apply_rope(r, cis)
        kv = torch.cat([kv[..., :-a.rope_dim], r], -1)
    return kv


def attend(x, qpos, ctx_kv, ctx_mask, inverse=True, block_kv=None):
    n, b, _ = x.shape
    q = at.wq_b(at.q_norm(at.wq_a(x))).unflatten(-1, (a.n_heads, a.head_dim))
    cis = D.rope_table(a.rope_dim, qpos, a.rope_theta)
    q = torch.cat([q[..., :-a.rope_dim], D.apply_rope(q[..., -a.rope_dim:], cis[:, :, None])], -1)
    bk = block_kv if block_kv is not None else kv_of(x, qpos)
    kv = torch.cat([ctx_kv, bk], 1)
    mask = torch.cat([ctx_mask, ctx_mask.new_ones(n, b)], 1)
    s = torch.einsum('nqhd,nkd->nqhk', q.float(), kv.float()) * a.head_dim ** -0.5
    s = s.masked_fill(~mask[:, None, None, :], float('-inf'))
    sink = at.attn_sink.float()[None, None, :, None].expand(n, b, a.n_heads, 1)
    p = torch.cat([s, sink], -1).softmax(-1)[..., :-1]
    o = torch.einsum('nqhk,nkd->nqhd', p, kv.float()).to(x.dtype)
    if inverse:
        o = torch.cat([o[..., :-a.rope_dim], D.apply_rope(o[..., -a.rope_dim:], cis[:, :, None], True)], -1)
    o = o.view(n, b, a.o_groups, -1)
    o = torch.einsum('nbgd,grd->nbgr', o, at.wo_a.weight.view(a.o_groups, a.o_lora, -1))
    if PARTIAL is not None:                       # only this TP rank's groups (heads) contribute
        keep = torch.zeros(a.o_groups, device=o.device, dtype=o.dtype)
        keep[PARTIAL] = 1
        o = o * keep[None, None, :, None]
    return at.wo_b(o.flatten(2))


variants = {
    'reference': {},
}
res = {k: [] for k in variants}
BINS = {}
for folder in sorted(Path('/data').glob('gate1-*')):
    dumps = sorted(Path('/debug').glob(f'*{folder.name}*/*.pt'))
    if not dumps:
        continue
    seq = load_sequence(folder)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        main_x = L0.main_norm(L0.main_proj(seq['aux'].to(dev)))
        pos_all = seq['positions'].to(dev)
        for dump in dumps[:40]:
            q = int(dump.stem)
            srv = torch.load(dump, weights_only=True)
            x = srv['attn_in0'].to(dev)[None].to(torch.bfloat16)          # [1, K, dim] serving's own input
            target = srv['attn_out0'].float().to(dev)
            qpos = (q + torch.arange(K, device=dev))[None]
            for name, v in variants.items():
                w = v.get('window', 128)
                c = q - 1
                idx = torch.arange(c - w + 1, c + 1, device=dev)
                mask = (idx >= 0)[None]
                idx = idx.clamp_min(0)
                cpos = pos_all[idx] + v.get('pos_shift', 0)
                ctx = kv_of(main_x[idx], cpos, **v.get('ctx', {}))[None]
                if v.get('no_ctx'):
                    mask = torch.zeros_like(mask)
                PARTIAL = v.get('partial')
                out = attend(x, qpos, ctx, mask, inverse=v.get('inverse', True))[0].float()
                cs = F.cosine_similarity(out.flatten(), target.flatten(), dim=0).item()
                res[name].append(cs)
                P = seq['meta']['prompt_len']
                BINS.setdefault((folder.name, min((q - P) // 10, 9)), []).append(cs)
for name, v in res.items():
    print(json.dumps(dict(variant=name, n=len(v), mean_cos=round(sum(v) / max(len(v), 1), 4))))

for (case, b), v in sorted(BINS.items()):
    print(case, 'steps_from_prompt_bin', b * 10, 'n', len(v), 'cos', round(sum(v) / len(v), 3))
