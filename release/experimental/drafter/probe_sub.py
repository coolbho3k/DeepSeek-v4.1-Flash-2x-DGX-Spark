# SPDX-License-Identifier: AGPL-3.0-only
"""Layer-0 attention, submodule by submodule: apply our port's module to serving's recorded input and
compare with serving's recorded output. Also prints every recorded submodule for orientation."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import dspark_torch as D

dev = torch.device('cuda')
with torch.device(dev):
    port = D.DSpark(D.Args())
D.load(port, '/draft', '/target', dev, experts='exl3', exl3_dir='/draft-exl3')
port.eval()
at = port.layers[0].attn
dumps = sorted(Path('/debug').glob('*/*.pt'))[:16]
first = torch.load(dumps[0], weights_only=True)
for k in sorted(first):
    if k.startswith('a0.'):
        print('recorded', k, tuple(first[k].shape), first[k].dtype)

ports = {
    'wq_a': lambda x: at.wq_a(x), 'q_norm': lambda x: at.q_norm(x), 'wq_b': lambda x: at.wq_b(x),
    'wkv': lambda x: at.wkv(x), 'kv_norm': lambda x: at.kv_norm(x), 'wo_b': lambda x: at.wo_b(x),
    'fused_wqa_wkv': lambda x: torch.cat([at.wq_a(x), at.wkv(x)], -1),
}
stats = {}
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for dump in dumps:
        d = torch.load(dump, weights_only=True)
        for k, v in d.items():
            if not k.startswith('a0.') or k.endswith('#in'):
                continue
            leaf = k.split('.')[-1]
            if leaf not in ports or k + '#in' not in d:
                continue
            x = d[k + '#in'].to(dev)
            try:
                out = ports[leaf](x.to(torch.bfloat16)).float()
            except Exception as e:
                stats.setdefault(k, []).append(('error', str(e)[:80]))
                continue
            s = v.float().to(dev).reshape(out.shape) if v.numel() == out.numel() else None
            if s is None:
                stats.setdefault(k, []).append(('shape', f'{tuple(v.shape)} vs {tuple(out.shape)}'))
                continue
            cos = F.cosine_similarity(out.flatten(), s.flatten(), dim=0).item()
            rel = ((out - s).norm() / (s.norm() + 1e-9)).item()
            stats.setdefault(k, []).append((cos, rel))
for k, v in stats.items():
    nums = [x for x in v if isinstance(x[0], float)]
    if nums:
        print(json.dumps(dict(module=k, n=len(nums), cos=round(sum(c for c, _ in nums) / len(nums), 5),
                              rel_err=round(sum(r for _, r in nums) / len(nums), 5))))
    else:
        print(json.dumps(dict(module=k, issue=v[0])))
