# SPDX-License-Identifier: AGPL-3.0-only
"""Layer-by-layer: serving's eager draft forward (debug dumps) vs our port at the same anchors.
Prints relative L2 error and cosine per recorded tensor, in forward order."""
import json
from pathlib import Path

import torch

import dspark_torch as D
from train import load_sequence

K = 3
dev = torch.device('cuda')
with torch.device(dev):
    port = D.DSpark(D.Args())
D.load(port, '/draft', '/target', dev, experts='exl3', exl3_dir='/draft-exl3')
port.eval()

rec = {}
for i, layer in enumerate(port.layers):
    layer.attn.register_forward_hook(lambda m, inp, out, i=i: rec.__setitem__(f'attn_in{i}', inp[0]) or rec.__setitem__(f'attn_out{i}', out))
    layer.ffn.register_forward_hook(lambda m, inp, out, i=i: rec.__setitem__(f'ffn_out{i}', out))
    layer.register_forward_hook(lambda m, inp, out, i=i: rec.__setitem__(f'stream{i}', out[0]))

errs = {}
debug_root = Path('/debug')
for folder in sorted(Path('/data').glob('gate1-*')):
    dumps = sorted((debug_root).glob(f'*{folder.name}*/*.pt'))
    if not dumps:
        continue
    seq = load_sequence(folder)
    tokens = seq['tokens'].to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ctx = port.context_kv(seq['aux'].to(dev), seq['positions'].to(dev))
        for dump in dumps:
            q = int(dump.stem)
            if q >= len(tokens):
                continue
            srv = torch.load(dump, weights_only=True)
            rec.clear()
            qt = torch.tensor([q], device=dev)
            h = port.draft_hidden(ctx, qt - 1, tokens[qt], qt, K)
            rec['head_hidden'] = port.layers[-1].norm.weight.new_zeros(0)  # compared below separately
            for key, s in srv.items():
                if key == 'head_hidden' or key not in rec:
                    continue
                p = rec[key].float().reshape(s.shape)
                s = s.float().to(dev)
                e = ((p - s).norm() / (s.norm() + 1e-9)).item()
                cos = torch.nn.functional.cosine_similarity(p.flatten(), s.flatten(), dim=0).item()
                errs.setdefault(key, []).append((e, cos))
order = sorted(errs, key=lambda k: (int(k[-1]), ['attn_in', 'attn_out', 'ffn_out', 'stream'].index(k[:-1])))
for k in order:
    v = errs[k]
    print(json.dumps(dict(tensor=k, n=len(v), mean_rel_err=round(sum(e for e, _ in v) / len(v), 4),
                          mean_cos=round(sum(c for _, c in v) / len(v), 5))))
