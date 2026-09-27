# SPDX-License-Identifier: AGPL-3.0-only
"""Ground truth: run DeepSeek's own reference DSpark (inference/model.py, tilelang
kernels, FP8/FP4 weights) on captured target features, one position at a time
like its generate() loop, and compare its greedy drafts with (a) the drafts
serving proposed and (b) our PyTorch port, at the same anchors.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, '/ref')
import model as R                                   # noqa: E402  DeepSeek reference

import dspark_torch as D                            # noqa: E402
from probe_drafts import offline_drafts             # noqa: E402
from train import load_sequence, ngram_drafts       # noqa: E402


def torch_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Same math as kernel.sparse_attn (index gather, -1 = invalid, sink in the denominator);
    the tilelang kernel needs 141 KB of shared memory, more than SM121 allows."""
    b, m, h, d = q.shape
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp_min(0).long()
    keys = torch.gather(kv[:, None].expand(b, m, *kv.shape[1:]), 2, idx[..., None].expand(*idx.shape, d))
    scores = torch.einsum('bmhd,bmkd->bmhk', q.float(), keys.float()) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float('-inf'))
    sink = attn_sink.float()[None, None, :, None].expand(b, m, h, 1)
    probs = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
    return torch.einsum('bmhk,bmkd->bmhd', probs, keys.float()).to(q.dtype)


R.sparse_attn = torch_sparse_attn


def build_reference(block):
    cfg = json.loads(Path('/ref/config.json').read_text())
    args = R.ModelArgs(**cfg)
    args.dspark_block_size = block
    args.temperature = 0.0
    args.max_batch_size = 1
    args.max_seq_len = 16384
    R.world_size, R.rank = 1, 0
    R.default_dtype = torch.float8_e4m3fn
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('cuda'):
        layers = [R.DSparkBlock(args.n_layers + i, args) for i in range(args.n_mtp_layers)]
        embed = R.ParallelEmbedding(args.vocab_size, args.dim)
        head = R.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    d = D._safetensors_index('/draft')
    t = D._safetensors_index('/target')
    for i, layer in enumerate(layers):
        prefix = f'mtp.{i}.'
        state = layer.state_dict()
        for name, value in state.items():
            full = prefix + name
            if full not in d:
                if 'bias_vl' in name or 'freqs_cis' in name or 'kv_cache' in name:
                    continue
                raise KeyError(full)
            src = D.read_tensor(d, full).to('cuda')
            if value.dtype == torch.float4_e2m1fn_x2:
                src = src.view(torch.float4_e2m1fn_x2)
            value.data.copy_(src.to(value.dtype) if value.dtype != src.dtype and value.dtype != torch.float4_e2m1fn_x2 else src)
        layer.embed, layer.head = embed, head
    with torch.no_grad():
        embed.weight.copy_(D.read_tensor(t, 'embed.weight').to('cuda'))
        head.weight.copy_(D.read_tensor(t, 'head.weight').to('cuda').float())
    return args, layers


@torch.inference_mode()
def reference_drafts(args, layers, seq, block):
    """drafts[c] for every context end c >= prompt_len, from one sequential pass."""
    P = seq['meta']['prompt_len']
    aux = seq['aux'].cuda()[None]                   # [1, L, 3*dim]
    tokens = seq['tokens'].cuda()
    L = aux.shape[1]
    hc = args.hc_mult

    def spec(anchor, main_hidden, start_pos):
        h, main_x = layers[0].forward_embed(main_hidden, anchor.view(1))
        pre = R.make_identity_pre_mix(h, hc)
        for layer in layers:
            h, pre = layer(h, start_pos, pre, main_x)
        if start_pos == 0:
            return None
        return layers[-1].forward_head(h, pre, anchor.view(1))[0]

    out = {}
    with torch.device('cuda'):                      # reference builds attention indices on the default device
        spec(tokens[P], aux[:, :P], 0)              # prefill seeds each layer's window cache
        for c in range(P, L - 1):
            ids = spec(tokens[c + 1], aux[:, c:c + 1], c)
            out[c] = ids[0, 1:1 + block].tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--glob', default='gate1-*')
    p.add_argument('--k', type=int, default=3)
    a = p.parse_args()
    args, layers = build_reference(a.k)
    with torch.device('cuda'):
        port = D.DSpark(D.Args())
    D.load(port, '/draft', '/target', torch.device('cuda'))
    port.eval()
    agree_srv = torch.zeros(a.k)
    agree_port = torch.zeros(a.k)
    port_srv = torch.zeros(a.k)
    total = 0
    for folder in sorted(Path('/data').glob(a.glob)):
        seq = load_sequence(folder)
        if 'serving_drafts' not in seq:
            continue
        ref = reference_drafts(args, layers, seq, a.k)
        hist = seq['tokens'].tolist()
        steps = [s for s in seq['serving_drafts'].tolist()
                 if (s[0] - 1) in ref and ngram_drafts(hist[:s[0] + 1], a.k) is None]
        if not steps:
            continue
        with torch.autocast('cuda', dtype=torch.bfloat16):
            off = offline_drafts(port, seq, [s[0] for s in steps], a.k, torch.device('cuda'))
        r = torch.tensor([ref[s[0] - 1] for s in steps])
        srv = torch.tensor([s[1:1 + a.k] for s in steps])
        agree_srv += (r == srv).float().sum(0)
        agree_port += (r == off).float().sum(0)
        port_srv += (off == srv).float().sum(0)
        total += len(steps)
        print(folder.name, len(steps), flush=True)
    f = lambda x: [round(v, 4) for v in (x / max(total, 1)).tolist()]
    print(json.dumps(dict(steps=total, reference_vs_serving=f(agree_srv), reference_vs_port=f(agree_port),
                          port_vs_serving=f(port_srv))), flush=True)


if __name__ == '__main__':
    main()
