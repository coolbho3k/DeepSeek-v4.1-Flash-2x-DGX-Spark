"""Stage 1 of the indexer FP4 probe: turn captured block inputs into indexer queries, weights and keys.

Follows the reference model (source/<rev>/inference/model.py) on CPU:
    x      = attn_norm(hc_pre(h, pre))                   h, pre from capture-source-v1/states/LL
    qr     = q_norm(wq_a(x))
    q      = indexer.wq_b(qr) -> [T, 32, 128]            saved before RoPE and FP4
    w      = indexer.weights_proj(x) * 128**-0.5 * 32**-0.5
    latent = compressor(x)                               KV-source layers only (2, 8, 14, 20)
    k      = indexer.k_norm(indexer.wk(latent))          saved before RoPE and FP4
FP8 weights are dequantized exactly (E4M3 x E8M0 per 32x32 block); GEMMs run in FP32 and round to
BF16 where the reference produces BF16. The reference's FP8 activation quantization before FP8 GEMMs
is omitted: it perturbs q and k identically for every format compared later.
Usage: project.py <layer> <states_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE = Path('/work/source/df42c109f1defefcbfcedbe7d905718a12266e40')
CONFIG = json.loads((SOURCE / 'inference/config.json').read_text())
EPS = CONFIG['norm_eps']
KV_SOURCES = (2, 8, 14, 20)
INDEX = json.loads((SOURCE / 'model.safetensors.index.json').read_text())['weight_map']


def tensor(name):
    with safe_open(SOURCE / INDEX[name], 'pt') as stream:
        return stream.get_tensor(name)


def fp8_weight(prefix):
    w, s = tensor(prefix + '.weight').float(), tensor(prefix + '.scale').float()
    return w * s.repeat_interleave(32, 0).repeat_interleave(32, 1)


def linear(x, weight):
    return (x.float() @ weight.float().T).to(torch.bfloat16)


def rms(x, weight):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    return (weight.float() * xf).to(torch.bfloat16)


def main():
    layer, states, out = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    torch.set_num_threads(6)
    ratio = CONFIG['compress_ratios'][layer]
    p = f'layers.{layer}.attn'
    attn_norm, q_norm = tensor(f'layers.{layer}.attn_norm.weight'), tensor(f'{p}.q_norm.weight')
    wq_a, wq_b = fp8_weight(f'{p}.wq_a'), fp8_weight(f'{p}.indexer.wq_b')
    weights_proj = tensor(f'{p}.indexer.weights_proj.weight')
    owns_k = layer in KV_SOURCES
    if owns_k:
        wkv, comp_norm = tensor(f'{p}.compressor.wkv.weight'), tensor(f'{p}.compressor.norm.weight')
        wgate = tensor(f'{p}.compressor.wgate.weight') if ratio > 1 else None
        wk, k_norm = tensor(f'{p}.indexer.wk.weight'), tensor(f'{p}.indexer.k_norm.weight')
    out.mkdir(parents=True, exist_ok=True)
    scale = CONFIG['index_head_dim'] ** -0.5 * CONFIG['index_n_heads'] ** -0.5
    for path in sorted(states.glob('text-*.safetensors')):
        with safe_open(path, 'pt') as stream:
            h, pre, tokens = stream.get_tensor('h')[0], stream.get_tensor('pre')[0], stream.get_tensor('tokens')
        x = (pre.unsqueeze(-1) * h.float()).sum(1).to(torch.bfloat16)
        x = rms(x, attn_norm)
        qr = rms(linear(x, wq_a), q_norm)
        result = dict(q=linear(qr, wq_b).view(-1, CONFIG['index_n_heads'], CONFIG['index_head_dim']).contiguous(),
                      w=(linear(x, weights_proj) * scale).float(), tokens=tokens)
        if owns_k:
            if ratio == 1:
                latent = rms(linear(x, wkv), comp_norm)
            else:
                xf = x.float()
                kv, score = xf @ wkv.float().T, xf @ wgate.float().T
                usable = kv.size(0) - kv.size(0) % ratio
                kv, score = kv[:usable].unflatten(0, (-1, ratio)), score[:usable].unflatten(0, (-1, ratio))
                latent = rms((kv * score.softmax(dim=1)).sum(1).to(torch.bfloat16), comp_norm)
            result['k'] = rms(linear(latent, wk), k_norm).contiguous()
        save_file(result, str(out / path.name))
        print(json.dumps(dict(layer=layer, record=path.stem, keys=int(result['k'].size(0)) if owns_k else 0)),
              flush=True)


if __name__ == '__main__':
    main()
