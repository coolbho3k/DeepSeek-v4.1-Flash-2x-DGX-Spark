# SPDX-License-Identifier: AGPL-3.0-only
"""Write a trained DSpark back into the original draft checkpoint layout.

Every tensor keeps its name, shape and dtype. FP8 (E4M3) weights are
requantized with one power-of-two E8M0 scale per 32x32 block (DeepSeek's
format); BF16/F32 tensors are cast. Routed experts and the frozen gate bias
are copied byte for byte. Then reloads the export and checks it against the
trained weights.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from dspark_torch import Args, DSpark, fp8_dequant

FP8_MAX = 448.0


def fp8_quant(w, block=32):
    rows, cols = w.shape
    pr, pc = -rows % block, -cols % block
    g = torch.nn.functional.pad(w.float(), (0, pc, 0, pr)).view((rows + pr) // block, block, (cols + pc) // block, block)
    amax = g.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (g / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).view(rows + pr, cols + pc)[:rows, :cols]
    return q.to(torch.float8_e4m3fn), scale.to(torch.float8_e8m0fnu)


def mapping(model):
    """checkpoint tensor name -> (trained tensor, kind)."""
    out = {}
    for i, layer in enumerate(model.layers):
        p = f'mtp.{i}.'
        at = layer.attn
        for n in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'):
            out[p + 'attn.' + n] = (getattr(at, n).weight, 'fp8')
        out[p + 'attn.attn_sink'] = (at.attn_sink, 'F32')
        out[p + 'attn.q_norm.weight'] = (at.q_norm.weight, 'BF16')
        out[p + 'attn.kv_norm.weight'] = (at.kv_norm.weight, 'BF16')
        out[p + 'attn_norm.weight'] = (layer.attn_norm.weight, 'BF16')
        out[p + 'ffn_norm.weight'] = (layer.ffn_norm.weight, 'BF16')
        for n in ('hc_attn_fn', 'hc_ffn_fn', 'hc_attn_base', 'hc_ffn_base', 'hc_attn_scale', 'hc_ffn_scale'):
            out[p + n] = (getattr(layer, n), 'F32')
        out[p + 'ffn.gate.weight'] = (layer.ffn.gate_weight, 'BF16')
        for n in ('w1', 'w2', 'w3'):
            out[p + 'ffn.shared_experts.' + n] = (getattr(layer.ffn.shared, n).weight, 'fp8')
        if i == 0:
            out[p + 'main_proj'] = (layer.main_proj.weight, 'fp8')
            out[p + 'main_norm.weight'] = (layer.main_norm.weight, 'BF16')
        if i == model.a.layers - 1:
            out[p + 'norm.weight'] = (layer.norm.weight, 'BF16')
            out[p + 'markov_head.embed.weight'] = (layer.markov_embed.weight, 'BF16')
            out[p + 'markov_head.head.weight'] = (layer.markov_head.weight, 'BF16')
            out[p + 'confidence_head.proj.weight'] = (layer.confidence.weight, 'BF16')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True, help='original draft checkpoint directory')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output directory required')
    with torch.device('meta'):
        model = DSpark(Args())
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state, strict=False, assign=True)
    names = mapping(model)
    a.output.mkdir(parents=True)
    written, report = set(), {}
    for shard in sorted(a.draft.glob('*.safetensors')):
        tensors = load_file(shard)
        for name, (value, kind) in names.items():
            if kind == 'fp8' and name + '.weight' in tensors:
                q, s = fp8_quant(value.detach())
                assert q.shape == tensors[name + '.weight'].shape and s.shape == tensors[name + '.scale'].shape, name
                err = (fp8_dequant(q, s) - value.float()).norm() / value.float().norm()
                tensors[name + '.weight'], tensors[name + '.scale'] = q, s
                report[name] = round(err.item(), 5)
                written.add(name)
            elif kind != 'fp8' and name in tensors:
                old = tensors[name]
                assert old.shape == value.shape, name
                tensors[name] = value.detach().to(old.dtype).contiguous()
                written.add(name)
        save_file(tensors, a.output / shard.name, metadata={'format': 'pt'})
    missing = set(names) - written
    if missing:
        raise RuntimeError(f'not exported: {sorted(missing)[:5]}')
    for extra in a.draft.iterdir():
        if extra.suffix != '.safetensors':
            shutil.copy2(extra, a.output / extra.name)
    worst = max(report.values())
    (a.output / 'export-report.json').write_text(json.dumps(dict(
        checkpoint=str(a.checkpoint), fp8_relative_error=report, worst_fp8_relative_error=worst), indent=1))
    print(json.dumps(dict(exported=len(written), worst_fp8_relative_error=worst)))


if __name__ == '__main__':
    main()
