"""Stage 2 of the indexer FP4 probe: does the index-key/query format change which positions are kept?

For every index-source layer, scores follow the reference Indexer:
    score[p, n] = sum_h w[p, h] * relu(q[p, h] . k[n]),  n < (p + 1) // ratio,  keep top 512
with RoPE (compress theta 160000, YaRN factor 16 over 65536) on the last 64 dims of q and k.
Formats applied to q and k after RoPE, as in the reference:
    ref    BF16 q and k (no FP4)
    mx     MXFP4: E2M1, power-of-two E8M0 scale per 32 (the reference and the current serving format)
    nv6    E2M1, E4M3 scale per 16, scale amax/6
    nv46   same, per group the lower-error of amax/6 and amax/4 (the main KV's four-over-six writer)
    kmx / knv46   BF16 query, only the key quantized (isolates the key's share)
Two context sets:
    native    each captured 2048-token record on its own; queries at positions 1024..2047
    stitched  the 8 records of one corpus concatenated (16384 tokens, keys and queries re-roped at
              their stitched positions); queries in the last record. Records were captured
              independently, so this approximates long-context key statistics, not true long-range
              relevance. At 16384 tokens layer 20's candidate blocks (2048 x 8) cover every position,
              so the candidate stage is a no-op and omitted.
Reports recall of the BF16 top-512 set and the share of BF16 top-64 / top-16 positions dropped.
Usage: analyze.py <proj_dir> <out.json>
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

RATIOS = {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}
KEY_SOURCE = {2: 2, 8: 8, 14: 14, 20: 20, 24: 20, 28: 20, 32: 20, 36: 20}
TOPK, ROPE = 512, 64
LEVELS = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
ORDER = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])  # even codes first: argmin gives nearest-even ties


def freqs_cis(seqlen, dim=ROPE, base=160000.0, original=65536, factor=16, beta_fast=32, beta_slow=1):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    corrected = lambda rotations: dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(corrected(beta_fast)), 0), min(math.ceil(corrected(beta_slow)), dim - 1)
    smooth = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return torch.polar(torch.ones(seqlen, dim // 2), torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs))


def rope(x, positions, table):
    """x: [..., n, (heads,) d] BF16; rotates the last 64 dims at `positions`, returns BF16."""
    y = x.clone()
    tail = torch.view_as_complex(y[..., -ROPE:].float().unflatten(-1, (-1, 2)).contiguous())
    f = table[positions]
    if tail.ndim == 3:
        f = f.unsqueeze(1)
    y[..., -ROPE:] = torch.view_as_real(tail * f).flatten(-2).to(y.dtype)
    return y


def e2m1(normalized):
    distance = (normalized.abs().unsqueeze(-1) - LEVELS[ORDER]).abs()
    return LEVELS[ORDER[distance.argmin(-1)]] * torch.where(torch.signbit(normalized), -1.0, 1.0)


def mxfp4(x):
    g = x.float().unflatten(-1, (-1, 32))
    amax = g.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0 ** -126) * torch.tensor(1 / 6, dtype=torch.float32)
    mantissa, exponent = torch.frexp(amax)
    scale = torch.ldexp(torch.ones_like(amax), torch.where(mantissa == 0.5, exponent - 1, exponent))
    return (e2m1((g / scale).clamp(-6, 6)) * scale).flatten(-2).to(torch.bfloat16)


def nvfp4(x, four_over_six, group=16):
    g = x.float().unflatten(-1, (-1, group))
    amax = g.abs().amax(-1, keepdim=True).clamp_min(6 * 2 ** -9)

    def candidate(divisor):
        scale = amax / divisor
        if divisor == 4:
            scale = scale.clamp_max(448)
        scale = scale.to(torch.float8_e4m3fn).float()
        restored = e2m1(g / scale) * scale
        return restored, (restored.double() - g.double()).square().sum(-1, keepdim=True)

    restored, sse = candidate(6)
    if four_over_six:
        r4, e4 = candidate(4)
        restored = torch.where(e4 < sse, r4, restored)
    return restored.flatten(-2).to(torch.bfloat16)


def fp8_row(x):
    """E4M3 with one FP32 scale per 128-value row (per token and head): the FP8 query option."""
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)


FORMATS = dict(ref=(None, None), mx=(mxfp4, mxfp4), nv6=(lambda t: nvfp4(t, False),) * 2,
               nv46=(lambda t: nvfp4(t, True),) * 2, kmx=(None, mxfp4), knv46=(None, lambda t: nvfp4(t, True)),
               k46g32=(None, lambda t: nvfp4(t, True, 32)), q8_k46=(fp8_row, lambda t: nvfp4(t, True)),
               q8_k46g32=(fp8_row, lambda t: nvfp4(t, True, 32)))


def scores(q, w, k, positions, ratio):
    """q: [P, 32, 128], w: [P, 32], k: [N, 128] (float); causal mask per query position."""
    s = torch.einsum('phd,nd->phn', q.float(), k.float()).relu_()
    s = (s * w.unsqueeze(-1)).sum(1)
    visible = (positions.unsqueeze(1) + 1) // ratio
    return s.masked_fill(torch.arange(k.size(0)).unsqueeze(0) >= visible, -torch.inf)


def compare(layer, q, w, k, positions, table, stats, context):
    ratio = RATIOS[layer]
    q_rot = rope(q, positions, table)
    k_rot = rope(k, torch.arange(k.size(0)) * ratio, table)
    top = {}
    for name, (fq, fk) in FORMATS.items():
        s = scores(fq(q_rot) if fq else q_rot, w, fk(k_rot) if fk else k_rot, positions, ratio)
        visible = ((positions + 1) // ratio).clamp_max(k.size(0))
        top[name] = (s, visible)
    s_ref, visible = top['ref']
    order = s_ref.argsort(-1, descending=True)
    for name, (s, _) in top.items():
        if name == 'ref':
            continue
        chosen = s.topk(TOPK, -1).indices
        for i in range(len(positions)):
            n = int(min(TOPK, visible[i]))
            if visible[i] <= TOPK:
                continue  # everything visible is kept in both
            kept = torch.zeros(k.size(0), dtype=torch.bool)
            kept[chosen[i, :n]] = True
            ref = order[i]
            st = stats[(context, layer, name)]
            st['recall512'].append(kept[ref[:n]].float().mean().item())
            st['drop_top64'].append(1 - kept[ref[:64]].float().mean().item())
            st['drop_top16'].append(1 - kept[ref[:16]].float().mean().item())
            st['candidates'].append(int(visible[i]))


def main():
    proj, out = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(6)
    table = freqs_cis(16384)
    records = sorted(p.stem for p in (proj / '20').glob('text-*.safetensors'))
    corpora = defaultdict(list)
    for r in records:
        corpora[r.split('-')[1]].append(r)
    for c in corpora:
        corpora[c].sort(key=lambda r: int(r.split('-')[2]))
    stats = defaultdict(lambda: defaultdict(list))
    for layer in RATIOS:
        ratio = RATIOS[layer]
        load = lambda r: load_file(str(proj / f'{layer:02d}' / f'{r}.safetensors'))
        keys = lambda r: load_file(str(proj / f'{KEY_SOURCE[layer]:02d}' / f'{r}.safetensors'))['k']
        for r in records:  # native 2048-token contexts
            d = load(r)
            positions = torch.arange(1024, 2048, 8)
            compare(layer, d['q'][positions], d['w'][positions], keys(r), positions, table, stats, 'native')
        for c, rs in corpora.items():  # stitched 16384-token contexts
            k = torch.cat([keys(r) for r in rs])
            d = load(rs[-1])
            local = torch.arange(0, 2048, 16)
            positions = local + 2048 * (len(rs) - 1)
            compare(layer, d['q'][local], d['w'][local], k, positions, table, stats, 'stitched')
        print(json.dumps(dict(layer=layer, done=True)), flush=True)
    report = defaultdict(dict)
    for (context, layer, name), st in sorted(stats.items()):
        mean = lambda v: sum(v) / len(v)
        report[context].setdefault(str(layer), {})[name] = dict(
            queries=len(st['recall512']), median_candidates=sorted(st['candidates'])[len(st['candidates']) // 2],
            recall512=round(mean(st['recall512']), 5), drop_top64=round(mean(st['drop_top64']), 5),
            drop_top16=round(mean(st['drop_top16']), 5))
    for context, layers in report.items():
        for layer, formats in layers.items():
            if 'mx' in formats and 'nv46' in formats and formats['mx']['recall512'] < 1:
                miss_mx, miss_nv = 1 - formats['mx']['recall512'], 1 - formats['nv46']['recall512']
                formats['nv46_recovers_share_of_mx_misses'] = round((miss_mx - miss_nv) / miss_mx, 4)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(stage='indexer_fp4_probe_done', out=str(out))))


if __name__ == '__main__':
    main()
