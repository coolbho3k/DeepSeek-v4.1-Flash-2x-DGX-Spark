# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded GPU proof of NVFP4 4/6 bytes, accuracy, RoPE, graphs and readers.

Run with the serving image; no weights, live KV or server hooks are touched.
--baseline-codec optionally verifies against the previous writer on the GPU.
The independent oracle uses nearest-code distances and FP64 reconstruction SSE.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from unittest.mock import patch
from pathlib import Path
import sys
import types

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def oracle(values, four_over_six=True):
    x = values.cpu().float().reshape(-1, 32, 16)
    # Even codes precede odd codes, so argmin implements nearest-even ties.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    amax = x.abs().amax(-1).clamp_min(6 * 2**-9)

    def candidate(divisor):
        scale = amax / divisor
        if divisor == 4:
            scale = scale.clamp_max(448)
        scale = scale.to(torch.float8_e4m3fn)
        normalized = x / scale.float()[..., None]
        distance = (normalized.abs()[..., None] - levels[order]).abs()
        codes = order[distance.argmin(-1)] | (torch.signbit(x).long() << 3)
        restored = levels[codes & 7] * scale.float()[..., None]
        restored = torch.where((codes & 8) != 0, -restored, restored)
        sse = (restored.double() - x.double()).square().sum(-1)
        return codes, scale.view(torch.uint8), restored, sse

    codes, scales, restored, sse = candidate(6)
    if four_over_six:
        c4, s4, r4, e4 = candidate(4)
        use4 = e4 < sse
        codes = torch.where(use4[..., None], c4, codes)
        scales = torch.where(use4, s4, scales)
        restored = torch.where(use4[..., None], r4, restored)
    codes = codes.reshape(-1, 512)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).byte()
    return torch.cat((packed, scales), -1), restored.reshape(-1, 512).bfloat16()


def errors(x, y):
    return (x.cpu().double() - y.cpu().double()).square().reshape(-1, 32, 16).sum(-1)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(416)
    torch.cuda.set_per_process_memory_fraction(.002)
    root = args.runtime.resolve()
    package = types.ModuleType('nvfp4_probe')
    package.__path__ = [str(root / 'serving/ds41')]
    sys.modules[package.__name__] = package
    codec = load('nvfp4_probe.fp4_main_kv', root / 'serving/ds41/fp4_main_kv.py')
    rope = load('nvfp4_probe.fp4_rope_store', root / 'serving/ds41/fp4_rope_store.py')
    assert codec.QUANTIZATION_MODE == 'nvfp4_4over6'
    old_package = types.ModuleType('nvfp4_legacy_probe')
    old_package.__path__ = package.__path__
    sys.modules[old_package.__name__] = old_package
    with patch.dict(os.environ, DS41_FP4_KV_MODE='legacy'):
        legacy = load('nvfp4_legacy_probe.fp4_main_kv', root / 'serving/ds41/fp4_main_kv.py')
        legacy_rope = load('nvfp4_legacy_probe.fp4_rope_store', root / 'serving/ds41/fp4_rope_store.py')
    baseline = load('nvfp4_baseline', args.baseline_codec) if args.baseline_codec else None
    cases = []
    total_old = total_new = 0.
    changed = groups = 0
    inputs = {
        'zero': torch.zeros(1, 512),
        'signed_zero_and_ties': torch.tensor([6., 0., -0., .25, -.25, .75, -.75,
            1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.]).repeat(32).reshape(1, 512),
        # Two distinct candidates have identical exact SSE; a FP32 tree
        # reduction alone used to choose /4 for this group.
        'exact_sse_tie': torch.tensor([-.28515625, -.13671875, .671875, .59375,
            -1.6171875, .62890625, .953125, -.0037689208984375, -1.3359375,
            1.0390625, -1.71875, -1.8359375, 1.8359375, .470703125,
            -.0615234375, -.20703125]).repeat(32).reshape(1, 512),
        'tiny': torch.randn(33, 512) * 1e-5,
        'normal': torch.randn(1056, 512),
        'trained_range': (torch.randn(513, 512) * 3).clamp(-22.5, 22.5),
        'wide_scales': torch.randn(257, 32, 16).clamp(-4, 4).mul(
            torch.logspace(-6, 2, 32)[None, :, None]).reshape(257, 512),
        'e4m3_upper_boundary': torch.linspace(-2688, 2688, 512).repeat(7, 1),
    }
    # Heavy outliers and correlated channels challenge amax-only selection.
    heavy = torch.randn(129, 32, 16)
    heavy[..., 0] *= 12
    inputs['outliers'] = heavy.clamp(-22.5, 22.5).reshape(129, 512)
    for label, cpu in inputs.items():
        values = cpu.bfloat16().cuda()
        rows, states = len(values), 64
        pages = (rows + 2 + states - 1) // states
        backing = torch.full((pages, states * 288 + 512), 165, device='cuda', dtype=torch.uint8)
        cache = backing[:, 256:-256].view(pages, states, 288)
        slots = torch.arange(rows, device='cuda', dtype=torch.int64) + 2
        want, restored = oracle(values)
        old_bytes, old = oracle(values, False)
        codec.store(cache, values, slots)
        actual = cache[slots // states, slots % states].cpu()
        assert torch.equal(actual, want), (label, 'packed oracle mismatch')
        got = codec.gather(cache, slots).cpu()
        assert torch.equal(got, restored), (label, 'gather oracle mismatch')
        for old_codec in (legacy, baseline) if baseline else (legacy,):
            old_cache = torch.empty_like(cache)
            old_codec.store(old_cache, values, slots)
            assert torch.equal(old_cache[slots // states, slots % states].cpu(), old_bytes)
            assert torch.equal(old_codec.gather(old_cache, slots).cpu(), old)
        old_error, new_error = errors(values, old), errors(values, got)
        assert torch.all(new_error <= old_error), (label, 'group regression')
        tied = new_error == old_error
        # All ties preserve BOTH the scale and signed E2M1 payload.
        old_groups = old_bytes[:, :256].reshape(rows, 32, 8)
        new_groups = actual[:, :256].reshape(rows, 32, 8)
        assert torch.equal(new_groups[tied], old_groups[tied])
        assert torch.equal(actual[:, 256:][tied], old_bytes[:, 256:][tied])
        assert torch.all(backing[:, :256] == 165) and torch.all(backing[:, -256:] == 165)
        assert torch.all(cache[0, :2] == 165)
        snapshot = backing.clone()
        invalid = torch.tensor([-1, -9, pages * states], device='cuda')
        codec.store(cache, values[:1].expand(3, -1).contiguous(), invalid, check_bounds=False)
        assert torch.equal(snapshot, backing)
        assert not codec.gather(cache, invalid, check_bounds=False).count_nonzero()
        selected = torch.cat((slots.flip(0).int(), invalid[:2].int()))
        gathered = codec.gather(cache, selected).cpu()
        assert torch.equal(gathered[:-2], got.flip(0)) and not gathered[-2:].count_nonzero()
        before, after = old_error.sum().item(), new_error.sum().item()
        improved = (new_error < old_error).sum().item()
        if label in ('normal', 'trained_range', 'wide_scales', 'outliers'):
            assert after < before and improved > 0, (label, 'no strict improvement')
        cases.append(dict(case=label, groups=rows * 32, improved=improved,
            regressed=0, baseline_sse=before, four_over_six_sse=after))
        print(json.dumps(cases[-1]), flush=True)
        total_old += before
        total_new += after
        changed += improved
        groups += rows * 32

    # Real native BF16 rotation is the input oracle for the fused writer.
    # Load the installed, unchanged native source without importing the full
    # serving stack. Only its platform/import adapters are supplied here.
    import triton
    import triton.language as tl
    native_path = Path(importlib.util.find_spec('vllm').origin).parent / 'models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py'
    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == 'cc486feefe40871f010c38ee97a2217ca70d86b7f42cdfc5884399045d5ac385'
    platform = types.ModuleType('vllm.platforms')
    platform.current_platform = types.SimpleNamespace(is_rocm=lambda: False, is_cuda=lambda: True)
    utils = types.ModuleType('vllm.triton_utils')
    utils.triton, utils.tl = triton, tl
    with patch.dict(sys.modules, {'vllm.platforms': platform, 'vllm.triton_utils': utils}):
        native = load('nvfp4_native_rope', native_path)
    rope_quant_insert = native.rope_quant_insert
    rope_cases = 0
    for ratio in (1, 2):
        for rank in (0, 1):
            for count, base in ((0, 0), (1, 0), (5, 1), (33, 127), (1056, 7)):
                positions = torch.arange(base, base + count, device='cuda')
                latent = torch.randn(count, 512, device='cuda').bfloat16()
                angle = torch.arange(max(base + count, 1) * 32, device='cuda').reshape(-1, 32).float() * .037
                cs = torch.cat((angle.cos(), angle.sin()), -1).contiguous()
                slots = torch.where((positions // ratio) % 2 == rank, positions // ratio // 2 + 2, -1)
                live = (slots >= 0) & ((positions + 1) % ratio == 0)
                latent[~live] = float('nan')
                backing = torch.full((16, 64 * 288 + 512), 197, device='cuda', dtype=torch.uint8)
                cache = backing[:, 256:-256].view(16, 64, 288)
                expected = backing.clone()
                plain = torch.empty(16, 64, 512, device='cuda', dtype=torch.bfloat16)
                rope_quant_insert(latent, positions, cs, plain, slots, ratio)
                rope.rope_quant_insert(latent, positions, cs, cache, slots, ratio)
                indices = slots[live]
                if indices.numel():
                    rotated = plain[indices // 64, indices % 64].contiguous()
                    packed, restored = oracle(rotated)
                    expected[indices[:, None] // 64,
                        256 + indices[:, None] % 64 * 288 + torch.arange(288, device='cuda')] = packed.cuda()
                    assert torch.equal(codec.gather(cache, indices).cpu(), restored)
                assert torch.equal(backing, expected), ('fused RoPE mismatch', ratio, rank, count)
                legacy_backing = torch.full_like(backing, 197)
                legacy_cache = legacy_backing[:, 256:-256].view(16, 64, 288)
                legacy_expected = torch.full_like(backing, 197)
                legacy_rope.rope_quant_insert(latent, positions, cs, legacy_cache, slots, ratio)
                if indices.numel():
                    old_packed, old_restored = oracle(rotated, False)
                    legacy_expected[indices[:, None] // 64,
                        256 + indices[:, None] % 64 * 288 + torch.arange(288, device='cuda')] = old_packed.cuda()
                    assert torch.equal(legacy.gather(legacy_cache, indices).cpu(), old_restored)
                assert torch.equal(legacy_backing, legacy_expected)
                rope_cases += 1

    # No host bounds synchronization is required inside the native CUDA graph.
    values = inputs['normal'][:33].bfloat16().cuda()
    slots = torch.arange(33, device='cuda')
    cache = torch.empty(1, 64, 288, device='cuda', dtype=torch.uint8)
    codec.store(cache, values, slots, check_bounds=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codec.store(cache, values, slots, check_bounds=False)
    values.mul_(2)
    graph.replay()
    assert torch.equal(cache[0, :33].cpu(), oracle(values)[0])

    # Exercise the production packed attention reader against an independent
    # dense attention oracle using the exact decoded cache.
    fused = load('nvfp4_probe.fused_sparse_attention', root / 'serving/ds41/fused_sparse_attention.py')
    query = (torch.randn(2, 32, 512, device='cuda') * .2).bfloat16()
    swa = torch.zeros(1, 64, 584, device='cuda', dtype=torch.uint8)
    si = torch.zeros(2, 1, device='cuda', dtype=torch.int32)
    sl = torch.zeros(2, device='cuda', dtype=torch.int32)
    ci = slots.int().expand(2, -1).contiguous()
    cl = torch.full((2,), 33, device='cuda', dtype=torch.int32)
    output, lse = fused.packed_sparse_attention_with_lse(query, swa, si, sl,
        compressed_cache=cache, compressed_indices=ci, compressed_lengths=cl)
    decoded = codec.gather(cache, slots).float()
    logits = query.float() @ decoded.T * 512**-.5
    expected = logits.softmax(-1) @ decoded
    nmse = ((output.float() - expected).square().sum() / expected.square().sum()).item()
    assert nmse < 1e-6, nmse
    original = values.float()
    original_logits = query.float() @ original.T * 512**-.5
    original_output = original_logits.softmax(-1) @ original
    legacy_values = oracle(values, False)[1].cuda().float()
    legacy_logits = query.float() @ legacy_values.T * 512**-.5
    legacy_output = legacy_logits.softmax(-1) @ legacy_values
    attention_old_sse = (legacy_output - original_output).double().square().sum().item()
    attention_new_sse = (output.float() - original_output).double().square().sum().item()
    assert attention_new_sse < attention_old_sse, (attention_old_sse, attention_new_sse)

    # Exercise the public 2048-token prefill bound, which the existing
    # combined runtime admits by raising both writer limits at startup.
    codec.MAX_WRITE_ROWS = rope.MAX_WRITE_ROWS = 2048
    batch = torch.randn(2048, 512, device='cuda').bfloat16()
    batch_slots = torch.arange(2048, device='cuda')
    batch_cache = torch.empty(1, 2048, 288, device='cuda', dtype=torch.uint8)
    codec.store(batch_cache, batch, batch_slots, check_bounds=False)
    assert torch.equal(batch_cache[0].cpu(), oracle(batch)[0])
    batch_positions = torch.arange(2048, device='cuda')
    identity_cs = torch.cat((torch.ones(2048, 32, device='cuda'),
                             torch.zeros(2048, 32, device='cuda')), -1)
    rope.rope_quant_insert(batch, batch_positions, identity_cs, batch_cache, batch_slots, 1, check_bounds=False)
    assert torch.equal(batch_cache[0].cpu(), oracle(batch)[0])

    # Record bounded timings after compilation; they are diagnostic under a
    # concurrently running server, not isolated end-to-end serving benchmarks.
    timings = {}
    for count in (1, 8, 1056):
        values = inputs['normal'][:count].bfloat16().cuda()
        slots = torch.arange(count, device='cuda')
        cache = torch.empty(1, count, 288, device='cuda', dtype=torch.uint8)
        timings[str(count)] = {}
        for label, impl in [('four_over_six', codec), ('legacy', legacy)] + ([('baseline', baseline)] if baseline else []):
            for _ in range(3):
                impl.store(cache, values, slots, check_bounds=False)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(30):
                impl.store(cache, values, slots, check_bounds=False)
            end.record()
            end.synchronize()
            timings[str(count)][label + '_ms'] = start.elapsed_time(end) / 30
    assert total_new < total_old
    assert torch.cuda.max_memory_allocated() < 128 * 2**20
    return dict(status='pass', device=torch.cuda.get_device_name(), cases=cases,
        groups=groups, improved_groups=changed, regressed_groups=0,
        baseline_sse=total_old, four_over_six_sse=total_new,
        reduction_percent=100 * (1 - total_new / total_old),
        state_bytes=288, bits_per_value=288 * 8 / 512,
        fused_rope_cases=rope_cases, cuda_graph_replay=True,
        packed_attention_nmse=nmse, writer_timings=timings,
        attention_vs_bf16=dict(legacy_sse=attention_old_sse, four_over_six_sse=attention_new_sse),
        public_prefill_2048_checked=True,
        baseline_gpu_checked=baseline is not None, legacy_exact=True,
        baseline_sha256=hashlib.sha256(args.baseline_codec.read_bytes()).hexdigest() if baseline else None,
        codec_sha256=hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(),
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
        full_model_quality_tested=False, actual_display_allocation_tested=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline-codec', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        result = run(args)
    raw = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(raw)
    print(raw, flush=True)


if __name__ == '__main__':
    main()
