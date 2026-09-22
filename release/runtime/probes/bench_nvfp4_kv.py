# SPDX-License-Identifier: AGPL-3.0-only
"""CUDA-graph KV writer timings without Python launch gaps; bounded test buffers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import types
from unittest.mock import patch

import torch


def load_package(root, name, mode):
    package = types.ModuleType(name)
    package.__path__ = [str(root / 'serving/ds41')]
    sys.modules[name] = package
    with patch.dict(os.environ, DS41_FP4_KV_MODE=mode):
        modules = []
        for filename in ('fp4_main_kv', 'fp4_rope_store'):
            spec = importlib.util.spec_from_file_location(name + '.' + filename,
                root / 'serving/ds41' / (filename + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules.append(module)
    return modules


def timing_pair(functions, repeats=21, nodes=128):
    """Interleave versions each round so serving/clock drift is shared."""
    import random
    graphs, kernels = [], []
    for fn in functions:
        for _ in range(5):
            kernel = fn()
        kernels.append(kernel)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                fn()
        graphs.append(graph)
    for _ in range(3):
        for graph in graphs:
            graph.replay()
    samples = [[] for _ in functions]
    rng = random.Random(416)
    for _ in range(repeats):
        order = list(range(len(graphs)))
        rng.shuffle(order)
        for index in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / nodes)
    return [dict(median_us=statistics.median(v), min_us=min(v), max_us=max(v),
                 p20_us=sorted(v)[len(v)//5], p80_us=sorted(v)[4*len(v)//5],
                 registers=k.n_regs, spills=k.n_spills, shared_bytes=k.metadata.shared)
            for v, k in zip(samples, kernels)]


def display_buffer(library):
    import ctypes
    lib = ctypes.CDLL(str(library))
    lib.ds41_display_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    lib.ds41_display_create.restype = ctypes.c_void_p
    lib.ds41_display_pointer.argtypes = [ctypes.c_void_p]
    lib.ds41_display_pointer.restype = ctypes.c_uint64
    lib.ds41_display_destroy.argtypes = [ctypes.c_void_p]
    lib.ds41_display_error.restype = ctypes.c_char_p
    handle = lib.ds41_display_create(0, 4 * 2**20)
    if not handle:
        raise RuntimeError(lib.ds41_display_error().decode())
    pointer = lib.ds41_display_pointer(handle)
    owner = types.SimpleNamespace(__cuda_array_interface__={
        'shape': (4 * 2**20,), 'strides': None, 'typestr': '|u1',
        'data': (pointer, False), 'version': 3})
    tensor = torch.as_tensor(owner, device='cuda:0')
    assert tensor.data_ptr() == pointer and tensor.numel() == 4 * 2**20
    return tensor, owner, lib, handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline-runtime', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--display-library', type=Path, help='Optional independent 4-MiB display-probe library')
    parser.add_argument('--warps', default='1,2,4,8')
    parser.add_argument('--selected', action='store_true', help='Benchmark the shipped decode/prefill dispatch')
    parser.add_argument('--tiles', default='', help='group-count:warps pairs; empty sweeps whole-row warps')
    parser.add_argument('--rows', default='1,4,8,24,128,512,1056,2048,3072')
    parser.add_argument('--ptx-directory', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(416)
    torch.cuda.set_per_process_memory_fraction(.002)
    versions = [('candidate', *load_package(args.runtime, 'kv_candidate', 'nvfp4_4over6')),
                ('legacy', *load_package(args.runtime, 'kv_legacy', 'legacy'))]
    if args.baseline_runtime:
        versions.append(('before', *load_package(args.baseline_runtime, 'kv_before', 'nvfp4_4over6')))
        if args.selected:
            versions.append(('legacy_before', *load_package(args.baseline_runtime, 'kv_legacy_before', 'legacy')))
    rows = []
    display = None
    if args.display_library:
        torch.empty(1, device='cuda')
        display, display_owner, display_lib, display_handle = display_buffer(args.display_library)
    for count in map(int, args.rows.split(',')):
        values = torch.randn(count, 512, device='cuda').bfloat16()
        slots = torch.arange(count, device='cuda', dtype=torch.int64)
        positions = torch.arange(1, count + 1, device='cuda', dtype=torch.int64)
        angle = torch.arange((count + 1) * 32, device='cuda').reshape(count + 1, 32).float() * .037
        cs = torch.cat((angle.cos(), angle.sin()), -1)
        states, stride = 128, 115200
        pages = (count + states - 1) // states
        backing = (torch.empty((pages, stride), device='cuda', dtype=torch.uint8) if display is None
                   else display[:pages * stride].view(pages, stride))
        cache = backing[:, :states * 288].view(-1, states, 288)
        for kind in ('plain', 'rope_cr1', 'rope_cr1_dcp', 'rope_cr2_dcp'):
            ratio = 2 if kind == 'rope_cr2_dcp' else 1
            live_slots = (torch.where((positions // ratio) % 2 == 0, positions // ratio // 2, -1)
                          if kind.endswith('_dcp') else slots)
            expected = {}
            pending = []
            for label, codec, rope in versions:
                historical = label in ('before', 'legacy_before')
                if args.selected:
                    geometries = [(32, 4)] if historical else [codec._writer_geometry(count, ratio)]
                else:
                    geometries = [tuple(map(int, pair.split(':'))) for pair in args.tiles.split(',')] if args.tiles and not historical else [(32, w) for w in map(int, args.warps.split(','))]
                for groups, warps in geometries:
                    extra = dict(GROUPS=groups) if not historical else {}
                    if kind == 'plain':
                        def fn(codec=codec, rope=rope, groups=groups, warps=warps, extra=extra):
                            return codec._store[(count, 32 // groups)](values, live_slots, cache,
                                CAPACITY=cache.shape[0]*states, VALUE_STRIDE=values.stride(0),
                                PAGE_STRIDE=stride, STATES=states, FOUR_OVER_SIX=codec.FOUR_OVER_SIX,
                                num_warps=warps, enable_fp_fusion=False, **extra)
                    else:
                        def fn(codec=codec, rope=rope, groups=groups, warps=warps, extra=extra):
                            return rope._rope_insert[(count, 32 // groups)](values, positions, cs, cache, live_slots,
                                CAPACITY=cache.shape[0]*states, POSITION_LIMIT=count + 1, COS_STRIDE=64,
                                PAGE_STRIDE=stride, STATES=states, RATIO=ratio,
                                FOUR_OVER_SIX=codec.FOUR_OVER_SIX, num_warps=warps, enable_fp_fusion=True, **extra)
                    cache.fill_(165)
                    kernel = fn()
                    mode = codec.QUANTIZATION_MODE
                    if mode not in expected:
                        expected[mode] = cache.clone()
                    else:
                        assert torch.equal(cache, expected[mode]), (count, kind, label, groups, warps)
                    pending.append((fn, dict(rows=count, kind=kind, version=label, groups=groups, warps=warps)))
                    if args.ptx_directory and count == 8:
                        args.ptx_directory.mkdir(parents=True, exist_ok=True)
                        (args.ptx_directory / f'{label}-{kind}-g{groups}-w{warps}.ptx').write_text(kernel.asm['ptx'])
            measurements = timing_pair([fn for fn, _ in pending])
            for (_, info), measured in zip(pending, measurements):
                result = dict(info, **measured)
                rows.append(result)
                print(json.dumps(result), flush=True)
    result = dict(device=torch.cuda.get_device_name(), methodology='128 CUDA graph nodes; 3 warmup and 21 timed replays, shuffled version order per round',
        concurrent_server_resident=True, actual_display_allocation_tested=display is not None,
        display_probe_bytes=0 if display is None else display.numel(),
        versions={label: {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (codec, rope)} for label,codec,rope in versions},
        max_allocated_bytes=torch.cuda.max_memory_allocated(), results=rows)
    assert result['max_allocated_bytes'] < 128 * 2**20
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if display is not None:
        del cache, backing, pending, expected, display, display_owner
        torch.cuda.synchronize()
        display_lib.ds41_display_destroy(display_handle)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
