"""Bounded, actual-GB10 EXL3 prefill diagnostic; never loads the full model.

Compare unchanged packed weights through the serving/default and forced
reconstruct paths. Separately compare eager and sort-once dispatch on real
captured routes. The dispatch fixture aliases four weights across 384 IDs:
it measures launch/routing behavior, NOT full-layer weight bandwidth or
end-to-end throughput. Nothing in this probe patches the serving runtime.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path('/work')
RESULTS = Path('/results')
BAKED = Path('/opt/ds41-dcp-v3')
FRACTION = .0075
PEAK_LIMIT = 512 * 2**20
CAPTURE = 'calibration/capture-smoke-v1/expert-inputs/00/text-0-262144.safetensors'
CAPTURE_SHA = '16be7aa5c09312442374e5ba4e83e255bccf6f40c5e77f9fe3fd9dcf9ea3b269'
EXPERTS = (0, 48, 96, 144)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusive(name, data):
    with (RESULTS / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def headroom(torch):
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.split(':')[0] in ('MemFree', 'MemAvailable')}
    free, total = torch.cuda.mem_get_info()
    assert 120 * 2**30 <= total <= 130 * 2**30
    assert memory['MemAvailable'] >= 96 * 2**30 and free >= 32 * 2**30, memory
    assert torch.cuda.max_memory_allocated() <= PEAK_LIMIT
    return dict(**memory, cuda_free_bytes=free, cuda_total_bytes=total)


def sort_once_moe(torch, experts, x, route_ids, route_weights, chunk_tokens=1024):
    """Experimental schedule only; preserves expert and row accumulation order."""
    if route_ids.shape != route_weights.shape or len(route_ids) != len(x):
        raise ValueError('Routing/input shape mismatch')
    output = torch.zeros_like(x, dtype=torch.float32)
    if not route_ids.numel():
        return output.to(x.dtype)
    sorted_ids, positions = torch.sort(route_ids.reshape(-1), stable=True)
    active, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    active_cpu, counts_cpu = torch.stack((active, counts)).cpu().tolist()
    all_rows = torch.div(positions, route_ids.shape[1], rounding_mode='floor')
    all_weights = route_weights.reshape(-1).index_select(0, positions)
    offset = 0
    for expert_id, count in zip(active_cpu, counts_cpu):
        expert = experts.get(expert_id)
        if expert is not None:
            for start in range(offset, offset + count, chunk_tokens):
                end = min(start + chunk_tokens, offset + count)
                rows = all_rows[start:end]
                values = expert(x[rows].half().contiguous(), all_weights[start:end, None])
                output.index_add_(0, rows, values.float())
        offset += count
    return output.to(x.dtype)


def measure_pair(torch, functions, repeats=5):
    """Interleave paths; stream elapsed includes launch gaps, not just kernel time."""
    records = {name: [] for name in functions}
    for function in functions.values():
        for _ in range(2):
            function()
    torch.cuda.synchronize()
    for iteration in range(repeats):
        names = list(functions)
        if iteration % 2:
            names.reverse()
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            result = functions[name]()
            end.record()
            end.synchronize()
            records[name].append(dict(wall_ms=(time.perf_counter() - wall) * 1000,
                                      stream_elapsed_ms=start.elapsed_time(end)))
            del result
    return {name: dict(samples=samples,
            median_wall_ms=statistics.median(s['wall_ms'] for s in samples),
            median_stream_elapsed_ms=statistics.median(s['stream_elapsed_ms'] for s in samples))
            for name, samples in records.items()}


def error(torch, actual, expected):
    assert actual.shape == expected.shape and torch.isfinite(actual).all()
    delta = actual.float() - expected.float()
    return dict(nmse=(delta.square().sum() / expected.float().square().sum().clamp_min(1e-30)).item(),
                max_abs=delta.abs().max().item() if delta.numel() else 0.,
                exact=torch.equal(actual, expected))


def reconstructed_expert(torch, expert, x, weights):
    from torch.nn import functional as F
    gate = expert.layers['w1'].forward(x, {'reconstruct': True}).float()
    up = expert.layers['w3'].forward(x, {'reconstruct': True}).float()
    if expert.limit > 0:
        gate = gate.clamp(max=expert.limit)
        up = up.clamp(min=-expert.limit, max=expert.limit)
    down = (F.silu(gate) * up * weights.float()).half().contiguous()
    return expert.layers['w2'].forward(down, {'reconstruct': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--image-id', required=True)
    args = parser.parse_args()
    assert not any((RESULTS / name).exists() for name in ('attempt.json', 'complete.json', 'failed.json'))
    build_path = ROOT / 'reports/runtime-overlay-v3-build/complete.json'
    build = json.loads(build_path.read_text())
    assert build['status'] == 'built_and_source_manifest_verified'
    pins = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
            str(build_path.relative_to(ROOT)): digest(build_path), CAPTURE: digest(ROOT / CAPTURE)}
    assert pins[CAPTURE] == CAPTURE_SHA
    inventory = ROOT / 'reports/quant-layer-00-3bit.json'
    entries = json.loads(inventory.read_text())
    assert entries['status'] == 'verified' and entries['bits'] == 3 and entries['experts'] == 384
    entries = {entry['expert']: entry for entry in entries['artifacts']}
    pins[str(inventory.relative_to(ROOT))] = digest(inventory)
    for expert in EXPERTS:
        relative = f'calibration/candidates-source-v1/layer-00/expert-{expert:03d}-3bit.safetensors'
        pins[relative] = digest(ROOT / relative)
        assert pins[relative] == entries[expert]['artifact_sha256']
    exclusive('attempt.json', dict(image_id=args.image_id, tp_rank=args.tp_rank, tp_world_size=2,
        input_sha256=pins, allocator_fraction=FRACTION, peak_limit_bytes=PEAK_LIMIT,
        full_model=False, actual_collectives=False, serving_modified=False))
    try:
        import torch
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(FRACTION)
        assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (12, 1)
        samples = [headroom(torch)]
        from safetensors.torch import load_file
        import ds41.exl3_moe as implementation
        from exllamav3.modules.quant import exl3
        actual_source = Path(implementation.__file__).resolve()
        assert actual_source == BAKED / 'ds41/exl3_moe.py'
        assert digest(actual_source) == build['source_sha256']['ds41/exl3_moe.py']
        assert Path(exl3.__file__).resolve().is_relative_to('/opt/exllamav3')
        fixture = load_file(ROOT / CAPTURE)
        bank = {}
        for expert_id in EXPERTS:
            tensors = load_file(ROOT / f'calibration/candidates-source-v1/layer-00/expert-{expert_id:03d}-3bit.safetensors', device='cuda')
            bank[expert_id] = implementation.PackedExpert(tensors, f'layers.0.ffn.experts.{expert_id}',
                                                         args.tp_rank, 2, limit=10.)
        del tensors
        input_x = fixture['inputs'][:1056].cuda()
        input_ids = fixture['route_ids'][:1056].cuda()
        input_weights = fixture['route_weights'][:1056].cuda()
        unique_ids, route_counts = torch.unique(input_ids, return_counts=True)
        route_distribution = dict(tokens=len(input_x), top_k=input_ids.shape[1],
            active_experts=len(unique_ids), min_rows=int(route_counts.min()),
            median_rows=float(route_counts.float().median()), max_rows=int(route_counts.max()),
            rows_above_auto_threshold=int((route_counts > exl3.AUTO_RECONSTRUCT_THRESHOLD).sum()),
            ids=unique_ids.cpu().tolist(), counts=route_counts.cpu().tolist())
        rows_results, projections, dispatch = [], [], []
        with torch.inference_mode():
            for expert_id, expert in bank.items():
                for rows in (1, 8, 16, 32, 64, 128, 256, 512):
                    x = input_x[:rows].half().contiguous()
                    weights = input_weights[:rows, :1].contiguous()
                    functions = {'default': lambda: expert(x, weights),
                                 'reconstruct': lambda: reconstructed_expert(torch, expert, x, weights)}
                    parity = error(torch, functions['reconstruct'](), functions['default']())
                    result = dict(expert=expert_id, rows=rows, parity=parity,
                                  timings=measure_pair(torch, functions))
                    rows_results.append(result)
                    print(json.dumps(dict(stage='expert', **result)), flush=True)
                    # A diagnostic records drift instead of blessing a new serving path.
                samples.append(headroom(torch))
            expert = bank[0]
            for name, layer in expert.layers.items():
                for rows in (16, 32, 128, 256):
                    x = input_x[:rows, :layer.in_features].half().contiguous()
                    functions = {'default': lambda: layer.forward(x, {}),
                                 'reconstruct': lambda: layer.forward(x, {'reconstruct': True})}
                    projections.append(dict(projection=name, rows=rows,
                        parity=error(torch, functions['reconstruct'](), functions['default']()),
                        timings=measure_pair(torch, functions)))
            # Real route spread; alias weights to keep the diagnostic safely small.
            aliased_bank = {index: bank[EXPERTS[index % len(EXPERTS)]] for index in range(384)}
            for rows in (1, 128, 1056):
                x, ids, weights = input_x[:rows], input_ids[:rows], input_weights[:rows]
                functions = {'eager': lambda: implementation.eager_moe(aliased_bank, x, ids, weights),
                             'sort_once': lambda: sort_once_moe(torch, aliased_bank, x, ids, weights)}
                parity = error(torch, functions['sort_once'](), functions['eager']())
                assert parity['exact'], ('Dispatch must preserve exact existing arithmetic', rows, parity)
                result = dict(rows=rows, aliased_weights=True, parity=parity,
                              timings=measure_pair(torch, functions, repeats=7))
                dispatch.append(result)
                print(json.dumps(dict(stage='dispatch', **result)), flush=True)
        samples.append(headroom(torch))
        gc.collect()
        assert pins == {relative: digest(ROOT / relative) for relative in pins}
        report = dict(status='bounded_exl3_prefill_diagnostic_complete', tp_rank=args.tp_rank,
            tp_world_size=2, image_id=args.image_id, input_sha256=pins,
            loaded_expert_module=str(actual_source), expert_source_sha256=digest(actual_source),
            linear_module=str(Path(exl3.__file__).resolve()), linear_source_sha256=digest(Path(exl3.__file__)),
            reconstruct_threshold=exl3.AUTO_RECONSTRUCT_THRESHOLD,
            fused_reconstruct_available=hasattr(implementation.exllamav3_ext, 'reconstruct_had_slice'),
            route_distribution=route_distribution, expert_cases=rows_results,
            projection_cases=projections, dispatch_cases=dispatch, headroom_samples=samples,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), allocator_fraction=FRACTION,
            full_model_loaded=False, actual_collectives=False, serving_modified=False,
            full_model_speedup_proven=False, quantization_quality_qualified=False,
            scope='Four existing real 3-bit layer-0 experts with true TP2 matrix slicing, measured independently per rank. Actual captured inputs/routes. Aliased four-weight dispatch excludes full-layer weight bandwidth. Interleaved wall/stream timings include CPU launch gaps. No attention, engrams, NCCL or full-model performance claim.')
        exclusive('complete.json', report)
        print(json.dumps(dict(status=report['status'], tp_rank=args.tp_rank,
            peak_allocated_bytes=report['peak_allocated_bytes'], dispatch_cases=dispatch)), flush=True)
    except Exception as exc:
        exclusive('failed.json', dict(status='failed', error=repr(exc),
            instruction='Inspect retained exact container and logs; no automatic retry or relaxed bound.'))
        raise


if __name__ == '__main__':
    main()
