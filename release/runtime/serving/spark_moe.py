"""Sort-once routing for the pinned eager EXL3 path; no kernel/weight changes.

The reference dispatcher synchronizes a variable-size torch.where for every
active expert. This version computes the stable schedule once, reads a small
expert/count table once, and retains the original per-expert projection,
clamping, routing-weight placement, 1024-row chunking and accumulation order.
No giant gathered-activation buffer or persistent workspace is introduced.
Registration is explicit; merely importing this file changes no runtime.
"""
import hashlib
from pathlib import Path

import torch

EXPECTED = {
    'ds41.exl3_moe': 'ee9244c0bf2dc07885fbf1c0954c90695bad847c668e0e5b80405e7dcaaffa62',
    'ds41.vllm_exl3': 'd23c1e0df03cc097cad69314e53cef161cf599bba0b7b726d082cf144ffdfaa3',
}
_original = None


def grouped_moe(experts, x, route_ids, route_weights, chunk_tokens=1024):
    if (x.ndim != 2 or not 0 <= len(x) <= 1056 or route_ids.ndim != 2
            or route_ids.shape != route_weights.shape or len(route_ids) != len(x)
            or not 1 <= route_ids.shape[1] <= 6
            or route_ids.dtype not in (torch.int32, torch.int64)
            or route_ids.device != x.device or route_weights.device != x.device
            or type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 1024):
        raise ValueError('Unsupported bounded DS41 EXL3 routing contract')
    if x.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
        raise ValueError('Sort-once EXL3 dispatcher requires eager execution')
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


def register():
    global _original
    import importlib
    modules = {name: importlib.import_module(name) for name in EXPECTED}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        if (not path.is_relative_to('/opt/ds41-dcp-v3/ds41')
                or hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED[name]):
            raise RuntimeError(f'Unreviewed EXL3 serving implementation: {name}')
    reference, adapter = modules.values()
    if _original is not None:
        if adapter.eager_moe is not grouped_moe or reference.eager_moe is not _original:
            raise RuntimeError('Registered EXL3 dispatcher was replaced')
        return
    if adapter.eager_moe is not reference.eager_moe:
        raise RuntimeError('Register sort-once routing before other EXL3 dispatch patches')
    _original = reference.eager_moe
    # Only change the pinned vLLM adapter's binding; retain the eager reference.
    adapter.eager_moe = grouped_moe
