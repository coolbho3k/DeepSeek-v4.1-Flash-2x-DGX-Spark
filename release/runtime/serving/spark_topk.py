"""GB10 decode top-k for the already-installed, pinned DS41 DCP indexer.

Do not change torch.ops or the upstream launch safety check. Replace only
the persistent-top-k call in our generated DS41 indexer. Each row uses a
stable CUDA sort over its real live length, with lower-index tie breaking.
The one-row-at-a-time scratch bound also covers a full million keys.
"""
import ast
import hashlib
from pathlib import Path

import torch

INDEXER_SHA = '5b094c4280ea615eb79db26734dd8633978bed1cc1189cf278e6a229894900a4'
_installed = None


def decode_topk(logits, lengths, output, workspace, k, max_seq_len):
    if (logits.device.type != 'cuda' or logits.dtype != torch.float32
            or logits.ndim != 2 or not 1 <= logits.shape[0] <= 64
            or not 0 < logits.shape[1] <= 1048576
            or logits.stride(1) != 1
            or lengths.device != logits.device or lengths.dtype != torch.int32
            or lengths.ndim not in (1, 2) or not lengths.is_contiguous()
            or lengths.numel() != logits.shape[0]
            or output.device != logits.device or output.dtype != torch.int32
            or tuple(output.shape) != (logits.shape[0], k)
            or k not in (512, 1024, 2048) or max_seq_len != logits.shape[1]):
        raise ValueError('Unsupported DS41 decode top-k contract')
    if torch.cuda.get_device_capability(logits.device) != (12, 1):
        raise ValueError('This serving top-k overlay is qualified only for GB10')
    if torch.cuda.is_current_stream_capturing():
        raise ValueError('GB10 decode top-k requires eager execution')
    counts = lengths.reshape(-1).cpu().tolist()
    if any(not 0 <= count <= logits.shape[1] for count in counts):
        raise ValueError('Decode top-k length exceeds supplied logits')
    output.fill_(-1)
    for row, count in enumerate(counts):
        if count <= k:
            # Match native short-row behavior: expose all live indices in
            # original order, and pad. The coordinated DCP merge subsequently
            # removes nonfinite candidate-masked scores before attention.
            output[row, :count].copy_(torch.arange(count, device=logits.device, dtype=torch.int32))
        else:
            # No cooperative residency/barrier or >=128KiB block requirement.
            # At most one live row is sorted; unused stride/padding is ignored.
            indices = torch.argsort(logits[row, :count], descending=True, stable=True)
            output[row].copy_(indices[:k])
            del indices


def register():
    global _installed
    from vllm.model_executor.layers import sparse_attn_indexer as op
    if hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() != INDEXER_SHA:
        raise RuntimeError('Unreviewed native sparse indexer')
    forward = op.SparseAttnIndexer.forward_cuda
    namespace = forward.__globals__
    original = namespace.get('_ds41_indexer')
    if original is None or not hasattr(original, '__ds41_patch_source__'):
        raise RuntimeError('Install coordinated DS41 DCP hooks before GB10 top-k')
    if _installed is not None:
        if original is not _installed:
            raise RuntimeError('Installed GB10 indexer was replaced')
        return
    source = original.__ds41_patch_source__
    before = 'torch.ops._C.persistent_topk('
    if source.count(before) != 1 or '_merge_dcp_topk_global(' not in source:
        raise RuntimeError('DS41 decode top-k patch anchor changed')
    source = source.replace(before, '_ds41_gb10_decode_topk(')
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    definition.decorator_list = []
    globals_copy = dict(original.__globals__)
    globals_copy['_ds41_gb10_decode_topk'] = decode_topk
    exec(compile(tree, '<ds41-gb10-topk:sparse_attn_indexer>', 'exec'), globals_copy)
    replacement = globals_copy[original.__name__]
    replacement.__ds41_patch_source__ = source
    namespace['_ds41_indexer'] = replacement
    _installed = replacement
