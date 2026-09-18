"""Canonicalize an already selected DCP top-k, without changing membership.

Candidate implementation, not registered into serving. Native stable top-k
stabilizes tie membership but emits selected IDs with atomic appends. Descending
integer ID order gives one reduction order, with negative padding at the end.
"""
import torch


def canonicalize_selected_keys_(indices):
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int32
            or indices.ndim != 2 or indices.shape[1] != 512
            or not 0 <= indices.shape[0] <= 1056):
        raise ValueError('Expected bounded int32 [0..1056,512] selected keys')
    # Sorting equal integer values cannot change the result, so a stable-sort
    # flag is unnecessary. Preserve duplicates and all padding values exactly.
    # The temporary values+int64 permutation are bounded by 1056*512*12 bytes.
    values = torch.sort(indices, dim=-1, descending=True).values
    indices.copy_(values)
    return indices
