"""Eager V4.1 compressed-state DCP ownership and paged-address primitives.

Compression precedes ownership. Only completed states are interleaved across
ranks; SWA/compressor rings stay replicated. Not installed into vLLM yet.
"""
import torch


def validate_coordinates(ratio, world_size, rank):
    if ratio not in (1, 2) or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Unsupported compression or DCP coordinates')


def local_compressed_lengths(global_lengths, ratio, world_size, rank):
    validate_coordinates(ratio, world_size, rank)
    states = global_lengths.clamp_min(0).div(ratio, rounding_mode='floor')
    return (states + world_size - 1 - rank).div(world_size, rounding_mode='floor')


def compressed_slot_mapping(num_tokens, query_start_loc, seq_lens, block_table,
                            storage_block_size, ratio, world_size, rank, out=None):
    """Physical write slots for complete compressed states, interleave=1.

    Supports decode, chunked prefill, padded requests and noncontiguous physical
    block IDs. No ownership is inferred from the raw source-token parity.
    Eager validation uses scalar synchronizations; graph capture is not supported.
    """
    validate_coordinates(ratio, world_size, rank)
    num_reqs = seq_lens.numel()
    if (num_tokens < 0 or storage_block_size < 1 or block_table.ndim != 2
            or block_table.shape[0] != num_reqs or query_start_loc.numel() != num_reqs + 1):
        raise ValueError('Invalid request metadata shapes')
    if out is None:
        out = torch.full((num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device)
    elif out.ndim != 1 or out.numel() < num_tokens or out.dtype != torch.int64:
        raise ValueError('Invalid output slot buffer')
    out.fill_(-1)
    slots = out[:num_tokens]
    if not num_reqs or not num_tokens:
        return slots
    if (query_start_loc[0].item() != 0 or query_start_loc[-1].item() > num_tokens
            or (query_start_loc[1:] < query_start_loc[:-1]).any().item()):
        raise ValueError('Invalid query boundaries')
    token = torch.arange(num_tokens, device=query_start_loc.device)
    req = torch.bucketize(token, query_start_loc[1:].contiguous(), right=True)
    valid_req = req < num_reqs
    req = req.clamp_max(num_reqs - 1)
    query_len = query_start_loc[req + 1] - query_start_loc[req]
    position = seq_lens[req] - query_len + token - query_start_loc[req]
    state = position.div(ratio, rounding_mode='floor')
    valid = valid_req & (position >= 0) & ((position + 1).remainder(ratio) == 0) & (state.remainder(world_size) == rank)
    local_state = state.div(world_size, rounding_mode='floor')
    block_index = local_state.div(storage_block_size, rounding_mode='floor')
    if ((block_index >= block_table.shape[1]) & valid).any().item():
        raise ValueError('Allocated block table is too short for an owned state')
    if block_table.shape[1] == 0:
        if valid.any().item():
            raise ValueError('No blocks allocated for valid states')
        return slots
    block = block_table[req, block_index.clamp(0, block_table.shape[1] - 1).long()]
    if ((block < 0) & valid).any().item():
        raise ValueError('Unallocated physical block for an owned state')
    physical = block.long() * storage_block_size + local_state.remainder(storage_block_size)
    slots.copy_(torch.where(valid, physical, -1))
    return slots


def sparse_global_to_local_slots(indices, lengths, req_ids, block_table,
                                 storage_block_size, world_size, rank):
    """Partition global compressed candidates and map owned rows into pages."""
    validate_coordinates(1, world_size, rank)
    if indices.ndim != 2 or req_ids.numel() != indices.shape[0] or lengths.numel() != indices.shape[0]:
        raise ValueError('Invalid sparse candidate metadata')
    width = indices.shape[1]
    position = torch.arange(width, device=indices.device)
    valid = (position < lengths[:, None]) & (indices >= 0) & (indices.remainder(world_size) == rank)
    local = indices.div(world_size, rounding_mode='floor')
    block_index = local.div(storage_block_size, rounding_mode='floor')
    if (req_ids < 0).any().item() or (req_ids >= block_table.shape[0]).any().item():
        raise ValueError('Invalid request index')
    if ((block_index >= block_table.shape[1]) & valid).any().item():
        raise ValueError('Sparse candidate exceeds allocated block table')
    if block_table.shape[1] == 0:
        if valid.any().item():
            raise ValueError('No blocks allocated for sparse candidates')
        return torch.full_like(indices, -1), torch.zeros_like(lengths)
    block = block_table[req_ids[:, None].long(), block_index.clamp(0, block_table.shape[1] - 1).long()]
    if ((block < 0) & valid).any().item():
        raise ValueError('Unallocated sparse candidate page')
    physical = block.long() * storage_block_size + local.remainder(storage_block_size)
    order = torch.argsort(torch.where(valid, position, width), dim=-1, stable=True)
    mapped = torch.where(valid, physical, -1).gather(-1, order).to(indices.dtype)
    return mapped.contiguous(), valid.sum(-1).to(lengths.dtype)
