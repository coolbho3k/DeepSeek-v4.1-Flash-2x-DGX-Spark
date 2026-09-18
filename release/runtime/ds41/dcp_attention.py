"""SM12 sparse-attention primitives used by the opt-in V4.1 DCP overlay.

Native metadata/indexing and local arithmetic are under test. Real two-node
collectives, full-model fit and end-to-end quality remain unqualified.
"""
import math

import torch


def partition_indices(indices, lengths, rank, world_size, *, localize):
    """Stable partition of logical sparse positions, retaining fixed width.

    With localize=True the cache rows must be interleaved by world_size.
    With localize=False rows stay replicated (e.g. small SWA state), but each
    logical entry contributes on only one rank. This is NOT a replacement
    for per-request paged block-table translation.
    """
    if not 0 <= rank < world_size or world_size < 1:
        raise ValueError("Invalid DCP rank/world size")
    width = indices.shape[-1]
    flat = indices.reshape(-1, width)
    if lengths.numel() != flat.shape[0]:
        raise ValueError("One sparse length is required per row")
    positions = torch.arange(width, device=indices.device)
    valid = ((positions < lengths.reshape(-1, 1)) & (flat >= 0)
             & (flat.remainder(world_size) == rank))
    order = torch.argsort(torch.where(valid, positions, width), dim=-1, stable=True)
    values = flat.div(world_size, rounding_mode="floor") if localize else flat
    values = torch.where(valid, values, -1).gather(-1, order)
    return values.reshape_as(indices).contiguous(), valid.sum(-1).to(torch.int32)


def compressed_local_positions(positions, ratio, rank, world_size):
    """Compress first, then DCP-interleave completed compressed states.

    Applying ownership to source-token positions before ratio-2 compression
    would send every completed state to one rank under token interleave=1.
    """
    if ratio not in (1, 2) or not 0 <= rank < world_size:
        raise ValueError("Unsupported compression or DCP coordinates")
    compressed = positions.div(ratio, rounding_mode="floor")
    valid = ((positions >= 0) & ((positions + 1).remainder(ratio) == 0)
             & (compressed.remainder(world_size) == rank))
    return torch.where(valid, compressed.div(world_size, rounding_mode="floor"), -1)


def sparse_attention_with_lse(query, swa_cache, swa_indices, swa_lengths, workspace,
                              *, compressed_cache=None, compressed_indices=None,
                              compressed_lengths=None, sinks=None, scale=None):
    """Call the real SM120/SM121 multi-segment kernel, preserving its LSE."""
    from flashinfer.mla._core import _SparseMLASegment, _trtllm_batch_decode_sparse_mla_sm120

    if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != 512:
        raise ValueError("V4.1 requires [tokens, heads, 512] BF16 queries")
    for cache in (swa_cache, compressed_cache):
        if cache is not None and (cache.dtype != torch.uint8 or cache.shape[-1] != 584):
            raise ValueError("The pinned SM12 kernel requires packed FP8 DSV4 cache pages (584 bytes/state)")
    if (compressed_indices is None) != (compressed_lengths is None):
        raise ValueError("Compressed sparse indices and lengths must be provided together")
    segments = [_SparseMLASegment(indices=swa_indices, lengths=swa_lengths)]
    if compressed_indices is not None:
        if compressed_cache is None:
            raise ValueError("Compressed indices require a cache")
        segments.append(_SparseMLASegment(indices=compressed_indices, lengths=compressed_lengths,
                                         kv_cache=compressed_cache))
    result, lse = _trtllm_batch_decode_sparse_mla_sm120(
        query=query.unsqueeze(1), kv_cache=swa_cache, workspace_buffer=workspace,
        sparse_mla_segments=segments, out=None, sm_scale=scale or 512**-0.5,
        sinks=sinks, lse=None, return_lse=True, kv_scale_format="auto")
    return result.squeeze(1), lse.reshape(query.shape[:2])


def split_sink(sinks, world_size):
    """Each rank gets 1/world_size of the single global sink's softmax mass."""
    if world_size < 1:
        raise ValueError("Invalid DCP world size")
    return sinks.float() - math.log(world_size)


def merge_outputs(outputs, lses, *, lse_base):
    """Numerical reference merge; runtime will use real DCP collectives."""
    if outputs.shape[:-1] != lses.shape or lse_base not in (2, math.e):
        raise ValueError("Mismatched DCP outputs/LSE or invalid logarithm base")
    log_normalizers = lses.float() * math.log(lse_base)
    merged_lse = torch.logsumexp(log_normalizers, dim=0)
    factors = torch.exp(log_normalizers - merged_lse.unsqueeze(0))
    factors = torch.where(torch.isfinite(factors), factors, 0)
    corrected = torch.where(factors.unsqueeze(-1) > 0,
                            outputs.float() * factors.unsqueeze(-1), 0)
    return corrected.sum(0).to(outputs.dtype), merged_lse / math.log(lse_base)


def gather_packed_cache(cache, slots):
    """Gather requested packed FP8 DSV4 rows into BF16, respecting page strides.

    Cache storage remains sharded/packed. Only the selected sparse rows are
    expanded, in bounded token chunks by the caller. Negative slots are zeros.
    """
    if cache.ndim != 3 or cache.dtype != torch.uint8 or cache.shape[-1] != 584:
        raise ValueError('Expected packed DSV4 pages [pages, states, 584]')
    valid = slots >= 0
    block_size = cache.shape[1]
    if ((slots >= cache.shape[0] * block_size) & valid).any().item():
        raise ValueError('Sparse slot exceeds allocated packed cache')
    if cache.shape[0] == 0:
        return torch.zeros((*slots.shape, 512), device=cache.device, dtype=torch.bfloat16)
    if cache.is_cuda:
        # Decode directly into the final BF16 buffer; the eager indexed path
        # below creates large int64 offset and FP32 conversion matrices.
        from .dcp_cache_gather import gather
        return gather(cache, slots)
    # Retain the original CPU arithmetic as a reference implementation.
    safe = slots.clamp_min(0).long()
    pages, rows = safe // block_size, safe % block_size
    byte_pages = cache.as_strided((cache.shape[0], block_size * 584), (cache.stride(0), 1))
    value_offsets = rows[..., None] * 576 + torch.arange(576, device=cache.device)
    scale_offsets = block_size * 576 + rows[..., None] * 8 + torch.arange(7, device=cache.device)
    values = byte_pages[pages[..., None], value_offsets].contiguous()
    scales = byte_pages[pages[..., None], scale_offsets].float() - 127
    nope = values[..., :448].contiguous().view(torch.float8_e4m3fn).float().reshape(*slots.shape, 7, 64)
    nope = (nope * torch.exp2(scales)[..., None]).reshape(*slots.shape, 448)
    rope = values[..., 448:].contiguous().view(torch.bfloat16).float()
    result = torch.cat((nope, rope), dim=-1).bfloat16()
    return torch.where(valid[..., None], result, 0)


def bf16_sparse_attention_with_lse(query, swa_cache, swa_indices, swa_lengths,
                                   workspace=None, *, compressed_cache=None,
                                   compressed_indices=None, compressed_lengths=None,
                                   sinks=None, scale=None, token_chunk=32):
    """Eager tensor-core BF16 sparse attention with FP32 accumulators/LSE.

    Supports both32/64-token SWA pages and compressed pages. Unlike the native
    SM12 kernel, this path does not quantize Q or probability*V back to FP8.
    A two-term BF16 probability expansion avoids rounding probabilities once
    per shard. Partial outputs remain FP32 through the DCP merge; the runtime
    should cast once when storing its final attention output. Temporary KV
    expansion is bounded by token_chunk and actual visible prefix lengths,
    not the request's full KV history or padded metadata-buffer capacity.
    """
    if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != 512 or token_chunk < 1:
        raise ValueError('Expected BF16 [tokens, heads, 512] queries and a positive chunk size')
    tokens, heads, _ = query.shape
    if (compressed_indices is None) != (compressed_lengths is None):
        raise ValueError('Compressed indices and lengths must be supplied together')
    if not tokens:
        return torch.empty_like(query, dtype=torch.float32), torch.empty((0, heads), device=query.device, dtype=torch.float32)
    indices = [swa_indices.reshape(tokens, -1)]
    lengths = [swa_lengths.reshape(tokens)]
    caches = [swa_cache]
    if compressed_indices is not None:
        if compressed_cache is None:
            raise ValueError('Compressed indices require a cache')
        indices.append(compressed_indices.reshape(tokens, -1))
        lengths.append(compressed_lengths.reshape(tokens))
        caches.append(compressed_cache)
    if sinks is not None and sinks.numel() != heads:
        raise ValueError('One sink is required per query head')
    outputs, lses = [], []
    for start in range(0, tokens, token_chunk):
        end = min(start + token_chunk, tokens)
        values, masks = [], []
        for index, length, cache in zip(indices, lengths, caches):
            # Native SWA buffers reserve a full image span even for ordinary
            # text; compressed candidates also retain a fixed top-k width.
            # Entries at/after every row's length were masked out below, so
            # omit their expansion entirely. This eager CPU observation adds
            # a synchronization, but prevents very large padded gather/BMM
            # temporaries without dropping any visible entry or changing KV.
            width = min(index.shape[1], max(0, int(length[start:end].max().item())))
            current = index[start:end, :width]
            valid = ((torch.arange(current.shape[1], device=query.device)[None, :] < length[start:end, None])
                     & (current >= 0))
            values.append(gather_packed_cache(cache, torch.where(valid, current, -1)))
            masks.append(valid)
        kv = torch.cat(values, dim=1).contiguous()
        valid = torch.cat(masks, dim=1)
        del values, masks
        scores = torch.bmm(query[start:end], kv.transpose(1, 2), out_dtype=torch.float32)
        scores *= 512**-0.5 if scale is None else scale
        scores.masked_fill_(~valid[:, None, :], -torch.inf)
        logits = scores if sinks is None else torch.cat(
            (scores, sinks.float().reshape(1, heads, 1).expand(end - start, -1, -1)), dim=-1)
        lse = logits.logsumexp(-1)
        probabilities = torch.exp(scores - lse[..., None])
        probabilities = torch.where(torch.isfinite(lse[..., None]), probabilities, 0)
        probability_hi = probabilities.bfloat16()
        probability_lo = (probabilities - probability_hi.float()).bfloat16()
        output = torch.bmm(probability_hi, kv, out_dtype=torch.float32)
        output.add_(torch.bmm(probability_lo, kv, out_dtype=torch.float32))
        outputs.append(output)
        lses.append(lse / math.log(2))
    return torch.cat(outputs), torch.cat(lses)
