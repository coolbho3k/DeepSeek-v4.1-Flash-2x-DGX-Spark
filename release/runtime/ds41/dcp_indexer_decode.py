"""Bounded eager FP8 indexer decode for V4.1's segregated128-state pages.

The native SM12 paged kernel accepts64-state FP8 pages. A128-state page
cannot be virtually split because all scales follow all values in each page.
Gather at most2048 owned states at a time and use the native unpaged logits
kernel. Cache storage is unchanged; no full-context BF16 expansion occurs.
"""
import torch


def paged_logits(q, kv, weights, lengths, table, schedule_metadata, *,
                 max_model_len, clean_logits=False, indices=None, state_chunk=2048):
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    values, q_scale = q
    if (q_scale is not None or values.ndim != 4 or values.shape[1:] != (1, 32, 128)
            or values.dtype != torch.float8_e4m3fn or indices is not None
            or lengths.shape != (values.shape[0], 1) or state_chunk < 1):
        raise ValueError('The eager V4.1 indexer decode supports FP8,32 heads,next_n1 without varlen indirection')
    if (kv.dtype != torch.uint8 or kv.ndim != 4 or kv.shape[2:] != (1, 132)
            or kv.shape[1] not in (64, 128) or kv.stride(1) != 132 or kv.stride(-1) != 1):
        raise ValueError('Expected segregated FP8 indexer pages [blocks,64/128,1,132]')
    if weights.shape != (values.shape[0], 32) or weights.dtype != torch.float32 or table.shape[0] != values.shape[0]:
        raise ValueError('Invalid per-request indexer weights or block table')
    counts = lengths[:, 0].cpu().tolist()
    if any(length < 0 or length > max_model_len for length in counts):
        raise ValueError('Local context length exceeds the bounded logits allocation')
    output = torch.full((values.shape[0], max_model_len), -torch.inf, device=values.device, dtype=torch.float32)
    physical_bs = kv.shape[1]
    pages = kv.as_strided((kv.shape[0], physical_bs * 132), (kv.stride(0), 1))
    key_offsets = torch.arange(128, device=kv.device)
    scale_offsets = torch.arange(4, device=kv.device)
    for request, length in enumerate(counts):
        for start in range(0, length, state_chunk):
            end = min(start + state_chunk, length)
            positions = torch.arange(start, end, device=kv.device)
            page_columns, rows = positions // physical_bs, positions % physical_bs
            if page_columns[-1].item() >= table.shape[1]:
                raise ValueError('Indexer block table is too short')
            block = table[request, page_columns].long()
            if (block < 0).any().item() or (block >= kv.shape[0]).any().item():
                raise ValueError('Invalid physical indexer page ID')
            key = pages[block[:, None], rows[:, None] * 128 + key_offsets].contiguous().view(torch.float8_e4m3fn)
            scales = pages[block[:, None], physical_bs * 128 + rows[:, None] * 4 + scale_offsets].contiguous().view(torch.float32).flatten()
            row_start = torch.zeros(1, device=kv.device, dtype=torch.int32)
            row_end = torch.full((1,), end - start, device=kv.device, dtype=torch.int32)
            logits = fp8_fp4_mqa_logits((values[request].contiguous(), None), (key, scales),
                weights[request:request + 1].contiguous(), row_start, row_end, clean_logits=False)
            output[request, start:end].copy_(logits[0, :end - start])
    return output
