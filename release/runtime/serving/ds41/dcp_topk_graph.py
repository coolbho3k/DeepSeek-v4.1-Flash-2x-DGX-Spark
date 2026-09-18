# SPDX-License-Identifier: AGPL-3.0-only
"""Owned-graph decode top-k with the existing stable/tie/padding semantics.

Keep lengths on the device. Scratch is reused one row at a time, including
the million-key case. Invalid lengths are masked and reported through the
same graph-owner boundary used by the indexer and sparse attention.
"""
import triton
import triton.language as tl


@triton.jit
def _rank_keys(Logits, Count, Keys, WIDTH:tl.constexpr, BLOCK:tl.constexpr):
    column=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    score=tl.load(Logits+column,column<WIDTH,other=-float('inf'))
    # Monotonic float32 bit ordering, with exact IEEE zero/NaN semantics
    # matching stable descending argsort. The low20 bits prefer smaller IDs.
    score=tl.where(score==0.,0.,score)
    bits=score.to(tl.uint32,bitcast=True)
    ordered=tl.where((bits&0x80000000)!=0,~bits,bits^0x80000000)
    ordered=tl.where(score!=score,0xffffffff,ordered).to(tl.uint32)
    ordered=tl.where(column<tl.load(Count),ordered,0).to(tl.int64)
    key=(ordered<<20)+(1048575-column).to(tl.int64)
    tl.store(Keys+column,key,column<WIDTH)


def decode_topk(logits, lengths, output, workspace, k, max_seq_len):
    import torch
    from .graph_validation import check_flags, require_capture_owner
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
    require_capture_owner()
    width = logits.shape[1]
    raw_counts = lengths.reshape(-1)
    errors = ((raw_counts < 0) | (raw_counts > width)).to(torch.int32)
    check_flags(errors, ((1, 'Decode top-k length exceeds supplied logits'),))
    counts = raw_counts.clamp(0, width)
    take = min(k, width)
    short_indices = torch.arange(take, device=logits.device, dtype=torch.int32)
    output.fill_(-1)
    for row in range(logits.shape[0]):
        keys=torch.empty(width,device=logits.device,dtype=torch.int64)
        _rank_keys[(triton.cdiv(width,256),)](logits[row],counts[row:row+1],keys,
            WIDTH=width,BLOCK=256,num_warps=4)
        # Unique exact integer keys permit partial selection without unstable
        # ties. No floating-point score or selected membership is approximated.
        indices = torch.topk(keys,take,sorted=True).indices
        sorted_ids = torch.where(indices < counts[row], indices, -1).to(torch.int32)
        short_ids = torch.where(short_indices < counts[row], short_indices, -1)
        output[row, :take].copy_(torch.where(counts[row] <= k, short_ids, sorted_ids))
