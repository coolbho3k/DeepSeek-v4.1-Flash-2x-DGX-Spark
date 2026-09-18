"""Global two-level candidate-block selection over DCP-local sparse logits.

Blocks are scored by the maximum of their token scores, with the newest
global block pinned, matching V4.1 semantics. Not installed into vLLM yet.
"""
import torch


def row_bounds(logits, starts, ends, row_repeat=1):
    if logits.ndim != 2 or row_repeat < 1:
        raise ValueError('Invalid logits or row repetition')
    rows = logits.shape[0]
    row = torch.arange(rows, device=logits.device) // row_repeat
    if row.numel() and row[-1].item() >= ends.numel():
        raise ValueError('Missing row bounds')
    first = torch.zeros(rows, device=logits.device, dtype=torch.int64) if starts is None else starts.reshape(-1)[row].long()
    last = ends.reshape(-1)[row].long()
    if (last < first).any().item() or (first < 0).any().item() or (last > logits.shape[1]).any().item():
        raise ValueError('Invalid local causal bounds')
    return first, last


def local_block_scores(logits, starts, ends, block_size, world, rank, nblocks, row_repeat=1):
    if block_size < 1 or nblocks < 0 or world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid candidate block/DCP coordinates')
    first, last = row_bounds(logits, starts, ends, row_repeat)
    scores = logits.new_full((logits.shape[0], nblocks), -torch.inf)
    if not nblocks or not logits.shape[1]:
        return scores
    # Work in strips: do not allocate an additional full logits-sized int64
    # address matrix when a long-context indexer already owns a large buffer.
    for start in range(0, logits.shape[1], 4096):
        columns = torch.arange(start, min(start + 4096, logits.shape[1]), device=logits.device)
        local = columns[None, :] - first[:, None]
        valid = (columns >= first[:, None]) & (columns < last[:, None])
        blocks = (local * world + rank).div(block_size, rounding_mode='floor')
        values = torch.where(valid, logits[:, start:start + len(columns)], -torch.inf)
        scores.scatter_reduce_(1, blocks.clamp(0, nblocks - 1), values, reduce='amax', include_self=True)
    return scores


def select_candidate_blocks(logits, starts, ends, topk_blocks, block_size, out,
                            group, row_repeat=1):
    """All ranks participate, including ranks owning zero rows of context."""
    world, rank = group.world_size, group.rank_in_group
    first, last = row_bounds(logits, starts, ends, row_repeat)
    local_lengths = (last - first).int().unsqueeze(1)
    global_lengths = group.all_gather(local_lengths, dim=1).sum(1)
    maximum = global_lengths.max().item() if global_lengths.numel() else 0
    nblocks = (maximum + block_size - 1) // block_size
    out.fill_(-1)
    if not nblocks:
        return
    local_scores = local_block_scores(logits, starts, ends, block_size, world, rank, nblocks, row_repeat)
    gathered = group.all_gather(local_scores, dim=1).reshape(logits.shape[0], world, nblocks)
    scores = gathered.amax(1)
    newest = (global_lengths - 1).div(block_size, rounding_mode='floor')
    rows = torch.arange(logits.shape[0], device=logits.device)
    live = global_lengths > 0
    scores[rows[live], newest[live]] = torch.inf
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    chosen = torch.where(top.values > -torch.inf, top.indices, -1).to(out.dtype)
    out[:, :chosen.shape[1]].copy_(chosen)


def apply_candidate_mask(logits, starts, ends, candidates, block_size, world, rank, row_repeat=1):
    if block_size < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid candidate block/DCP coordinates')
    first, last = row_bounds(logits, starts, ends, row_repeat)
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = (width * world + block_size - 1) // block_size
    # One sentinel column consumes invalid/out-of-range candidate IDs.
    flags = torch.zeros((rows, nblocks + 1), device=logits.device, dtype=torch.bool)
    valid_candidates = (candidates >= 0) & (candidates < nblocks)
    flag_index = torch.where(valid_candidates, candidates, nblocks).long()
    flags.scatter_(1, flag_index, True)
    for start in range(0, width, 4096):
        columns = torch.arange(start, min(start + 4096, width), device=logits.device)
        local = columns[None, :] - first[:, None]
        valid = (columns >= first[:, None]) & (columns < last[:, None])
        blocks = (local * world + rank).div(block_size, rounding_mode='floor')
        keep = flags.gather(1, blocks.clamp(0, nblocks - 1)) & valid
        logits[:, start:start + len(columns)].masked_fill_(~keep, -torch.inf)
