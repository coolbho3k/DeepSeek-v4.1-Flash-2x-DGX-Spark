# SPDX-License-Identifier: AGPL-3.0-only
# DCP merge adapted from ds41/dcp_head_exchange.py, in the MiaAI-derived
# AGPLv3 serving recipe. See the parent kit's complete attribution/notices.
"""Schedule existing attention head tiles; preserve key/channel reductions.

The only new numerical kernel is an address-adapted copy of the existing
two-rank FP32 merge. No new quantizer, approximation, key selection, or softmax.
Bit-exactness remains a required, UNRUN qualification gate.
"""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .policy import head_schedule
from .packed import attention as packed_attention, merge as packed_merge

_deferred_errors = ContextVar('ds41_overlap_deferred_errors', default=None)


def check_flags(errors, messages):
    from ds41.graph_validation import check_flags as native_check
    collected = _deferred_errors.get()
    if collected is None or torch.cuda.is_current_stream_capturing():
        native_check(errors, messages)
    else:
        collected.append((errors, tuple(messages)))


@contextmanager
def joined_errors():
    """Check both masked-load error flags after the streams rejoin, not sooner.

Capture keeps the native owner checks. Eager execution coalesces only the
host observation, and still throws before the attention forward returns.
"""
    from ds41.graph_validation import check_flags as native_check
    if _deferred_errors.get() is not None:
        raise RuntimeError('Nested deferred attention validation')
    collected = []
    token = _deferred_errors.set(collected)
    try:
        yield
        if collected:
            messages = collected[0][1]
            if len(collected) != 2 or any(m != messages or e.numel() != 1 for e, m in collected):
                raise RuntimeError('Changed bounded attention validation contract')
            native_check(torch.stack([e.reshape(()) for e, _ in collected]), messages)
    finally:
        _deferred_errors.reset(token)


class PrefillDispatch:
    """Original MMA tile selection, with head-major storage for 32-head calls."""
    def __getitem__(self, grid):
        def launch(*args, **options):
            if (options['BH'] != 16 or options['BN'] != 32
                    or options['num_stages'] != 1):
                raise ValueError('Unexpected baseline attention launch envelope')
            wide = grid[0] >= 32
            heads = 32 if wide else 16
            if options['HEADS'] not in (16, 32) or options['HEADS'] % heads:
                raise ValueError('Head split would change the original attention MMA tile')
            head_major = (wide and args[-3].stride() == (513, grid[0] * 513, 1))
            return packed_attention[(grid[0], options['HEADS'] // heads)](
                *args, **dict(options, BH=heads, SINGLE_ACC=wide,
                              OUTPUT_HEAD_MAJOR=head_major))
        return launch


def allocate_outputs(query, split_k):
    if split_k == 1 and len(query) >= 32 and query.shape[1] == 32:
        storage = torch.empty((32, len(query), 513), device=query.device, dtype=torch.float32)
        return storage[..., :512].transpose(0, 1), storage[..., 512].transpose(0, 1)
    storage = torch.empty((*query.shape[:2], 513), device=query.device, dtype=torch.float32)
    return storage[..., :512], storage[..., 512]


def make_head_attention(original):
    """Clone the already selected one-pass wrapper, changing only addressing.

    Decode still calls the SAME _online and _online_merge kernel objects, with
    the same split-K/BH/BN values. Prefill calls the SAME online attention JIT
    function, same per-head key order and two-term BF16 probability expansion.
    """
    source = getattr(original, '__ds41_patch_source__', '')
    anchor = 'query.shape[1] not in (32, 64)'
    required = ('_online(grid, arguments, partial, local_lse, error, common, split_k)',
                'output, lse = _ds41_attention_outputs(query, split_k)',
                '_ds41_check_flags(error')
    if source.count(anchor) != 1 or any(text not in source for text in required):
        raise RuntimeError('DCP overlap requires the selected owned one-pass attention')
    source = source.replace(anchor, 'query.shape[1] not in (16, 32)')
    source = source.replace('[0..512 tokens, 32/64 heads, 512]',
                            '[0..512 tokens, 16/32 heads, 512]')
    source = source.replace('output, lse = _ds41_attention_outputs(query, split_k)',
        'output, lse = (_ds41_attention_outputs(query, split_k) if _outputs is None else _outputs)')
    tree = ast.parse(source)
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    definition.decorator_list = []
    definition.args.kwonlyargs.append(ast.arg(arg='_outputs'))
    definition.args.kw_defaults.append(ast.Constant(value=None))
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _attention=PrefillDispatch(),
                     _ds41_attention_outputs=allocate_outputs, _online_merge=packed_merge,
                     _ds41_check_flags=check_flags)
    exec(compile(tree, __file__ + ':head-wrapper', 'exec'), namespace)
    result = namespace[definition.name]
    result.__ds41_patch_source__ = source
    return result


def pack_result(output, lse):
    """Export 32 remote-head partials without a copy on wide prefill.

    Transport deliberately uses one rank-major contiguous [rows,32,513]
    interpretation even when physically head-major. Both peers agree on the
    layout by row count, and merge uses the corresponding physical indexing.
    """
    rows = len(output)
    head_major = (rows >= 32 and output.stride() == (513, rows * 513, 1)
                  and lse.stride() == (513, rows * 513)
                  and output.storage_offset() == 0 and lse.storage_offset() == 512
                  and output.untyped_storage().data_ptr() == lse.untyped_storage().data_ptr())
    if head_major:
        # This is a flat view of owned storage, not a fake logical head view.
        return output.as_strided((rows, 32, 513), (32 * 513, 513, 1)), True
    if (output.stride() == (32 * 513, 513, 1) and lse.stride() == (32 * 513, 513)
            and output.storage_offset() == 0 and lse.storage_offset() == 512
            and output.untyped_storage().data_ptr() == lse.untyped_storage().data_ptr()):
        return output.as_strided((rows, 32, 513), (32 * 513, 513, 1)), False
    return torch.cat((output, lse.unsqueeze(-1)), dim=-1), False


@tr.jit
def _merge(Local0, Lse0, Local1, Lse1, Peers, Output,
           O0T: tl.constexpr, O0H: tl.constexpr, O0C: tl.constexpr,
           L0T: tl.constexpr, L0H: tl.constexpr,
           O1T: tl.constexpr, O1H: tl.constexpr, O1C: tl.constexpr,
           L1T: tl.constexpr, L1H: tl.constexpr,
           DT: tl.constexpr, DH: tl.constexpr, DC: tl.constexpr,
           ROWS: tl.constexpr, RANK: tl.constexpr, SPLIT_OWN: tl.constexpr,
           HEAD_MAJOR: tl.constexpr, LOG_BASE: tl.constexpr):
    row = tl.program_id(0)
    token, head = row // 32, row % 32
    col = tl.arange(0, 512)
    remote = ((1 - RANK) * ROWS * 32 + row) * 513
    if HEAD_MAJOR:
        remote = ((1 - RANK) * 32 * ROWS + head * ROWS + token) * 513
    if SPLIT_OWN and head >= 16:
        local_lse = tl.load(Lse1 + token * L1T + (head - 16) * L1H).to(tl.float32) * LOG_BASE
        own = tl.load(Local1 + token * O1T + (head - 16) * O1H + col * O1C).to(tl.float32)
    else:
        local_lse = tl.load(Lse0 + token * L0T + head * L0H).to(tl.float32) * LOG_BASE
        own = tl.load(Local0 + token * O0T + head * O0H + col * O0C).to(tl.float32)
    peer_lse = tl.load(Peers + remote + 512).to(tl.float32) * LOG_BASE
    if RANK == 0:
        a, b = local_lse, peer_lse
    else:
        a, b = peer_lse, local_lse
    maximum = tl.maximum(a, b)
    shift = tl.where(tl.abs(maximum) == float('inf'), 0., maximum)
    normalizer = libdevice.log(libdevice.exp(a - shift) + libdevice.exp(b - shift)) + shift
    wa, wb = libdevice.exp(a - normalizer), libdevice.exp(b - normalizer)
    wa = tl.where((wa == wa) & (tl.abs(wa) != float('inf')), wa, 0.)
    wb = tl.where((wb == wb) & (tl.abs(wb) != float('inf')), wb, 0.)
    peer = tl.load(Peers + remote + col).to(tl.float32)
    if RANK == 0:
        x, y = own, peer
    else:
        x, y = peer, own
    value = tl.where(wa > 0, x * wa, 0.) + tl.where(wb > 0, y * wb, 0.)
    tl.store(Output + token * DT + head * DH + col * DC, value)


@dataclass
class PendingMerge:
    transfer: object
    locals: tuple
    destination: object
    rank: int
    head_major: bool

    def finish(self):
        peers = self.transfer.join()
        (a, al), (b, bl) = (self.locals if len(self.locals) == 2
                             else (self.locals[0], self.locals[0]))
        rows = len(a)
        if (tuple(self.destination.shape) != (rows, 32, 512)
                or self.destination.dtype != torch.bfloat16
                or peers.shape != (2, rows, 32, 513)
                or peers.dtype != torch.float32 or not peers.is_contiguous()
                or any(t.device != a.device for t in (al, b, bl, peers, self.destination))):
            raise ValueError('Changed DCP local/remote merge contract')
        _merge[(rows * 32,)](a, al, b, bl, peers, self.destination,
            *a.stride(), *al.stride(), *b.stride(), *bl.stride(),
            *self.destination.stride(), rows, self.rank, len(self.locals) == 2,
            self.head_major, math.log(2.), num_warps=4, enable_fp_fusion=False)


def step(transfer, query, swa, indices, lengths, sinks, scale, extra,
         destination, previous, *, attention, mode):
    """Q exchange overlaps own heads; result exchange overlaps the tail.

    Small-row balanced mode: own16 / remote32 / own16, retaining BH=16.
    Wide mode: own32 / remote32, retaining BH=32; defer result merge until
    the next slab's local heads finish. Only one prior result is retained.
    """
    transport = transfer.transport
    rank = transport.group.rank_in_group
    if mode == 'concurrent':
        if previous is not None:
            raise RuntimeError('Concurrent slabs must finish their own join')
        # Allocate outputs on the caller stream. Side-stream temporaries are
        # consumed only there; shared wire buffers return to their allocator
        # stream before any reference can be released or reused.
        split_k = 8 if len(query) <= 2 else 2 if len(query) <= 8 else 1
        remote = allocate_outputs(query, split_k)
        payload, head_major = pack_result(*remote)
        def produce():
            attention(transfer.destination[1 - rank], swa, indices, lengths,
                sinks=sinks[(1 - rank) * 32:(2 - rank) * 32], scale=scale,
                _outputs=remote, **extra)
        with joined_errors():
            result = transport.remote_result(transfer, payload, produce)
            own = attention(query, swa, indices, lengths,
                sinks=sinks[rank * 32:(rank + 1) * 32], scale=scale, **extra)
            PendingMerge(result, (own,), destination, rank, head_major).finish()
        return None
    schedule = head_schedule(len(query), mode)

    def local(begin, end):
        start = rank * 32 + begin
        return attention(query[:, begin:end], swa, indices, lengths,
                         sinks=sinks[start:start + end - begin], scale=scale, **extra)

    first = local(*schedule[0])
    rank_queries = transfer.join()  # Also joins any preceding result on this stream.
    if previous is not None:
        previous.finish()
    remote = attention(rank_queries[1 - rank], swa, indices, lengths,
                       sinks=sinks[(1 - rank) * 32:(2 - rank) * 32], scale=scale, **extra)
    payload, head_major = pack_result(*remote)
    result = transport.gather(payload, kind='result')
    locals_ = (first, local(*schedule[1])) if len(schedule) == 2 else (first,)
    pending = PendingMerge(result, locals_, destination, rank, head_major)
    if len(schedule) == 2:
        pending.finish()
        return None
    return pending
