# SPDX-License-Identifier: AGPL-3.0-only
"""Record drafter training features from the serving target (capture kit only).

Wraps DSpark propose, which runs after sampling on every step. For every row
whose input token is committed it stores, on TP rank 0:
  * aux: the exact tensor serving hands the drafter, torch.cat(aux_hidden_states)
    [T, 3*5120] bf16 (the input of the trainable main_proj);
  * the target's next-token distribution at that row, from the same final
    hidden states sampling uses: top-64 log-probs (fp16), ids (int32) and
    logsumexp (fp32), so the tail mass is exact;
  * absolute positions and input token ids.
Prefill rows are all committed. In a decode verify block (anchor + drafts) only
the first num_sampled rows are: the anchor and the accepted drafts; rejected
draft rows are dropped. Features therefore carry exactly the numerics serving
uses: prefill for prompts, decode (FP4 KV, verify batches) for generated text.
Rows are buffered per request and flushed as <request id>/<first position>.pt
when the request leaves the batch or the buffer is large. Speculation, sampling
and outputs are unchanged. Only requests whose id contains CAPTURE_PREFIX
('cap-', set by our clients via X-Request-Id) are recorded, so the kit can
serve client traffic without recording it.
"""
import functools
import json
import os
from pathlib import Path

import torch

CAPTURE_DIR = Path('/cache/drafter-capture')
TOPK = 64
LOGIT_ROWS = 256
FLUSH_ROWS = 2048
DEBUG_DIR = Path('/cache/drafter-debug')
DEBUG_STEPS = 8          # per gate1 request: eager re-run of the draft forward with per-layer hooks
_debug_left = {}
_current = {}             # this step's request ids and anchors, set by the propose wrapper
CAPTURE_PREFIX = 'cap-'   # only our own data requests are recorded; client traffic never is


def _ours(req_id):
    return CAPTURE_PREFIX in req_id
_runner = None
_pending = {}
_buffers = {}
_drafts = {}          # req_id -> list of [anchor_position, draft_1..draft_K] actually proposed by serving
_draft_top = {}       # req_id -> list of (top-32 values [K,32], ids [K,32]) of serving's cached draft logits


def attach(runner):
    global _runner
    _runner = runner
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[drafter-capture] attached, writing to {CAPTURE_DIR}', flush=True)
    return True


def _target_distribution(hidden):
    vals, ids, lse = [], [], []
    for start in range(0, hidden.shape[0], LOGIT_ROWS):
        logits = _runner.model.compute_logits(hidden[start:start + LOGIT_ROWS]).float()
        logits = logits[:, :_runner.vocab_size]
        norm = torch.logsumexp(logits, dim=-1)
        top = torch.topk(logits, TOPK, dim=-1)
        vals.append((top.values - norm[:, None]).to(torch.float16))
        ids.append(top.indices.to(torch.int32))
        lse.append(norm)
    return torch.cat(vals), torch.cat(ids), torch.cat(lse)


def _write(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # keep page cache out of the container limit
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _flush(req_id):
    parts = _buffers.pop(req_id, None)
    proposals = _drafts.pop(req_id, None)
    tops = _draft_top.pop(req_id, None)
    if not parts:
        return
    payload = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    if proposals:
        payload['serving_drafts'] = torch.tensor(proposals, dtype=torch.int64)
    if tops:
        payload['serving_draft_top_values'] = torch.stack([v for v, _ in tops])
        payload['serving_draft_top_ids'] = torch.stack([i for _, i in tops])
    folder = CAPTURE_DIR / req_id.replace('/', '_')
    folder.mkdir(exist_ok=True)
    first = int(payload['positions'][0])
    _write(folder / f'{first:08d}.pt', payload)
    side = folder / f'{first:08d}.json'
    side.with_suffix('.jtmp').write_text(json.dumps(dict(rows=len(payload['positions']), first=first,
                                                         last=int(payload['positions'][-1]))))
    os.replace(side.with_suffix('.jtmp'), side)


def wrap_sample(original):
    """Target distributions from the same final hidden states sampling uses."""
    @functools.wraps(original)
    def sample(self, hidden_states, input_batch, *args, **kwargs):
        _pending.clear()
        if _runner is not None and input_batch.num_reqs and any(_ours(r) for r in input_batch.req_ids):
            n = input_batch.num_tokens
            _pending.update(key=(n, input_batch.req_ids[0]), dist=_target_distribution(hidden_states[:n]))
        return original(self, hidden_states, input_batch, *args, **kwargs)
    return sample


def wrap(original):
    @functools.wraps(original)
    def propose(self, input_batch, *args, **kwargs):
        aux_hidden_states = args[3] if len(args) > 3 else kwargs.get('aux_hidden_states')
        num_sampled = args[4] if len(args) > 4 else kwargs.get('num_sampled')
        n = input_batch.num_tokens if input_batch.num_reqs else 0
        anchors = None
        if (_runner is not None and not kwargs.get('dummy_run') and not kwargs.get('is_profile')
                and aux_hidden_states and n and _pending.get('key') == (n, input_batch.req_ids[0])):
            from vllm.distributed import get_tp_group
            values, ids, lse = _pending.pop('dist')
            # Anchors on every rank (the debug decision below must be identical on both ranks).
            starts0 = input_batch.query_start_loc_np
            drafts0 = input_batch.num_draft_tokens_per_req
            sampled0 = num_sampled[:input_batch.num_reqs].cpu().tolist() if num_sampled is not None else None
            pos_cpu = input_batch.positions[:n].cpu()
            anchors = [None] * input_batch.num_reqs
            for r in range(input_batch.num_reqs):
                a0, b0 = int(starts0[r]), int(starts0[r + 1])
                if drafts0 is not None and drafts0[r] > 0:
                    if sampled0 is None:
                        continue
                    b0 = min(b0, a0 + int(sampled0[r]))
                if b0 > a0:
                    anchors[r] = int(pos_cpu[b0 - 1]) + 1
            if get_tp_group().rank_in_group == 0:
                aux = torch.cat(aux_hidden_states, dim=-1)[:n].cpu()
                positions = input_batch.positions[:n].cpu()
                tokens = input_batch.input_ids[:n].cpu()
                values, ids, lse = values.cpu(), ids.cpu(), lse.cpu()
                sampled = num_sampled[:input_batch.num_reqs].cpu().tolist() if num_sampled is not None else None
                starts = input_batch.query_start_loc_np
                drafts = input_batch.num_draft_tokens_per_req
                live = set(input_batch.req_ids)
                for req_id in [r for r in _buffers if r not in live]:
                    _flush(req_id)
                for r, req_id in enumerate(input_batch.req_ids):
                    if not _ours(req_id):
                        continue
                    a, b = int(starts[r]), int(starts[r + 1])
                    if drafts is not None and drafts[r] > 0:
                        if sampled is None:
                            continue
                        b = min(b, a + int(sampled[r]))      # anchor + accepted drafts only
                    if b <= a:
                        continue
                    _buffers.setdefault(req_id, []).append(dict(
                        positions=positions[a:b].clone(), tokens=tokens[a:b].clone(), aux=aux[a:b].clone(),
                        top_logprobs=values[a:b].clone(), top_ids=ids[a:b].clone(), logsumexp=lse[a:b].clone()))
                    if sum(len(p['positions']) for p in _buffers[req_id]) >= FLUSH_ROWS:
                        _flush(req_id)
        _current.update(req_ids=list(input_batch.req_ids), anchors=anchors,
                        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound)
        drafts_out = original(self, input_batch, *args, **kwargs)
        if not kwargs.get('dummy_run') and not kwargs.get('is_profile') and n:
            debug_rerun(self, input_batch.num_reqs)
        _current.clear()
        from vllm.distributed import get_tp_group as _tp
        if (_runner is not None and n and not kwargs.get('dummy_run') and anchors is not None
                and _tp().rank_in_group == 0):
            proposed = drafts_out[:input_batch.num_reqs].cpu().tolist()
            cache = getattr(self, 'draft_logits', None)
            top = None
            if cache is not None:
                k = drafts_out.shape[1]
                rows = cache[input_batch.idx_mapping[:input_batch.num_reqs].long(), :k].float()
                t = rows.topk(32, dim=-1)
                top = (t.values.to(torch.float16).cpu(), t.indices.to(torch.int32).cpu())
            for r, req_id in enumerate(input_batch.req_ids):
                if anchors[r] is not None and _ours(req_id):
                    _drafts.setdefault(req_id, []).append([anchors[r]] + proposed[r])
                    if top is not None:
                        _draft_top.setdefault(req_id, []).append((top[0][r].clone(), top[1][r].clone()))
        return drafts_out
    return propose


def flush_all():
    for req_id in list(_buffers):
        _flush(req_id)


def debug_rerun(spec, num_reqs):
    """Diagnostic for gate1 requests: re-run this step's draft forward eagerly with per-layer hooks.

    FULL-graph replay never calls _generate_draft, so rebuild its inputs exactly as propose does,
    run it with CUDAGraphMode.NONE, then restore draft_tokens / draft_logits so serving is unaffected.
    Decision is identical on both TP ranks (no collectives); only rank 0 writes.
    """
    ids, anchors = _current.get('req_ids'), _current.get('anchors')
    if not ids or anchors is None or torch.cuda.is_current_stream_capturing():
        return
    wanted = [r for r, rid in enumerate(ids[:num_reqs]) if 'gate1' in rid and anchors[r] is not None
              and _debug_left.setdefault(rid, DEBUG_STEPS) > 0]
    if not wanted:
        return
    import sys
    from vllm.config.compilation import CUDAGraphMode
    from vllm.distributed import get_tp_group
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
    dflash = sys.modules['vllm.v1.worker.gpu.spec_decode.dflash.speculator']
    n = spec.num_query_per_req
    tokens = num_reqs * n
    meta = spec._build_draft_attn_metadata(num_reqs=num_reqs, num_reqs_padded=num_reqs, num_tokens_padded=tokens,
                                           seq_lens_cpu_upper_bound=_current['seq_lens_cpu_upper_bound'],
                                           step=n, causal=spec._group_causal)
    slots = dflash.build_slot_mappings_by_layer(spec.block_tables.slot_mappings[:, :tokens], spec.kv_cache_config)
    saved_tokens = spec.draft_tokens.clone()
    saved_logits = spec.draft_logits.clone() if getattr(spec, 'draft_logits', None) is not None else None
    rec, hooks = {}, []
    for i, layer in enumerate(spec.model.model.layers):
        hooks.append(layer.attn.register_forward_hook(
            lambda m, inp, out, i=i: rec.__setitem__(f'attn_in{i}', inp[1].detach().clone()) or
            rec.__setitem__(f'attn_out{i}', (out[0] if isinstance(out, tuple) else out).detach().clone())))
        hooks.append(layer.ffn.register_forward_hook(
            lambda m, inp, out, i=i: rec.__setitem__(f'ffn_out{i}', (out[0] if isinstance(out, tuple) else out).detach().clone())))
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, i=i: rec.__setitem__(f'stream{i}', mhc_post_tilelang(out[0], out[1], out[2], out[3]).detach().clone())))
    def keep(name):
        def hook(m, inp, out):
            v = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(v, torch.Tensor):
                rec[name] = v.detach().clone()
            if inp and isinstance(inp[0], torch.Tensor):
                rec[name + '#in'] = inp[0].detach().clone()
        return hook
    for sub, mod in spec.model.model.layers[0].attn.named_modules():
        if sub:
            hooks.append(mod.register_forward_hook(keep('a0.' + sub)))
    save = get_tp_group().rank_in_group == 0
    try:
        head = spec._run_model(tokens, meta, slots, None, CUDAGraphMode.NONE)
        rec['head_hidden'] = head.detach().clone()
        first = dict(rec)
        # Controlled experiment: rank 0 rewrites each wanted request's context window from the exact
        # captured target features (serving's own insert path + slot formula), then reruns.
        if save:
            for r in wanted:
                parts = _buffers.get(ids[r])
                if not parts:
                    continue
                pos = torch.cat([p['positions'] for p in parts])
                aux = torch.cat([p['aux'] for p in parts])
                c = anchors[r] - 1
                sel = (pos <= c) & (pos > c - 128)
                if not bool(sel.any()):
                    continue
                wpos = pos[sel].cuda()
                main_x = spec.model.combine_hidden_states(aux[sel].cuda())
                by_gid = {}
                for gid in spec.draft_kv_cache_group_ids:
                    bt = spec.block_tables.input_block_tables[gid][r]
                    bs = int(spec.block_tables.kernel_block_sizes[gid])
                    by_gid[gid] = (bt[(wpos // bs).long()].long() * bs + wpos % bs).long()
                if spec._layer_group_idx is not None:
                    ctx_slots = [by_gid[spec.draft_kv_cache_group_ids[g]] for g in spec._layer_group_idx]
                else:
                    ctx_slots = by_gid[spec.draft_kv_cache_group_ids[0]]
                spec.model.precompute_and_store_context_kv(main_x, wpos, ctx_slots)
        rec.clear()
        head = spec._run_model(tokens, meta, slots, None, CUDAGraphMode.NONE)
        rec['head_hidden'] = head.detach().clone()
        rewritten = dict(rec)
        rec.clear()
        rec.update(first)
        rec.update({k + '@rw': v for k, v in rewritten.items()})
    finally:
        for h in hooks:
            h.remove()
        spec.draft_tokens.copy_(saved_tokens)
        if saved_logits is not None:
            spec.draft_logits.copy_(saved_logits)
    for r in wanted:
        rid = ids[r]
        _debug_left[rid] -= 1
        if not save:
            continue
        folder = DEBUG_DIR / rid.replace('/', '_')
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({k: (v[r * n:(r + 1) * n] if v.dim() and v.shape[0] == tokens else v).cpu() for k, v in rec.items()},
                   folder / f'{anchors[r]:08d}.pt')
