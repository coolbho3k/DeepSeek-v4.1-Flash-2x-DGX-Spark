# SPDX-License-Identifier: AGPL-3.0-only
"""Fine-tune DSpark on captured target features (DSpark paper objective).

Loss per draft position j (weight w_j = exp(-j/B)):
  0.9 * TV   ||p_draft - p_target||_1 over the stored top-64 support + tail bucket
  0.1 * CE   -log p_draft(actual next token)
  1.0 * BCE  confidence head vs c* = 1 - TV/2 (detached)
Block size B is sampled per step from --blocks so the drafter is trained for
every verify length serving uses (vLLM builds the draft block from K).

Data: capture folders <root>/<request>/<position>.pt plus <root>/<request>/meta.json
({"prompt_len": n, "split": "train"|"heldout", "category": ...}).
Runs single-GPU or DDP (torchrun / explicit env). Checkpoints are resumable.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dspark_torch import Args, DSpark, load

NGRAM = 3


# ------------------------------------------------------------------ data

def load_sequence(folder):
    parts = sorted(folder.glob('*.pt'))
    chunks = [torch.load(p, map_location='cpu', weights_only=True) for p in parts]
    proposals = [c.pop('serving_drafts') for c in chunks if 'serving_drafts' in c]
    top_v = [c.pop('serving_draft_top_values') for c in chunks if 'serving_draft_top_values' in c]
    top_i = [c.pop('serving_draft_top_ids') for c in chunks if 'serving_draft_top_ids' in c]
    seq = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    order = torch.argsort(seq['positions'])
    seq = {k: v[order] for k, v in seq.items()}
    if proposals:
        seq['serving_drafts'] = torch.cat(proposals)          # [steps, 1 + K]: anchor position, drafts
    if top_v:
        seq['serving_draft_top_values'] = torch.cat(top_v)    # [steps, K, 32] serving draft logits (top-32)
        seq['serving_draft_top_ids'] = torch.cat(top_i)
    if not torch.equal(seq['positions'], torch.arange(len(order))):
        raise ValueError(f'{folder}: capture positions are not contiguous from 0')
    seq['meta'] = json.loads((folder / 'meta.json').read_text())
    return seq


def sequences(root, split):
    out = []
    for folder in sorted(Path(root).iterdir()):
        meta = folder / 'meta.json'
        if folder.is_dir() and meta.exists() and json.loads(meta.read_text()).get('split') == split:
            out.append(folder)
    return out


def valid_anchors(seq, block, response_only=True):
    length = len(seq['tokens'])
    # First real draft step: context ends at the last prompt token, anchor = first reply token.
    lo = max(0, seq['meta']['prompt_len'] - 1) if response_only else 0
    hi = length - 2 - block          # needs rows c+1..c+block and token c+block+1
    return torch.arange(lo, hi + 1) if hi >= lo else torch.empty(0, dtype=torch.long)


# ------------------------------------------------------------------ loss

def position_loss(logits, conf_logit, top_lp, top_ids, target_token):
    """logits [M, V] fp32; top_lp/top_ids [M, 64]; target_token [M]. Returns tv, ce, conf, accept."""
    logp = logits.log_softmax(-1)
    p = logp.exp()
    pt = top_lp.float().exp()
    pd_top = p.gather(1, top_ids.long())
    tail_t = (1 - pt.sum(-1)).clamp_min(0)
    tail_d = (1 - pd_top.sum(-1)).clamp_min(0)
    tv = (pd_top - pt).abs().sum(-1) + (tail_d - tail_t).abs()
    ce = -logp.gather(1, target_token[:, None].long()).squeeze(1)
    accept = (1 - tv / 2).clamp(0, 1)
    conf = F.binary_cross_entropy_with_logits(conf_logit.float(), accept.detach(), reduction='none')
    return tv, ce, conf, accept


def batch_loss(model, hidden, prev, tops, weights, chunk=512):
    """hidden [N,B,dim]; prev [N,B] Markov input tokens; tops: dict of [N,B,...] targets."""
    n, b, _ = hidden.shape

    def part(h, pv, lp, ids, tok, w):
        logits, conf = model.logits(h, pv)
        tv, ce, cf, acc = position_loss(logits.flatten(0, 1), conf.flatten(), lp.flatten(0, 1),
                                        ids.flatten(0, 1), tok.flatten())
        w = w.expand(h.shape[0], -1).flatten()
        loss = (w * (0.9 * tv + 0.1 * ce + 1.0 * cf)).sum()
        return loss, torch.stack([(w * tv).sum(), (w * ce).sum(), (w * cf).sum(), acc.sum()]).detach()

    total, stats = 0, torch.zeros(4, device=hidden.device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        loss, st = checkpoint(part, hidden[s:e], prev[s:e], tops['lp'][s:e], tops['ids'][s:e], tops['tok'][s:e],
                              weights, use_reentrant=False)
        total = total + loss
        stats += st
    norm = n * weights.sum()
    return total / norm, stats / torch.tensor([norm, norm, norm, n * b], device=hidden.device)


def make_batch(model, seqs, block, anchors_per_seq, device, rng, generator):
    hid, prev, lp, ids, tok = [], [], [], [], []
    for seq in seqs:
        cand = valid_anchors(seq, block)
        if not len(cand):
            continue
        pick = cand[torch.randperm(len(cand), generator=generator)[:anchors_per_seq]]
        aux = seq['aux'].to(device)
        positions = seq['positions'].to(device)
        ctx = model.context_kv(aux, positions)
        c = pick.to(device)
        tokens = seq['tokens'].to(device)
        anchor_tok = tokens[c + 1]
        h = model.draft_hidden(ctx, c, anchor_tok, c + 1, block)
        rows = (c[:, None] + 1 + torch.arange(block, device=device))            # target rows c+1+j
        hid.append(h)
        prev.append(tokens[rows])                                               # Markov input: token c+1+j
        lp.append(seq['top_logprobs'].to(device).float()[rows])
        ids.append(seq['top_ids'].to(device)[rows])
        tok.append(tokens[rows + 1])                                            # actual next token
    if not hid:
        return None
    return torch.cat(hid), torch.cat(prev), dict(lp=torch.cat(lp), ids=torch.cat(ids), tok=torch.cat(tok))


# ------------------------------------------------------------------ offline acceptance (serving emulation)

def greedy_target(seq):
    """Target argmax chain per row. For greedy (T=0) captures use the generated tokens themselves:
    they are what serving's decode-time argmax produced (prefill argmax differs at near-ties)."""
    top = seq['top_ids'][:, 0].clone()
    if seq['meta'].get('temperature') == 0:
        top[:-1] = seq['tokens'][1:].to(top.dtype)
    return top


@torch.no_grad()
def greedy_accepts(model, seq, block, device, chunk=256):
    """Per anchor c: leading drafts equal to the target's argmax (exact for greedy serving)."""
    anchors = valid_anchors(seq, block)
    if not len(anchors):
        return anchors, torch.empty(0)
    tokens = seq['tokens'].to(device)
    ctx = model.context_kv(seq['aux'].to(device), seq['positions'].to(device))
    target = greedy_target(seq).to(device)                                     # target argmax at each row
    out = []
    for s in range(0, len(anchors), chunk):
        c = anchors[s:s + chunk].to(device)
        rows = c[:, None] + 1 + torch.arange(block, device=device)
        h = model.draft_hidden(ctx, c, tokens[c + 1], c + 1, block)
        logits, _ = model.logits(h, target[rows - 1].where(rows > c[:, None] + 1, tokens[c + 1][:, None]))
        match = (logits.argmax(-1) == target[rows]).int()
        out.append(match.cumprod(-1).sum(-1).cpu())
    return anchors, torch.cat(out)


def ngram_drafts(history, block):
    """Prompt-lookup emulation of ds41/ngram_draft.py on a token list (latest earlier match)."""
    if len(history) < 2 * NGRAM + block:
        return None
    tail = history[-NGRAM:]
    for i in range(len(history) - NGRAM - block, -1, -1):
        if history[i:i + NGRAM] == tail:
            return history[i + NGRAM:i + NGRAM + block]
    return None


def walk(seq, anchors, accepts, block, use_ngram=True):
    """Replay serving over a greedy sequence: tokens per verify step."""
    acc = dict(zip(anchors.tolist(), accepts.tolist()))
    tokens = seq['tokens'].tolist()
    target = greedy_target(seq).tolist()
    c, steps, produced = int(anchors[0]) if len(anchors) else 0, 0, 0
    while c in acc:
        a = acc[c]
        if use_ngram:
            drafts = ngram_drafts(tokens[:c + 2], block)
            if drafts is not None:
                a = 0
                while a < block and drafts[a] == target[c + 1 + a]:
                    a += 1
        steps += 1
        produced += a + 1
        c += a + 1
    return produced, steps


# ------------------------------------------------------------------ main

def setup_distributed():
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group('nccl')
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--blocks', default='3,4,5')
    p.add_argument('--block-weights', default='0.4,0.2,0.4')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--seqs-per-step', type=int, default=4)
    p.add_argument('--anchors-per-seq', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--quant-kv', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--seed', type=int, default=41)
    a = p.parse_args()
    rank, world = setup_distributed()
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.manual_seed(a.seed + rank)
    with torch.device(device):
        model = DSpark(Args(), quant_kv=a.quant_kv)
    load(model, a.draft, a.target, device)
    params = model.trainable()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    a.out.mkdir(parents=True, exist_ok=True)
    step = 0
    ckpt = a.out / 'last.pt'
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state['model'], strict=False)
        opt.load_state_dict(state['opt'])
        step = state['step']
    train = sequences(a.data, 'train')[rank::world]
    heldout = sequences(a.data, 'heldout')
    blocks = [int(x) for x in a.blocks.split(',')]
    bw = [float(x) for x in a.block_weights.split(',')]
    rng = random.Random(a.seed + rank)
    gen = torch.Generator().manual_seed(a.seed + rank)
    log = (a.out / 'log.jsonl').open('a') if rank == 0 else None

    def evaluate():
        model.eval()
        rows = {}
        for folder in heldout[:64]:
            seq = load_sequence(folder)
            for k in (3, 5):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    anchors, acc = greedy_accepts(model, seq, k, device)
                if not len(anchors):
                    continue
                produced, steps = walk(seq, anchors, acc, k)
                cat = seq['meta'].get('category', 'other')
                r = rows.setdefault(f'K{k}/{cat}', [0, 0])
                r[0] += produced
                r[1] += steps
        model.train()
        return {k: round(v[0] / max(v[1], 1), 4) for k, v in sorted(rows.items())}

    if rank == 0:
        base = evaluate()
        print(json.dumps(dict(step=step, eval_tokens_per_step=base)), flush=True)
        log.write(json.dumps(dict(step=step, eval=base)) + '\n'); log.flush()
    if a.eval_only:
        return
    started = time.time()
    while step < a.steps:
        lr = a.lr * min(1, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps))
        for g in opt.param_groups:
            g['lr'] = lr
        block = rng.choices(blocks, bw)[0]
        seqs = [load_sequence(f) for f in rng.sample(train, min(a.seqs_per_step, len(train)))]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            batch = make_batch(model, seqs, block, a.anchors_per_seq, device, rng, gen)
            if batch is None:
                continue
            hidden, prev, tops = batch
            weights = torch.exp(-torch.arange(block, device=device) / block)
            loss, stats = batch_loss(model, hidden, prev, tops, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:
            for q in params:
                if q.grad is not None:
                    dist.all_reduce(q.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if rank == 0 and step % 10 == 0:
            row = dict(step=step, block=block, lr=lr, loss=round(loss.item(), 5), tv=round(stats[0].item(), 4),
                       ce=round(stats[1].item(), 4), conf=round(stats[2].item(), 4), accept=round(stats[3].item(), 4),
                       sec_per_step=round((time.time() - started) / 10, 2))
            started = time.time()
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n'); log.flush()
        if rank == 0 and (step % a.eval_every == 0 or step == a.steps):
            torch.save(dict(model={k: v for k, v in model.state_dict().items()}, opt=opt.state_dict(), step=step),
                       a.out / 'last.tmp')
            os.replace(a.out / 'last.tmp', ckpt)
            ev = evaluate()
            print(json.dumps(dict(step=step, eval_tokens_per_step=ev)), flush=True)
            log.write(json.dumps(dict(step=step, eval=ev)) + '\n'); log.flush()
    if world > 1:
        dist.barrier()


if __name__ == '__main__':
    main()
