"""Bounded source-final-head and teacher-forced logit comparison primitives.

Inputs must already be final full-model states for quality claims. Unit probes
may use synthetic/early states, but those do not measure model accuracy.
"""
import torch
import torch.nn.functional as F


class SourceFinalHead:
    def __init__(self, runtime):
        self.runtime = runtime
        args = runtime.args
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Source-final FP32 head scoring requires TF32 disabled explicitly')
        if runtime.ref.world_size != 1:
            raise ValueError('Layerwise reference scorer requires the full unsharded head')
        with torch.device('cuda'), runtime.ref.set_dtype(torch.bfloat16):
            self.norm = runtime.ref.RMSNorm(args.dim, args.norm_eps)
            self.head = runtime.ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        runtime.load_parameters(self.norm, 'norm')
        runtime.load_parameters(self.head, 'head')
        self.norm.requires_grad_(False)
        self.head.requires_grad_(False)

    def validate(self, h, pre):
        args = self.runtime.args
        if h.ndim != 4 or h.shape[0] != 1 or h.shape[2:] != (args.hc_mult, args.dim) or h.dtype != torch.bfloat16:
            raise ValueError('Expected BF16 final hidden state [1,tokens,hc_mult,dim]')
        if pre.shape != h.shape[:3] or pre.dtype != torch.float32:
            raise ValueError('Expected FP32 final pre-mix [1,tokens,hc_mult]')
        if not torch.isfinite(h).all() or not torch.isfinite(pre).all():
            raise ValueError('Nonfinite final state')

    @torch.inference_mode()
    def logits(self, h, pre):
        self.validate(h, pre)
        # Call the pinned native operations, including BF16 rounding between
        # hyper-connection collapse and RMSNorm, and the FP32 output head.
        collapsed = self.runtime.ref.Block.hc_pre(None, h, pre)
        return self.head(self.norm(collapsed), full_logits=True)[0]

    @torch.inference_mode()
    def compare(self, reference, candidate, tokens, masks, token_chunk=32):
        for h, pre in (reference, candidate):
            self.validate(h, pre)
            if h.shape[1] != len(tokens):
                raise ValueError('State/token length differs')
        if token_chunk < 1 or tokens.ndim != 1 or tokens.dtype != torch.int64 or len(tokens) < 2:
            raise ValueError('Invalid token vector or scoring chunk size')
        if (tokens < 0).any() or (tokens >= self.runtime.args.vocab_size).any():
            raise ValueError('Token ID outside the source vocabulary')
        if not masks or any(mask.shape != (len(tokens) - 1,) or mask.dtype != torch.bool for mask in masks.values()):
            raise ValueError('Expected bool next-token masks indexed by prediction position')
        union = torch.stack([mask.cpu() for mask in masks.values()]).any(0)
        positions = union.nonzero().flatten()
        columns = {name: [] for name in ('positions', 'reference_nll', 'candidate_nll', 'reference_to_candidate_kl',
                                         'top1_agreement', 'reference_target_top1', 'candidate_target_top1')}
        for start in range(0, len(positions), token_chunk):
            selected = positions[start:start + token_chunk]
            h, pre = reference
            first = self.logits(h[:, selected].cuda(), pre[:, selected].cuda())
            if reference[0] is candidate[0] and reference[1] is candidate[1]:
                # Source-only scoring and identity checks need only one GEMM.
                second = first
            else:
                h, pre = candidate
                second = self.logits(h[:, selected].cuda(), pre[:, selected].cuda())
            inputs = [first, second]
            labels = tokens[selected + 1].cuda()
            logp, logq = (F.log_softmax(logits, dim=-1) for logits in inputs)
            best_p, best_q = (logits.argmax(-1) for logits in inputs)
            values = dict(positions=selected,
                reference_nll=F.nll_loss(logp, labels, reduction='none'),
                candidate_nll=F.nll_loss(logq, labels, reduction='none'),
                reference_to_candidate_kl=(logp.exp() * (logp - logq)).sum(-1, dtype=torch.float64),
                top1_agreement=best_p == best_q,
                reference_target_top1=best_p == labels, candidate_target_top1=best_q == labels)
            for name, value in values.items():
                columns[name].append(value.cpu())
        result = {name: torch.cat(values) if values else torch.empty(0, dtype=torch.int64 if name == 'positions' else torch.float64)
                  for name, values in columns.items()}
        if any(not torch.isfinite(value).all() for name, value in result.items() if value.is_floating_point()):
            raise ValueError('Nonfinite teacher-forced metric')
        result['masks'] = {name: mask.cpu()[positions] for name, mask in masks.items()}
        return result
