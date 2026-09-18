"""Device-only routing counts for bounded decode/speculative EXL3 batches.

The existing cooperative K3/MUL1 kernel, weights and shared scratch are
unchanged. At most 128 total assignments implies no expert can overflow its
128-row capacity. Larger batches use the original dispatcher unchanged.
"""
import hashlib
from pathlib import Path

import torch
import spark_fused_moe as base

BASE_SHA = 'c5b573e624be1d4d92ebfcf39e1968130a57e85c3c8c1dde7c9982120639b3de'


class AsyncSmallDispatcher(base.Dispatcher):
    def __call__(self, experts, x, ids, weights, chunk_tokens=1024):
        base.validate_inputs(experts, x, ids, weights, chunk_tokens)
        if not len(x) or not experts or (ids.numel() > 128 and not globals().get('_ds41_coop_eligible', lambda *_: False)(x.shape, ids.shape)):
            return super().__call__(experts, x, ids, weights, chunk_tokens)
        with self.lock, torch.cuda.device(x.device):
            if self.failed:
                raise RuntimeError('MUL1 dispatcher is poisoned after an earlier failure')
            try:
                stream = torch.cuda.current_stream(x.device)
                if self.workspace is None:
                    self.workspace = base.Workspace(self.module, x.device)
                work = self.workspace
                if work.device != x.device:
                    raise ValueError('One visible GPU per DS41 worker is required')
                if work.pending and work.stream_id != stream.cuda_stream:
                    stream.wait_event(work.ready)
                    self.stream_waits += 1
                bank = self.banks.get(id(experts))
                if bank is None:
                    if len(self.banks) >= 43:
                        raise ValueError('More than 43 immutable target-plus-draft expert banks')
                    bank = base.Bank(experts, x.device)
                    self.banks[id(experts)] = bank
                if bank.owner is not experts or len(experts) != len(bank.keys):
                    raise ValueError('Expert bank changed after native pointer capture')
                flat = ids.reshape(-1).long()
                safe = torch.where((flat >= 0) & (flat < 384), flat, 384)
                mapped = bank.mapping.index_select(0, safe)
                order = torch.argsort(mapped, stable=True)
                # Fixed-size scatter avoids both bincount's dynamic-size
                # synchronization and the explicit counts.cpu() of the base.
                counts = torch.zeros(len(bank.keys)+1, device=x.device, dtype=torch.int64)
                counts.scatter_add_(0, mapped, torch.ones_like(mapped))
                tokens = torch.div(order, ids.shape[1], rounding_mode='floor')
                sorted_weights = weights.reshape(-1).index_select(0, order).float().contiguous()
                out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                # Zero-count experts (including an all-missing batch) are
                # skipped by the existing device scheduler. No overflow is possible.
                self.module.forward(x.half().contiguous(), out, counts, tokens,
                                    sorted_weights, bank.ptrs, work.temps, work.locks)
                result = out.to(x.dtype)
                self.last_schedule = dict(mode='device_only_small_routes',
                    assignments=ids.numel(), fallback_experts=0,
                    host_count_readback=False, counts=None)
                work.ready.record(stream)
                work.pending, work.stream_id = True, stream.cuda_stream
                return result
            except Exception:
                self.failed = True
                raise


def register(kernel_path=None):
    """Opt in before the first forward; do not replace a live workspace."""
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError('Unreviewed base MUL1 dispatcher')
    base.register(kernel_path)
    if isinstance(base._dispatcher, AsyncSmallDispatcher):
        return
    if base._dispatcher.workspace is not None or base._dispatcher.banks:
        raise RuntimeError('Register asynchronous routing before any forward')
    base._dispatcher = AsyncSmallDispatcher(base._dispatcher.module)
    base.register(kernel_path)
