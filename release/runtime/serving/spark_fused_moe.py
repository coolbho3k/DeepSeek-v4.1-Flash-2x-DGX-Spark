"""Pinned K3/MUL1 eager TP2 dispatcher with ONE shared GPU workspace.

Registration changes only the baked vLLM adapter's eager_moe binding. Expert
weights, dense/vision paths and distributed reduction remain untouched.
Banks are immutable after load. Scratch is allocated on the first nonempty
forward, even if every expert overflows, so profiling accounts for it.
"""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import threading

import torch
import spark_moe

BINARY_NAME = 'ds41_moe_mul1_v1.so'
BINARY_SHA256 = '66de4aa31e49462fd00fb4b26ebd4e16dfd995b631157e3b0194c27fc1239342'
WORKSPACE_BYTES = 23470344
_dispatcher = None
_reference = None


def load_kernel(path):
    path = Path(path).resolve()
    if path.name != BINARY_NAME or hashlib.sha256(path.read_bytes()).hexdigest() != BINARY_SHA256:
        raise RuntimeError('Unreviewed DS41 MUL1 kernel binary')
    name = 'ds41_moe_mul1_v1'
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    if Path(module.__file__).resolve() != path or module.contract_version() != 1:
        raise RuntimeError('Unexpected DS41 MUL1 module origin or ABI')
    return module


def validate_inputs(experts, x, ids, weights, chunk):
    if (type(experts) is not dict or len(experts) > 384
            or x.device.type != 'cuda' or x.dtype not in (torch.float16, torch.bfloat16)
            or x.ndim != 2 or x.shape[1] != 5120 or not 0 <= len(x) <= 1056
            or ids.ndim != 2 or ids.shape != weights.shape or len(ids) != len(x)
            or not 1 <= ids.shape[1] <= 6 or ids.dtype not in (torch.int32, torch.int64)
            or weights.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or ids.device != x.device or weights.device != x.device
            or type(chunk) is not int or not 1 <= chunk <= 1024):
        raise ValueError('Unsupported bounded DS41 MUL1 routing contract')
    if torch.cuda.is_current_stream_capturing():
        raise ValueError('DS41 MUL1 shared workspace requires eager execution')


class Bank:
    def __init__(self, experts, device):
        keys = sorted(experts)
        if not keys or any(type(key) is not int or not 0 <= key < 384 for key in keys):
            raise ValueError('DS41 expert keys must be in [0, 384)')
        self.owner = experts  # Prevent dict ID reuse; model banks are immutable.
        self.keys = tuple(keys)
        self.experts = tuple(experts[key] for key in keys)
        self.tensors = []  # Retain every native pointer's actual tensor owner.
        for expert in self.experts:
            if expert.limit != 10. or set(expert.layers) != {'w1', 'w3', 'w2'}:
                raise ValueError('Unreviewed DS41 expert activation contract')
            for name in ('w1', 'w3', 'w2'):
                layer = expert.layers[name]
                inside, outside = (1152, 5120) if name == 'w2' else (5120, 1152)
                if (layer.K != 3 or not layer.mul1 or layer.mcg
                        or (layer.in_features, layer.out_features) != (inside, outside)):
                    raise ValueError('Fused path requires K3/MUL1 TP2 expert shapes')
                for field, shape, dtype in (
                        ('trellis', (inside//16, outside//16, 48), torch.int16),
                        ('suh', (inside,), torch.float16), ('svh', (outside,), torch.float16)):
                    tensor = getattr(layer, field)
                    if (tensor.device != device or tensor.dtype != dtype
                            or tuple(tensor.shape) != shape or not tensor.is_contiguous()):
                        raise ValueError('Invalid native expert tensor storage')
                    self.tensors.append(tensor)
        self.ptrs = [torch.tensor([getattr(expert.layers[name], field).data_ptr()
                                  for expert in self.experts], device=device, dtype=torch.int64)
                     for name in ('w1', 'w3', 'w2') for field in ('trellis', 'suh', 'svh')]
        # Compact sparse bank IDs. Missing, negative and padded routes map to
        # one trailing sentinel, never to a null pointer or negative bincount.
        mapping = [len(keys)] * 385
        for local, key in enumerate(keys):
            mapping[key] = local
        self.mapping = torch.tensor(mapping, device=device, dtype=torch.int64)


class Workspace:
    def __init__(self, module, device):
        self.device = device
        self.resources = module.resources()
        sms, occupancy, smem, static, regs, local, threads, locks, sms_per_expert = self.resources
        if (sms != 48 or occupancy < 1 or smem + static > 101376
                or threads != 512 or locks != 1050690 or sms_per_expert != 8):
            raise RuntimeError('Unreviewed GB10 cooperative kernel resources')
        self.temps = [torch.empty((6, 128, width), device=device, dtype=torch.float16)
                      for width in (5120, 5120, 1152, 1152)]
        self.locks = torch.zeros(locks, device=device, dtype=torch.int32)
        self.bytes = sum(t.numel()*t.element_size() for t in (*self.temps, self.locks))
        if self.bytes != WORKSPACE_BYTES:
            raise RuntimeError('Unexpected persistent MUL1 workspace size')
        self.ready = torch.cuda.Event()
        self.pending = False
        self.stream_id = None


class Dispatcher:
    def __init__(self, module):
        self.module = module
        self.workspace = None
        self.banks = {}
        self.lock = threading.Lock()
        self.failed = False
        self.last_schedule = None
        self.stream_waits = 0

    def __call__(self, experts, x, ids, weights, chunk_tokens=1024):
        validate_inputs(experts, x, ids, weights, chunk_tokens)
        with self.lock, torch.cuda.device(x.device):
            if self.failed:
                raise RuntimeError('MUL1 dispatcher is poisoned after an earlier failure')
            if not len(x) or not experts:
                self.last_schedule = dict(fused_experts=0, fallback_experts=0,
                                          skipped_assignments=ids.numel(), counts=[])
                return torch.zeros_like(x)
            try:
                stream = torch.cuda.current_stream(x.device)
                if self.workspace is None:
                    self.workspace = Workspace(self.module, x.device)
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
                    bank = Bank(experts, x.device)
                    self.banks[id(experts)] = bank
                if bank.owner is not experts or len(experts) != len(bank.keys):
                    raise ValueError('Expert bank changed after native pointer capture')
                flat = ids.reshape(-1).long()
                safe = torch.where((flat >= 0) & (flat < 384), flat, 384)
                mapped = bank.mapping.index_select(0, safe)
                order = torch.argsort(mapped, stable=True)
                counts = torch.bincount(mapped, minlength=len(bank.keys)+1)
                cpu_counts = counts.cpu().tolist()
                tokens = torch.div(order, ids.shape[1], rounding_mode='floor')
                sorted_weights = weights.reshape(-1).index_select(0, order).float().contiguous()
                out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                fused = sum(0 < count <= 128 for count in cpu_counts[:-1])
                fallback = sum(count > 128 for count in cpu_counts[:-1])
                if fused:
                    self.module.forward(x.half().contiguous(), out, counts, tokens,
                                        sorted_weights, bank.ptrs, work.temps, work.locks)
                offset = 0
                for expert, count in zip(bank.experts, cpu_counts[:-1]):
                    if count > 128:
                        for start in range(offset, offset+count, chunk_tokens):
                            end = min(start+chunk_tokens, offset+count)
                            rows = tokens[start:end]
                            values = expert(x[rows].half().contiguous(), sorted_weights[start:end, None])
                            out.index_add_(0, rows, values.float())
                    offset += count
                result = out.to(x.dtype)
                self.last_schedule = dict(fused_experts=fused, fallback_experts=fallback,
                                          skipped_assignments=cpu_counts[-1], counts=cpu_counts)
                # Serialize shared scratch and pointer-table initialization even
                # if profiling and serving use different current CUDA streams.
                work.ready.record(stream)
                work.pending, work.stream_id = True, stream.cuda_stream
                return result
            except Exception:
                self.failed = True
                # No further CUDA call after a launch/allocation exception.
                raise


def fused_moe(experts, x, route_ids, route_weights, chunk_tokens=1024):
    if _dispatcher is None:
        raise RuntimeError('Register the pinned MUL1 dispatcher before use')
    return _dispatcher(experts, x, route_ids, route_weights, chunk_tokens)


def register(kernel_path=None):
    global _dispatcher, _reference
    modules = {name: importlib.import_module(name) for name in spark_moe.EXPECTED}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        if (not path.is_relative_to('/opt/ds41-dcp-v3/ds41')
                or hashlib.sha256(path.read_bytes()).hexdigest() != spark_moe.EXPECTED[name]):
            raise RuntimeError(f'Unreviewed EXL3 serving implementation: {name}')
    reference, adapter = modules.values()
    path = Path(__file__).resolve().parent/BINARY_NAME if kernel_path is None else Path(kernel_path)
    module = load_kernel(path)
    if _dispatcher is not None:
        if (adapter.eager_moe is not fused_moe or reference.eager_moe is not _reference
                or _dispatcher.module is not module):
            raise RuntimeError('Registered MUL1 dispatcher was replaced')
        return
    if adapter.eager_moe not in (reference.eager_moe, spark_moe.grouped_moe):
        raise RuntimeError('Unexpected preexisting EXL3 dispatch patch')
    _reference = reference.eager_moe
    _dispatcher = Dispatcher(module)
    adapter.eager_moe = fused_moe
