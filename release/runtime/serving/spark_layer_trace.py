"""Private, opt-in, bounded CPU snapshots of the actual eager model path.

Hooks never replace inputs/outputs or change precision. Copies synchronize
the observed tensors and can perturb timing; a stable traced run alone does
not prove the uninstrumented path is race-free. No trace is armed at startup.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

MAX_TOKENS = 1056
MAX_FORWARDS = 64
MAX_TENSOR_BYTES = 64 * 2**20
MAX_FRAME_BYTES = 8 * 2**20
MIN_AVAILABLE = 1536 * 2**20
SOURCES = {
    'models/deepseek_v4_1/nvidia/model.py':
        '1f1419b164f62fa9067f63b31d65409023fd0d73c158cf37528ceee76de268b7',
    'models/deepseek_v4_1/nvidia/vl_model.py':
        'ab698e56c83a345ea73e41359cab79b5e484ccfc116ed131347d50cdfe896251',
}


def memory_available():
    return next(int(line.split()[1]) * 1024
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate trace-control key')
        result[key] = value
    return result


def control(path, now=None):
    if not path.exists():
        if path.is_symlink():
            raise ValueError('Redirected trace control')
        return None
    if path.resolve() != path or not path.is_file() or path.stat().st_size > 4096:
        raise ValueError('Trace control must be a small regular private file')
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if (set(value) != {'format', 'tag', 'expires_unix', 'max_forwards'}
            or value['format'] != 'ds41_layer_trace_control_v1'
            or not isinstance(value['tag'], str)
            or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,39}', value['tag'])
            or type(value['max_forwards']) is not int
            or not 1 <= value['max_forwards'] <= MAX_FORWARDS
            or type(value['expires_unix']) not in (int, float)
            or not math.isfinite(value['expires_unix'])):
        raise ValueError('Invalid bounded trace control')
    now = time.time() if now is None else now
    if value['expires_unix'] <= now:
        return None
    if value['expires_unix'] > now + 1800:
        raise ValueError('Trace control may arm at most30 minutes ahead')
    return value


def exclusive(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError('Private trace record exceeds8MiB')
    with path.open('xb') as stream:
        stream.write(raw)
    return len(raw)


class Recorder:
    def __init__(self, root, rank, torch, device_type='cuda'):
        self.root, self.rank, self.torch = root, rank, torch
        self.device_type = device_type
        self.current = None
        self.tag = None
        self.count = 0
        self.last_frame = None
        self.last_position = None
        self.request_index = -1
        self.total_bytes = 0

    def snapshot(self, value, rows):
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            if len(value) > 10:
                raise ValueError('Unexpected trace tuple')
            return [self.snapshot(item, rows) for item in value]
        if not isinstance(value, self.torch.Tensor) or value.device.type != self.device_type:
            raise ValueError('Expected a tensor on the model device')
        if value.ndim < 1 or value.ndim > 4:
            raise ValueError('Unexpected boundary tensor rank')
        if rows is not None:
            if type(rows) is not int or not 1 <= rows <= MAX_TOKENS or value.shape[0] < rows:
                raise ValueError('Invalid valid-token prefix')
            selected = value[:rows]
        else:
            selected = value
        if selected.numel() * selected.element_size() > MAX_TENSOR_BYTES:
            raise ValueError('Boundary snapshot exceeds64MiB')
        # Contiguity conversion happens on CPU, never by cloning GPU state.
        cpu = selected.detach().to(device='cpu').contiguous()
        raw = cpu.view(self.torch.uint8).numpy()
        last = cpu[-1:].view(self.torch.uint8).numpy()
        sample = base64.b64encode(last).decode() if last.nbytes <= 128 * 2**10 else None
        return dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
            sha256=hashlib.sha256(raw).hexdigest(), bytes=raw.nbytes,
            last_row_base64=sample, last_row_omitted=sample is None)

    def begin(self, input_ids, positions, inputs_embeds, lookback, counts):
        if self.current is not None:
            raise ValueError('Nested/concurrent model trace is unsupported')
        self.last_frame = None
        armed = control(self.root / 'control.json')
        if armed is None:
            return
        if self.tag is not None and self.tag != armed['tag']:
            raise ValueError('One immutable trace tag per worker')
        if self.count >= armed['max_forwards']:
            return
        if memory_available() < MIN_AVAILABLE:
            raise ValueError('Trace requires1536MiB available host RAM')
        if (set(counts) != {'num_decodes', 'num_prefills', 'num_decode_tokens', 'num_prefill_tokens'}
                or any(type(value) is not int or value < 0 for value in counts.values())
                or counts['num_decodes'] + counts['num_prefills'] != 1):
            raise ValueError('Trace requires one actual request, not a profile batch')
        rows = counts['num_decode_tokens'] + counts['num_prefill_tokens']
        if not 1 <= rows <= MAX_TOKENS:
            raise ValueError('Unsupported trace token count')
        if self.tag is None:
            self.tag = armed['tag']
            (self.root / self.tag).mkdir(mode=0o700, exist_ok=False)
        pos = positions[:rows].detach().to(device='cpu').tolist()
        if len(pos) != rows or any(type(p) is not int for p in pos):
            raise ValueError('Expected flat integer positions')
        if pos != list(range(pos[0], pos[0] + rows)):
            raise ValueError('Expected contiguous positions for a single request')
        if self.last_position is None or pos[0] <= self.last_position:
            self.request_index += 1
        self.last_position = pos[-1]
        self.current = dict(format='ds41_layer_boundary_trace_v1', rank=self.rank,
            frame=self.count, request_index=self.request_index,
            started_unix=time.time(), valid_tokens=rows, counts=counts,
            position_start=pos[0], position_end=pos[-1], stages=[],
            inputs=dict(input_ids=self.snapshot(input_ids, rows),
                positions=self.snapshot(positions, rows),
                inputs_embeds=self.snapshot(inputs_embeds, rows),
                lookback_token_ids=self.snapshot(lookback, None)))

    def record(self, name, output):
        if self.current is None:
            return
        stages = self.current['stages']
        if len(stages) >= 256 or any(row['name'] == name for row in stages):
            raise ValueError('Unexpected duplicate/excess model boundary')
        if len(stages) % 16 == 0 and memory_available() < MIN_AVAILABLE:
            raise ValueError('Host RAM fell below the bounded trace floor')
        stages.append(dict(name=name, value=self.snapshot(output, self.current['valid_tokens'])))

    def finish(self, output):
        if self.current is None:
            return
        self.record('model.output', output)
        value = self.current
        value['finished_unix'] = time.time()
        value['available_bytes'] = memory_available()
        value['model_forward_returned_none'] = output is None
        value['timing_perturbed_by_cpu_copies'] = True
        path = self.root / self.tag / f'frame-{self.count:03d}.json'
        self.total_bytes += exclusive(path, value)
        self.last_frame = self.count
        self.count += 1
        self.current = None

    def logits(self, output):
        if self.last_frame is None:
            return
        value = dict(format='ds41_layer_trace_logits_v1', rank=self.rank,
            frame=self.last_frame, time_unix=time.time(), output=self.snapshot(output, None))
        if output is not None:
            cpu = output.detach().to(device='cpu').float()
            if cpu.ndim != 2 or not 1 <= cpu.shape[0] <= 4 or cpu.shape[1] != 129280:
                raise ValueError('Expected bounded generated-token logits')
            scores, tokens = cpu[-1].topk(8)
            value['last_row_top8'] = dict(tokens=tokens.tolist(), logits=scores.tolist())
        self.total_bytes += exclusive(self.root / self.tag / f'logits-{self.last_frame:03d}.json', value)
        self.last_frame = None


def attach(model, rank):
    import torch
    import vllm
    from vllm.forward_context import get_forward_context
    native = Path(vllm.__file__).resolve().parent
    for name, expected in SOURCES.items():
        if hashlib.sha256((native/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Unreviewed native model boundary source')
    if type(rank) is not int or rank not in (0, 1) or type(model).__name__ != 'DeepseekV41ForCausalLM':
        raise ValueError('Trace requires the actual two-node native model wrapper')
    core = model.language_model.model
    if (len(core.layers) != 40 or core.use_sequence_parallel or
            any(type(layer).__name__ != 'DeepseekV4DecoderLayer' for layer in core.layers)):
        raise ValueError('Trace requires the complete40-layer non-SP backbone')
    root = Path('/cache/ds41-layer-trace')
    if root.resolve() != root or root.exists():
        raise ValueError('Private trace directory must be fresh')
    root.mkdir(mode=0o700)
    recorder = Recorder(root, rank, torch)

    def begin(module, args, kwargs):
        if control(root/'control.json') is None:
            recorder.last_frame = None
            return
        fields = ('input_ids', 'positions', 'intermediate_tensors',
                  'inputs_embeds', 'lookback_token_ids')
        values = dict(zip(fields, args)) | kwargs
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or core.engram_swa_prefix not in metadata:
            raise ValueError('Actual native SWA metadata required for valid rows')
        swa = metadata[core.engram_swa_prefix]
        counts = {name: getattr(swa, name) for name in
                  ('num_decodes', 'num_prefills', 'num_decode_tokens', 'num_prefill_tokens')}
        recorder.begin(values.get('input_ids'), values.get('positions'),
            values.get('inputs_embeds'), values.get('lookback_token_ids'), counts)

    handles = [core.register_forward_pre_hook(begin, with_kwargs=True),
               core.register_forward_hook(lambda module, args, out: recorder.finish(out), always_call=True)]
    if isinstance(core.engram_hash, torch.nn.Module):
        handles.append(core.engram_hash.register_forward_hook(
            lambda module, args, out: recorder.record('engram.hashes', out)))
    for i, layer in enumerate(core.layers):
        prefix = f'layer.{i:02d}'
        for name, child, input_index in (('attn', layer.attn, 1), ('ffn', layer.ffn, 0)):
            label = prefix+'.'+name
            handles.append(child.register_forward_pre_hook(
                lambda module, args, label=label, index=input_index: recorder.record(label+'.input', args[index])))
            handles.append(child.register_forward_hook(
                lambda module, args, out, label=label: recorder.record(label+'.output', out)))
        if layer.engram is not None:
            handles.append(layer.engram.register_forward_pre_hook(
                lambda module, args, label=prefix: recorder.record(label+'.engram.input', args[0])))
            handles.append(layer.engram.register_forward_hook(
                lambda module, args, out, label=prefix: recorder.record(label+'.engram.output', out)))
        handles.append(layer.register_forward_hook(
            lambda module, args, out, label=prefix: recorder.record(label+'.state', out)))
    handles.append(model.language_model.logits_processor.register_forward_hook(
        lambda module, args, out: recorder.logits(out)))
    model._ds41_private_layer_trace = (recorder, handles)
    exclusive(root/'ready.json', dict(status='private_layer_trace_attached_unarmed',
        rank=rank, pid=os.getpid(), hook_count=len(handles), native_source_sha256=SOURCES,
        trace_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_valid_tokens=MAX_TOKENS, max_forwards=MAX_FORWARDS,
        max_tensor_bytes=MAX_TENSOR_BYTES, max_frame_bytes=MAX_FRAME_BYTES,
        min_available_bytes=MIN_AVAILABLE, gpu_snapshot_allocations=False,
        outputs_replaced=False, weight_mutations=False, publication_approved=False))
