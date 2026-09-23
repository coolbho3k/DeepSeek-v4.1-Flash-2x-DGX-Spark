# SPDX-License-Identifier: AGPL-3.0-only
"""Clone a verified runtime and defer full-graph error-flag readback.

Previously every full-graph replay synchronously copied its device error flags
to the host before returning, stalling the GPU between the target graph and
the LM head. Now each captured full graph ends with a cross-rank summary
(an all-reduce of per-rank nonzero-flag counts), and replay enqueues only
non-blocking copies of the flags and summary into pinned host buffers.
Pending checks are drained (a) in AsyncOutput.get_output before any sampled
token leaves the worker, and (b) at the start of every graph execution. A
peer rank's errors therefore also block rank 0's output. Error codes,
messages, masking and poisoning are unchanged; piecewise graphs and eager
checks keep their synchronous boundary.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'release/runtime'))
from verify import verify

ASYNC_UTILS_SHA256 = '77e17a4570ead2be30ae9b00888cf077a3b873018c12ca0549b853bfda02c1ba'

VALIDATION_EDITS = [
    # Module docstring: record the new boundary precisely.
    ('Graph replay is checked before its\noutputs leave the execution boundary; errors poison the owner, not the GPU.\n',
     'Full-graph replay flags, plus a captured\ncross-rank summary, are checked before sampled output leaves the worker\n'
     '(AsyncOutput.get_output) and before the next graph executes; errors poison\nthe owner, not the GPU.\n'),
    ('_live_owners = set()\n',
     '_live_owners = set()\n_pending = set()\n_pending_lock = threading.RLock()\n'),
    # Owner state: a device summary allocated before capture, pinned host
    # mirrors allocated on first deferred replay.
    ('        self.flags = torch.zeros(MAX_ERROR_VALUES, device=device, dtype=torch.int32)\n',
     '        self.flags = torch.zeros(MAX_ERROR_VALUES, device=device, dtype=torch.int32)\n'
     '        # [rank0 nonzero flags, rank1 nonzero flags], all-reduced in-graph.\n'
     '        self.summary = torch.zeros(2, device=device, dtype=torch.float32)\n'
     '        self.deferred = False\n'
     '        self.rank = None\n'
     '        self.host_flags = self.host_summary = self.host_event = None\n'),
    # Drain before any execution (fail closed on either rank).
    ("            self.capture_only = capture_only\n            token = _current.set(self)\n",
     "            drain_pending()\n"
     "            self.capture_only = capture_only\n            token = _current.set(self)\n"),
    ("                if not capture_only and self.used:\n"
     "                    flags = self.flags[:self.used].cpu().tolist()\n"
     "                    for start, count, messages in self.checks:\n"
     "                        _raise_flags(flags[start:start + count], messages)\n",
     "                if not capture_only and self.deferred:\n"
     "                    self._defer(stream)\n"
     "                elif not capture_only and self.used:\n"
     "                    flags = self.flags[:self.used].cpu().tolist()\n"
     "                    for start, count, messages in self.checks:\n"
     "                        _raise_flags(flags[start:start + count], messages)\n"),
    ('    def wait_before_graph_destruction(self):\n',
     '''    def capture_tail(self):
        """Append the cross-rank error summary to a full graph being captured.

        Captured unconditionally at the end of every full graph on both TP
        ranks, so collective order is identical even if flag counts differ.
        """
        import torch
        from vllm.distributed import (get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce)
        if (current_owner() is not self or not self.capture_only or self.captured
                or not torch.cuda.is_current_stream_capturing()
                or get_tensor_model_parallel_world_size() != 2):
            raise RuntimeError('Cross-rank graph validation requires an owned TP2 full-graph capture')
        rank = get_tensor_model_parallel_rank()
        self.summary.zero_()
        if self.used:
            count = self.flags[:self.used].ne(0).sum().to(torch.float32)
            self.summary[rank:rank + 1].copy_(count.reshape(1))
        reduced = tensor_model_parallel_all_reduce(self.summary)
        if reduced is not self.summary:
            self.summary.copy_(reduced)
        self.rank = rank
        self.deferred = True

    def _defer(self, stream):
        import torch
        if self.host_event is None:
            self.host_flags = torch.zeros(max(self.used, 1), dtype=torch.int32, pin_memory=True)
            self.host_summary = torch.zeros(2, dtype=torch.float32, pin_memory=True)
            self.host_event = torch.cuda.Event()
        with _pending_lock:
            if self in _pending:
                raise RuntimeError('Deferred graph validation was not drained before replay')
            if self.used:
                self.host_flags[:self.used].copy_(self.flags[:self.used], non_blocking=True)
            self.host_summary.copy_(self.summary, non_blocking=True)
            self.host_event.record(stream)
            _pending.add(self)

    def _check_deferred(self):
        self.host_event.synchronize()
        flags = self.host_flags[:self.used].tolist() if self.used else []
        for start, count, messages in self.checks:
            _raise_flags(flags[start:start + count], messages)
        summary = self.host_summary.tolist()
        if any(summary):
            raise ValueError('Tensor-parallel peer reported graph validation errors')

    def wait_before_graph_destruction(self):
'''),
]

VALIDATION_APPEND = '''

def drain_pending():
    """Check every deferred full-graph replay; poison and raise on any error."""
    with _pending_lock:
        for owner in list(_pending):
            _pending.discard(owner)
            try:
                owner._check_deferred()
            except BaseException:
                owner.failed = True
                raise


def capture_tail():
    owner = current_owner()
    if owner is None:
        raise RuntimeError('Full graph capture requires an owned validation boundary')
    owner.capture_tail()
'''

GRAPH_EDITS = [
    ("    'vllm.compilation.cuda_graph':\n        'cbc474f9098386d2eef2e3ff61364c611fbc1338d9c259ad5a24439aa7f06412',\n",
     "    'vllm.compilation.cuda_graph':\n        'cbc474f9098386d2eef2e3ff61364c611fbc1338d9c259ad5a24439aa7f06412',\n"
     f"    'vllm.v1.worker.gpu.async_utils':\n        '{ASYNC_UTILS_SHA256}',\n"),
    ('from .graph_validation import GraphOwner, _execution_lock\n',
     'from .graph_validation import GraphOwner, _execution_lock, capture_tail, drain_pending\n'),
    ("    capture = _compile(original_capture, [\n"
     "        ('with torch.cuda.graph(',\n"
     "         'with _ds41_capture_context(self, desc, graph), torch.cuda.graph('),\n"
     "    ], {'_ds41_capture_context': capture_context})\n",
     "    capture = _compile(original_capture, [\n"
     "        ('with torch.cuda.graph(',\n"
     "         'with _ds41_capture_context(self, desc, graph), torch.cuda.graph('),\n"
     "        ('get_offloader().join_after_forward()',\n"
     "         'get_offloader().join_after_forward(); _ds41_capture_tail()'),\n"
     "    ], {'_ds41_capture_context': capture_context, '_ds41_capture_tail': capture_tail})\n"
     "    output_type = modules['vllm.v1.worker.gpu.async_utils'].AsyncOutput\n"
     "    original_output = output_type.get_output\n"
     "\n"
     "    def get_output(output):\n"
     "        # Deferred full-graph validation must pass before tokens leave.\n"
     "        drain_pending()\n"
     "        return original_output(output)\n"),
    ("        (pw.CUDAGraphWrapper, 'clear_graphs', clear),\n    ]\n",
     "        (pw.CUDAGraphWrapper, 'clear_graphs', clear),\n"
     "        (output_type, 'get_output', get_output),\n    ]\n"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def edit(path, edits, append=''):
    text = path.read_text()
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError(f'Changed source anchor in {path}: {before[:60]!r}')
        text = text.replace(before, after)
    path.write_text(text + append)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-kit', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    parent = a.parent_kit.absolute()
    verify(parent, a.parent_sha256)
    kit = a.output.absolute()
    if kit.exists() or a.receipt.exists():
        raise ValueError('Fresh output required')
    shutil.copytree(parent, kit)
    edit(kit / 'serving/ds41/graph_validation.py', VALIDATION_EDITS, VALIDATION_APPEND)
    edit(kit / 'serving/ds41/vllm_owned_graphs.py', GRAPH_EDITS)
    manifest = json.loads((parent / 'bundle-manifest.json').read_bytes())
    history = {name: {row['sha256'], sha((kit / name).read_bytes())}
               for name, row in manifest['files'].items()
               if name.endswith('.py') and name.startswith(('serving/', 'tools/'))}
    for _ in range(32):
        updates = {}
        for name, old in history.items():
            new = sha((kit / name).read_bytes())
            for digest in old:
                if digest != new:
                    if digest in updates and updates[digest] != new:
                        raise RuntimeError('Ambiguous pin')
                    updates[digest] = new
            old.add(new)
        changed = False
        for name in history:
            path = kit / name
            text = before = path.read_text()
            for old, new in updates.items():
                text = text.replace(old, new)
            if text != before:
                path.write_text(text)
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError('Hash cycle')
    for path in (kit / 'serving').rglob('*.py'):
        ast.parse(path.read_text())
    requirements = json.loads((kit / 'runtime-requirements.json').read_bytes())
    requirements['loaded_backend_verification']['sha256'] = sha(
        (kit / 'serving/spark_backend_attestation.py').read_bytes())
    requirements['deferred_graph_validation'] = dict(full_graphs=True, piecewise_graphs=False,
        cross_rank_summary='in-graph all-reduce', drained_before='AsyncOutput.get_output and next graph execution',
        async_utils_sha256=ASYNC_UTILS_SHA256)
    (kit / 'runtime-requirements.json').write_bytes(encoded(requirements))
    overlay = {q.relative_to(kit / 'serving').as_posix(): sha(q.read_bytes())
               for q in sorted((kit / 'serving').rglob('*')) if q.is_file() and q.name != 'overlay-manifest.json'}
    (kit / 'serving/overlay-manifest.json').write_bytes(encoded(overlay))
    manifest['parent_manifest_sha256'] = a.parent_sha256
    manifest['files'] = {q.relative_to(kit).as_posix(): dict(bytes=q.stat().st_size, sha256=sha(q.read_bytes()))
                         for q in sorted(kit.rglob('*')) if q.is_file() and q.name != 'bundle-manifest.json'}
    (kit / 'bundle-manifest.json').write_bytes(encoded(manifest))
    digest = sha((kit / 'bundle-manifest.json').read_bytes())
    proof = verify(kit, digest)
    changes = {n: dict(before=sha((parent / n).read_bytes()) if (parent / n).is_file() else None,
                       after=sha((kit / n).read_bytes())) for n in manifest['files']
               if not (parent / n).is_file() or (parent / n).read_bytes() != (kit / n).read_bytes()}
    a.receipt.write_bytes(encoded(dict(parent_kit=str(parent), parent_sha256=a.parent_sha256,
        candidate_kit=str(kit), candidate_kit_sha256=digest, changed_files=changes, verification=proof)))
    print(json.dumps(dict(kit=str(kit), sha256=digest, changed_files=sorted(changes))))


if __name__ == '__main__':
    main()
