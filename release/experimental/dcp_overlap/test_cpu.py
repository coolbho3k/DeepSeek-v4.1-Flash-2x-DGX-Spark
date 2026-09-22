# SPDX-License-Identifier: AGPL-3.0-only
"""Deferred standard-library tests. No CUDA, serving imports, or network."""
import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from . import integration, policy, prepare, transport

RUNTIME = Path(__file__).resolve().parents[2] / 'runtime'


def function_source(path, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def baseline_forward():
    """Reconstruct the installed forward source without importing GPU modules."""
    source = function_source(RUNTIME / 'serving/ds41/vllm_dcp.py', 'attention_forward')
    rules = function_source(RUNTIME / 'serving/ds41/dcp_head_exchange.py', 'forward_replacements')
    namespace = {}
    exec(rules, namespace)
    for old, new in namespace['forward_replacements']():
        if source.count(old) != 1:
            raise AssertionError('Parent head-exchange anchor changed')
        source = source.replace(old, new)
    source = source.replace('range(0, count, 32)', 'range(0, count, _ds41_collective_chunk)')
    source = source.replace('start + 32', 'start + _ds41_collective_chunk')
    namespace = {k: object() for k in ('bf16_sparse_attention_with_lse', 'sparse_global_to_local_slots',
                 'partition_indices', '_ds41_pack_result', '_ds41_merge_packed')}
    exec(source, namespace)
    original = namespace['attention_forward']
    original.__ds41_patch_source__ = source
    return original


class PolicyTests(unittest.TestCase):
    def test_tiles_preserve_parent_shape(self):
        self.assertEqual(policy.MODE, 'off')
        for rows in range(1, 513):
            self.assertEqual(policy.head_schedule(rows, 'query'), ((0, 32),))
            tiles = policy.head_schedule(rows, 'balanced')
            self.assertEqual(tiles, ((0, 16), (16, 32)) if rows < 32 else ((0, 32),))
            self.assertEqual(sum(b - a for a, b in tiles), 32)

    def test_invalid_policy(self):
        for mode in ('auto', '', 1, None):
            with self.assertRaises(ValueError):
                policy.validate_mode(mode)
        for rows in (-1, 0, 513, 1., True):
            with self.assertRaises(ValueError):
                policy.head_schedule(rows, 'balanced')
        with self.assertRaises(ValueError):
            policy.head_schedule(1, 'off')

    def test_transport_accounting(self):
        self.assertEqual(policy.working_bytes(512), dict(query_receive=33554432,
            result_send=33619968, result_receive=67239936))


class IntegrationTests(unittest.TestCase):
    def test_disabled_is_identity(self):
        original = baseline_forward()
        self.assertIs(integration.wrap_forward(original), original)

    def test_real_parent_rewrite_and_late_mapper(self):
        stub = ModuleType(integration.__package__ + '.attention')
        stub.make_head_attention = lambda original: object()
        stub.step = lambda *a, **kw: None
        with patch.dict(sys.modules, {stub.__name__: stub}), patch.object(policy, 'MODE', 'balanced'):
            original = baseline_forward()
            candidate = integration.wrap_forward(original)
            self.assertEqual(inspect.signature(candidate), inspect.signature(original))
            self.assertTrue(integration.forward_admitted(candidate, 'unused'))
            self.assertNotIn('local_q = group.all_gather', candidate.__ds41_patch_source__)
            self.assertIn('pending.finish()', candidate.__ds41_patch_source__)
            self.assertIn('swa_metadata.is_valid_token[rows]', candidate.__ds41_patch_source__)
            replacement = object()
            integration.bind_sparse_mapper(candidate, replacement)
            for function in (candidate, original, candidate.__ds41_overlap_scheduled__):
                self.assertIs(function.__globals__['sparse_global_to_local_slots'], replacement)
            self.assertTrue(integration.forward_admitted(candidate, 'unused'))
            candidate.__ds41_overlap_scheduled__.__globals__['partition_indices'] = object()
            self.assertFalse(integration.forward_admitted(candidate, 'unused'))

    def test_changed_source_fails_closed(self):
        stub = ModuleType(integration.__package__ + '.attention')
        stub.make_head_attention = lambda original: object()
        stub.step = lambda *a, **kw: None
        with patch.dict(sys.modules, {stub.__name__: stub}), patch.object(policy, 'MODE', 'query'):
            original = baseline_forward()
            original.__ds41_patch_source__ = original.__ds41_patch_source__.replace(
                'local_q = group.all_gather(local_q, dim=1)', 'local_q = something_else(local_q)')
            with self.assertRaises(RuntimeError):
                integration.wrap_forward(original)


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Minimal unpatched source fixture: the shipped runtime now already
        # includes overlap, and must continue to reject double installation.
        cls.payload = {name: source.encode() for name, source in {
            'serving/ds41/vllm_fp4_main.py': 'def register():\n        workspace.register()\n',
            'serving/spark_dcp_communication.py': ('import hashlib\n'
                'def admitted(forward):\n    return '
                "'all_packed = group.all_gather(_ds41_pack_result(partial, lse), dim=0)' not in forward.__ds41_patch_source__\n"),
            'serving/combined_worker.py': 'class Worker:\n    init_device = _init_device\n',
            'serving/spark_sparse_slots.py': 'def install(forward, candidate):\n    forward.__globals__[KEY]=candidate\n',
            'serving/spark_combined_miaai.py': 'DESCRIPTOR = dict(kernel_batch=KERNEL_BATCH,)\n',
            'serving/spark_backend_attestation.py': 'PRIVATE_SOURCES = {}\n',
        }.items()}
        cls.payload['runtime-requirements.json'] = prepare.encoded(dict(
            loaded_backend_verification=dict(sha256=prepare.sha(cls.payload['serving/spark_backend_attestation.py']))))
        for name in ('serving/ds41/combined_config.py', 'serving/ds41/display_kv.py'):
            cls.payload[name] = (RUNTIME / name).read_bytes()

    def test_modes_and_attestation(self):
        before = dict(self.payload)
        for mode in policy.MODES:
            result = prepare.transform(self.payload, mode)
            self.assertEqual(before, self.payload)
            self.assertIn(('MODE = ' + repr(mode)).encode(), result['serving/ds41/dcp_overlap/policy.py'])
            source = result['serving/spark_backend_attestation.py'].decode()
            assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PRIVATE_SOURCES' for t in n.targets))
            for name, digest in ast.literal_eval(assignment.value).items():
                self.assertEqual(prepare.sha(result['serving/' + name]), digest, name)
            requirements = json.loads(result['runtime-requirements.json'])
            self.assertFalse(requirements['dcp_overlap_candidate']['gpu_tests_run'])
            self.assertEqual(result['serving/ds41/combined_config.py'], before['serving/ds41/combined_config.py'])
            self.assertEqual(result['serving/ds41/display_kv.py'], before['serving/ds41/display_kv.py'])

    def test_builder_cannot_patch_same_bundle_twice(self):
        result = prepare.transform(self.payload, 'balanced')
        with self.assertRaises(ValueError):
            prepare.transform(result, 'balanced')

    def test_bad_digest_and_paths(self):
        with self.assertRaises(ValueError):
            prepare.load_parent(RUNTIME, '0' * 64)
        for name in ('../bad', '/absolute', 'a//b', 'a/./b', 'a\\b', '', '.'):
            with self.assertRaises(ValueError):
                prepare.safe_name(name)
        with self.assertRaises(ValueError):
            prepare.prepare(RUNTIME, '0' * 64, RUNTIME, 'balanced')

    def test_fresh_output_remains_unqualified(self):
        with tempfile.TemporaryDirectory(prefix='ds41-overlap-cpu-') as temporary:
            parent = Path(temporary) / 'parent'
            parent.mkdir()
            for name, raw in self.payload.items():
                path = parent / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            manifest = dict(format='ds41_runtime_inputs_v5', standalone_runtime=False,
                clean_rebuild_qualified=False, publication_approved=False,
                files={name: dict(bytes=len(raw), sha256=prepare.sha(raw)) for name, raw in self.payload.items()})
            raw = prepare.encoded(manifest)
            (parent / 'bundle-manifest.json').write_bytes(raw)
            target = Path(temporary) / 'candidate'
            digest = prepare.sha(raw)
            result = prepare.prepare(parent, digest, target, 'balanced')
            self.assertFalse(result['deployed'])
            manifest, _ = prepare.load_parent(target, result['manifest_sha256'])
            self.assertFalse(manifest['serving_qualified'])
            self.assertFalse(manifest['publication_approved'])
            with self.assertRaises(ValueError):
                prepare.prepare(parent, digest, target, 'query')


class TransportTests(unittest.TestCase):
    def test_concurrent_join_owns_query_and_remote_allocations(self):
        from contextlib import contextmanager
        log = []
        class Stream:
            def __init__(self, number): self.cuda_stream = number
            def wait_event(self, event): log.append(('wait', self.cuda_stream, event))
        class Event:
            def record(self, stream): log.append(('record', stream.cuda_stream, self))
        class Tensor:
            ndim, dtype, device, requires_grad = 3, 'bf16', 'device', False
            def __init__(self, shape, dtype): self.shape, self.dtype = shape, dtype
            def __len__(self): return self.shape[0]
            def is_contiguous(self): return True
        main, side = Stream(1), Stream(2)
        active = [main]
        @contextmanager
        def stream_context(stream):
            old, active[0] = active[0], stream
            try: yield
            finally: active[0] = old
        torch = SimpleNamespace(bfloat16='bf16', float32='fp32',
            empty=lambda shape, **kw: Tensor(shape, kw['dtype']), cuda=SimpleNamespace(Event=Event,
            current_stream=lambda *a: active[0], is_current_stream_capturing=lambda: False,
            stream=stream_context))
        validation = SimpleNamespace(_execution_lock=threading.RLock(),
            require_capture_owner=lambda: None, current_owner=lambda: None)
        value = transport.Transport.__new__(transport.Transport)
        value.stream, value.device, value.lock = side, 'device', threading.RLock()
        value.pending, value.failed, value.thread, value.caller_stream = [], False, None, None
        value.comm = SimpleNamespace(disabled=False, available=True,
            all_gather=lambda output, input, stream: log.append(('nccl', stream.cuda_stream)))
        with patch.dict(sys.modules, {'torch': torch, 'ds41.graph_validation': validation}):
            with value.session():
                query = value.gather(Tensor((4, 32, 512), 'bf16'), kind='query')
                produce = lambda: log.append(('remote_attention', active[0].cuda_stream))
                result = value.remote_result(query, Tensor((4, 32, 513), 'fp32'), produce)
                self.assertIs(result.resources, produce)
                self.assertEqual(value.pending, [query, result])
                log.append(('local_attention', active[0].cuda_stream))
                result.join()
                self.assertTrue(query.joined)
                self.assertEqual(value.pending, [])
                with self.assertRaises(RuntimeError): query.join()
            self.assertEqual([row[:2] for row in log], [('record', 1), ('wait', 2),
                ('nccl', 2), ('record', 2), ('record', 1), ('wait', 2),
                ('remote_attention', 2), ('nccl', 2), ('record', 2),
                ('local_attention', 1), ('wait', 1)])

    def test_fork_join_and_failure_retention_without_cuda(self):
        log = []
        class Stream:
            cuda_stream = 1
            def wait_event(self, event):
                log.append(('wait', self.cuda_stream, event))
        class Event:
            def record(self, stream):
                log.append(('record', stream.cuda_stream, self))
        class Tensor:
            shape, ndim, dtype, device, requires_grad = (1, 32, 512), 3, 'bf16', 'device', False
            def __len__(self): return 1
            def is_contiguous(self): return True
        main, side = Stream(), Stream()
        side.cuda_stream = 2
        torch = SimpleNamespace(bfloat16='bf16', float32='fp32',
            empty=lambda *a, **kw: Tensor(), cuda=SimpleNamespace(Event=Event,
            current_stream=lambda *a: main, is_current_stream_capturing=lambda: False))
        validation = SimpleNamespace(_execution_lock=threading.RLock(),
            require_capture_owner=lambda: None, current_owner=lambda: None)
        value = transport.Transport.__new__(transport.Transport)
        value.stream, value.device, value.lock = side, 'device', threading.RLock()
        value.pending, value.failed, value.thread, value.caller_stream = [], False, None, None
        value.comm = SimpleNamespace(disabled=False, available=True,
            all_gather=lambda output, input, stream: log.append(('nccl', stream.cuda_stream)))
        with patch.dict(sys.modules, {'torch': torch, 'ds41.graph_validation': validation}):
            with value.session():
                ticket = value.gather(Tensor(), kind='query')
                self.assertEqual(len(value.pending), 1)
                ticket.join()
                self.assertFalse(value.pending)
                with self.assertRaises(RuntimeError):
                    ticket.join()
            self.assertEqual([row[:2] for row in log],
                             [('record', 1), ('wait', 2), ('nccl', 2), ('record', 2), ('wait', 1)])
            retained = len(transport._failed_resources)
            try:
                with self.assertRaises(RuntimeError):
                    with value.session():
                        value.gather(Tensor(), kind='query')
                        # Deliberately do not join; the session must poison.
                self.assertTrue(value.failed)
                self.assertEqual(len(transport._failed_resources), retained + 1)
                with self.assertRaises(RuntimeError):
                    with value.session():
                        pass
            finally:
                del transport._failed_resources[retained:]  # Mock CPU objects only.


class PackedKernelTests(unittest.TestCase):
    def test_eager_errors_are_checked_at_join_and_capture_keeps_owner_boundary(self):
        from contextlib import contextmanager
        from contextvars import ContextVar
        seen, capturing = [], [False]
        class Error:
            def __init__(self, value): self.value = value
            def numel(self): return 1
            def reshape(self, shape): return self
        def native(errors, messages):
            errors = errors if isinstance(errors, list) else [errors]
            seen.append(([e.value for e in errors], tuple(messages)))
            if not capturing[0] and any(e.value for e in errors):
                raise ValueError('invalid sparse slot')
        namespace = dict(_deferred_errors=ContextVar('test_errors', default=None),
            torch=SimpleNamespace(stack=lambda errors: errors,
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing[0])))
        source = Path(__file__).parent / 'attention.py'
        for name in ('check_flags', 'joined_errors'):
            exec(function_source(source, name), namespace)
        joined = contextmanager(namespace['joined_errors'])
        check = namespace['check_flags']
        module = SimpleNamespace(check_flags=native)
        with patch.dict(sys.modules, {'ds41.graph_validation': module}):
            with self.assertRaisesRegex(ValueError, 'invalid sparse slot'):
                with joined():
                    check(Error(0), ((1, 'invalid sparse slot'),))
                    check(Error(1), ((1, 'invalid sparse slot'),))
                    self.assertEqual(seen, [])
            self.assertEqual(seen[0][0], [0, 1])
            self.assertIsNone(namespace['_deferred_errors'].get())
            capturing[0] = True
            with joined():
                check(Error(0), ((1, 'invalid sparse slot'),))
                self.assertEqual(len(seen), 2)
                check(Error(1), ((1, 'invalid sparse slot'),))
                self.assertEqual(len(seen), 3)
            self.assertEqual(len(seen), 3)

    def test_only_output_addresses_changed(self):
        packed = RUNTIME / 'serving/ds41/dcp_overlap/packed.py'
        prefill = function_source(RUNTIME / 'serving/ds41/online_sparse_attention.py', 'attention')
        prefill = prefill.replace('(token*HEADS+head[:,None])*512', '(token*HEADS+head[:,None])*513')
        prefill = prefill.replace('normalizers+token*HEADS+head,', 'normalizers+(token*HEADS+head)*513,')
        merge = function_source(RUNTIME / 'serving/ds41/online_decode_attention.py', '_merge')
        merge = merge.replace('def _merge(', 'def merge(')
        merge = merge.replace('output + row * 512 + d', 'output + row * 513 + d')
        merge = merge.replace('normalizers + row,', 'normalizers + row * 513,')
        for name, expected in (('attention', prefill), ('merge', merge)):
            self.assertEqual(ast.dump(ast.parse(function_source(packed, name))),
                             ast.dump(ast.parse(expected)))


if __name__ == '__main__':
    unittest.main()
