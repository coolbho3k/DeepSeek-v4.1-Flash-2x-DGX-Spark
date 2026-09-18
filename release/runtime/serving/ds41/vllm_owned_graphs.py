# SPDX-License-Identifier: AGPL-3.0-only
# DS41 adaptation for the MiaAI serving integration. Calls into pinned vLLM
# retain its Apache-2.0 implementation and native graph-pool/output policy.
"""Owned validation/callback lifetime around native V2 and piecewise graphs.

No independent graph manager, global Tensor monkeypatch, or lazy capture policy.
The upstream capture monitor remains authoritative about when capture is legal.
"""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import weakref

from .graph_validation import GraphOwner, _execution_lock
from .vllm_dcp import _compile

UPSTREAM = {
    'vllm.v1.worker.gpu.cudagraph_utils':
        '211de2232fb71aeb761dedbd7fadb7e0832db9331eac6951a1eafb6d77f1be9c',
    'vllm.compilation.cuda_graph':
        'cbc474f9098386d2eef2e3ff61364c611fbc1338d9c259ad5a24439aa7f06412',
}
_failed_resources = []


class GraphResources:
    def __init__(self):
        self.owners = {}
        self.graphs = {}
        self.poisoned = False

    def create(self, key, device):
        if self.poisoned or key in self.owners:
            raise RuntimeError('Cannot replace a live or poisoned graph owner')
        owner = GraphOwner(device)
        self.owners[key] = owner
        return owner

    def bind(self, key, graph):
        if self.poisoned or key not in self.owners or key in self.graphs:
            raise RuntimeError('Native CUDA graph must have exactly one live owner')
        self.graphs[key] = graph
        return graph

    def clear(self, *, finalizing=False):
        with _execution_lock:
            try:
                if self.poisoned or any(o.failed for o in self.owners.values()):
                    raise RuntimeError('Retaining poisoned native graph resources until process exit')
                for owner in self.owners.values():
                    owner.wait_before_graph_destruction()
                for graph in self.graphs.values():
                    graph.reset()
                self.graphs.clear()
                for owner in self.owners.values():
                    owner.release_after_graph_destruction()
                self.owners.clear()
            except BaseException:
                if not self.poisoned:
                    self.poisoned = True
                    _failed_resources.append(self)
                if not finalizing:
                    raise


def _finalize(resources):
    resources.clear(finalizing=True)


def _resources(instance):
    # Bypass CUDAGraphWrapper.__getattr__, which delegates to its runnable.
    resources = vars(instance).get('_ds41_graph_resources')
    if resources is None:
        resources = GraphResources()
        instance._ds41_graph_resources = resources
        finalizer = weakref.finalize(instance, _finalize, resources)
        # Interpreter teardown is not a safe place to call CUDA APIs.
        finalizer.atexit = False
    return resources


def make_graph_patches():
    import torch
    modules = {name: importlib.import_module(name) for name in UPSTREAM}
    for name, module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM[name]:
            raise RuntimeError(f'Unreviewed native graph runtime: {name}')
    v2 = modules['vllm.v1.worker.gpu.cudagraph_utils']
    pw = modules['vllm.compilation.cuda_graph']
    original_capture = v2.CudaGraphManager.capture
    original_replay = v2.CudaGraphManager.run_fullgraph
    original_call = pw.CUDAGraphWrapper.__call__
    original_clear = pw.CUDAGraphWrapper.clear_graphs

    @contextmanager
    def capture_context(manager, key, graph):
        pw.validate_cudagraph_capturing_enabled()
        if manager.use_breakable_cg:
            raise ValueError('Combined stack uses native full/compiled-piecewise graphs, not breakable graphs')
        resources = _resources(manager)
        owner = resources.create(key, manager.device)
        resources.bind(key, graph)
        with owner.execution(capture_only=True):
            yield

    capture = _compile(original_capture, [
        ('with torch.cuda.graph(',
         'with _ds41_capture_context(self, desc, graph), torch.cuda.graph('),
    ], {'_ds41_capture_context': capture_context})

    def replay(manager, key):
        resources = vars(manager).get('_ds41_graph_resources')
        if (resources is None or key not in resources.owners
                or resources.graphs.get(key) is not manager.graphs.get(key)):
            raise RuntimeError('Full graph replay has no matching validation/callback owner')
        with resources.owners[key].execution(capture_only=False):
            return original_replay(manager, key)

    def new_piecewise_graph(wrapper, key):
        return _resources(wrapper).bind(key, torch.cuda.CUDAGraph())

    compiled_call = _compile(original_call, [
        ('cudagraph = torch.cuda.CUDAGraph()',
         'cudagraph = _ds41_new_piecewise_graph(self, batch_descriptor)'),
    ], {'_ds41_new_piecewise_graph': new_piecewise_graph})

    def call(wrapper, *args, **kwargs):
        if not pw.is_forward_context_available():
            return original_call(wrapper, *args, **kwargs)
        context = pw.get_forward_context()
        if (context.cudagraph_runtime_mode == pw.CUDAGraphMode.NONE
                or context.cudagraph_runtime_mode != wrapper.runtime_mode):
            return original_call(wrapper, *args, **kwargs)
        key = context.batch_descriptor
        if key is None:
            raise ValueError('Native piecewise graph requires a batch descriptor')
        entry = wrapper.concrete_cudagraph_entries.get(key)
        capture_only = entry is None or entry.cudagraph is None
        if capture_only:
            # Check BEFORE allocating ownership buffers; no lazy graph memory
            # can be introduced after the native KV-admission boundary closes.
            pw.validate_cudagraph_capturing_enabled()
            resources = _resources(wrapper)
            owner = resources.create(key, torch.device('cuda', torch.cuda.current_device()))
        else:
            resources = vars(wrapper).get('_ds41_graph_resources')
            if (resources is None or key not in resources.owners
                    or resources.graphs.get(key) is not entry.cudagraph):
                raise RuntimeError('Piecewise replay has no matching graph owner')
            owner = resources.owners[key]
        with owner.execution(capture_only=capture_only):
            return compiled_call(wrapper, *args, **kwargs)

    def clear(wrapper):
        resources = vars(wrapper).get('_ds41_graph_resources')
        if resources is not None:
            resources.clear()
        return original_clear(wrapper)

    return [
        (v2.CudaGraphManager, 'capture', capture),
        (v2.CudaGraphManager, 'run_fullgraph', replay),
        (pw.CUDAGraphWrapper, '__call__', call),
        (pw.CUDAGraphWrapper, 'clear_graphs', clear),
    ]
