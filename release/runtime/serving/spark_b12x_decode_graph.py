"""Load-time, one-token replay of the unchanged native B12X operation.

Experimental: not enabled by a frozen serving kit. Registration is CPU-only;
each native post-load call packs normally, then prewarms its graph before KV
profiling. No lazy first-request capture, weight conversion or vision changes.
All other input shapes/dtypes/biases use the original native apply method.
"""
import hashlib
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

from spark_dense_decode_graph import DenseDecodeGraph, DenseGraphPool

NATIVE_SHA = '7abc42bccf03114e880871fa2ffd67d11466483b2ea636d466e762c17417f3d9'
CORE_SHA = '89bb94bf73ebb2859399abc61350794b9cffe205c79d5329b3498e927a60acd2'
ATTR = '_ds41_b12x_decode_graph'
_installed = None


def require_opt_in():
    required = dict(DS41_ENABLE_B12X_GRAPHS='1', DS41_B12X_NATIVE_SELECTION='1',
                    DS41_B12X_FP32_REDUCER='1', B12X_DENSE_SPLITK_TURBO='0')
    if any(os.environ.get(name) != value for name, value in required.items()):
        raise ValueError('Dense graphs require explicit opt-in and qualified FP32 B12X selection')


def storage_signature(packed):
    """Metadata only; detects replacement/resize without tensor readback."""
    tensors = (packed.weight.values, packed.weight.scale_rows, packed.weight.scale_mma)
    return (int(packed.in_features), int(packed.out_features), int(packed.padded_in_features),
            tuple((id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                  for t in tensors))


class NativeDenseGraphBinding:
    def __init__(self, native_apply, kernel, layer, pool):
        import torch
        self.packed = layer.b12x_mxfp8_packed_weight
        self.storage = storage_signature(self.packed)
        # Do not capture layer itself: layer -> binding -> callable -> layer
        # would retain CUDA weights/buffers until cyclic GC.
        proxy = SimpleNamespace(b12x_mxfp8_packed_weight=self.packed)
        example = torch.zeros((1, int(self.packed.in_features)), dtype=torch.bfloat16,
                              device=self.packed.weight.values.device)
        self.graph = DenseDecodeGraph(lambda x: native_apply(kernel, proxy, x, None),
                                      example, pool=pool)

    def validate(self, layer):
        if (layer.b12x_mxfp8_packed_weight is not self.packed
                or storage_signature(self.packed) != self.storage):
            raise ValueError('Dense graph packed weight storage changed after prewarming')
        if self.graph.failed or self.graph.pool.failed:
            raise ValueError('Dense graph binding is poisoned')

    def describe(self):
        return dict(prewarmed=True, input_shape=list(self.graph.signature[0]),
                    input_dtype=str(self.graph.signature[1]),
                    output_shape=list(self.graph.output.shape),
                    shared_pool_token=list(self.graph.pool.token),
                    capture_live_tensor_bytes=self.graph.capture_live_tensor_bytes,
                    input_output_bytes=(self.graph.input.numel()*self.graph.input.element_size()
                                        + self.graph.output.numel()*self.graph.output.element_size()))


def _install(backend):
    """Install only after source/selector admission; separated for CPU tests."""
    global _installed
    if _installed is not None:
        if (backend is not _installed['backend']
                or backend.process_weights_after_loading is not _installed['process']
                or backend.apply_weights is not _installed['apply']):
            raise RuntimeError('Native B12X graph hooks changed')
        return
    native_process = backend.process_weights_after_loading
    native_apply = backend.apply_weights
    pool = DenseGraphPool()

    def process(self, layer):
        import torch
        if hasattr(layer, ATTR) or pool.failed or pool.graphs >= 256:
            raise ValueError('Dense layer already prewarmed, or graph arena unavailable')
        native_process(self, layer)
        try:
            with torch.inference_mode():
                binding = NativeDenseGraphBinding(native_apply, self, layer, pool)
            setattr(layer, ATTR, binding)
        except BaseException:
            pool.failed = True
            raise

    def apply(self, layer, x, bias=None):
        import torch
        binding = getattr(layer, ATTR, None)
        if binding is None:
            raise ValueError('Native B12X layer was not graph-prewarmed during loading')
        binding.validate(layer)
        if (isinstance(x, torch.Tensor) and x.ndim >= 2 and bias is None
                and x.dtype == torch.bfloat16 and x.device == binding.graph.signature[2]
                and x.shape[-1] == binding.graph.signature[0][1]
                and x.numel() == x.shape[-1] and not torch.is_grad_enabled()
                and not torch.cuda.is_current_stream_capturing()):
            output = binding.graph(x.reshape(1, x.shape[-1]))
            return output.view(*x.shape[:-1], int(binding.packed.out_features))
        return native_apply(self, layer, x, bias)

    backend.process_weights_after_loading = process
    backend.apply_weights = apply
    _installed = dict(backend=backend, process=process, apply=apply, pool=pool,
                      native_process=native_process, native_apply=native_apply)


def register():
    require_opt_in()
    native = importlib.import_module('vllm.model_executor.kernels.linear.mxfp8.b12x')
    core = importlib.import_module('spark_dense_decode_graph')
    for module, expected in ((native, NATIVE_SHA), (core, CORE_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError('Unqualified dense graph source: '+module.__name__)
    # This also verifies the precision reducer and SM121; no capture/allocation.
    from spark_b12x_linear import register as register_selection
    register_selection()
    _install(native.B12xMxfp8LinearKernel)
