"""Experimental bounded replay of an unchanged dense decode operation.

Not registered in any serving kit. Capture must be prepared/accounted for
before serving admission. Inputs are copied into private buffers and outputs
are cloned, so later replays cannot overwrite a caller's previous result.
Shared pools contain scratch only: all input/output buffers live outside the
capture pool, no capture-owned tensors may survive, and a pool-wide lock/event
serializes capture and replay across streams. No numerical/weight changes.
"""
import threading


def scratch_pool_usage(segments, token):
    """Inspect the actual arena, not unrelated global allocator changes."""
    if not segments or any('segment_pool_id' not in row for row in segments):
        raise ValueError('CUDA allocator snapshot lacks graph-pool identities')
    own = [row for row in segments if tuple(row['segment_pool_id']) == tuple(token)]
    if not own:
        raise ValueError('Captured scratch pool is missing from allocator snapshot')
    return dict(active_bytes=sum(row['active_size'] for row in own),
                reserved_bytes=sum(row['total_size'] for row in own), segments=len(own))


class DenseGraphPool:
    """One-device scratch arena; graph output lifetimes never depend on it."""
    def __init__(self):
        self.lock = threading.Lock()
        self.device = None
        self.pending_stream = None
        self.failed = False
        self.graphs = 0

    def validate(self, device):
        if self.failed or self.graphs >= 256:
            raise ValueError('Dense graph pool is poisoned or at its graph bound')
        if self.device is not None and self.device != device:
            raise ValueError('A dense graph pool cannot span devices')

    def prepare(self, torch, device):
        self.validate(device)
        if self.device is None:
            self.device = device
            self.token = torch.cuda.graph_pool_handle()
            self.capture_stream = torch.cuda.Stream()
            self.ready = torch.cuda.Event()

    def wait(self, stream):
        if self.pending_stream is not None and self.pending_stream != stream.cuda_stream:
            stream.wait_event(self.ready)

    def record(self, stream):
        self.ready.record(stream)
        self.pending_stream = stream.cuda_stream


class DenseDecodeGraph:
    def __init__(self, function, example, pool=None):
        import torch
        if (not callable(function) or torch.is_grad_enabled()
                or not isinstance(example, torch.Tensor)
                or example.device.type != 'cuda' or example.ndim != 2
                or not 1 <= example.shape[0] <= 8
                or not 1 <= example.shape[1] <= 8192
                or example.dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError('An unarmed, bounded inference-only dense graph is required')
        if pool is not None and not isinstance(pool, DenseGraphPool):
            raise ValueError('An owned dense scratch pool is required')
        self.signature = (tuple(example.shape), example.dtype, example.device)
        self.pool = pool if pool is not None else DenseGraphPool()
        self.pool.validate(example.device)
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('Nested dense graph capture is not supported')
        self.failed = False
        self.function = function  # Retain packed-weight owners captured by the callable.
        with self.pool.lock, torch.cuda.device(example.device):
            try:
                self.pool.prepare(torch, example.device)
                before = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                current = torch.cuda.current_stream()
                self.pool.wait(current)
                self.input = torch.empty_like(example, memory_format=torch.contiguous_format)
                self.input.copy_(example)
                # Every capture uses the SAME stream for allocator reuse.
                capture_stream = self.pool.capture_stream
                capture_stream.wait_stream(current)
            except BaseException:
                self.failed = self.pool.failed = True
                raise
            try:
                with torch.cuda.stream(capture_stream):
                    for _ in range(2):
                        warm = function(self.input)
                        if (not isinstance(warm, torch.Tensor) or warm.device != example.device
                                or warm.ndim != 2 or warm.shape[0] != example.shape[0]
                                or warm.dtype != example.dtype or warm.numel()*warm.element_size() > 2**20):
                            raise ValueError('Unexpected dense graph output contract')
                    self.output = torch.empty_like(warm, memory_format=torch.contiguous_format)
                    del warm
                capture_stream.synchronize()
                live_before_capture = torch.cuda.memory_allocated()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=capture_stream, pool=self.pool.token):
                    # Destroy the temporary inside capture. Retaining a pool
                    # output would impose cross-graph replay-order constraints.
                    self.output.copy_(function(self.input))
                self.capture_global_tensor_delta_bytes = torch.cuda.memory_allocated()-live_before_capture
                usage = scratch_pool_usage(torch.cuda.memory_snapshot(), self.pool.token)
                self.capture_live_tensor_bytes = usage['active_bytes']
                if self.capture_live_tensor_bytes != 0:
                    raise ValueError('Capture retained live tensor storage in shared scratch: '+repr(usage))
                current.wait_stream(capture_stream)
                self.pool.record(current)
                self.pool.graphs += 1
                self.allocated_growth_bytes = torch.cuda.memory_allocated()-before
                self.reserved_growth_bytes = torch.cuda.memory_reserved()-reserved
            except BaseException:
                self.failed = self.pool.failed = True
                raise

    def __call__(self, x):
        import torch
        if (self.failed or self.pool.failed or torch.is_grad_enabled()
                or not isinstance(x, torch.Tensor)
                or (tuple(x.shape), x.dtype, x.device) != self.signature
                or torch.cuda.is_current_stream_capturing()):
            raise ValueError('Dense graph replay contract changed or graph is poisoned')
        with self.pool.lock:
            if self.failed or self.pool.failed:
                raise ValueError('Dense graph is poisoned')
            try:
                with torch.cuda.device(x.device):
                    stream = torch.cuda.current_stream()
                    self.pool.wait(stream)
                    self.input.copy_(x)
                    self.graph.replay()
                    result = self.output.clone()
                    self.pool.record(stream)
                    return result
            except BaseException:
                self.failed = self.pool.failed = True
                # Do not attempt another CUDA operation after an allocation/launch error.
                raise
