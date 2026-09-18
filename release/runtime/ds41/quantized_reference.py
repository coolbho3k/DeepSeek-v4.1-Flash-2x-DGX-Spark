"""Layerwise quantized reference blocks with no routed source-weight loading.

Native attention, HC, router, shared expert and SSD engrams are retained. The
routed bank uses the frozen EXL3 kernel and serving-style BF16 accumulation
boundary. Whole-model quality requires a complete, explicitly selected bank
for every layer; a prefix probe is not such a qualification.
"""
from contextlib import contextmanager

import torch
from safetensors.torch import load_file

from scripts import run_quant_queue as baseline
from .exl3_moe import PackedExpert, eager_moe
from .quantized_manifest import ExpertArtifact, LayerPlan, baseline_layer_plan, digest


def fingerprint(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class QuantizedReferenceMoE(torch.nn.Module):
    def __init__(self, layer, args, reference):
        super().__init__()
        if args.get_moe_config(layer) != (384, 6) or args.n_shared_experts != 1:
            raise ValueError('Expected the V4.1 main-model top6/384-expert configuration')
        self.layer, self.dim = layer, args.dim
        self.gate = reference.Gate(layer, args)
        self.shared_experts = reference.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        self._experts = {}
        self._ready = False
        self.inventory_sha256 = None

    def load_bank(self, plan, limit):
        plan.validate()
        if plan.layer != self.layer or self._ready or self._experts:
            raise ValueError('Wrong layer or an already initialized expert bank')
        for artifact in plan.artifacts:
            before = fingerprint(artifact.path)
            if (digest(artifact.path.with_suffix('.json')) != artifact.report_sha256
                    or baseline.tensor_inventory(artifact.path, artifact.prefix, 3) != artifact.payload_bytes):
                raise ValueError('Selected expert report or3-bit tensor inventory changed')
            if digest(artifact.path) != artifact.sha256:
                raise ValueError(f'Packed expert checksum changed: {artifact.prefix}')
            tensors = load_file(artifact.path, device='cuda')
            if fingerprint(artifact.path) != before:
                raise ValueError('Packed artifact changed while being read')
            self._experts[artifact.expert] = PackedExpert(tensors, artifact.prefix, limit=limit)
        if set(self._experts) != set(range(384)):
            raise ValueError('Incomplete expert bank')
        self.inventory_sha256 = plan.inventory_sha256
        self._ready = True

    def forward(self, x, image_mask=None):
        if not self._ready or len(self._experts) != 384:
            raise ValueError('Refusing to execute an incomplete quantized expert bank')
        shape = x.shape
        if x.dtype != torch.bfloat16 or shape[-1] != self.dim:
            raise ValueError('Expected native BF16 MoE inputs')
        x = x.reshape(-1, self.dim)
        if not torch.isfinite(x.half()).all():
            raise ValueError('Quantized-model FP16 expert inputs overflowed')
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        routed = eager_moe(self._experts, x, indices, weights)
        # Match the vLLM MoERunner contract already checked by the serving probe:
        # routed accumulation rounds to BF16 before adding BF16 shared output.
        return (routed + self.shared_experts(x)).view(shape)


@contextmanager
def quantized_moe_factory(runtime):
    original = runtime.ref.MoE
    def factory(layer, args):
        return QuantizedReferenceMoE(layer, args, runtime.ref)
    runtime.ref.MoE = factory
    try:
        yield
    finally:
        runtime.ref.MoE = original


def load_quantized_block(runtime, plan):
    plan.validate()
    if runtime.ref.world_size != 1 or runtime.args.n_layers != 40:
        raise ValueError('Quantized reference blocks require the unsharded40-layer model')
    runtime.close_tables()
    with torch.device('cuda'), runtime.ref.set_dtype(torch.bfloat16), runtime._ssd_embedding(), quantized_moe_factory(runtime):
        layout = runtime.ref.EngramLayout.from_args(runtime.args)
        block = runtime.ref.Block(plan.layer, runtime.args, layout)
    prefix = f'layers.{plan.layer}'
    loaded = runtime.load_parameters(block, prefix)
    expected = {key for key in runtime.weights.index if key.startswith(prefix + '.')
                and '.ffn.experts.' not in key and '.engram.embed.' not in key}
    if loaded != expected:
        raise ValueError({'unloaded_native_parameters': sorted(expected - loaded), 'unexpected': sorted(loaded - expected)})
    if not isinstance(block.ffn, QuantizedReferenceMoE):
        raise ValueError('The native block did not adopt the explicit quantized MoE')
    block.ffn.load_bank(plan, runtime.args.swiglu_limit)
    return block
