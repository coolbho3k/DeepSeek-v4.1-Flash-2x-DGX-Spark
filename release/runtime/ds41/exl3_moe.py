"""Small correctness-first EXL3 expert dispatcher, including TP-safe slicing.

No full expert matrix is reconstructed at load. The upstream LinearEXL3
implementation may reconstruct one matrix temporarily during larger prefills.
This eager dispatcher is a reference for later fused MoE integration.
"""
import torch
import torch.nn.functional as F
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.ext import exllamav3_ext


def make_linear_exl3(**kwargs):
    layer = LinearEXL3(**kwargs)
    # The proven GB10 native extension predates this optional fused prefill
    # entry point. Its existing reconstruct + Hadamard path remains supported.
    if not hasattr(exllamav3_ext, "reconstruct_had_slice"):
        layer._fused_reconstruct = False
    return layer


def shard_packed(packed, projection, rank=0, world_size=1):
    if not 0 <= rank < world_size or projection not in ("w1", "w2", "w3"):
        raise ValueError("Invalid projection or TP rank")
    if set(packed) != {"trellis", "suh", "svh", "mul1"}:
        raise ValueError("Expected the current MUL1 packed format")
    if packed["trellis"].shape[-1] not in (48, 64):
        raise ValueError("Expected an explicit 3- or 4-bit trellis")
    marker = int(packed["mul1"].item()) & 0xffffffff
    if marker != 0x83DCD12D:
        raise ValueError("MUL1 marker mismatch")
    axis = 0 if projection == "w2" else 1
    scale = "suh" if projection == "w2" else "svh"
    width = packed[scale].numel()
    # Hadamard blocks must not straddle a rank boundary.
    if width % (128 * world_size):
        raise ValueError("TP partition must align to complete 128-channel Hadamard blocks")
    result = dict(packed)
    result["trellis"] = packed["trellis"].chunk(world_size, dim=axis)[rank].contiguous()
    result[scale] = packed[scale].chunk(world_size)[rank].contiguous()
    return result


class PackedExpert:
    def __init__(self, tensors, prefix, rank=0, world_size=1, limit=10.0):
        self.limit = limit
        self.layers = {}
        for name in ("w1", "w3", "w2"):
            stem = prefix + "." + name + "."
            packed = {key.removeprefix(stem): value for key, value in tensors.items() if key.startswith(stem)}
            packed = shard_packed(packed, name, rank, world_size)
            self.layers[name] = make_linear_exl3(config=None, in_features=packed["suh"].numel(),
                out_features=packed["svh"].numel(), out_dtype=torch.float16, key=stem[:-1], **packed)
        if self.layers["w1"].out_features != self.layers["w2"].in_features:
            raise ValueError("Gate/down partition dimensions disagree")

    def __call__(self, x, weights):
        gate = self.layers["w1"].forward(x, {}).float()
        up = self.layers["w3"].forward(x, {}).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        down = (F.silu(gate) * up * weights.float()).half().contiguous()
        return self.layers["w2"].forward(down, {})


def eager_moe(experts, x, route_ids, route_weights, chunk_tokens=1024):
    """Sum local expert contributions; the caller owns distributed reduction."""
    if route_ids.shape != route_weights.shape or len(route_ids) != len(x):
        raise ValueError("Routing/input shape mismatch")
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert_id in torch.unique(route_ids).cpu().tolist():
        expert = experts.get(expert_id)
        if expert is None:
            continue
        rows, slots = torch.where(route_ids == expert_id)
        for token_rows, token_slots in zip(rows.split(chunk_tokens), slots.split(chunk_tokens)):
            values = expert(x[token_rows].half().contiguous(), route_weights[token_rows, token_slots, None])
            output.index_add_(0, token_rows, values.float())
    return output.to(x.dtype)
