"""Explicit layer-mixed EXL3 routed experts with native V4.1 dense quantization.

Initial runtime path: eager, tensor parallel, no EPLB or expert parallel.
Each layer has an integer K. Ten 4-bit and thirty 3-bit layers give exactly
3.25 backbone weight bits before scales/metadata. Allocation is external.
"""
import re
import weakref

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config

from .exl3_moe import PackedExpert, eager_moe


@register_quantization_config("ds41_exl3")
class DS41EXL3Config(DeepseekV4FP8Config):
    def __init__(self, layer_bits, target_bpw, **kwargs):
        super().__init__(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                         weight_block_size=[32, 32], **kwargs)
        if len(layer_bits) != 40 or any(type(bits) is not int or bits not in (3, 4) for bits in layer_bits):
            raise ValueError("layer_bits must explicitly specify 3 or 4 for all 40 backbone layers")
        if target_bpw not in (3.0, 3.25) or sum(layer_bits) / 40 != target_bpw:
            raise ValueError("Layer allocation does not exactly match target_bpw")
        self.layer_bits = list(layer_bits)
        self.target_bpw = target_bpw

    @classmethod
    def get_name(cls):
        return "ds41_exl3"

    @classmethod
    def get_min_capability(cls):
        return 120

    @classmethod
    def from_config(cls, config):
        if config.get("codebook") != "mul1" or config.get("scope") != "deepseek_v41_backbone_routed_experts":
            raise ValueError("Unsupported EXL3 codebook or conversion scope")
        return cls(config["layer_bits"], config["target_bpw"], ignored_layers=config.get("ignored_layers"))

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (hf_quant_cfg or {}).get("quant_method") == "ds41_exl3":
            return "ds41_exl3"
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
            if not match or not 0 <= int(match[1]) < 40:
                raise ValueError(f"No calibrated backbone allocation for {prefix}")
            return DS41EXL3MoEMethod(layer.moe_config, self.layer_bits[int(match[1])])
        # Preserve the V4.1 32x32 MXFP8 dense path (including E8M0 scales).
        # Vision constructs its native BF16 linears without this quant config.
        return super().get_quant_method(layer, prefix)


class DS41EXL3MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, bits):
        super().__init__(moe)
        self.bits = bits
        self.loaded = set()

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                       params_dtype, **extra_weight_attrs):
        current = get_current_vllm_config()
        if not current.model_config.enforce_eager or current.parallel_config.enable_expert_parallel:
            raise ValueError("Initial DS41 EXL3 backend requires eager TP without expert parallel")
        if current.parallel_config.enable_eplb:
            raise ValueError("EXL3 EPLB has not been validated")
        h, i = hidden_size, intermediate_size_per_partition
        if h % 128 or i % 128:
            raise ValueError("EXL3 dimensions must align to 128-channel Hadamard blocks")
        shapes = {"w13_trellis": (num_experts, 2, h // 16, i // 16, self.bits * 16),
                  "w13_suh": (num_experts, 2, h), "w13_svh": (num_experts, 2, i),
                  "w13_mul1": (num_experts, 2, 1),
                  "w2_trellis": (num_experts, i // 16, h // 16, self.bits * 16),
                  "w2_suh": (num_experts, i), "w2_svh": (num_experts, h),
                  "w2_mul1": (num_experts, 1)}
        for name, shape in shapes.items():
            suffix = name.rsplit("_", 1)[1]
            dtype = {"trellis": torch.int16, "suh": torch.float16, "svh": torch.float16, "mul1": torch.int32}[suffix]
            parameter = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            parameter.weight_loader = self._load
            parameter._ds41_owner = weakref.ref(layer)
            parameter._ds41_suffix = suffix
            layer.register_parameter(name, parameter)
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("Dense expert allocation is forbidden for EXL3")
        self.num_experts = num_experts

    def _load(self, param, loaded_weight, weight_name, shard_id="w1", expert_id=0, return_success=False):
        layer = param._ds41_owner()
        local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if local_id < 0:
            return False if return_success else None
        if shard_id not in ("w1", "w3", "w2"):
            raise ValueError(shard_id)
        suffix = param._ds41_suffix
        loaded = loaded_weight.detach()
        rank, size = self.moe.tp_rank, self.moe.tp_size
        axis = None
        if suffix == "trellis":
            if loaded.shape[-1] != self.bits * 16:
                raise ValueError(f"Trellis bitrate disagrees with allocation: {weight_name}")
            axis = 0 if shard_id == "w2" else 1
        elif (shard_id == "w2" and suffix == "suh") or (shard_id != "w2" and suffix == "svh"):
            axis = 0
        if axis is not None:
            if loaded.shape[axis] % size:
                raise ValueError(f"Nonuniform TP slice: {weight_name}")
            loaded = loaded.chunk(size, dim=axis)[rank]
        if suffix == "mul1":
            loaded = loaded.reshape(1)
        dest = param.data[local_id] if shard_id == "w2" else param.data[local_id, int(shard_id == "w3")]
        if dest.shape != loaded.shape or dest.dtype != loaded.dtype:
            raise ValueError(f"Packed load shape/dtype mismatch: {weight_name}: {dest.shape}/{dest.dtype} vs {loaded.shape}/{loaded.dtype}")
        dest.copy_(loaded)
        self.loaded.add((local_id, shard_id, suffix))
        return True if return_success else None

    def process_weights_after_loading(self, layer):
        expected = {(expert, projection, suffix) for expert in range(self.num_experts)
                    for projection in ("w1", "w2", "w3") for suffix in ("trellis", "suh", "svh", "mul1")}
        if self.loaded != expected:
            raise ValueError(f"Missing packed expert tensors: {sorted(expected - self.loaded)[:20]}")
        bank = {}
        for expert in range(self.num_experts):
            tensors = {}
            for projection in ("w1", "w3", "w2"):
                group = "w2" if projection == "w2" else "w13"
                for suffix in ("trellis", "suh", "svh", "mul1"):
                    param = getattr(layer, group + "_" + suffix)
                    value = param[expert] if projection == "w2" else param[expert, int(projection == "w3")]
                    tensors[f"expert.{projection}.{suffix}"] = value
            bank[expert] = PackedExpert(tensors, "expert", limit=float(layer.swiglu_limit or 0))
        layer._ds41_experts = bank

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        # TP-only: local expert IDs equal global IDs, weights are matrix-sharded.
        # The vLLM runner retains routing, shared-expert execution and reduction.
        return eager_moe(layer._ds41_experts, x, topk_ids, topk_weights)
