"""Pinned V4.1 integration, enabled explicitly by DS41_SSD_ENGRAM_SOURCE.

This first implementation intentionally requires eager execution. It retains
the upstream hasher/request-state machinery and replaces only table storage
and lookup. Model load must remain lazy and exclude table tensors entirely.
"""
import hashlib
import json
import os
from pathlib import Path

from .ssd_embedding import SSDHeadEmbedding, nonengram_weights
from .ssd_rows import EngramRows

UPSTREAM = {
    "engram": "ac787367615f7e8f25511ea81cb4286c9a3211fe8d27af08a502bf54ea1244c3",
    "default_loader": "9c9d54b1b650bf5affc924ebb8b8c73711e187ae7486cadd9f737bb1279cc7b8",
}
_registered = False
_dcp_mode = None


def register():
    global _registered, _dcp_mode
    source_env = os.environ.get("DS41_SSD_ENGRAM_SOURCE")
    if not source_env:
        return
    setting = os.environ.get('DS41_ENABLE_DCP2', '0')
    if setting not in ('0', '1'):
        raise ValueError('DS41_ENABLE_DCP2 must be exactly0 or1')
    dcp_enabled = setting == '1'
    if _registered:
        if dcp_enabled != _dcp_mode:
            raise RuntimeError('DCP startup mode changed after plugin registration; use a fresh process')
        if dcp_enabled:
            from .vllm_dcp_runtime import register as register_dcp
            register_dcp()
        return
    from vllm.config import get_current_vllm_config
    from vllm.distributed import (
        get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.model_executor.model_loader import default_loader
    from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
    from vllm.models.deepseek_v4_1.common import engram

    for name, module in (("engram", engram), ("default_loader", default_loader)):
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != UPSTREAM[name]:
            raise RuntimeError(f"SSD integration has not been reviewed for this {name} revision: {actual}")
    # Registration is separate from selecting the backend; the checkpoint must
    # explicitly request ds41_exl3 with its measured integer layer allocation.
    from . import vllm_exl3  # noqa: F401
    source = Path(source_env).resolve()
    metadata = json.loads((source / "config.json").read_text())
    config = metadata.get("text_config", metadata)
    layer_for_rows = dict(zip(config["engram_num_embeddings"], config["engram_layer_ids"]))
    if len(layer_for_rows) != len(config["engram_layer_ids"]):
        raise ValueError("Ambiguous source engram table sizes")
    cache_bytes = int(os.environ.get("DS41_ENGRAM_CACHE_MIB", "64")) * 1024**2
    if not 0 <= cache_bytes <= 1024**3:
        raise ValueError("DS41_ENGRAM_CACHE_MIB must be between 0 and 1024")

    def validate_config():
        current = get_current_vllm_config()
        if not current.model_config.enforce_eager:
            raise ValueError("SSD engrams currently require --enforce-eager")
        if current.model_config.hf_config.model_type != "deepseek_v41":
            raise ValueError("DS41 SSD integration only supports the pinned DeepSeek V4.1 architecture")
        load = current.load_config
        if load.load_format not in ("auto", "hf", "safetensors"):
            raise ValueError("SSD engrams require the default lazy safetensors loader")
        if load.safetensors_load_strategy not in (None, "lazy") or load.model_loader_extra_config.get("enable_multithread_load"):
            raise ValueError("Eager/prefetch/multithread loading would materialize SSD engrams")
        from .vllm_vision_inputs import validate_config as validate_vision_config
        validate_vision_config(current)
        from .vllm_prefill_workspace import row_bound
        row_bound(current)
        if dcp_enabled:
            from .vllm_dcp_runtime import validate_config as validate_dcp_config
            validate_dcp_config(current)

    class VllmSSDHeadEmbedding(SSDHeadEmbedding):
        def __init__(self, num_embeddings, dim, head_sizes, block_size=32, cpu_offload=False):
            validate_config()
            if block_size != 32 or dim != config["engram_head_dim"] or num_embeddings not in layer_for_rows:
                raise ValueError("Runtime engram shape does not match SSD source")
            reader = EngramRows(source, layer_for_rows[num_embeddings], cache_bytes)
            try:
                if reader.weight.rows != num_embeddings or reader.weight.row_bytes != dim:
                    raise ValueError("SSD tensor shape mismatch")
                super().__init__(num_embeddings, dim, head_sizes, get_tensor_model_parallel_world_size(),
                                 get_tensor_model_parallel_rank(), reader, tensor_model_parallel_all_gather)
            except Exception:
                reader.close()
                raise

    def iterator(files, use_tqdm_on_load, safetensors_load_strategy=None, local_expert_ids=None, **kwargs):
        validate_config()
        if safetensors_load_strategy not in (None, "lazy"):
            raise ValueError("Only lazy SSD-safe loading is supported")
        yield from nonengram_weights(files, lambda name: should_skip_weight(name, local_expert_ids))

    original_iterator = default_loader.DefaultModelLoader._get_weights_iterator

    def guarded_iterator(loader, source_spec):
        validate_config()
        return original_iterator(loader, source_spec)

    # The runtime pin predates the checkpoint's named reasoning-budget values.
    # Install only after all source/storage checks, before changing live hooks.
    from .vllm_prompt import register as register_prompt
    from .vllm_vision_inputs import register as register_vision_inputs
    from .vllm_prefill_workspace import register as register_prefill_workspace
    register_prompt()
    register_vision_inputs()
    # Install both workspace consumers before any DCP adapter copies globals.
    register_prefill_workspace()
    if dcp_enabled:
        from .vllm_dcp_runtime import register as register_dcp
        register_dcp()
    engram.ParallelEngramEmbedding = VllmSSDHeadEmbedding
    default_loader.safetensors_weights_iterator = iterator
    default_loader.DefaultModelLoader._get_weights_iterator = guarded_iterator
    _dcp_mode = dcp_enabled
    _registered = True
