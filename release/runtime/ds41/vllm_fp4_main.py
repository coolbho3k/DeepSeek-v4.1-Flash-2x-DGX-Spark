"""Opt-in FP4 main KV, installed atomically with the existing DCP2 hooks.

This thin overlay leaves the baked plugin/runtime paths intact. Enable before
their first registration with DS41_ENABLE_FP4_MAIN_KV=1. SWA stays FP8;
DS41_ENABLE_FP4_INDEXER=1 additionally selects the coordinated MXFP4 indexer.
No weights, CUDA buffers or groups are created.
"""
from dataclasses import replace
import hashlib
import importlib
import os
from pathlib import Path
import threading

import torch

from . import fp4_main_kv as codec
from .fp4_rope_store import rope_quant_insert

FORMAT = 'ds41_fp4_e2m1_e4m3_g16'
UPSTREAM = {
    'vllm.models.deepseek_v4_1.attention':
        'ef13a8503b54172a63cca6932e2ee5a6d5d6ced81445949067f01b3f61ab6e5e',
    'vllm.models.deepseek_v4_1.compressor':
        'ed66374bb9e4c8c0201383d3fef811e061fd14f3cb0cb9db74bcb740f5a59009',
    'vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache':
        'cc486feefe40871f010c38ee97a2217ca70d86b7f42cdfc5884399045d5ac385',
}
_lock = threading.Lock()
_installed = ()
_forward = None
_installed_collective_chunk = None
_installed_indexer_mode = None


def _collective_chunk_size():
    """Opt-in communication batching; inner attention/gather stays at32 tokens."""
    value = os.environ.get('DS41_FP4_DCP_COLLECTIVE_CHUNK', '32')
    if value not in ('32', '64'):
        raise ValueError('DS41_FP4_DCP_COLLECTIVE_CHUNK must be exactly32 or64')
    return int(value)


def _collective_chunk_replacements():
    # Both choices are multiples of the unchanged32-token inner attention
    # batch. Increase only exchanged Q/partial-output slabs, not KV expansion.
    return [
        ('for start in range(0, count, 32):',
         'for start in range(0, count, _ds41_collective_chunk):'),
        ('end = min(start + 32, count)',
         'end = min(start + _ds41_collective_chunk, count)'),
    ]


def _main_pages(cache):
    if cache.ndim == 4 and cache.shape[-2] == 1:
        cache = cache.squeeze(-2)
    codec._layout(cache)
    return cache


def _mixed_gather(cache, slots):
    if cache.shape[-1] == codec.STATE_BYTES:
        return codec.gather(cache, slots)
    from .dcp_attention import gather_packed_cache
    return gather_packed_cache(cache, slots)


def _insert(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
            compress_ratio, fp8_scale=None):
    # Slots/positions come from the coordinated native allocator/metadata,
    # not untrusted user tensors. Retain all structural and kernel bounds.
    return rope_quant_insert(latent, positions, cos_sin_cache, kv_cache,
        slot_mapping, compress_ratio, fp8_scale, check_bounds=False)


def register():
    global _installed, _forward, _installed_collective_chunk, _installed_indexer_mode
    collective_chunk = _collective_chunk_size()
    indexer_mode = os.environ.get('DS41_ENABLE_FP4_INDEXER', '0')
    if indexer_mode not in ('0', '1'):
        raise ValueError('DS41_ENABLE_FP4_INDEXER must be exactly0 or1')
    setting = os.environ.get('DS41_ENABLE_FP4_MAIN_KV', '0')
    if setting not in ('0', '1'):
        raise ValueError('DS41_ENABLE_FP4_MAIN_KV must be exactly0 or1')
    if setting == '0':
        if _installed:
            raise RuntimeError('FP4 mode cannot change after startup')
        if collective_chunk != 32:
            raise ValueError('Larger collective batches require the FP4 main-cache variant')
        if indexer_mode != '0':
            raise ValueError('FP4 indexer requires FP4 main-cache registration')
        return
    if os.environ.get('DS41_ENABLE_DCP2') != '1':
        raise ValueError('FP4 main KV requires the coordinated DCP2 runtime')
    from . import vllm_dcp as arithmetic, vllm_dcp_runtime as runtime
    from . import vllm_prefill_workspace as workspace
    with _lock:
        if _installed:
            if indexer_mode != _installed_indexer_mode:
                raise RuntimeError('FP4 indexer format cannot change after startup')
            if collective_chunk != _installed_collective_chunk:
                raise RuntimeError('FP4 collective batching cannot change after startup')
            if (arithmetic.attention_forward is not _forward
                    or any(getattr(owner, name) is not function for owner,name,function in _installed)):
                raise RuntimeError('Registered FP4 main-cache hooks changed')
            runtime.register()
            return
        if runtime._installed:
            raise RuntimeError('Enable FP4 before the first DCP registration; no hot conversion')
        modules = {name:importlib.import_module(name) for name in UPSTREAM}
        for name,module in modules.items():
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM[name]:
                raise RuntimeError('Unreviewed FP4 native target: '+name)
        attention, compressor, native_writer = modules.values()
        if compressor.rope_quant_insert is not native_writer.rope_quant_insert:
            raise RuntimeError('Native main-cache writer already changed')
        from vllm.v1.kv_cache_interface import MLAAttentionSpec, KVQuantMode
        original_spec = attention.DeepseekV4Attention.get_kv_cache_spec

        def main_spec(self, config):
            runtime.validate_config(config)
            spec = original_spec(self, config)
            if spec is None:
                return None
            if (type(spec) is not MLAAttentionSpec or spec.head_size != 512
                    or spec.dtype != torch.uint8 or spec.cache_dtype_str != 'fp8_ds_mla'
                    or spec.state_content_size_bytes != 584 or spec.tokens_per_state not in (1,2)):
                raise ValueError('FP4 requires the pinned 512-channel packed main cache')
            return replace(spec, cache_dtype_str=FORMAT, state_content_bytes=288,
                alignment=512, page_size_padded=None, kv_quant_mode=KVQuantMode.NONE)

        mixed_attention = arithmetic._compile(arithmetic.bf16_sparse_attention_with_lse,
            [], {'gather_packed_cache':_mixed_gather})
        old_forward = arithmetic.attention_forward
        forward = arithmetic._compile(old_forward, [
            ('compressed = None if swa_only else _packed_pages(self_kv_cache)',
             'compressed = None if swa_only else _fp4_main_pages(self_kv_cache)'),
            *_collective_chunk_replacements(),
        ], {'_fp4_main_pages':_main_pages,'bf16_sparse_attention_with_lse':mixed_attention,
            '_ds41_collective_chunk':collective_chunk})
        workspace.register()
        arithmetic.attention_forward = forward
        try:
            dcp_hooks = runtime.prepare_hooks()
            hooks = (*dcp_hooks,
                (attention.DeepseekV4Attention,'get_kv_cache_spec',main_spec),
                (compressor,'rope_quant_insert',_insert))
            expected = 15 if indexer_mode == '1' else 13
            if len(hooks) != expected or len({(id(o),n) for o,n,_ in hooks}) != expected:
                raise RuntimeError('Incomplete coordinated FP4/DCP hook set')
            runtime.install_hooks(hooks)
        except Exception:
            arithmetic.attention_forward = old_forward
            raise
        # Runtime owns the DCP targets (plus two optional FP4 selectors);
        # this module also owns the main-cache spec and writer replacements.
        runtime._installed = dcp_hooks
        _installed, _forward = hooks, forward
        _installed_collective_chunk = collective_chunk
        _installed_indexer_mode = indexer_mode
