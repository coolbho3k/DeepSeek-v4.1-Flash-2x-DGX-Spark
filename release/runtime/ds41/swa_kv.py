# SPDX-License-Identifier: Apache-2.0
"""Selectable group-32/64 FP8 SWA with the original BF16 RoPE tail.

Selection is immutable before model allocation/graph capture. Group 64 uses
vLLM's original fused writer. Group 32 uses the same native kernel adapted
only for two-lane (32-value) scale groups and 16 scale bytes per state.
"""
from dataclasses import replace
import hashlib
import importlib
import os
from pathlib import Path

import torch


def selected_group_size():
    value = os.environ.get('DS41_SWA_KV_GROUP_SIZE', '32')
    if value not in ('32', '64'):
        raise ValueError('DS41_SWA_KV_GROUP_SIZE must be 32 or 64; RoPE stays BF16')
    return int(value)


GROUP_SIZE = selected_group_size()
STATE_BYTES = 576 + 512 // GROUP_SIZE
BINARY_SHA256 = '7f1780ae862047e01b45f50a7788f6c5da2e6bb01c5310fd5772790905eec982'
NATIVE_SWA_SHA256 = 'cc4c5f3d5dd914ce05f491b03b3bd7c45ba532bd7e566e2b12af348768acecbe'
_native = None


def validate_selection():
    if selected_group_size() != GROUP_SIZE:
        raise RuntimeError('SWA KV group size cannot change after import; restart workers')


def load_native():
    global _native
    path = Path(__file__).resolve().parent.parent / 'libds41_swa32.so'
    if hashlib.sha256(path.read_bytes()).hexdigest() != BINARY_SHA256:
        raise RuntimeError('SWA32 native library differs from its qualified source build')
    if _native is None:
        torch.ops.load_library(str(path))
        _native = torch.ops.ds41_swa32.insert
    return _native


def make_hooks(attention):
    """Prepare writer and allocation hooks for the atomic FP4/DCP installer."""
    validate_selection()
    from .vllm_dcp import _compile
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    native = importlib.import_module('vllm.v1.attention.backends.mla.sparse_swa')
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != NATIVE_SWA_SHA256:
        raise RuntimeError('Unreviewed native SWA cache specification')
    original_spec = native.DeepseekV4SWACache.get_kv_cache_spec
    original_writer = attention._fused_qnorm_rope_kv_insert

    def spec(layer, config):
        validate_selection()
        result = original_spec(layer, config)
        if (type(result) is not SlidingWindowMLASpec or result.head_size != 512
                or result.dtype != torch.uint8 or result.cache_dtype_str != 'fp8_ds_mla'
                or result.state_content_size_bytes != 584 or result.alignment != 576):
            raise ValueError('SWA adapter requires the pinned FP8/BF16 native layout')
        # 32-token pages remain 19008 bytes after native 576-byte alignment:
        # the additional 256 scale bytes fit inside the old page's padding.
        return replace(result, state_content_bytes=STATE_BYTES, page_size_padded=None)

    writer = original_writer
    if GROUP_SIZE == 32:
        writer = _compile(original_writer, [
            ('torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(',
             '_ds41_swa32_insert('),
        ], {'_ds41_swa32_insert': load_native()})
    return [(native.DeepseekV4SWACache, 'get_kv_cache_spec', spec),
            (attention, '_fused_qnorm_rope_kv_insert', writer)]


def gather(cache, slots):
    """Bounded eager fallback; selected sparse rows only, no persistent copy."""
    if (cache.ndim != 3 or cache.dtype != torch.uint8 or cache.shape[-1] != 592):
        raise ValueError('Expected 592-byte group-32 SWA pages')
    if (slots >= cache.shape[0] * cache.shape[1]).any().item():
        raise ValueError('Sparse SWA slot exceeds allocated capacity')
    from .dcp_cache_gather import gather as decode
    return decode(cache, slots)
