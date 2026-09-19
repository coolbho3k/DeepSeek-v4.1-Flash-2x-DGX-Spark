# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental zero-copy context KV projection for the native DSpark drafter.

Uses B12X's existing packed MXFP8 weights and FP32-reduced native operations.
No dequantization/requantization, scale changes, copied weights, or cache format
changes. The query path retains its full 1792-output fused projection.
Not enabled until component and serving comparison pass.
"""
from dataclasses import replace
from types import SimpleNamespace


def slice_kv_weight(packed):
    """Select the 512 KV outputs after 1280 Q outputs (all 128-row aligned)."""
    weight=packed.weight
    if (packed.in_features!=5120 or packed.padded_in_features!=5120 or packed.out_features!=1792
            or tuple(weight.values.shape)!=(1792,5120)
            or tuple(weight.scale_rows.shape)!=(1,1792,160)
            or tuple(weight.scale_mma.shape)!=(32,4,14,4,40,1)
            or weight.values_tiled is not None):
        raise ValueError('Expected the unchanged 1792x5120 native MXFP8 packed weight')
    values=weight.values.narrow(0,1280,512)
    rows=weight.scale_rows.narrow(1,1280,512)
    mma=weight.scale_mma.narrow(2,10,4)
    for original,view in ((weight.values,values),(weight.scale_rows,rows),(weight.scale_mma,mma)):
        if original.untyped_storage().data_ptr()!=view.untyped_storage().data_ptr():
            raise ValueError('KV projection must alias, never copy, existing weight storage')
    return replace(packed,out_features=512,
        weight=replace(weight,values=values,scale_rows=rows,scale_mma=mma))


class KVProjection:
    def __init__(self,attn):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare KV projection views eagerly before graph capture')
        if attn.q_lora_rank!=1280:raise ValueError('Changed DSpark query rank')
        self.parent=attn.fused_wqa_wkv.b12x_mxfp8_packed_weight
        self.proxy=SimpleNamespace(b12x_mxfp8_packed_weight=slice_kv_weight(self.parent))

    def __call__(self,attn,x):
        import torch
        if (attn.fused_wqa_wkv.b12x_mxfp8_packed_weight is not self.parent
                or x.dtype!=torch.bfloat16 or x.ndim!=2 or x.shape[1]!=5120
                or not x.is_cuda or not x.is_contiguous()):
            raise ValueError('Changed KV-only projection input or packed-weight owner')
        if 1<=x.shape[0]<=4:
            # Same input quantizer and tile as the installed 1792-output
            # fused-input decode path; only fewer output tiles are computed.
            from b12x._lib.dense_gemm import dense_gemm_fused_quant_a
            packed=self.proxy.b12x_mxfp8_packed_weight
            return dense_gemm_fused_quant_a(x,packed.weight.values.reshape(512,5120,1),
                packed.weight.scale_mma,expected_m=len(x),mma_tiler_mn=(16,128))[:,:,0]
        from vllm.model_executor.kernels.linear.mxfp8.b12x import _apply_b12x_mxfp8_packed_linear
        return _apply_b12x_mxfp8_packed_linear(self.proxy,x,None)


def context_kv(attn,x):
    projection=getattr(attn,'_ds41_context_kv_projection',None)
    if projection is None:
        projection=KVProjection(attn)
        attn._ds41_context_kv_projection=projection
    return projection(attn,x)
