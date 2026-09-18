# SPDX-License-Identifier: AGPL-3.0-only
"""Select the native fused-input MXFP8 GEMM for six qualified decode shapes.

No weight conversion, altered activation quantizer, precision reduction or
new graph owner. Batches above four, biased and other dense layouts retain the
original native operation. The shared FP32 split-K reducer remains selected.
"""
TILES={(5120,1152):(16,64),(4096,1280):(16,128),
       (16384,1280):(16,64),(5120,4096):(32,64),
       (1792,5120):(16,128),(2304,5120):(16,64)}


def wrap_apply(original):
    if hasattr(original,'_ds41_fused_input_original'):
        raise RuntimeError('Dense fused-input selection already installed')
    def apply(kernel,layer,x,bias=None):
        import torch
        if (bias is not None or x.dtype!=torch.bfloat16 or x.ndim<2
                or not x.is_cuda or not 1<=x.numel()//x.shape[-1]<=4 or not x.is_contiguous()):
            return original(kernel,layer,x,bias)
        packed=layer.b12x_mxfp8_packed_weight
        n,k=int(packed.out_features),int(packed.in_features)
        tile=TILES.get((n,k))
        if tile is None or packed.padded_in_features!=k or x.shape[-1]!=k:
            return original(kernel,layer,x,bias)
        from b12x._lib.dense_gemm import dense_gemm_fused_quant_a
        rows=x.numel()//k
        y=dense_gemm_fused_quant_a(x.reshape(rows,k),
            packed.weight.values.reshape(n,k,1),packed.weight.scale_mma,
            expected_m=rows,mma_tiler_mn=tile)[:,:,0]
        return y.view(*x.shape[:-1],n)
    apply._ds41_fused_input_original=original
    apply._ds41_fused_input_tiles=dict(TILES)
    return apply
