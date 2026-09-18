# SPDX-License-Identifier: AGPL-3.0-only
"""Unselected one-row MXFP8 GEMV, using the native activation quantizer.

Reads the original contiguous FP8 values and compact UE8M0 scales directly.
No converted weight copy, FP16 accumulator, or atomic output. Numerical and
GPU performance qualification is required before any serving registration.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _ue8(value):
    bits = value.to(tl.int32) << 23
    normal = bits.to(tl.float32, bitcast=True)
    return tl.where(value == 0, 2.0**-127,
                    tl.where(value == 255, float('nan'), normal))


@tr.jit
def _gemv(X, XS, W, WS, Y, N:tl.constexpr, K:tl.constexpr,
          BN:tl.constexpr, BK:tl.constexpr, SPLITS:tl.constexpr):
    rows = tl.program_id(0)*BN+tl.arange(0,BN)
    split = tl.program_id(1)
    cols = tl.arange(0,BK)
    span = tr.cdiv(K, SPLITS*BK)*BK
    accum = tl.full((BN,BK),0.,tl.float32)
    for offset in range(split*span,(split+1)*span,BK):
        k = offset+cols
        xv = tl.load(X+k,k<K,other=0.).to(tl.float32)
        xs = _ue8(tl.load(XS+k//32,k<K,other=127))
        wv = tl.load(W+rows[:,None]*K+k[None,:],
            (rows[:,None]<N)&(k[None,:]<K),other=0.).to(tl.float32)
        ws = _ue8(tl.load(WS+rows[:,None]*(K//32)+k[None,:]//32,
            (rows[:,None]<N)&(k[None,:]<K),other=127))
        accum += (wv*ws)*(xv*xs)[None,:]
    value = tl.sum(accum,1)
    tl.store(Y+split*N+rows,value,rows<N)


@tr.jit
def _sum_parts(P,Y,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B)
    part=tl.arange(0,S)
    values=tl.load(P+part[:,None]*N+row[None,:],row[None,:]<N,other=0.)
    tl.store(Y+row,tl.sum(values,0),row<N)


def forward(x,packed,*,tile_rows=8,tile_k=256,splits=1):
    from b12x.gemm._shared.block_fp8 import quantize_block_fp8_linear_input_mxfp8
    n,k=int(packed.out_features),int(packed.in_features)
    if (x.shape!=(1,k) or x.dtype!=torch.bfloat16 or not x.is_cuda
            or not x.is_contiguous() or packed.padded_in_features!=k or k%128
            or tile_rows not in (4,8,16) or tile_k not in (128,256,512)
            or splits not in (1,2,4)):
        raise ValueError('Unqualified one-token MXFP8 GEMV layout')
    w=packed.weight.values
    ws=packed.weight.scale_rows
    if (w.shape!=(n,k) or w.dtype!=torch.float8_e4m3fn or not w.is_contiguous()
            or ws.numel()!=n*(k//32) or not ws.is_contiguous()
            or w.device!=x.device or ws.device!=x.device):
        raise ValueError('Original native contiguous MXFP8 weights required')
    q=quantize_block_fp8_linear_input_mxfp8(x)
    output=torch.empty((1,n),device=x.device,dtype=x.dtype)
    partial=output if splits==1 else torch.empty((splits,n),device=x.device,dtype=torch.float32)
    _gemv[(tr.cdiv(n,tile_rows),splits)](q.values,q.scale_rows.view(torch.uint8),
        w,ws.view(torch.uint8),partial,n,k,tile_rows,tile_k,splits,
        num_warps=4,enable_fp_fusion=False)
    if splits>1:
        _sum_parts[(tr.cdiv(n,256),)](partial,output,n,splits,256,num_warps=4)
    return output
