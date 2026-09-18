# SPDX-License-Identifier: AGPL-3.0-only
"""Unselected launch sweep of the original packed wo_a arithmetic.

Uses the existing kernel's BF16 reconstruction, FP32 products/reduction and
final BF16 output. No weight repacking, new kernel math or serving hook.
"""
def forward(x,weight,scale,*,splits=8,tile_n=16,tile_k=256):
    import torch
    from spark_packed_wo_a import kernels
    if (x.shape!=(1,4,4096) or x.dtype!=torch.bfloat16 or not x.is_cuda
            or weight.shape!=(4096,4096) or weight.dtype!=torch.float8_e4m3fn
            or scale.shape!=(4096,128) or scale.dtype!=torch.uint8
            or any(t.device!=x.device or not t.is_contiguous() for t in (weight,scale))
            or any(s<=0 for s in x.stride())
            or splits not in (4,8,16) or tile_n not in (8,16,32)
            or tile_k not in (128,256,512)):
        raise ValueError('Expected the bounded one-token native packed projection')
    tr,gemv,_,finish,_=kernels()
    out=torch.empty((1,4,1024),device=x.device,dtype=x.dtype)
    partial=torch.empty((splits,1,4,1024),device=x.device,dtype=torch.float32)
    gemv[(tr.cdiv(1024,tile_n),1,4*splits)](x,weight,scale,partial,
        1,4,1024,4096,*x.stride(),splits,tile_n,tile_k,
        num_warps=4,enable_fp_fusion=False)
    finish[(tr.cdiv(out.numel(),256),)](partial,out,out.numel(),splits,256,num_warps=4)
    return out
