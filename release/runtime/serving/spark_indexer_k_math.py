# SPDX-License-Identifier: Apache-2.0
"""Bounded native RMSNorm plus fused GPT-J/MXFP4 index-key store.

The pinned fused native writer differs from staged native arithmetic at rare
BF16 rounding boundaries. Keep native RMSNorm, then fuse RoPE, quantization
and the original paged store with explicit CUDA-compatible sine-term FMA.
FP8 calls retain the original kernel. This module never changes cache sizing.
"""
import ast
import hashlib
import importlib
import inspect
import linecache
import os
from pathlib import Path
import threading

import torch

NATIVE_MODULE='vllm.models.deepseek_v4_1.common.ops.indexer_k_store'
NATIVE_SHA='3ffe978236978ca9247a1db144a5c9cb683d794cad7483f002b7f7f822024e64'
KERNEL_SHA='64acf26b3a96dd1a44797bff597f62ebf15c5977d0f305d9ef646f142753dffc'
MAX_ROWS=1056
MAX_TEMPORARY_BYTES=MAX_ROWS*128*2
_installed=None
_lock=threading.Lock()


def build_kernel(module):
    """Rewrite only the pinned arithmetic; preserve native masks and layout."""
    from vllm.triton_utils import triton
    tree=ast.parse(inspect.getsource(module._indexer_k_norm_rope_quant_store_kernel.fn))
    fn=tree.body[0]
    fn.decorator_list=[]
    body=[]
    removed=changed=0
    for statement in fn.body:
        if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name):
            target=statement.targets[0].id
            expression=ast.unparse(statement.value)
            if target in ('rms_w','variance') or (target=='k' and expression==
                    '(k * tl.rsqrt(variance + rms_norm_eps) * rms_w).to(tl.bfloat16)'):
                removed+=1
                continue
            replacement=None
            if target=='new_even' and expression=='even * cos_v - odd * sin_v':
                replacement='tl.fma(-odd, sin_v, even * cos_v)'
            elif target=='new_odd' and expression=='odd * cos_v + even * sin_v':
                replacement='tl.fma(even, sin_v, odd * cos_v)'
            if replacement is not None:
                statement.value=ast.parse(replacement,mode='eval').body
                changed+=1
        body.append(statement)
    if (removed,changed)!=(3,2):
        raise ValueError('Unreviewed indexer normalization/rotary source structure')
    fn.body=body
    ast.fix_missing_locations(tree)
    source=ast.unparse(tree)+'\n'
    if hashlib.sha256(source.encode()).hexdigest()!=KERNEL_SHA:
        raise ValueError('Indexer kernel differs from qualified arithmetic')
    filename=__file__+'::native_rotary_fp4_store'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(module.__dict__)
    exec(compile(source,filename,'exec'),namespace)
    return triton.jit(namespace[fn.name])


class KernelLaunch:
    def __init__(self,original,candidate):
        self.original=original
        self.candidate=candidate

    def __getitem__(self,grid):
        native_launch=self.original[grid]
        candidate_launch=self.candidate[grid]

        def launch(*args,**kwargs):
            if not kwargs.get('USE_FP4',False):
                return native_launch(*args,**kwargs)
            if (not isinstance(grid,tuple) or len(grid)!=1 or type(grid[0]) is not int
                    or not 1<=grid[0]<=MAX_ROWS or len(args)!=10
                    or kwargs.get('HEAD_SIZE')!=128 or kwargs.get('ROPE_HEAD_DIM')!=64
                    or kwargs.get('COMPRESS_RATIO') not in (1,2)
                    or kwargs.get('TOKEN_STRIDE')!=64 or kwargs.get('SCALE_DIM')!=4
                    or kwargs.get('SHUFFLE') is not False
                    or 'enable_fp_fusion' in kwargs):
                raise ValueError('Expected bounded native MXFP4 index-key launch')
            count=grid[0]
            kpre,weight=args[0],args[3]
            if (not kpre.is_cuda or kpre.dtype!=torch.bfloat16 or kpre.ndim!=2
                    or kpre.shape[1]!=128 or kpre.shape[0]<count or kpre.stride(1)!=1
                    or args[1]!=kpre.stride(0) or args[8].numel()!=count
                    or weight.shape!=(128,) or weight.dtype!=torch.bfloat16
                    or weight.device!=kpre.device or not weight.is_contiguous()):
                raise ValueError('Expected bounded native BF16 index-key normalization')
            from vllm import _custom_ops as ops
            normalized=torch.empty((count,128),device=kpre.device,dtype=torch.bfloat16)
            ops.rms_norm(normalized,kpre[:count],weight,args[4])
            return candidate_launch(normalized,128,*args[2:],**kwargs,enable_fp_fusion=False)

        return launch


def register():
    global _installed
    setting=os.environ.get('DS41_ENABLE_INDEXER_K_PARITY','0')
    if setting not in ('0','1'):
        raise ValueError('DS41_ENABLE_INDEXER_K_PARITY must be exactly0 or1')
    if setting=='0':
        if _installed is not None:
            raise RuntimeError('Indexer arithmetic cannot change after registration')
        return
    if any(os.environ.get(key)!='1' for key in
           ('DS41_ENABLE_DCP2','DS41_ENABLE_FP4_MAIN_KV','DS41_ENABLE_FP4_INDEXER')):
        raise ValueError('Indexer parity requires the full-FP4 DCP runtime')
    from vllm.platforms import current_platform
    if tuple(current_platform.get_device_capability() or ())!=(12,1):
        raise ValueError('Private index-key arithmetic requires SM121')
    with _lock:
        module=importlib.import_module(NATIVE_MODULE)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=NATIVE_SHA:
            raise ValueError('Unreviewed native index-key implementation')
        if _installed is not None:
            if module._indexer_k_norm_rope_quant_store_kernel is not _installed:
                raise RuntimeError('Registered index-key arithmetic hook changed')
            return
        original=module._indexer_k_norm_rope_quant_store_kernel
        candidate=build_kernel(module)
        _installed=KernelLaunch(original,candidate)
        module._indexer_k_norm_rope_quant_store_kernel=_installed
