# SPDX-License-Identifier: AGPL-3.0-only
"""Strict draft-only packed tensor names; independent of CUDA/vLLM imports."""
import re

PATTERN=re.compile(r'mtp\.([0-2])\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(trellis|suh|svh|mul1)')


def packed_key(name):
    match=PATTERN.fullmatch(name)
    if not match or not 0<=int(match[2])<128:raise ValueError('Expected an exact draft-only EXL3 tensor: '+name)
    layer,expert,projection,suffix=int(match[1]),int(match[2]),match[3],match[4]
    group='w2' if projection=='w2' else 'w13'
    param=f'model.layers.{layer}.ffn.experts.{group}_{suffix}'
    return dict(layer=layer,expert=expert,projection=projection,suffix=suffix,param=param)


def expected_keys(layer,expert):
    if not 0<=layer<3 or not 0<=expert<128:raise ValueError('Draft only')
    return {f'mtp.{layer}.ffn.experts.{expert}.{projection}.{suffix}'
        for projection in ('w1','w2','w3') for suffix in ('trellis','suh','svh','mul1')}


def full_shape(projection,suffix):
    # Full unsharded dimensions; serving's existing packed loader splits TP2.
    h,i=5120,2304
    output,input_=(h,i) if projection=='w2' else (i,h)
    if suffix=='trellis':return [input_//16,output//16,48],'I16'
    if suffix=='suh':return [input_],'F16'
    if suffix=='svh':return [output],'F16'
    if suffix=='mul1':return [],'I32'
    raise ValueError('Unsupported packed suffix')
