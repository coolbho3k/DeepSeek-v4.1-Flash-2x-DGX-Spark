# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare fused communication before the existing atomic FP4 installation.

The original FP4 source, cache writer, bounds checks, metadata and native
attention kernel are untouched. No live worker may acquire this hook late.
"""
import hashlib
from ds41.dcp_overlap.integration import forward_admitted as _ds41_overlap_forward_admitted
import os
from pathlib import Path

CORE_SHA='7aa4a5e6d978f4be72db26e65619e75c7c09e75a218426afbf4a8aab1d56ce20'
FP4_SHA='4d2783a7182755b7577d9a2f8905ecae9dac8e652835ef6184fbc06039954547'
DESCRIPTOR=dict(implementation='fused_dcp2_communication_v1',license='AGPL-3.0-only',
    kernel_sha256=CORE_SHA,maximum_rows=64,maximum_sparse_width=8192,
    stable_sparse_partition=True,duplicate_entries_preserved=True,
    packed_output_lse_collective=True,per_chunk_collectives=2,
    sink_exchange_unchanged=True,query_exchange_unchanged=True,
    partial_dtype='float32',lse_base=2,original_image_visibility=True,
    synchronous_cache_bounds_checks=True,persistent_gpu_workspace_bytes=0,
    maximum_packed_send_bytes=8404992,full_model_graph_capture_enabled=False)
_installed=None


def register():
    global _installed
    mode=os.environ.get('DS41_ENABLE_DCP_COMMUNICATION','0')
    if mode not in ('0','1'):raise ValueError('DCP communication mode must be exactly0 or1')
    if mode=='0':
        if _installed is not None:raise RuntimeError('DCP communication mode changed after startup')
        return
    from ds41 import vllm_fp4_main as fp4,vllm_dcp as arithmetic
    from ds41 import vllm_dcp_runtime as runtime,dcp_attention as reference,dcp_communication as fused
    for module,pin in ((fp4,FP4_SHA),(fused,CORE_SHA)):
        path=Path(module.__file__)
        if path.resolve()!=path or path.stat().st_size>64*1024 or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:
            raise RuntimeError('Unqualified DCP communication or FP4 source')
    bindings={'partition_indices':fused.partition_indices,
        '_ds41_pack_result':fused.pack_result,'_ds41_merge_packed':fused.merge_packed}
    if _installed is not None:
        if (fp4._collective_chunk_replacements is not _installed['rules']
                or any(_installed['bindings'][k] is not v or arithmetic.__dict__.get(k) is not v
                       for k,v in bindings.items())):
            raise RuntimeError('Prepared DCP communication binding changed')
        if fp4._installed:
            fp4.register()
            forward=fp4._forward
            if (forward is not arithmetic.attention_forward
                    or any(forward.__globals__.get(k) is not v for k,v in bindings.items())
                    or not _ds41_overlap_forward_admitted(forward, 'all_packed = group.all_gather(_ds41_pack_result(partial, lse), dim=0)')
                    or 'all_lses = group.all_gather' in forward.__ds41_patch_source__):
                raise RuntimeError('Native attention did not install fused DCP communication')
        return
    if (fp4._installed or runtime._installed
            or arithmetic.partition_indices is not reference.partition_indices
            or any(k in arithmetic.__dict__ for k in ('_ds41_pack_result','_ds41_merge_packed'))):
        raise RuntimeError('Install DCP communication before atomic FP4 startup, not into a live model')
    original=fp4._collective_chunk_replacements
    baseline=original()
    def rules():
        current=original()
        if current!=baseline:raise RuntimeError('Original DCP collective rules changed')
        return current+fused.forward_replacements()
    # Only pure-Python bindings change here. The existing FP4 installer still
    # compiles/validates every coordinated replacement before installing any.
    fp4._collective_chunk_replacements=rules
    arithmetic.__dict__.update(bindings)
    _installed=dict(rules=rules,original_rules=original,bindings=bindings)
