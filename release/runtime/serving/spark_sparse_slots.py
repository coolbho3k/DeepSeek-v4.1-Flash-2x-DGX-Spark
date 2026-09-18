"""Bind the exact GPU-qualified mapper before loading, with no tensor allocation."""
import hashlib
import os
from pathlib import Path

SOURCE_SHA='acbd5dce12e3a988697268c946f7c1a178cc38a3dc738dbb5a94287b7cc43edb'
KEY='sparse_global_to_local_slots'
DESCRIPTOR=dict(implementation='stable_fused_dcp2_sparse_slots_v1',kernel_sha256=SOURCE_SHA,
    stable_candidate_order=True,duplicate_candidates_preserved=True,
    synchronous_bounds_checks=True,image_key_membership_unchanged=True,
    maximum_rows=64,maximum_width=8192,persistent_gpu_workspace_bytes=0)
_installed=None


def register():
    global _installed
    mode=os.environ.get('DS41_ENABLE_FUSED_SPARSE_SLOTS','0')
    if mode not in ('0','1'):
        raise ValueError('Fused sparse mapping requires an explicit0/1 setting')
    if mode=='0':
        if _installed is not None:
            raise RuntimeError('Sparse mapping mode cannot change after startup')
        return
    from ds41 import vllm_fp4_main as fp4, vllm_dcp as arithmetic
    from ds41 import dcp_metadata as eager, dcp_sparse_slots as fused
    path=Path(fused.__file__)
    if path.resolve()!=path or path.stat().st_size>64*1024 or hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError('Sparse mapper source differs from exact GPU qualification')
    if (not fp4._installed or fp4._forward is None or fp4._installed_attention_mode!='1'
            or arithmetic.attention_forward is not fp4._forward):
        raise RuntimeError('Fused sparse mapping requires the already-installed FP4 attention path')
    fp4.register()
    forward=fp4._forward
    candidate=fused.sparse_global_to_local_slots
    if _installed is not None:
        if (_installed['forward'] is not forward or _installed['candidate'] is not candidate
                or forward.__globals__.get(KEY) is not candidate):
            raise RuntimeError('Installed native sparse mapper binding changed')
        return
    if forward.__globals__.get(KEY) is not eager.sparse_global_to_local_slots:
        raise RuntimeError('Unexpected original native sparse mapper binding')
    from ds41.dcp_overlap.integration import bind_sparse_mapper
    bind_sparse_mapper(forward, candidate)
    _installed=dict(forward=forward,candidate=candidate)
