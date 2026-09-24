# SPDX-License-Identifier: AGPL-3.0-only
"""Isolated register-dequantization pipeline; no fused/persistent experiment."""
import hashlib
from pathlib import Path
from fused_gateup import WRAPPER_SHA,KERNEL_SHA,once


def transform(wrapper,kernel):
    if hashlib.sha256(wrapper).hexdigest()!=WRAPPER_SHA or hashlib.sha256(kernel).hexdigest()!=KERNEL_SHA:
        raise ValueError('Changed qualified cooperative parent')
    w=wrapper.decode();k=kernel.decode()
    start=k.index('    for (int ib = 0; ib < myn; ib += PF)')
    end=k.index('    // Cross-warp reduction over the k splits.',start)
    original=k[start:end]
    replacement='    if constexpr (REG && bits == 3)\n    {\n#include "dequant_pipeline.cuh"\n    }\n    else\n    {\n'+original+'    }\n\n'
    k=k[:start]+replacement+k[end:]
    w=once(w,'extern "C" int goal50_coop_abi() { return 2; }',
        'extern "C" int goal50_coop_abi() { return 2; }\nextern "C" int goal50_coop_experiment() { return 301; }')
    for old,new in (('exl3_moe_coop_a_kernel','exl3_moe_coop_dq_a_kernel'),('exl3_moe_coop_b_kernel','exl3_moe_coop_dq_b_kernel')):
        w=w.replace(old,new);k=k.replace(old,new)
    return w.encode(),k.encode(),Path(__file__).with_suffix('.cuh').read_bytes()
