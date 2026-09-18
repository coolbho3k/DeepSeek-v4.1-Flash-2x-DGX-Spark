# SPDX-License-Identifier: AGPL-3.0-only
"""Bind staged experts; preserve the native large-batch forward.

The route metadata aliases24 FP16 cells in an otherwise unused scratch row
for one-token execution. No new allocation, scratch owner or capture policy.
Every staged invocation produces metadata before consuming it. Original
multi-row kernels may overwrite this row freely on later invocations.
"""


def wrap_module(module):
    existing=getattr(module,'_ds41_staged_decode_binding',None)
    if existing is not None:
        if module.forward is not existing['selected'] or module.resources is not existing['resources']:
            raise RuntimeError('Staged expert bindings changed after startup')
        return module
    if not hasattr(module,'forward_staged1') or not hasattr(module,'staged1_resources'):
        raise RuntimeError('Paired-qualified staged native API required')
    original_forward,original_resources=module.forward,module.resources
    checked=None
    small_checked=None
    grouped_checked=None
    # A newly built additive binary is the explicit opt-in. Frozen older
    # runtimes still expose only staged1 and keep their exact old selection.
    small_enabled=hasattr(module,'forward_staged_small')
    grouped_enabled=hasattr(module,'forward_staged_grouped_small')
    if grouped_enabled and not small_enabled:
        raise RuntimeError('Grouped experts require the original small-batch baseline')

    def resources():
        nonlocal checked,small_checked,grouped_checked
        result=original_resources() # Keep all original device/residency checks.
        if checked is None:
            raw=module.staged1_resources()
            if len(raw)!=56:
                raise RuntimeError('Unexpected staged resource schema')
            rows=[raw[i:i+7] for i in range(0,len(raw),7)]
            selected=[r for r in rows if r[0]==2]
            if (len(selected)!=2 or {r[1] for r in selected}!={0,1}
                    or any(r[2]!=256 or r[3]!=2048 or r[4]<3 or r[6]>8 for r in selected)):
                raise RuntimeError('Staged register/shared-memory/stack contract changed')
            checked=rows
        if small_enabled and small_checked is None:
            raw=module.staged_small_resources()
            if len(raw)!=56:
                raise RuntimeError('Unexpected small-batch staged resource schema')
            rows=[raw[i:i+7] for i in range(0,len(raw),7)]
            selected=[r for r in rows if r[0]==2]
            if (len(selected)!=2 or {r[1] for r in selected}!={0,1}
                    or any(r[2]!=256 or r[3]!=2048 or r[4]<2 or r[6]>8 for r in selected)):
                raise RuntimeError('Small-batch staged resource contract changed')
            small_checked=rows
        if grouped_enabled and grouped_checked is None:
            raw=module.staged_grouped_small_resources()
            if len(raw)!=56:
                raise RuntimeError('Unexpected grouped staged resource schema')
            rows=[raw[i:i+7] for i in range(0,len(raw),7)]
            selected=[r for r in rows if r[0]==2]
            if (len(selected)!=2 or {r[1] for r in selected}!={0,1}
                    or any(r[2]!=256 or r[3]!=8192 or r[4]<2 or r[6]>8 for r in selected)):
                raise RuntimeError('Grouped staged resource contract changed')
            grouped_checked=rows
        return result

    def selected(x,out,counts,tokens,weights,ptrs,temps,locks):
        if x.ndim!=2 or (x.shape[0]!=1 and not (small_enabled and 2<=x.shape[0]<=4)):
            return original_forward(x,out,counts,tokens,weights,ptrs,temps,locks)
        import torch
        if checked is None:
            raise RuntimeError('Prewarm staged resources through the original workspace before capture')
        if (locks.dtype!=torch.int32 or locks.device!=x.device
                or locks.ndim!=1 or locks.numel()!=1050690 or not locks.is_contiguous()
                or len(temps)!=4 or temps[0].shape!=(6,128,5120)):
            raise ValueError('Original staged workspace/lock contract required')
        if x.shape[0]!=1:
            if small_checked is None:
                raise RuntimeError('Prewarm small-batch resources before capture')
            # First24 scratch rows are data; final row is unused by every
            # small-batch kernel. Original large-batch kernels may reuse it.
            meta=temps[0][-1,-1,:24*x.shape[0]].view(torch.int64)
            if grouped_enabled:
                if grouped_checked is None:
                    raise RuntimeError('Prewarm grouped resources before capture')
                return module.forward_staged_grouped_small(x,out,counts,tokens,weights,ptrs,temps,meta,2,2)
            return module.forward_staged_small(x,out,counts,tokens,weights,ptrs,temps,meta,2,2)
        meta=temps[0][0,1,:24].view(torch.int64)
        return module.forward_staged1(x,out,counts,tokens,weights,ptrs,temps,meta,2,2)

    module.forward=selected
    module.resources=resources
    module._ds41_staged_decode_binding=dict(selected=selected,resources=resources,
        original_forward=original_forward,original_resources=original_resources,
        variant=(2,2),extra_gpu_allocation_bytes=0,scratch_alias_byte_offset=10240,
        grouped_small_rows=(2,3,4) if grouped_enabled else (),
        small_rows=(2,3,4) if small_enabled else ())
    return module
