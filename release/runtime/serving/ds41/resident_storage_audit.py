# SPDX-License-Identifier: AGPL-3.0-only
"""Unselected metadata-only census of explicit model/runtime tensor owners.

No tensor contents, CUDA kernels, synchronization, garbage collection, cache
flush, or allocation-policy changes. A storage gap is unexplained ownership,
NOT evidence that the gap is reclaimable. Install only in a subsequent build.
"""
from collections import Counter
from types import FunctionType, ModuleType

MAX_OBJECTS = 1_000_000
MAX_STORAGES = 250_000
MAX_DEPTH = 40


def category(path):
    lower = path.lower()
    if 'vision' in lower or 'aligner' in lower:
        return 'vision'
    if 'embed_tokens' in lower or 'lm_head' in lower:
        return 'vocabulary'
    if 'experts' in lower or '.banks' in lower:
        return 'experts'
    if 'b12x' in lower or 'wo_a' in lower:
        return 'dense_packed'
    if 'graph' in lower or 'workspace' in lower or 'dispatcher' in lower:
        return 'runtime_workspace'
    return 'other_model'


def census(roots, tensor_type, module_type, *, object_limit=MAX_OBJECTS):
    """Duck-typed metadata walker also testable without importing Torch.

    Traverse instance dictionaries only for explicit recognized owner types.
    Never follow function closures, imports, descriptors, arbitrary iterators,
    or the GC heap. Unknown object types are counted as incomplete coverage.
    """
    if (type(roots) is not dict or not roots or len(roots)>16
            or any(type(k) is not str or len(k)>128 for k in roots)
            or type(object_limit) is not int or not 1<=object_limit<=MAX_OBJECTS):
        raise ValueError('Bounded explicitly named model/runtime roots required')
    stack=[(k,v,0) for k,v in roots.items()]
    seen=set(); storages={}; skipped=Counter(); tensors=logical_bytes=0
    allowed=('ds41.', 'spark_', 'exllamav3.', 'b12x.')
    while stack:
        path,value,depth=stack.pop()
        if isinstance(value,(type(None),bool,int,float,str,bytes,FunctionType,ModuleType,type)):
            continue
        identity=id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen)>object_limit or depth>MAX_DEPTH:
            raise ValueError('Storage census ownership graph exceeds bound')
        if isinstance(value,tensor_type):
            tensors+=1
            logical_bytes+=value.numel()*value.element_size()
            # Storage identity counts backing allocations, not overlapping
            # tensor views or duplicate Parameters as independent memory.
            storage=value.untyped_storage()
            size=storage.nbytes()
            if not size:
                continue
            key=(str(storage.device),storage.data_ptr())
            if key in storages:
                if storages[key]['bytes']!=size:
                    raise ValueError('Storage resized during census')
                storages[key]['tensor_views']+=1
            else:
                if len(storages)>=MAX_STORAGES:
                    raise ValueError('Storage census allocation bound exceeded')
                storages[key]=dict(bytes=size,category=category(path),
                    tensor_views=1,first_owner=path[:240],device=key[0])
            continue
        if type(value) is dict:
            children=((str(k),v) for k,v in value.items() if type(k) in (str,int))
        elif type(value) in (tuple,list,set,frozenset):
            children=((str(i),v) for i,v in enumerate(value))
        elif (isinstance(value,module_type) or type(value).__module__.startswith(allowed)):
            # vars() does not invoke properties or __getattr__ hooks.
            try:
                children=vars(value).items()
            except TypeError:
                skipped[type(value).__module__+'.'+type(value).__name__]+=1
                continue
        else:
            name=type(value).__module__+'.'+type(value).__name__
            if name not in skipped and len(skipped)>=128:
                raise ValueError('Storage census unknown-owner type bound exceeded')
            skipped[name]+=1
            continue
        for key,child in children:
            if len(stack)+len(seen)>=object_limit:
                raise ValueError('Storage census pending-owner bound exceeded')
            stack.append((path+'.'+key[:80],child,depth+1))
    per_device=Counter(); per_category=Counter()
    for row in storages.values():
        per_device[row['device']]+=row['bytes']
        per_category[row['device']+'/'+row['category']]+=row['bytes']
    largest=sorted(storages.values(),key=lambda r:r['bytes'],reverse=True)[:24]
    return dict(status='bounded_explicit_owner_storage_census',objects=len(seen),
        tensors=tensors,logical_tensor_bytes=logical_bytes,unique_storages=len(storages),
        unique_storage_bytes_by_device=dict(per_device),
        unique_storage_bytes_by_category=dict(per_category),largest_storages=largest,
        skipped_owner_types=dict(skipped),complete_global_allocation_census=False,
        tensor_contents_read=False,reclaimable_bytes_established=False)


def audit(roots):
    import torch
    if not torch.cuda.is_initialized() or torch.cuda.is_current_stream_capturing():
        raise ValueError('Audit only the initialized idle worker outside capture')
    result=census(roots,torch.Tensor,torch.nn.Module)
    result['torch_allocated_bytes']=torch.cuda.memory_allocated()
    result['torch_reserved_bytes']=torch.cuda.memory_reserved()
    result['native_cuda_allocation_inventory_complete']=False
    return result
