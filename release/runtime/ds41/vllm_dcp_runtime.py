"""Opt-in coordinated DCP2 startup for pinned V4.1, not a serving qualification.

The source plugin calls register only with DS41_ENABLE_DCP2=1. Cache ownership,
metadata/indexing and attention are installed together after workspace setup.
No CUDA allocation, process-group creation or memory-limit changes occur here.
"""
import threading

from . import vllm_dcp as arithmetic
from . import vllm_dcp_cache as cache
from . import vllm_prefill_workspace as workspace

_lock = threading.Lock()
_installed = ()


def validate_config(config):
    workspace.row_bound(config)
    arithmetic.validate_config(config)


def prepare_hooks():
    if workspace._original is None:
        raise RuntimeError('Register the bounded workspace before DCP startup')
    workspace.register()
    hooks = arithmetic.make_probe_patches() + cache.make_cache_patches()
    for _, _, replacement in hooks:
        namespace = getattr(replacement, '__globals__', {})
        if '_ds41_validate' in namespace:
            namespace['_ds41_validate'] = validate_config
    expected = 13 if arithmetic._FP4_INDEXER_MODE == '1' else 11
    if len(hooks) != expected or len({(id(owner), name) for owner, name, _ in hooks}) != len(hooks):
        raise RuntimeError('Incomplete or overlapping coordinated DCP hooks')
    return tuple(hooks)


def install_hooks(hooks):
    """Restore earlier targets if installing a later target raises."""
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in hooks]
    applied = []
    try:
        for (owner, name, replacement), (_, _, original) in zip(hooks, originals):
            setattr(owner, name, replacement)
            applied.append((owner, name, original))
    except Exception:
        for owner, name, original in reversed(applied):
            setattr(owner, name, original)
        raise


def register():
    global _installed
    with _lock:
        if _installed:
            workspace.register()
            if any(getattr(owner, name) is not replacement for owner, name, replacement in _installed):
                raise RuntimeError('A registered coordinated DCP hook changed; refusing partial repair')
            return
        hooks = prepare_hooks()
        install_hooks(hooks)
        _installed = hooks
