"""Install the reviewed image-input module before vLLM parent/spawn imports.

Read-only process-local replacement. The baked image and model are untouched.
The public overlay carries both this bootstrap and its exact pinned source.
"""
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

BAKED_SHA = '69c4a3187092f31bab8fac5e3a4ada8c867f0b9f44642d4c4f1d639044c5c712'
OVERRIDE_SHA = 'dd2b90570f6f42a70ce3b2b997e0027d98a6baa75889b441ae5c60c6443173aa'
NAME = 'ds41.vllm_vision_inputs'


def checked(path, expected):
    if path.resolve() != path or not path.is_file() or path.stat().st_size > 64*1024:
        raise RuntimeError('Expected a small unredirected image-input source')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Image-prefix bootstrap source differs from its reviewed digest')
    return raw


def install():
    import ds41
    # Optimized serving uses a thin package whose __path__ falls through to
    # the baked package. Resolve the source without importing it (importing
    # it first would bind the old hooks in native plugin modules).
    spec = importlib.machinery.PathFinder.find_spec(NAME, ds41.__path__)
    if spec is None or not isinstance(spec.origin, str):
        raise RuntimeError('Cannot resolve the reviewed baked image-input source')
    checked(Path(spec.origin), BAKED_SHA)
    path = Path(__file__).resolve().with_name('vision_inputs_override.py')
    raw = checked(path, OVERRIDE_SHA)
    current = sys.modules.get(NAME)
    if current is not None:
        if (getattr(current, '__file__', None) != str(path)
                or getattr(current, '__ds41_image_prefix_sha256__', None) != OVERRIDE_SHA
                or getattr(ds41, 'vllm_vision_inputs', None) is not current):
            raise RuntimeError('Image-prefix bootstrap must run before the baked module is imported')
        return current
    if hasattr(ds41, 'vllm_vision_inputs'):
        raise RuntimeError('Inconsistent preexisting image-input module binding')
    spec = importlib.util.spec_from_file_location(NAME, path)
    if spec is None:
        raise RuntimeError('Unable to construct the reviewed image module')
    module = importlib.util.module_from_spec(spec)
    # Compile the already verified bytes; never load a stale/unreviewed pyc.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    module.__ds41_image_prefix_sha256__ = OVERRIDE_SHA
    sys.modules[NAME] = module
    ds41.vllm_vision_inputs = module
    return module
