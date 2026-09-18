# SPDX-License-Identifier: AGPL-3.0-only
"""Return unused CPU tokenizer heap after large prompts, never live tokens.

The native renderer already serializes tokenization on one executor. This
adds no queue, CUDA operation, token transformation, or per-decode cleanup.
"""
import ctypes
import functools
import hashlib
import inspect
import json
from pathlib import Path

NATIVE_SHA256 = '9f4e1a860c836fe9383385fa8188e5d3ea499ad4300d26091a1526dc28d3654e'
MIN_TOKENS = 65536
_installed = None


def trim_unused():
    trim = ctypes.CDLL('libc.so.6').malloc_trim
    trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
    return int(trim(0))


def wrap(original):
    @functools.wraps(original)
    def tokenize(self, prompt, params):
        result = original(self, prompt, params)
        count = len(result['prompt_token_ids'])
        if count >= MIN_TOKENS:
            value = trim_unused()
            print(json.dumps(dict(stage='ds41_large_prompt_cpu_heap_trim',
                prompt_tokens=count, malloc_trim_result=value,
                tokens_modified=False, cuda_calls=False)), flush=True)
        return result
    return tokenize


def register():
    global _installed
    from vllm.renderers import base
    owner = base.BaseRenderer
    if _installed is not None:
        if owner._tokenize_prompt is not _installed:
            raise RuntimeError('Tokenizer cleanup binding changed')
        return
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != NATIVE_SHA256:
        raise RuntimeError('Unreviewed native tokenizer cleanup source')
    original = owner._tokenize_prompt
    if (tuple(inspect.signature(original).parameters) != ('self', 'prompt', 'params')
            or original.__module__ != base.__name__):
        raise RuntimeError('Unexpected native tokenizer signature/binding')
    _installed = wrap(original)
    owner._tokenize_prompt = _installed
