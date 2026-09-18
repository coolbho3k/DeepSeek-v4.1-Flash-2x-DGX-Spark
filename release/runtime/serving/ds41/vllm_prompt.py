# SPDX-License-Identifier: AGPL-3.0-only
"""Preserve native reasoning budgets and add OpenAI-client effort aliases.

Official df42c109 uses low=50/high=75/max=100, default high. The runtime pin
uses low=25/high=50/xhigh=75/max=100 instead. Preserve the official presets
and add minimal=25/medium=60/xhigh=90. Numeric1..100 and non-thinking controls
retain their native behavior; these are prompt hints, not token limits.
"""
import hashlib
from pathlib import Path
import threading

UPSTREAM = {
    'tokenizer': 'd6e9cbce7c5f127819452fa5d03dd4f22ad9f136dd202a54b5b31fe42402cae2',
    'encoding': '829d14a024c6f88146fa02b2735d1451f0177a0ddd758be82717d9cfecf10b4c',
}
ORIGINAL = {'low': 25, 'high': 50, 'xhigh': 75, 'max': 100}
SOURCE_BUDGETS = {'low': 50, 'high': 75, 'max': 100}
SERVING_BUDGETS = {**SOURCE_BUDGETS, 'minimal': 25, 'medium': 60, 'xhigh': 90}
_registered = False
_lock = threading.Lock()


def register():
    global _registered
    from vllm.tokenizers import deepseek_v41 as tokenizer
    from vllm.tokenizers import deepseek_v41_encoding as encoding

    with _lock:
        for name, module in (('tokenizer', tokenizer), ('encoding', encoding)):
            actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            if actual != UPSTREAM[name]:
                raise RuntimeError(f'Prompt compatibility has not been reviewed for this {name} revision: {actual}')
        mapping = encoding.REASONING_EFFORT_MAPPINGS
        if (mapping is not tokenizer.REASONING_EFFORT_MAPPINGS
                or encoding.DEFAULT_REASONING_EFFORT != 'high'
                or mapping != (SERVING_BUDGETS if _registered else ORIGINAL)):
            raise RuntimeError('Unexpected runtime reasoning-budget state; refusing a partial prompt patch')
        if not _registered:
            # In-place update preserves the tokenizer wrapper's imported alias.
            mapping.clear()
            mapping.update(SERVING_BUDGETS)
            _registered = True
