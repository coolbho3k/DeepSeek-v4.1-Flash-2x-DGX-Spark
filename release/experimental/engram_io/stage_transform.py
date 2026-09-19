# SPDX-License-Identifier: AGPL-3.0-only
"""Small, exact-anchor transformations of the pinned MiaAI-derived adapter.

Retains its arithmetic, masking, callbacks and lifetime management. Never edits
the parent. Component mode includes the production 96-thread/external-event
adjustments normally applied by spark_combined_miaai at startup.
"""
import argparse
import hashlib
from pathlib import Path
import re

PARENT_SHA = 'c9b751ec4ee4251acc26dece3c7794408bb305a45ea32cf666a784e49033aacb'


def transform(raw, library_sha256, *, component=True):
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError('Unrecognized parent staging source')
    if not re.fullmatch('[0-9a-f]{64}', library_sha256):
        raise ValueError('Explicit native library digest required')
    text = raw.decode()
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Changed staging anchor: '+old)
        text = text.replace(old, new)
    replace("BINARY_SHA = 'cacd266e8b218327854c394a5c519e63ed22045dc77fcd058a75ca427a776e36'",
            'BINARY_SHA = '+repr(library_sha256))
    replace('if lib.ds41_row_store_abi() != 1:', 'if lib.ds41_row_store_abi() != 2:')
    replace('    lib.ds41_row_store_abi.restype = U\n',
        '    lib.ds41_row_store_abi.restype = U\n'
        '    lib.ds41_row_store_attach_packed.argtypes = [P, C.c_char_p, U]\n'
        '    lib.ds41_row_store_attach_packed.restype = C.c_int\n'
        '    lib.ds41_row_store_profile.argtypes = [P, C.POINTER(U)]\n'
        '    lib.ds41_row_store_clear_cache.argtypes = [P]\n')
    replace('def __init__(self, embedding, library, *, device=None):',
            'def __init__(self, embedding, library, *, device=None, packed=None, packed_layer=None, mapped=False):')
    replace('        self.embedding = embedding\n',
        "        if type(mapped) is not bool or ((packed is None) != (packed_layer is None)):\n"
        "            raise ValueError('Explicit mapped mode and paired packed path/layer required')\n"
        '        self.mapped = mapped\n'
        '        self.embedding = embedding\n')
    replace('            self.dev_w = torch.empty_like(self.host_w, device=self.device)\n'
            '            self.dev_s = torch.empty_like(self.host_s, device=self.device)\n',
        '            if self.mapped:\n'
        '                from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor\n'
        '                self.dev_w = get_accelerator_view_from_cpu_tensor(self.host_w)\n'
        '                self.dev_s = get_accelerator_view_from_cpu_tensor(self.host_s)\n'
        '                if self.dev_w.device != self.device or self.dev_s.device != self.device:\n'
        "                    raise RuntimeError('Mapped staging is not on the current GPU')\n"
        '            else:\n'
        '                self.dev_w = torch.empty_like(self.host_w, device=self.device)\n'
        '                self.dev_s = torch.empty_like(self.host_s, device=self.device)\n')
    replace('            self.staging_bytes = cap * (8 + 264 * 2)\n',
        '            if packed is not None and self.lib.ds41_row_store_attach_packed(\n'
        '                    self.store, str(packed).encode(), packed_layer) != 1:\n'
        "                raise ValueError('Explicit packed Engram artifact did not attach')\n"
        '            self.staging_bytes = cap * (8 + 264 * (1 if self.mapped else 2))\n')
    replace('            self.dev_w[:rows].copy_(self.host_w[:rows], non_blocking=True)\n'
            '            self.dev_s[:rows].copy_(self.host_s[:rows], non_blocking=True)\n',
        '            if not self.mapped:\n'
        '                self.dev_w[:rows].copy_(self.host_w[:rows], non_blocking=True)\n'
        '                self.dev_s[:rows].copy_(self.host_s[:rows], non_blocking=True)\n')
    if component:
        replace("not in ('32', '64')", "not in ('96',)")
        replace('self.event = torch.cuda.Event()', 'self.event = torch.cuda.Event(external=True)')
        replace('MAX_TOKENS = 1056', 'MAX_TOKENS = 2048')
    compile(text, 'engram_candidate_stage.py', 'exec')
    return text.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = transform(a.parent.read_bytes(), hashlib.sha256(a.library.read_bytes()).hexdigest())
    with a.output.open('xb') as f:
        f.write(result)


if __name__ == '__main__':
    main()
