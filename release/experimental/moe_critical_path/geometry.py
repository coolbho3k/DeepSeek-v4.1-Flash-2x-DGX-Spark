# SPDX-License-Identifier: AGPL-3.0-only
"""Pure, source-pinned geometry experiment. No deployment/GPU side effects.

Transforms the MiaAI-derived local adapter for component tests only. Both
resource preparation and launch must choose the same existing compiled
geometry. Arithmetic, scratch, locks and original native digest are unchanged.
"""
import ast
import hashlib

PARENT_SHA='22edf8a5aed2194ac0fd2050a9fd27ffea48dd82de021c40f68fd17109ec6bc5'
GEOMETRIES={0:'narrow_gateup_narrow_down',1:'wide_gateup_wide_down',2:'wide_gateup_narrow_down'}


def transform(source, geometry):
    if type(geometry) is not int or geometry not in GEOMETRIES:
        raise ValueError('Only already-compiled native geometries 0, 1 and 2')
    if hashlib.sha256(source).hexdigest()!=PARENT_SHA:
        raise ValueError('Cooperative adapter differs from the reviewed baseline')
    text=source.decode()
    edits=[('self.library.goal50_coop_info(3, 1, info)',
            f'self.library.goal50_coop_info(3, {geometry}, info)'),
           ('self.launch(pointers, 3, len(x), len(bank.keys), 10., 1, 0,',
            f'self.launch(pointers, 3, len(x), len(bank.keys), 10., {geometry}, 0,')]
    for before,after in edits:
        if text.count(before)!=1:raise ValueError('Changed cooperative launch anchor')
        text=text.replace(before,after)
    ast.parse(text)
    return text.encode()
