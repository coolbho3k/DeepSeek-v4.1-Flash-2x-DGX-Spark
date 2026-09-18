# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests of our actual adapter/patches; no GPU correctness claim."""
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ds41 import cooperative_contract as contract, cooperative_moe as adapter


def source_definition(path, name):
    source = path.read_text(); nodes = ast.parse(source).body
    for part in name.split('.'):
        node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
        nodes = node.body
    return textwrap.dedent(''.join(source.splitlines(True)[node.lineno-1:node.end_lineno]))


def patched_source():
    path = ROOT/'artifacts/ds41-runtime-optimize-v53/serving'
    source = source_definition(path/'spark_fused_moe_async.py', 'AsyncSmallDispatcher.__call__')
    ns = {}
    exec(source_definition(ROOT/'ds41/staged_route_prepare.py', 'forward_replacements'), ns)
    for old, new in ns['forward_replacements']()+contract.forward_replacements():
        if source.count(old) != 1:
            raise AssertionError('Patch anchor changed: '+old)
        source = source.replace(old, new)
    return source


class Tensor(NS):
    def __len__(self): return self.shape[0]
    def numel(self):
        import math
        return math.prod(self.shape)


class Tests(unittest.TestCase):
    def test_ranges_capacity_alignment_and_no_locks(self):
        ranges = contract.intervals()
        self.assertEqual(len(ranges), 7)
        self.assertEqual(ranges[-1][-1]-ranges[-1][-2], 851*4)
        self.assertEqual(sum(b-a for _, _, a, b in ranges), 2301260)
        self.assertTrue(all(p in range(4) for _, p, _, _ in ranges))

    def test_shape_selection_preserves_draft_and_prefill(self):
        for rows in (0, 1, 2, 3, 4, 6, 8, 9, 24, 2048):
            for topk in (1, 3, 6, 8):
                self.assertEqual(contract.eligible_shape((rows,5120),(rows,topk)),
                    1 <= rows <= 8 and topk == 6)
                self.assertEqual(contract.selected_shape((rows,5120),(rows,topk)),
                    5 <= rows <= 8 and topk == 6)
        self.assertFalse(contract.eligible_shape((1,4096),(1,6)))
        self.assertFalse(contract.eligible_shape((5120,),(1,6)))

    def fixture(self, rows=5, topk=6, fail=False):
        log=[]; experts={0:object()}; stream=NS(cuda_stream=17, wait_event=lambda event:log.append('wait'))
        work=NS(device='cuda:0', pending=True, stream_id=9, temps=[], locks=None,
            ready=NS(record=lambda s:log.append('record')))
        bank=NS(owner=experts, keys=(0,), mapping=None, ptrs=[])
        dispatch=NS(lock=nullcontext(), failed=False, workspace=work, banks={id(experts):bank},
            module=NS(forward=lambda *args:log.append('fallback_native')), stream_waits=0)
        x=Tensor(shape=(rows,5120), device='cuda:0', dtype='bf16')
        ids=Tensor(shape=(rows,topk)); weights=Tensor(shape=ids.shape)
        def call(*args):
            log.append('cooperative')
            if fail: raise RuntimeError('native launch error; no retry')
            return 'cooperative_result'
        torch=NS(cuda=NS(device=lambda d:nullcontext(), current_stream=lambda d:stream))
        out=NS(to=lambda dtype:'fallback_result')
        namespace=dict(torch=torch, base=NS(validate_inputs=lambda *a:None,
            Dispatcher=NS(__call__=lambda *args:'large_or_empty_fallback')),
            _ds41_coop_eligible=contract.selected_shape, _ds41_coop_call=call,
            _ds41_prepare_routes=lambda *args:(None,None,None,out,None))
        exec(patched_source(),namespace)
        return namespace['__call__'], dispatch, experts, x, ids, weights, log

    def test_exact_dispatcher_uses_existing_fence(self):
        fn,d,e,x,i,w,log=self.fixture()
        self.assertEqual(fn(d,e,x,i,w),'cooperative_result')
        self.assertEqual(log,['wait','cooperative','record'])
        self.assertEqual(d.workspace.stream_id,17)
        self.assertEqual(d.stream_waits,1)
        self.assertEqual(d.last_schedule['mode'],'miaai_two_stage_cooperative')

    def test_no_retry_after_native_error(self):
        fn,d,e,x,i,w,log=self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError,'no retry'): fn(d,e,x,i,w)
        self.assertTrue(d.failed)
        self.assertEqual(log,['wait','cooperative'])
        with self.assertRaisesRegex(RuntimeError,'poisoned'): fn(d,e,x,i,w)
        self.assertEqual(log,['wait','cooperative'])

    def test_top3_fallback_unchanged(self):
        fn,d,e,x,i,w,log=self.fixture(rows=1,topk=3)
        self.assertEqual(fn(d,e,x,i,w),'fallback_result')
        self.assertEqual(log,['wait','fallback_native','record'])

    def test_large_and_empty_fallback_unchanged(self):
        for rows in (0,24,2048):
            fn,d,e,x,i,w,log=self.fixture(rows=rows)
            self.assertEqual(fn(d,e,x,i,w),'large_or_empty_fallback')
            self.assertEqual(log,[])

    def test_small_rows_retain_faster_staged_path(self):
        for rows in (1,2,3,4):
            fn,d,e,x,i,w,log=self.fixture(rows=rows)
            self.assertEqual(fn(d,e,x,i,w),'fallback_result')
            self.assertEqual(log,['wait','fallback_native','record'])

    def test_capture_cannot_initialize_native(self):
        torch=NS(cuda=NS(is_current_stream_capturing=lambda:True))
        with patch.dict(sys.modules,torch=torch), self.assertRaisesRegex(RuntimeError,'prewarm'):
            adapter.Native(None,None,None)

    def test_native_abi_pointer_order_and_no_host_routes(self):
        source=(ROOT/'ds41/cooperative_moe.py').read_text()
        self.assertIn('[half_x, local, rw, *bank.ptrs, *self.scratch, out]',source)
        self.assertNotIn('.cpu()',source)
        self.assertNotIn('.item()',source)
        routes=(ROOT/'ds41/cooperative_routes.py').read_text()
        self.assertIn('tl.store(Counters+ctr, 0, ctr < 851)',routes)
        self.assertIn('rw.to(tl.float16)',routes)

    def test_atomic_registration_compiles_and_preserves_parent_pins(self):
        spec=importlib.util.spec_from_file_location('prepare_coop',ROOT/'scripts/prepare_cooperative_runtime.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        names=('cooperative_contract','cooperative_moe','cooperative_routes')
        raw=module.combined_source(module.PARENT,{f'ds41.{n}':(ROOT/f'ds41/{n}.py').read_bytes() for n in names})
        compile(raw,'<cooperative-register>','exec')
        self.assertIn(b'Cooperative MoE selection changed after startup',raw)
        self.assertIn(b'cooperative_quality_qualified=False',raw)
        self.assertIn(b'put(owner, name, replacement)',raw)


def torch_alias_test():
    import torch
    from ds41.cooperative_moe import alias_scratch
    torch.set_num_threads(2)
    temps=[torch.full((size//2,),-7.,dtype=torch.float16) for size in contract.CAPACITIES]
    work=NS(temps=temps,device=torch.device('cpu'))
    views=alias_scratch(work)
    for view, (_,parent,start,end) in zip(views,contract.intervals()):
        assert view.data_ptr()==temps[parent].data_ptr()+start
        assert view.numel()*view.element_size()==end-start
        assert view.untyped_storage().data_ptr()==temps[parent].untyped_storage().data_ptr()
        view.zero_()
    for parent,temp in enumerate(temps):
        covered=max(b for _,p,a,b in contract.intervals() if p==parent)
        assert torch.all(temp[covered//2:]==-7.)
    print(json.dumps(dict(status='actual_torch_cpu_alias_pass',device_allocations=0,views=len(views))))


if __name__=='__main__':
    if '--torch-alias' in sys.argv: torch_alias_test()
    else: unittest.main()
