# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'release/experimental/dspark'


def module(name):
    spec = importlib.util.spec_from_file_location('dspark_test_'+name, HERE/(name+'.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


c = module('contracts')
n = module('native_capacity')


class Contracts(unittest.TestCase):
    def test_existing_capacity_and_nonoverlap(self):
        self.assertEqual(c.COUNTERS, 3819)
        self.assertEqual(c.MAX_SLOTS, 216)
        self.assertLessEqual(c.MAX_SLOTS, 256)
        ends = [0]*4
        for _, parent, _, end in c.intervals():
            ends[parent] = max(ends[parent], end)
        self.assertEqual(ends, [6635520, 2211840, 995328, 512940])
        self.assertTrue(all(a <= b for a, b in zip(ends, c.CAPACITIES)))

    def test_fixed_graph_envelope(self):
        self.assertEqual(c.Policy(3).graph_sizes(), (1,2,3,4,6,8,9,12,15,16,18,20,24))
        for k in (3,4,5):
            policy = c.Policy(k)
            self.assertEqual(max(policy.graph_sizes()), 6*(k+1))
            for count in range(1,7):
                self.assertIn(count*(k+1), policy.graph_sizes())
                self.assertIn(count*k, policy.graph_sizes())

    def test_adaptive_keys_do_not_alias(self):
        policy = c.Policy(5, 'ema', (1,2,3,4,5))
        self.assertNotEqual(policy.target_key(1,3), policy.target_key(2,1))
        keys = [policy.target_key(n,k) for n in range(1,7) for k in range(1,6)]
        self.assertEqual(len(set(keys)),30)
        for total, _, _ in keys:
            self.assertIn(total,policy.graph_sizes())

    def test_sparse_ema_inventory_keeps_full_checkpoint_envelope(self):
        policy = c.Policy(5, 'ema', (1,3,5))
        keys = [policy.target_key(n,k) for n in range(1,7) for k in policy.prefix_lengths]
        self.assertEqual(len(set(keys)),18)
        self.assertEqual(max(policy.graph_sizes()),36)
        for total, _, _ in keys:self.assertIn(total,policy.graph_sizes())
        with self.assertRaises(ValueError):policy.target_key(1,2)

    def test_fail_closed(self):
        for args in ((True,), (0,), (6,), (5,'other'), (5,'fixed',(3,5)),
                     (5,'ema',(0,5)), (5,'ema',(3,)), (5,'ema',(5,3)),
                     (5,'ema',(3,3,5)), (5,'ema',(True,5))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                c.Policy(*args)

    def test_pinned_native_capacity_only(self):
        wrapper=(ROOT/'release/runtime/sources/cooperative24.cu').read_bytes()
        kernel=(ROOT/'release/runtime/vendor/miaai-cooperative-moe-agpl/extensions/cooperative_moe/native/cooperative_moe_kernel.cuh').read_bytes()
        kernel=kernel.replace(b'p.slots_max = 48; p.rows_max = 8;',b'p.slots_max = 144; p.rows_max = 24;')
        kernel=kernel.replace(b'p.ctr_a_len = 432; p.ctr_b_len = 320;',b'p.ctr_a_len = 1296; p.ctr_b_len = 960;')
        kernel=b'// Local DS41 C6 adaptation: 24 physical rows / 144 routed slots, ABI 2.\n'+kernel
        out, extended=n.transform(wrapper,kernel)
        self.assertIn(b'ROWS_MAX = 36;',out)
        self.assertIn(b'goal50_coop_experiment() { return 401; }',out)
        # The complete numerical kernel body is byte-identical.
        anchor=b'constexpr int WK = 16;'
        self.assertEqual(extended.split(anchor)[1],kernel.split(anchor)[1])
        with self.assertRaises(ValueError): n.transform(wrapper+b'\n',kernel)
        with self.assertRaises(ValueError): n.transform(out,extended)


if __name__=='__main__': unittest.main()
