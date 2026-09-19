# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from test_dspark_contracts import module

adaptive=module('adaptive')


class Adaptive(unittest.TestCase):
    def test_warmup_and_structured_keep_full_prefix(self):
        p=adaptive.PrefixEMA()
        self.assertEqual(p.choose(['new']),5)
        for _ in range(3):p.observe('r',5,0)
        self.assertEqual(p.choose(['r']),5)
        p.observe('r',5,0)
        self.assertLess(p.choose(['r']),5)
        self.assertEqual(p.choose(['r'],structured=['r']),5)

    def test_censored_tail_is_not_fabricated_and_exploration_recovers(self):
        p=adaptive.PrefixEMA(alpha=1.,warmup=1,explore_after=3)
        p.observe('r',5,1)
        self.assertEqual(p.choose(['r']),2)
        for _ in range(2):p.observe('r',2,2)
        self.assertEqual(p.state['r'].survival,[1.,1.,0.,0.,0.])
        self.assertLess(p.choose(['r']),5)
        p.observe('r',2,2)
        self.assertEqual(p.choose(['r']),5)
        p.observe('r',5,5)
        self.assertEqual(p.choose(['r']),5)

    def test_one_hard_request_does_not_force_batch_minimum(self):
        p=adaptive.PrefixEMA(alpha=1.,warmup=1)
        p.observe('hard',5,0);p.observe('easy',5,5)
        self.assertEqual(p.choose(['hard']),1)
        self.assertEqual(p.choose(['hard','easy']),4)

    def test_cost_aware_decision_prices_total_step_and_padding(self):
        p=adaptive.PrefixEMA(alpha=1.,warmup=1)
        p.observe('r',5,3)
        self.assertEqual(p.choose(['r'],costs={1:70,2:70,3:70,4:100,5:100}),3)
        self.assertEqual(p.choose(['r'],costs={1:10,2:70,3:70,4:100,5:100}),1)
        with self.assertRaises(ValueError):p.choose(['r'],costs={1:10})

    def test_state_reset_and_empty_placeholder_observation(self):
        p=adaptive.PrefixEMA()
        p.observe('r',0,0)
        self.assertEqual(p.state,{})
        p.observe('r',5,0);p.prune([])
        self.assertEqual(p.choose(['r']),5)

    def test_sparse_inventory_never_selects_an_uncaptured_length(self):
        p=adaptive.PrefixEMA(lengths=(1,3,5),alpha=1.,warmup=1)
        for accepted in range(6):
            p.observe('r',5,accepted)
            self.assertIn(p.choose(['r']),p.lengths)
        self.assertEqual(p.choose(['r']),5)

    def test_invalid_observations_and_configs(self):
        p=adaptive.PrefixEMA()
        for drafted,accepted in ((6,1),(3,4),(-1,0),(True,1),(3,True)):
            with self.assertRaises(ValueError):p.observe('r',drafted,accepted)
        for kwargs in ({'alpha':float('nan')},{'alpha':0},{'lengths':(2,6)},
                       {'warmup':0},{'explore_after':0},{'margin':-1}):
            with self.assertRaises(ValueError):adaptive.PrefixEMA(**kwargs)
        with self.assertRaises(ValueError):p.choose(['r','r'])


if __name__=='__main__':unittest.main()
