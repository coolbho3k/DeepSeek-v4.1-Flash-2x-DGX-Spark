# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import ast
from test_dspark_contracts import module, ROOT

confidence = module('confidence')


class Confidence(unittest.TestCase):
    def test_cache_change_preserves_dcp_transaction(self):
        raw=(ROOT/'release/runtime/serving/ds41/vllm_v2_cache.py').read_bytes()
        changed=confidence.cache_initializer_source(raw)
        inserted=('        '+repr(confidence.GRAPH_MODE)+',\n').encode()
        restored=changed.replace(inserted,b'').replace(confidence.SLOT_VIEW[1].encode(),confidence.SLOT_VIEW[0].encode())
        self.assertEqual(restored,raw)
        self.assertIn(b'_ds41_block_tables(kv_cache_config,',changed)
        ast.parse(changed)
        for invalid in (raw+b'\n',changed):
            with self.assertRaises(ValueError):confidence.cache_initializer_source(invalid)

    def test_padded_prefix_bounds(self):
        for requests in range(7):
            for length in range(9):
                self.assertEqual(confidence.valid_start_shape(requests,6,(length,),1),requests+1<=length<=7)
        for shape,stride in (((7,),2),((7,1),1),((),1)):
            self.assertFalse(confidence.valid_start_shape(1,6,shape,stride))
        self.assertFalse(confidence.valid_start_shape(7,6,(8,),1))

    def test_budget_receipt_counts_actual_and_zero_work(self):
        counts=dict(steps=0,request_rounds=0,scheduled_drafts=0,verified_drafts=0,trimmed_steps=0)
        self.assertTrue(confidence.observe_budget(counts,[5,5],10))
        self.assertTrue(confidence.observe_budget(counts,[5,5],3))
        self.assertFalse(confidence.observe_budget(counts,[5],0))
        self.assertEqual(counts,dict(steps=3,request_rounds=5,scheduled_drafts=25,verified_drafts=13,trimmed_steps=2))
        for scheduled,verified in (([5],6),([5],-1),([True],0),([5],True)):
            with self.assertRaises(ValueError):confidence.observe_budget(counts,scheduled,verified)

    def test_only_add_sm121_flattened_support(self):
        for cuda in (False,True):
            for capability in (None,80,90,100,110,120,121,122):
                for native in (False,True):
                    self.assertEqual(confidence.supports_platform(native,cuda,capability),
                        native or (cuda and capability==121))


if __name__=='__main__':unittest.main()
