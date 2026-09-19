# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
import ast
import unittest
from test_dspark_contracts import module

integration=module('integration')


class Integration(unittest.TestCase):
    def test_observer_patch_is_at_the_native_dedented_depth(self):
        source='''def update_from_output(self):
    for req_id in ids:
        if scheduled_spec_token_ids:
            num_accepted = max(len(generated_token_ids) - num_sampled, 0)
            num_rejected = num_draft_tokens - num_accepted
'''
        before,after=integration.OBSERVE
        tree=ast.parse(source.replace(before,after))
        body=tree.body[0].body[0].body[0].body
        self.assertEqual([type(n).__name__ for n in body],['Assign','If','Assign'])
        self.assertEqual(body[1].body[0].value.func.attr,'observe')

    def setUp(self):
        self.settings=integration.settings
        integration.settings=lambda:SimpleNamespace(DRAFT_TOKENS=5,PREFIX_LENGTHS=(1,2,3,4,5))
    def tearDown(self):integration.settings=self.settings

    def test_target_and_draft_graph_lengths_stay_distinct(self):
        target=type('ModelCudaGraphManager',(),{'decode_query_len':6})()
        draft=type('DFlashCudaGraphManager',(),{'decode_query_len':5})()
        self.assertEqual(integration.graph_lens(target),[2,3,4,5,6])
        self.assertEqual(integration.graph_lens(draft),[5])
        target.decode_query_len=4
        with self.assertRaises(ValueError):integration.graph_lens(target)
        with self.assertRaises(ValueError):integration.graph_lens(SimpleNamespace(decode_query_len=6))

    def test_only_active_decode_requests_influence_batch_choice(self):
        observed={}
        class Policy:
            def prune(self,live):observed['live']=set(live)
            def choose(self,ids,structured):
                observed['ids']=ids;observed['structured']=structured;return 3
        requests={k:SimpleNamespace(request_id=k,is_prefill_chunk=p,use_structured_output=s)
                  for k,p,s in [('decode',False,False),('schema',False,True),('prompt',True,False)]}
        scheduler=SimpleNamespace(requests=requests)
        old=integration.prefix_policy;integration.prefix_policy=lambda _:Policy()
        try:self.assertEqual(integration.choose_prefix(scheduler,{'decode':6,'schema':6,'prompt':100,'finished':1}),3)
        finally:integration.prefix_policy=old
        self.assertEqual(observed['ids'],['decode','schema'])
        self.assertEqual(observed['structured'],['schema'])
        self.assertEqual(observed['live'],set(requests))


if __name__=='__main__':unittest.main()
