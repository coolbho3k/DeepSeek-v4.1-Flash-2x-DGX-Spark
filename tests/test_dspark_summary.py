# SPDX-License-Identifier: AGPL-3.0-only
import copy
import unittest
from test_dspark_contracts import module

stats = module('summarize')


class Summary(unittest.TestCase):
    def report(self):
        counters = {
            'spec_decode_num_drafts_total': 10,
            'spec_decode_num_draft_tokens_total': 30,
            'spec_decode_num_accepted_tokens_total': 15,
        }
        counters = {'vllm:'+key+'{engine="0"}': value for key, value in counters.items()}
        for position, value in enumerate((8, 5, 2)):
            counters['vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="'+str(position)+'"}'] = value
        return dict(status='complete', cases=[dict(label='example', request=dict(temperature=0),
            speculative_delta=counters, timing=dict(decode_tokens_per_second=30),wall_ms_per_step=80)])

    def test_unconditional_survival_and_scheduled_prefix(self):
        group = stats.summary(self.report())['example/T0.0']
        self.assertEqual(group['mean_scheduled_drafts'], 3)
        self.assertNotIn('mean_verified_drafts',group)
        self.assertIn('not the confidence-trimmed',group['draft_counter_scope'])
        self.assertEqual(group['acceptance_fraction'], .5)
        self.assertEqual(group['position_survival'], [.8, .5, .2])

    def test_reject_failed_or_inconsistent_evidence(self):
        report = self.report(); report['status'] = 'running'
        with self.assertRaises(ValueError): stats.summary(report)
        report = self.report()
        report['cases'][0]['speculative_delta']['vllm:spec_decode_num_accepted_tokens_total{engine="0"}'] = 16
        with self.assertRaises(ValueError): stats.summary(report)

    def test_descriptive_comparison(self):
        baseline = stats.summary(self.report()); candidate = copy.deepcopy(baseline)
        candidate['example/T0.0']['median_decode_tps'] = 33
        self.assertAlmostEqual(stats.compare(baseline,candidate)['geometric_mean_speed_ratio'], 1.1)
        with self.assertRaises(ValueError): stats.compare(baseline,{})


if __name__ == '__main__': unittest.main()
