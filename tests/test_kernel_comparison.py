# SPDX-License-Identifier: AGPL-3.0-only
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'probes/compare_kernel_batch.py'
SPEC = importlib.util.spec_from_file_location('kernel_comparison', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    return dict(status='complete', source_sha256='same-harness', deployment=dict(
        run_id='baseline', kit_manifest_sha256='old', serving=dict(gpu_memory_utilization=.92),
        nodes=[dict(kit='old', model='same-model', draft='same-draft', image='same-image')]),
        cases=[dict(label='reference', request=dict(temperature=0., seed=41, messages=['same']),
                    usage=dict(completion_tokens=400), reply='same', acceptance_fraction=.5,
                    tokens_per_step=2.5, wall_ms_per_step=100., timing=dict(
                        decode_tokens=399, server_decode_seconds=15.96, decode_tokens_per_second=25.))])


class Comparison(unittest.TestCase):
    def test_matched_gain(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['deployment'].update(run_id='candidate', kit_manifest_sha256='new')
        after['deployment']['nodes'][0]['kit'] = 'new'
        row = after['cases'][0]
        row['timing'].update(server_decode_seconds=13.3, decode_tokens_per_second=30.)
        row['wall_ms_per_step'] = 100. / 1.2
        report = MODULE.compare(before, after)
        self.assertAlmostEqual(report['pooled_decode_change_percent'], 20.)
        self.assertAlmostEqual(report['rows'][0]['step_change_percent'], -100. / 6.)
        self.assertEqual(report['reply_identical_count'], 1)

    def test_mismatched_request_refused(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['cases'][0]['request']['seed'] = 1729
        with self.assertRaisesRegex(ValueError, 'Request identity'):
            MODULE.compare(before, after)

    def test_changed_memory_refused(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['deployment']['serving']['gpu_memory_utilization'] = .93
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            MODULE.compare(before, after)

    def test_changed_weights_refused(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['deployment']['nodes'][0]['draft'] = 'different-draft'
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            MODULE.compare(before, after)

    def test_changed_harness_refused(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['source_sha256'] = 'new'
        with self.assertRaisesRegex(ValueError, 'harness changed'):
            MODULE.compare(before, after)

    def test_incomplete_refused(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'completed trials'):
            MODULE.compare(before, after)

    def test_different_output_lengths_refused(self):
        before = fixture()
        after = copy.deepcopy(before)
        after['cases'][0]['usage']['completion_tokens'] = 200
        with self.assertRaisesRegex(ValueError, 'token counts differ'):
            MODULE.compare(before, after)
