# SPDX-License-Identifier: AGPL-3.0-only
"""Offline contracts for selectable NVFP4 KV and its display-backed release."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from test_release import deployment, node_module, settings

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'release/runtime'


class Nvfp4Configuration(unittest.TestCase):
    def test_new_default_and_explicit_rollback(self):
        self.assertEqual(settings()['serving']['fp4_kv_mode'], 'nvfp4_4over6')
        self.assertEqual(settings(DS41_FP4_KV_MODE='legacy')['serving']['fp4_kv_mode'], 'legacy')
        for mode in ('', '0', '1', 'nvfp4', 'LEGACY'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'DS41_FP4_KV_MODE'):
                settings(DS41_FP4_KV_MODE=mode)

    def test_profiles_validate_modes_and_default_older_descriptors(self):
        host = KIT / 'tools/launch_profile.py'
        worker = KIT / 'serving/ds41/launch_profile.py'
        self.assertEqual(host.read_bytes(), worker.read_bytes())
        profile = runpy.run_path(str(host))
        old = dict(settings()['serving'])
        del old['fp4_kv_mode']
        self.assertEqual(profile['validate'](old)['fp4_kv_mode'], 'nvfp4_4over6')
        self.assertNotIn('fp4_kv_mode', old)
        for mode in ('legacy', 'nvfp4_4over6'):
            with patch.dict('os.environ', DS41_FP4_KV_MODE=mode, DS41_KV_CAP_MIB='0'):
                self.assertEqual(profile['from_environment']()['fp4_kv_mode'], mode)
        for mode in (False, None, 'wrong'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                profile['validate'](dict(old, fp4_kv_mode=mode))

    def test_both_workers_get_mode_and_keep_display_only_pool(self):
        node = node_module()
        for mode in ('nvfp4_4over6', 'legacy'):
            config = deployment()
            config['serving']['fp4_kv_mode'] = mode
            for rank in (0, 1):
                result = node.docker_command(config, rank)
                self.assertEqual(result['env']['DS41_FP4_KV_MODE'], mode)
                self.assertEqual(result['env']['DS41_KV_CAP_MIB'], '0')
                self.assertNotIn('--fp4-kv-mode', result['cmd'])
                self.assertIn('--device=' + config['nodes'][rank]['drm_card'] + ':/dev/dri/card0', result['command'])

    def test_cli_override(self):
        import contextlib
        import io
        import sys
        import launch
        from config import load
        base = settings()['values']
        def configured(root, overrides):
            return load(root, overrides, base)
        with patch.object(sys, 'argv', ['launch.py', '--dry-run', '--fp4-kv-mode', 'legacy']), \
             patch.object(launch, 'load', side_effect=configured), contextlib.redirect_stdout(io.StringIO()) as output:
            launch.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result['settings']['serving']['fp4_kv_mode'], 'legacy')
        self.assertFalse(result['changed'])

    def test_recorded_gpu_accuracy_matches_shipped_sources(self):
        report = json.loads((ROOT / 'release/experimental/nvfp4_kv/gpu-results.json').read_bytes())
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['codec_sha256'], hashlib.sha256((KIT / 'serving/ds41/fp4_main_kv.py').read_bytes()).hexdigest())
        self.assertEqual(report['probe_sha256'], hashlib.sha256((KIT / 'probes/check_nvfp4_four_over_six.py').read_bytes()).hexdigest())
        self.assertEqual(report['bits_per_value'], 4.5)
        self.assertEqual(report['regressed_groups'], 0)
        self.assertGreater(report['improved_groups'], 0)
        self.assertLess(report['four_over_six_sse'], report['baseline_sse'])
        self.assertTrue(report['legacy_exact'])
        self.assertTrue(report['baseline_gpu_checked'])
        self.assertTrue(report['public_prefill_2048_checked'])
        self.assertFalse(report['full_model_quality_tested'])

    def test_native_rounding_and_display_speed_receipts(self):
        root = ROOT / 'release/experimental/nvfp4_kv'
        codec_sha = hashlib.sha256((KIT / 'serving/ds41/fp4_main_kv.py').read_bytes()).hexdigest()
        for report_name, probe_name in (('rounding-results', 'check_nvfp4_rounding.py'),
                                       ('performance-results', 'bench_nvfp4_kv.py')):
            report = json.loads((root / (report_name + '.json')).read_bytes())
            self.assertEqual(report['probe_sha256'], hashlib.sha256((KIT / 'probes' / probe_name).read_bytes()).hexdigest())
            if report_name == 'rounding-results':
                self.assertEqual(report['codec_sha256'], codec_sha)
                self.assertEqual(report['status'], 'pass')
                self.assertEqual(report['combinations'], 4461660)
                self.assertTrue(report['all_midpoints_checked'])
            else:
                self.assertEqual(report['versions']['candidate']['fp4_main_kv.py'], codec_sha)
                self.assertTrue(report['actual_display_allocation_tested'])
                self.assertEqual(report['display_probe_bytes'], 4 * 2**20)
                self.assertEqual({r['version'] for r in report['results']},
                                 {'candidate', 'legacy', 'before', 'legacy_before'})
                self.assertEqual({r['rows'] for r in report['results']},
                                 {1, 4, 8, 24, 128, 512, 1056, 2048, 3072})
                self.assertTrue(all(r['spills'] == 0 for r in report['results']))

    def test_release_copies_and_writer_attestation(self):
        source = (KIT / 'serving/spark_backend_attestation.py').read_text()
        pins = next(ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PRIVATE_SOURCES' for t in n.targets))
        overlay = json.loads((KIT / 'serving/overlay-manifest.json').read_bytes())
        for name in ('fp4_main_kv.py', 'fp4_rope_store.py'):
            data = (KIT / 'serving/ds41' / name).read_bytes()
            self.assertEqual(data, (KIT / 'ds41' / name).read_bytes())
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(pins['ds41/' + name], digest)
            self.assertEqual(overlay['ds41/' + name], digest)
        # The display allocation implementation and its exact budget are
        # retained from the existing release, for either writer.
        self.assertEqual(hashlib.sha256((KIT / 'serving/ds41/display_kv.py').read_bytes()).hexdigest(),
            '81ae6b42cb400f5986b853e200eab101323ab992296f43659fc062db985d0c4f')


if __name__ == '__main__':
    unittest.main()
