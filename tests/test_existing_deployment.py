# SPDX-License-Identifier: AGPL-3.0-only
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from test_release import deployment, settings, launch


class ExistingDeployment(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ds41-reuse-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'existing.json'
        self.original = deployment()
        self.source.write_bytes(launch.encoded(self.original))
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.settings = settings(EXISTING_DEPLOYMENT=str(self.source),
                                 EXISTING_DEPLOYMENT_SHA256=self.digest, API_PORT='9999')

    def test_configuration_requires_both_settings(self):
        for values in ({'EXISTING_DEPLOYMENT': str(self.source)},
                       {'EXISTING_DEPLOYMENT_SHA256': self.digest},
                       {'EXISTING_DEPLOYMENT': 'relative', 'EXISTING_DEPLOYMENT_SHA256': self.digest},
                       {'EXISTING_DEPLOYMENT': str(self.source), 'EXISTING_DEPLOYMENT_SHA256': 'bad'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                settings(**values)

    def test_source_pin_checked_before_loading_runtime(self):
        self.source.write_text('{}')
        with patch.object(launch, 'module', side_effect=AssertionError('No runtime imports')):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                launch.prepare_existing(self.settings)

    def test_reuse_creates_new_run_and_keeps_asset_paths(self):
        checked = []
        checker = SimpleNamespace(verify=lambda *args: checked.append(args))
        node = SimpleNamespace(validate_config=lambda config: config)
        with patch.object(launch, 'STATE', self.root/'state'), \
             patch.object(launch, 'module', side_effect=[checker, node]), \
             patch.object(launch, 'host_info', return_value={'uid': 1000, 'gid': 1000, 'drm_gid': 44}), \
             patch.object(launch, 'node_action', return_value={'status': 'portable_node_preflight_pass'}) as action, \
             patch.object(launch, 'run', side_effect=AssertionError('No download/sync/start')), \
             contextlib.redirect_stdout(io.StringIO()):
            path, result = launch.prepare_existing(self.settings)
        self.assertEqual(len(checked), 1)
        self.assertNotEqual(result['run_id'], self.original['run_id'])
        self.assertEqual(result['api']['port'], 9999)
        self.assertEqual(action.call_count, 2)
        self.assertTrue(all(call.args[2] == 'preflight' for call in action.call_args_list))
        for i in (0, 1):
            for key in ('kit', 'model', 'draft', 'cache', 'image', 'model_receipt', 'runs'):
                self.assertEqual(result['nodes'][i][key], self.original['nodes'][i][key])
        self.assertEqual(json.loads(path.read_bytes()), result)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.digest)
        self.assertFalse((self.root/'state/current.json').exists())

    def test_reuse_does_not_retarget_another_worker(self):
        self.settings['worker'] = 'different@worker'
        checker = SimpleNamespace(verify=lambda *args: None)
        node = SimpleNamespace(validate_config=lambda config: config)
        with patch.object(launch, 'module', side_effect=[checker, node]), \
             patch.object(launch, 'host_info', side_effect=AssertionError('No remote access')):
            with self.assertRaisesRegex(ValueError, 'different worker'):
                launch.prepare_existing(self.settings)

    def test_prepare_skips_download_path_after_idle_checks(self):
        lock = {'runtime': {'image': 'unused'}}
        with patch.object(launch, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=b'')) as run, \
             patch.object(launch, 'prepare_existing', return_value=('new', {})) as reuse, \
             patch.object(launch, 'remote_home', side_effect=AssertionError('No download preparation')):
            self.assertEqual(launch.prepare(self.settings, lock), ('new', {}))
        self.assertEqual(run.call_count, 2)
        reuse.assert_called_once_with(self.settings)


if __name__ == '__main__':
    unittest.main()
