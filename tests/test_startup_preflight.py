# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only regressions for issue #1 control IP and issue #3 driver checks."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_release import ROOT, deployment, launch, node_module, settings


class ControlNetwork(unittest.TestCase):
    def test_each_node_uses_its_fabric_ip_not_inherited_lan_ip(self):
        node = node_module()
        for dual in (False, True):
            config = deployment(dual)
            for index in (0, 1):
                with self.subTest(dual=dual, index=index), patch.dict(
                        os.environ, VLLM_HOST_IP='192.168.50.99'):
                    result = node.docker_command(config, index)
                    address = config['nodes'][index]['fabric_ip']
                    self.assertEqual(result['env'].get('VLLM_HOST_IP'), address)
                    self.assertEqual(result['command'].count('VLLM_HOST_IP='+address), 1)
                    self.assertEqual(result['env']['NCCL_DEBUG'], 'INFO')


class DriverPreflight(unittest.TestCase):
    def policy(self):
        return launch.module(ROOT/'release/runtime/tools/display_driver.py', 'test_display_driver')

    def test_qualified_pair(self):
        policy = self.policy()
        sample = dict(loaded='580.173.02', reported=['580.173.02'])
        self.assertEqual(policy.validate(sample, 'worker'), sample)

    def test_unqualified_or_inconsistent_or_missing_versions_fail(self):
        policy = self.policy()
        for sample in (
            dict(loaded='595.84', reported=['595.84']),
            dict(loaded='580.178.04', reported=['580.178.04']),
            dict(loaded='580.173.02', reported=['595.84']),
            dict(loaded='', reported=['580.173.02']),
            dict(loaded='580.173.02', reported=[]),
            dict(loaded='580.173.02', reported=['580.173.02', '595.84']),
            {}, None,
        ):
            with self.subTest(sample=sample), self.assertRaisesRegex(ValueError, '580.173.02'):
                policy.validate(sample, 'worker')

    def test_observation_is_read_only_and_no_cuda(self):
        policy = self.policy()
        with patch.object(Path, 'read_text', return_value='580.173.02\n'), patch.object(
                policy.subprocess, 'check_output', return_value='580.173.02\n') as command:
            self.assertEqual(policy.observe(), dict(loaded='580.173.02', reported=['580.173.02']))
        self.assertEqual(command.call_args.args[0],
                         ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
        self.assertNotIn('torch', sys.modules)

    def test_both_hosts_checked_without_downloads_or_container_actions(self):
        sample = dict(loaded='580.173.02', reported=['580.173.02'])
        with patch.object(launch, 'json_run', return_value=sample) as command, \
                contextlib.redirect_stdout(io.StringIO()):
            launch.check_display_drivers(settings())
        self.assertEqual(command.call_count, 2)
        self.assertEqual([call.args[1] for call in command.call_args_list], [None, 'alice@worker.example'])
        for call in command.call_args_list:
            self.assertEqual(call.args[0], ['python3', '-B', '-'])
            self.assertLessEqual(call.kwargs['timeout'], 30)

    def test_peer_failure_identifies_peer(self):
        good = dict(loaded='580.173.02', reported=['580.173.02'])
        bad = dict(loaded='595.84', reported=['595.84'])
        with patch.object(launch, 'json_run', side_effect=[good, bad]), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, 'worker.*595.84'):
            launch.check_display_drivers(settings())

    def test_restart_checks_before_stopping_or_preparing(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(launch, 'STATE', Path(directory)), \
                patch.object(sys, 'argv', ['launch.py', '--restart']), \
                patch.object(launch, 'load', return_value=settings()), \
                patch.object(launch, 'read_state', return_value=None), \
                patch.object(launch, 'check_display_drivers', create=True,
                             side_effect=ValueError('incompatible driver')) as check, \
                patch.object(launch, 'stop', side_effect=AssertionError('Do not stop')) as stop, \
                patch.object(launch, 'prepare', side_effect=AssertionError('Do not download')) as prepare:
            with self.assertRaisesRegex(ValueError, 'incompatible driver'):
                launch.main()
        check.assert_called_once()
        stop.assert_not_called()
        prepare.assert_not_called()

    def test_node_rechecks_before_memory_or_asset_validation(self):
        node = node_module()
        with self.assertRaisesRegex(ValueError, '580.173.02'):
            node.validate_start_sample(deployment()['nodes'][0],
                                       {'driver': dict(loaded='595.84', reported=['595.84'])})


class StartupDiagnostics(unittest.TestCase):
    def test_control_handshake_hint_does_not_claim_nccl_failure(self):
        node = node_module()
        text = "Using ['PYNCCL'] all-reduce backends out of ['PYNCCL'] for group 'tp:0'"
        row = node.startup_log_summary(text)
        self.assertEqual(row['last_observed_stage'], 'tp_communicator_selected')
        self.assertIn('ZeroMQ', row['hint'])
        self.assertIn('not proof', row['hint'])
        self.assertNotIn(text, json.dumps(row))

    def test_driver_hint_preserves_stage_without_returning_logs(self):
        node = node_module()
        row = node.startup_log_summary('private request text\nregister display IO: CUDA_ERROR_INVALID_VALUE (1)')
        self.assertEqual(row['last_observed_stage'], 'display_io_registration_failed')
        self.assertIn('580.173.02', row['hint'])
        self.assertNotIn('private request text', json.dumps(row))

    def test_loading_marker_prevents_misleading_early_handshake_hint(self):
        node = node_module()
        for text in ('Starting to load model', 'Loading safetensors checkpoint shards',
                     'Model loading took 123 seconds'):
            row = node.startup_log_summary("Using ['PYNCCL'] all-reduce backends\n"+text)
            self.assertEqual(row['last_observed_stage'], 'model_load_or_later')
        self.assertEqual(node.startup_log_summary('')['last_observed_stage'], 'unknown')

    def test_logs_only_for_exact_owned_container_and_include_stderr(self):
        node = node_module()
        cid = 'd'*64
        with patch.object(node, 'inspect_owned', return_value={'container': cid}), \
                patch.object(node.subprocess, 'check_output', return_value='private text') as command:
            row = node.startup_diagnostics(deployment(), 1)
        self.assertEqual(row['control_ip'], '192.168.40.2')
        self.assertEqual(command.call_args.args[0], ['docker', 'logs', '--tail', '120', cid])
        self.assertEqual(command.call_args.kwargs['stderr'], subprocess.STDOUT)
        self.assertLessEqual(command.call_args.kwargs['timeout'], 5)
        self.assertNotIn('private text', json.dumps(row))

    def test_unowned_container_never_read(self):
        node = node_module()
        with patch.object(node, 'inspect_owned', side_effect=ValueError('not owned')), \
                patch.object(node.subprocess, 'check_output') as command:
            with self.assertRaisesRegex(ValueError, 'not owned'):
                node.startup_diagnostics(deployment(), 0)
        command.assert_not_called()

    def test_diagnostic_failure_is_not_a_new_stop_condition(self):
        node_module()
        pair = launch.module(ROOT/'release/runtime/tools/portable_pair.py', 'test_startup_pair')
        cids = ['a'*64, 'b'*64]
        records, calls = [], []
        tick = [0]

        def caller(config, index, action):
            calls.append((tick[0], index, action))
            if action == 'inspect':
                return dict(container=cids[index], node=index,
                            state=dict(Running=tick[0] < 2, OOMKilled=False, StartedAt='fixed'),
                            memory=dict(MemAvailable=4*2**30))
            if action == 'health':
                return dict(healthy=False)
            if action == 'startup-diagnostics':
                raise TimeoutError('CPU-only simulated log timeout')
            if action == 'stop':
                return dict(state=dict(Running=False))
            raise AssertionError(action)

        def sleep(seconds):
            tick[0] += 1

        result = pair.watch(deployment(), cids, records.append, caller=caller,
                            sleep=sleep, clock=lambda: tick[0]*200)
        self.assertEqual(result['reason'], 'node0_worker_stopped_or_unhealthy')
        self.assertEqual([t for t, i, a in calls if a == 'stop'], [2, 2])
        self.assertEqual(len([1 for t, i, a in calls if a == 'startup-diagnostics']), 2)
        self.assertIn('startup_control_plane', [r.get('stage') for r in records])
        self.assertIn('startup_diagnostics', [r.get('stage') for r in records])


if __name__ == '__main__':
    unittest.main()
