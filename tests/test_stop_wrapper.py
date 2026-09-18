# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the stop wrapper using a fake launcher or the real dry-run only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StopWrapper(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ds41-stop-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.recipe = self.directory / 'recipe with spaces'
        self.recipe.mkdir()
        shutil.copy2(ROOT / 'stop-server.sh', self.recipe / 'stop-server.sh')
        fake = self.recipe / 'start-server.sh'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n'
                        'exit "${WRAPPER_TEST_EXIT_CODE:-0}"\n')
        fake.chmod(0o755)
        # Do not inherit the real recipe config, credentials or server state.
        self.env = {'PATH': os.defpath, 'DS41_ENV_FILE': '/dev/null'}

    def run_wrapper(self, *args, exit_code=0):
        return subprocess.run([str(self.recipe / 'stop-server.sh'), *args],
                              cwd=self.directory, capture_output=True, timeout=10,
                              env={**self.env, 'WRAPPER_TEST_EXIT_CODE': str(exit_code)})

    def test_stops_by_delegating_from_another_directory(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'stop\0')

    def test_preserves_arguments_without_shell_evaluation(self):
        result = self.run_wrapper('--dry-run', 'literal value;not-a-command')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout,
                         b'stop\0--dry-run\0literal value;not-a-command\0')

    def test_failure_is_not_hidden(self):
        self.assertEqual(self.run_wrapper(exit_code=7).returncode, 7)

    def test_real_launcher_dry_run_only(self):
        result = subprocess.run([str(ROOT / 'stop-server.sh'), '--dry-run'],
                                cwd=self.directory, env=self.env, text=True,
                                capture_output=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout), {'action': 'stop', 'changed': False})


if __name__ == '__main__':
    unittest.main()
