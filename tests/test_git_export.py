# SPDX-License-Identifier: AGPL-3.0-only
"""The Git allowlist must retain the complete recipe and reject private state."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'release'))
import export


class GitExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='ds41-export-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.checkout = Path(cls.temporary.name) / 'recipe'
        with contextlib.redirect_stdout(io.StringIO()):
            export.export(cls.checkout)
        cls.expected = {str(p.relative_to(cls.checkout)) for p in cls.checkout.rglob('*') if p.is_file()}
        cls.git('init', '--quiet')

    @classmethod
    def git(cls, *args, **kwargs):
        return subprocess.run(['git', '-c', 'core.autocrlf=false', '-c', 'core.excludesFile=/dev/null',
                               *args], cwd=cls.checkout, capture_output=True, check=True, **kwargs)

    def test_all_export_files_are_trackable(self):
        self.git('add', '--all')
        tracked = set(self.git('ls-files', '-z').stdout.decode().rstrip('\0').split('\0'))
        self.assertEqual(tracked, self.expected)
        self.assertIn('probes/compare_kernel_batch.py', tracked)
        self.assertIn('docs/kernel-batch-performance.md', tracked)
        self.assertIn('stop-server.sh', tracked)

    def test_shell_launchers_remain_executable(self):
        for name in ('start-server.sh', 'stop-server.sh'):
            with self.subTest(script=name):
                self.assertTrue((self.checkout / name).stat().st_mode & 0o111)

    def test_private_paths_are_ignored(self):
        paths = ['.env.ds41', '.env', '.env.local', '.state/public/current.json',
                 '.assets/weights.bin', 'artifacts/runtime.tar', 'reports/private.json',
                 'scripts/campaign.py', 'probes/private-benchmark.py',
                 'release/review.local.json', 'release/runtime/review.local.json',
                 'release/.env.local', 'tests/__pycache__/test.pyc']
        result = self.git('check-ignore', '--no-index', '-z', '--stdin',
                          input=('\0'.join(paths) + '\0').encode())
        self.assertEqual(set(result.stdout.decode().rstrip('\0').split('\0')), set(paths))
