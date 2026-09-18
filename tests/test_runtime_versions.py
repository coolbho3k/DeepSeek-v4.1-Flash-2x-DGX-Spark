# SPDX-License-Identifier: AGPL-3.0-only
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'release'))
import bootstrap


class RuntimeVersions(unittest.TestCase):
    def test_stable_for_same_runtime(self):
        first = dict(image='first', cache_manifest_sha256='cache')
        second = dict(cache_manifest_sha256='cache', image='first')
        self.assertEqual(bootstrap.runtime_storage('/example', first), bootstrap.runtime_storage('/example', second))

    def test_changed_image_or_assets_get_new_directory(self):
        before = dict(image='first', files={'source': 'old'})
        for after in (dict(image='second', files={'source': 'old'}), dict(image='first', files={'source': 'new'})):
            self.assertNotEqual(bootstrap.runtime_storage('/example', before), bootstrap.runtime_storage('/example', after))

    def test_existing_downloads_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'downloads').mkdir()
            previous = root / 'downloads/kernel-cache.tar'
            previous.write_bytes(b'old-cache')
            new = bootstrap.runtime_storage(root, dict(image='new'))
            self.assertTrue(new.is_relative_to(root / 'runtime-assets'))
            self.assertFalse(new.exists())
            self.assertEqual(previous.read_bytes(), b'old-cache')
