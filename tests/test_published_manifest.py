# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PublishedManifest(unittest.TestCase):
    def test_post_upload_manifest_not_pre_upload_pin(self):
        path=ROOT/'release/model-release-manifest.json'
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest,lock['model']['manifest_sha256'])
        self.assertEqual(digest,'828b7b5d7a672c5f2e44caca3d72d9c114c39634e46eb51deabd067a0f44a2c9')

    def test_published_manifest_passes_native_verifier(self):
        spec=importlib.util.spec_from_file_location('manifest_regression_verifier',ROOT/'release/runtime/tools/verify_public_download.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        _,summary=module.load_manifest(ROOT/'release/model-release-manifest.json',lock['model']['manifest_sha256'])
        self.assertEqual(summary['weight_shards'],55)


if __name__=='__main__':unittest.main()
