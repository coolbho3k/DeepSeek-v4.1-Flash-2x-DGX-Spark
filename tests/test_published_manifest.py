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
        self.assertEqual(digest,'6d79a9ae5cfd121df7c559b76cde87b50551adbe68ae0dc94c97973e85e8e1d1')

    def test_published_manifest_passes_native_verifier(self):
        spec=importlib.util.spec_from_file_location('manifest_regression_verifier',ROOT/'release/runtime/tools/verify_public_download.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        _,summary=module.load_manifest(ROOT/'release/model-release-manifest.json',lock['model']['manifest_sha256'])
        self.assertEqual(summary['weight_shards'],55)


if __name__=='__main__':unittest.main()
