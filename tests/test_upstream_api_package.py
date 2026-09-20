# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only checks for the source-overlay refresh and preserved runtime."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('upstream_api_prepare',
    ROOT/'release/experimental/upstream_api/prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class Packaging(unittest.TestCase):
    def test_transitive_source_and_metadata_pins(self):
        a = b'VALUE = 1\n'
        b = ('PIN = '+repr(prepare.sha(a))+'\n').encode()
        c = ('PIN = '+repr(prepare.sha(b))+'\n').encode()
        old = {'serving/a.py':a, 'serving/b.py':b, 'tools/c.py':c,
               'runtime-requirements.json':prepare.encoded({'sha256':prepare.sha(c)}),
               'serving/overlay-manifest.json':b'{}\n'}
        candidate = dict(old, **{'serving/a.py':b'VALUE = 2\n'})
        output = prepare.repin(candidate, {n:prepare.sha(raw) for n,raw in old.items()})
        self.assertEqual(old['serving/a.py'], a)
        self.assertIn(prepare.sha(output['serving/a.py']).encode(), output['serving/b.py'])
        self.assertIn(prepare.sha(output['serving/b.py']).encode(), output['tools/c.py'])
        self.assertEqual(json.loads(output['runtime-requirements.json'])['sha256'],
                         prepare.sha(output['tools/c.py']))
        self.assertEqual(json.loads(output['serving/overlay-manifest.json']),
                         {n.removeprefix('serving/'):prepare.sha(output[n])
                          for n in ('serving/a.py','serving/b.py')})

    def test_public_overlay_and_backend_source_pins(self):
        kit = ROOT/'release/runtime'
        overlay = json.loads((kit/'serving/overlay-manifest.json').read_text())
        for name, digest in overlay.items():
            self.assertEqual(hashlib.sha256((kit/'serving'/name).read_bytes()).hexdigest(), digest, name)
        tree = ast.parse((kit/'serving/spark_backend_attestation.py').read_text())
        pins = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'PRIVATE_SOURCES' for t in n.targets))
        for name, digest in pins.items():
            self.assertEqual(hashlib.sha256((kit/'serving'/name).read_bytes()).hexdigest(), digest, name)

    def test_profile_sources_match_and_zero_rollback(self):
        kit = ROOT/'release/runtime'
        self.assertEqual((kit/'tools/launch_profile.py').read_bytes(),
                         (kit/'serving/ds41/launch_profile.py').read_bytes())
        source = (kit/'serving/ds41/speculative_prefix_retention.py').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         'c183e62741c8bf6176514414d2c80cf2514a5ee7a8f5f9cedccba06a8d3e5f65')


if __name__ == '__main__':
    unittest.main()
