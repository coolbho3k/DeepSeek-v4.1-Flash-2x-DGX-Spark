# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only release checks: exact tested overlap payload and installed hooks."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'release/runtime'
EXPECTED = {
    '__init__': '60cd06da8801cf7403d06e37398b92409f2bab7c9dea47744ba56df72b83599d',
    'attention': '9d186d63cc310c4e3a4fb81448e0bc7b862a2e87d22f69af265ae310a065e466',
    'integration': '2b40d6f1c2bcfc204b006c97d5b737c816f34ea1e80954253b5b22c3160e58cb',
    'packed': '8788de737b1feeda2c4003b27f531547bbf685602d0c28bf6342ab081233b944',
    'policy': '74d7e3f8527f9079110f8033dde63ed9b0c2d3b135cb1035e043b259e4f2d806',
    'transport': '10f1e338b31db54db30959d127f6bfa8964281aa909f579570297b7015217af3',
}


def assignment(path, name):
    return next(ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


class OverlapRelease(unittest.TestCase):
    def test_exact_canary_source_and_attestation(self):
        pins = assignment(KIT / 'serving/spark_backend_attestation.py', 'PRIVATE_SOURCES')
        overlay = json.loads((KIT / 'serving/overlay-manifest.json').read_bytes())
        for module, digest in EXPECTED.items():
            name = 'ds41/dcp_overlap/' + module + '.py'
            with self.subTest(module=module):
                self.assertEqual(hashlib.sha256((KIT / 'serving' / name).read_bytes()).hexdigest(), digest)
                self.assertEqual(pins[name], digest)
                self.assertEqual(overlay[name], digest)

    def test_concurrent_default_and_startup_hooks(self):
        self.assertEqual(assignment(KIT / 'serving/ds41/dcp_overlap/policy.py', 'MODE'), 'concurrent')
        descriptor = assignment(KIT / 'serving/spark_combined_miaai.py', 'DCP_OVERLAP')
        self.assertEqual(descriptor['mode'], 'concurrent')
        self.assertTrue(descriptor['memory_limits_unchanged'])
        for path, anchor in (
            ('serving/combined_worker.py', 'init_device = _wrap_overlap_init(_init_device)'),
            ('serving/ds41/vllm_fp4_main.py', 'forward = wrap_forward(forward)'),
            ('serving/spark_sparse_slots.py', 'bind_sparse_mapper(forward, candidate)'),
            ('serving/spark_dcp_communication.py', '_ds41_overlap_forward_admitted(forward,')):
            self.assertIn(anchor, (KIT / path).read_text())

    def test_honest_qualification_and_unchanged_image(self):
        requirements = json.loads((KIT / 'runtime-requirements.json').read_bytes())
        info = requirements['dcp_overlap_candidate']
        self.assertTrue(info['serving_tests_run'])
        self.assertFalse(info['public_fresh_clone_gpu_qualified'])
        self.assertFalse(info['broad_quality_qualified'])
        lock = json.loads((ROOT / 'recipe-lock.json').read_bytes())
        self.assertTrue(lock['runtime']['image'].endswith(
            '@sha256:30557154f0d56b613a95867d41918c50d61f94107fa46861343d08f28c54bce3'))


def load_tests(loader, tests, pattern):
    from release.experimental.dcp_overlap import test_cpu
    tests.addTests(loader.loadTestsFromModule(test_cpu))
    return tests
