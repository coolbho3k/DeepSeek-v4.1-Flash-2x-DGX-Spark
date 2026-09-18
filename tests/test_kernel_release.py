# SPDX-License-Identifier: AGPL-3.0-only
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'release/runtime'


class KernelRelease(unittest.TestCase):
    def test_both_tested_optimizations_enabled(self):
        tree = ast.parse((KIT / 'serving/spark_combined_miaai.py').read_text())
        selected = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'KERNEL_BATCH' for t in node.targets))
        self.assertEqual(selected, dict(online_decode_attention=True, length_aware_radix_topk=True))

    def test_exact_qualified_kernel_payloads(self):
        expected = {
            'serving/ds41/online_decode_attention.py': '198b1dc6e54b6980c6db96e465778773d75f334b7dade43f1b73ce71e46abca7',
            'serving/ds41/length_aware_topk.py': '83990e153b30aad6f98dd299ec5db57afc85f0654c326455aca535613c1ec20f',
            'serving/ds41/length_aware_topk_native.py': 'fc98239082a24a0125ce891ba702618ca28f42072fcd3fb43ea1ad11c8d1d377',
            'serving/topk-native/topk.so': '2ee6bde332658f6c3fd0eba299a6ed9cbdc3a72b78f17fb9c7bd149d0246f672',
            'kernels/length_aware_topk.cu': 'afa778cfbdb5650ccc0cc063ff41eeabe4682c6190e19b37763a376f9a56fb3e',
        }
        for name, digest in expected.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((KIT / name).read_bytes()).hexdigest(), digest)

    def test_native_binary_has_matching_source_and_receipt(self):
        receipt = json.loads((KIT / 'serving/topk-native/complete.json').read_bytes())
        self.assertEqual(hashlib.sha256((KIT / 'serving/topk-native/topk.so').read_bytes()).hexdigest(), receipt['binary_sha256'])
        self.assertEqual(hashlib.sha256((KIT / 'kernels/length_aware_topk.cu').read_bytes()).hexdigest(), receipt['source_sha256'])
        self.assertIn('AGPL-3.0-only', (KIT / 'kernels/length_aware_topk.cu').read_text())
