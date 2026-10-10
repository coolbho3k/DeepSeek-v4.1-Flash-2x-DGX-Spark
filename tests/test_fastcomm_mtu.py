# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only coverage of the MTU policy used by the candidate native build."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FastcommMtu(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_peer_negotiation(self):
        source = r'''
#include "fastcomm_mtu.h"
#include <cassert>
#include <cstdint>
using ds41_fastcomm::negotiate_mtu;
int main() {
  // In particular, hardware max=4096 must not override active MTU=1024.
  static_assert(negotiate_mtu(3, 3) == 3);
  static_assert(negotiate_mtu(5, 3) == 3);
  static_assert(negotiate_mtu(3, 5) == 3);
  static_assert(negotiate_mtu(5, 5) == 5);
  for (uint32_t local = 1; local <= 5; ++local)
    for (uint32_t peer = 1; peer <= 5; ++peer) {
      auto mtu = negotiate_mtu(local, peer);
      assert(mtu == (local < peer ? local : peer));
      assert(mtu == negotiate_mtu(peer, local));
    }
  for (uint32_t invalid : {0u, 6u, UINT32_MAX})
    for (uint32_t valid = 1; valid <= 5; ++valid) {
      assert(negotiate_mtu(invalid, valid) == 0);
      assert(negotiate_mtu(valid, invalid) == 0);
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'test.cpp').write_text('#include <initializer_list>\n' + source)
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'release/experimental/fastcomm'),
                            str(path / 'test.cpp'), '-o', str(path / 'test')], check=True)
            subprocess.run([str(path / 'test')], check=True)

    def test_replace_integrated_kit_refreshes_verified_pins(self):
        parent = ROOT / 'release/runtime'
        manifest_sha = hashlib.sha256((parent / 'bundle-manifest.json').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            library = directory / 'candidate.so'
            # This test checks packaging/integrity only, not ELF execution.
            library.write_bytes(b'synthetic candidate for pin-refresh regression')
            kit = directory / 'kit'
            receipt = directory / 'receipt.json'
            subprocess.run([sys.executable, '-B',
                            str(ROOT / 'release/experimental/fastcomm/prepare_kit.py'),
                            '--parent-kit', str(parent), '--parent-sha256', manifest_sha,
                            '--library', str(library), '--output', str(kit),
                            '--receipt', str(receipt), '--replace-existing'], check=True,
                           capture_output=True, text=True)
            proof = json.loads(receipt.read_text())
            subprocess.run([sys.executable, '-B', str(parent / 'verify.py'), str(kit),
                            '--manifest-sha256', proof['candidate_kit_sha256']],
                           check=True, capture_output=True, text=True)
            def assignment(path, name):
                return next(ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
                            if isinstance(n, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == name for t in n.targets))
            module_sha = hashlib.sha256((kit / 'serving/ds41/fastcomm.py').read_bytes()).hexdigest()
            pins = assignment(kit / 'serving/spark_combined_miaai.py', 'PINS')
            attestation = assignment(kit / 'serving/spark_backend_attestation.py', 'PRIVATE_SOURCES')
            self.assertEqual(pins['ds41.fastcomm'], module_sha)
            self.assertEqual(attestation['ds41/fastcomm.py'], module_sha)
            self.assertEqual(attestation['libfastcomm.so'], hashlib.sha256(library.read_bytes()).hexdigest())
            self.assertTrue((kit / 'native-source/fastcomm_mtu.h').is_file())
        self.assertEqual(hashlib.sha256((parent / 'bundle-manifest.json').read_bytes()).hexdigest(), manifest_sha)


if __name__ == '__main__':
    unittest.main()
