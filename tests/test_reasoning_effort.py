# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only reasoning alias and release-overlay checks; no vLLM/GPU imports."""
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'release/runtime/ds41/vllm_prompt.py'
EXPECTED = {'minimal': 25, 'low': 50, 'medium': 60,
            'high': 75, 'xhigh': 90, 'max': 100}


class ReasoningEffort(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('prompt_adapter_test', ADAPTER)
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.mapping = dict(self.adapter.ORIGINAL)
        self.tokenizer = ModuleType('vllm.tokenizers.deepseek_v41')
        self.encoding = ModuleType('vllm.tokenizers.deepseek_v41_encoding')
        self.encoding.DEFAULT_REASONING_EFFORT = 'high'
        payloads = {}
        for name, module in (('tokenizer', self.tokenizer), ('encoding', self.encoding)):
            module.__file__ = '/fixture/' + name + '.py'
            module.REASONING_EFFORT_MAPPINGS = self.mapping
            raw = ('reviewed ' + name).encode()
            payloads[module.__file__] = raw
            self.adapter.UPSTREAM[name] = hashlib.sha256(raw).hexdigest()
        tokenizers = ModuleType('vllm.tokenizers')
        tokenizers.deepseek_v41 = self.tokenizer
        tokenizers.deepseek_v41_encoding = self.encoding
        vllm = ModuleType('vllm')
        vllm.tokenizers = tokenizers
        self.enterContext(patch.dict(sys.modules, {
            'vllm': vllm, 'vllm.tokenizers': tokenizers,
            self.tokenizer.__name__: self.tokenizer,
            self.encoding.__name__: self.encoding,
        }))
        self.enterContext(patch.object(Path, 'read_bytes', autospec=True,
                                       side_effect=lambda path: payloads[str(path)]))

    def test_native_presets_and_default_preserved(self):
        self.adapter.register()
        self.assertEqual(self.adapter.SOURCE_BUDGETS, {'low': 50, 'high': 75, 'max': 100})
        for name, value in self.adapter.SOURCE_BUDGETS.items():
            self.assertEqual(self.mapping[name], value)
        self.assertEqual(self.encoding.DEFAULT_REASONING_EFFORT, 'high')

    def test_compatibility_aliases_are_distinct_and_ordered(self):
        self.adapter.register()
        self.assertEqual(self.mapping, EXPECTED)
        values = [self.mapping[name] for name in EXPECTED]
        self.assertEqual(values, sorted(set(values)))
        self.assertTrue(all(1 <= value <= 100 for value in values))
        # "none" remains a tokenizer control, not an out-of-range numeric alias.
        self.assertNotIn('none', self.mapping)

    def test_imported_mapping_alias_updated_in_place(self):
        self.adapter.register()
        self.assertIs(self.tokenizer.REASONING_EFFORT_MAPPINGS, self.mapping)
        self.assertIs(self.encoding.REASONING_EFFORT_MAPPINGS, self.mapping)

    def test_registration_is_idempotent(self):
        self.adapter.register()
        self.adapter.register()
        self.assertEqual(self.mapping, EXPECTED)

    def test_unreviewed_source_rejected_before_mutation(self):
        self.adapter.UPSTREAM['tokenizer'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'not been reviewed'):
            self.adapter.register()
        self.assertEqual(self.mapping, self.adapter.ORIGINAL)

    def test_changed_default_rejected_before_mutation(self):
        self.encoding.DEFAULT_REASONING_EFFORT = 'low'
        with self.assertRaisesRegex(RuntimeError, 'Unexpected runtime'):
            self.adapter.register()
        self.assertEqual(self.mapping, self.adapter.ORIGINAL)

    def test_detached_tokenizer_mapping_rejected(self):
        self.tokenizer.REASONING_EFFORT_MAPPINGS = dict(self.mapping)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected runtime'):
            self.adapter.register()

    def test_unexpected_initial_mapping_rejected(self):
        self.mapping['medium'] = 50
        with self.assertRaisesRegex(RuntimeError, 'Unexpected runtime'):
            self.adapter.register()

    def test_post_registration_tampering_rejected(self):
        self.adapter.register()
        self.mapping['xhigh'] = 75
        with self.assertRaisesRegex(RuntimeError, 'Unexpected runtime'):
            self.adapter.register()


class ReasoningRelease(unittest.TestCase):
    def test_source_and_serving_overlay_match(self):
        self.assertEqual((ROOT / 'release/runtime/serving/ds41/vllm_prompt.py').read_bytes(),
                         ADAPTER.read_bytes())

    def test_serving_overlay_shadows_baked_adapter(self):
        overlay = ROOT / 'release/runtime/serving/ds41'
        spec = importlib.machinery.PathFinder.find_spec(
            'ds41.vllm_prompt', [str(overlay), '/opt/ds41-dcp-v3/ds41'])
        self.assertEqual(Path(spec.origin), overlay / 'vllm_prompt.py')


if __name__ == '__main__':
    unittest.main()
