# SPDX-License-Identifier: AGPL-3.0-only
"""SWA format selection, reproducible native sources and GPU evidence."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

from test_release import deployment, node_module, settings

ROOT=Path(__file__).resolve().parents[1]
KIT=ROOT/'release/runtime'

class SwaKv(unittest.TestCase):
    def test_default_and_explicit_rollback(self):
        self.assertEqual(settings()['serving']['swa_kv_group_size'],32)
        self.assertEqual(settings(DS41_SWA_KV_GROUP_SIZE='64')['serving']['swa_kv_group_size'],64)
        for value in ('','16','032','32.0','bf16','0'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'DS41_SWA_KV_GROUP_SIZE'):
                settings(DS41_SWA_KV_GROUP_SIZE=value)

    def test_profiles_inherit_and_validate(self):
        path=KIT/'tools/launch_profile.py'
        self.assertEqual(path.read_bytes(),(KIT/'serving/ds41/launch_profile.py').read_bytes())
        profile=runpy.run_path(str(path));old=dict(settings()['serving'])
        del old['swa_kv_group_size'];del old['fp4_kv_mode']
        current=profile['validate'](old)
        self.assertEqual((current['swa_kv_group_size'],current['fp4_kv_mode']),(32,'nvfp4_4over6'))
        self.assertNotIn('swa_kv_group_size',old)
        for size in (32,64):
            with patch.dict('os.environ',DS41_SWA_KV_GROUP_SIZE=str(size),DS41_KV_CAP_MIB='0'):
                self.assertEqual(profile['from_environment']()['swa_kv_group_size'],size)
        for invalid in (True,'32',32.0,None,16):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):profile['validate'](dict(old,swa_kv_group_size=invalid))

    def test_modes_reach_both_workers_without_unknown_vllm_flags(self):
        node=node_module()
        for size in (32,64):
            for mode in ('legacy','nvfp4_4over6'):
                config=deployment();config['serving'].update(swa_kv_group_size=size,fp4_kv_mode=mode)
                for rank in (0,1):
                    command=node.docker_command(config,rank)
                    self.assertEqual(command['env']['DS41_SWA_KV_GROUP_SIZE'],str(size))
                    self.assertEqual(command['env']['DS41_FP4_KV_MODE'],mode)
                    self.assertEqual(command['env']['DS41_KV_CAP_MIB'],'0')
                    self.assertNotIn('--swa-kv-group-size',command['cmd'])

    def test_cli_override(self):
        import launch
        from config import load
        base=settings()['values']
        with patch.object(sys,'argv',['launch.py','--dry-run','--swa-kv-group-size','64']), \
             patch.object(launch,'load',side_effect=lambda root,overrides:load(root,overrides,base)), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            launch.main()
        report=json.loads(output.getvalue())
        self.assertEqual(report['settings']['serving']['swa_kv_group_size'],64)
        self.assertEqual(report['settings']['serving']['fp4_kv_mode'],'nvfp4_4over6')
        self.assertFalse(report['changed'])

    def test_native_corresponding_source_and_binary(self):
        root=ROOT/'release/experimental/swa_kv';vendor=KIT/'vendor/vllm-swa32-apache'
        build=runpy.run_path(str(root/'build_native.py'))
        manifest=json.loads((vendor/'UPSTREAM.json').read_bytes())
        for name,row in manifest['files'].items():
            self.assertEqual(hashlib.sha256((vendor/name).read_bytes()).hexdigest(),row['sha256'])
        generated=build['transform']((vendor/'csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu').read_text())
        self.assertEqual(generated,(KIT/'sources/swa32.cu').read_text())
        receipt=json.loads((root/'native-build.json').read_bytes())
        self.assertEqual(hashlib.sha256(generated.encode()).hexdigest(),receipt['source_sha256'])
        self.assertEqual(hashlib.sha256((KIT/'serving/libds41_swa32.so').read_bytes()).hexdigest(),receipt['binary_sha256'])
        for name in ('swa_kv.py','dcp_cache_gather.py','vllm_fp4_main.py','vllm_dcp.py'):
            self.assertEqual((KIT/'ds41'/name).read_bytes(),(KIT/'serving/ds41'/name).read_bytes())

    def test_source_verifier_admits_the_pinned_native_library(self):
        source=(KIT/'serving/spark_backend_attestation.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='file_sha')
        namespace={'hashlib':hashlib}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<file_sha>','exec'),namespace)
        binary=KIT/'serving/libds41_swa32.so'
        self.assertEqual(namespace['file_sha'](binary),hashlib.sha256(binary.read_bytes()).hexdigest())

    def test_actual_display_accuracy_and_reader_evidence(self):
        report=json.loads((ROOT/'release/experimental/swa_kv/display-results.json').read_bytes())
        self.assertEqual(report['status'],'pass')
        self.assertTrue(report['actual_display_allocation_tested'])
        self.assertEqual(report['group32_rope_dtype'],'bfloat16')
        self.assertEqual(report['group64_rope_dtype'],'bfloat16')
        # A later reviewed change may only ADD exact text to an evidenced file.
        later=json.loads((ROOT/'release/experimental/prefill/packed-supersession.json').read_bytes())
        for name,digest in report['source_sha256'].items():
            text=(KIT/'serving'/name).read_bytes()
            if name==later['path'] and later['evidenced_sha256']==digest:
                added=later['added_text'].encode()
                self.assertEqual(text.count(added),1)
                text=text.replace(added,b'')
            self.assertEqual(hashlib.sha256(text).hexdigest(),digest)
        for row in report['cases']:
            self.assertEqual(row['regressed_groups'],0)
            self.assertTrue(all(row[k] for k in ('q_bit_exact','rope_bit_exact','canaries_preserved')))
        self.assertGreater(sum(row['improved_groups'] for row in report['cases']),0)
        self.assertEqual({r['path'] for r in report['readers']},{'two_pass','prefill','dcp_prefill','decode'})
        self.assertTrue(all(r['nmse']<1e-8 for r in report['readers']))
        self.assertFalse(report['full_model_quality_ab'])

if __name__=='__main__':unittest.main()
