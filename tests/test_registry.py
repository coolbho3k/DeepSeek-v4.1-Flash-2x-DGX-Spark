# SPDX-License-Identifier: AGPL-3.0-only
"""Offline tests of GHCR packaging and pull/extraction; no actual containers."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'release'))
import registry
import package_ghcr


def spec():
    return {'transport': 'ghcr', 'image': 'ghcr.io/example/ds41@sha256:' + 'a'*64,
        'cache_manifest_sha256': 'b'*64,
        'files': {name: {'bytes': 2, 'sha256': hashlib.sha256(b'ok').hexdigest()} for name in registry.ASSETS}}


class Registry(unittest.TestCase):
    def test_immutable_references_only(self):
        registry.validate(spec())
        for image in ('ghcr.io/example/ds41:latest', 'docker.io/example/ds41@sha256:'+'a'*64,
                      'ghcr.io/example/ds41@sha256:bad', '-oProxyCommand=bad'):
            value = spec(); value['image'] = image
            with self.subTest(image=image), self.assertRaises(ValueError):
                registry.validate(value)

    def test_assets_are_bounded_and_exact(self):
        for update in ('path', 'size', 'hash', 'manifest'):
            value = spec()
            if update == 'path':value['files']['../escape'] = value['files'].pop('kernel-cache.tar')
            if update == 'size':value['files']['kernel-cache.tar']['bytes'] = 2**40
            if update == 'hash':value['files']['kernel-cache.tar']['sha256'] = 'bad'
            if update == 'manifest':value['cache_manifest_sha256'] = 'bad'
            with self.subTest(update=update), self.assertRaises(ValueError):registry.validate(value)

    def test_valid_cached_assets_need_no_container(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'downloads').mkdir()
            for name in registry.ASSETS:(root/'downloads'/name).write_bytes(b'ok')
            checker=SimpleNamespace(inspect=lambda _: {'Id': 'sha256:'+'c'*64}, verify=lambda *_: None)
            with patch.object(registry.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, \
                 patch.object(registry.subprocess, 'check_output', side_effect=AssertionError('No container')):
                image=registry.prepare(spec(), root, {}, checker, lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(image, 'sha256:'+'c'*64)
            self.assertEqual(run.call_count, 1)

    def test_extraction_never_starts_or_forces_container(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as d:
                calls=[]; container='d'*64
                def run(argv, **kwargs):
                    calls.append(argv)
                    if argv[1]=='cp':
                        if fail:raise subprocess.CalledProcessError(1, argv)
                        Path(argv[-1]).write_bytes(b'ok')
                    return subprocess.CompletedProcess(argv, 0)
                checker=SimpleNamespace(inspect=lambda _: {'Id': 'sha256:'+'c'*64}, verify=lambda *_: None)
                with patch.object(registry.subprocess,'run', side_effect=run), \
                     patch.object(registry.subprocess,'check_output',return_value=container+'\n') as create:
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):
                            registry.prepare(spec(), d, {}, checker, lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
                    else:
                        registry.prepare(spec(), d, {}, checker, lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
                self.assertEqual(calls[-1], ['docker','rm',container])
                self.assertIn('--runtime=runc',create.call_args.args[0])
                self.assertFalse(any(word in argv for argv in calls for word in ('start','run','--force','--gpus')))

    def test_identity_failure_prevents_asset_extraction(self):
        checker=SimpleNamespace(inspect=lambda _: {}, verify=lambda *_: (_ for _ in ()).throw(ValueError('identity')))
        with patch.object(registry.subprocess,'run',return_value=subprocess.CompletedProcess([],0)), \
             patch.object(registry.subprocess,'check_output',side_effect=AssertionError('No container')):
            with self.assertRaisesRegex(ValueError,'identity'):registry.prepare(spec(), Path('/unused'), {}, checker, None)


class Packaging(unittest.TestCase):
    def node(self):
        return {'Architecture':'arm64','Os':'linux','Config':{'Env':['PATH=/opt/venv/bin:/usr/bin'],
                'Entrypoint':['/opt/venv/bin/python','-m','vllm'],'WorkingDir':'/opt/app','Labels':{}}}

    def test_no_compilation_and_full_source_credit(self):
        text=package_ghcr.dockerfile(self.node())
        self.assertIn('FROM scratch',text)
        self.assertIn('COPY --from=compiled / /',text)
        self.assertNotIn('\nRUN ',text)
        self.assertIn('MiaAI-Lab',text)
        self.assertIn('AGPL-3.0-only',text)
        self.assertIn('runtime-source.tar.gz',text)

    def test_unknown_metadata_not_silently_lost(self):
        node=self.node();node['Config']['Volumes']={'/model':{}}
        with self.assertRaises(ValueError):package_ghcr.dockerfile(node)

    def test_sensitive_environment_not_packaged(self):
        node=self.node();node['Config']['Env'].append('HF_TOKEN_WRITE=not-real')
        with self.assertRaises(ValueError):package_ghcr.dockerfile(node)

    def test_metadata_cannot_inject_dockerfile(self):
        for value in ('/safe\nRUN bad','${OTHER}', 'nul\0'):
            node=self.node();node['Config']['Env']=['PATH='+value]
            with self.subTest(value=value),self.assertRaises(ValueError):package_ghcr.dockerfile(node)

    def test_namespace_is_explicit(self):
        self.assertTrue(package_ghcr.image_name('example').startswith('ghcr.io/example/'))
        for owner in ('Example','../bad','x;cmd',''):
            with self.assertRaises(ValueError):package_ghcr.image_name(owner)


if __name__=='__main__':unittest.main()
