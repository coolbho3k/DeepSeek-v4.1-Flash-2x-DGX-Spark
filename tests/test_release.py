# SPDX-License-Identifier: AGPL-3.0-only
"""Offline release checks: no SSH, Docker, GPU allocation or serving requests."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'release'))
import config
import launch
import bootstrap


def settings(**extra):
    env=dict(WORKER_HOST='alice@worker.example',HEAD_FABRIC_IP='192.168.40.1',
        WORKER_FABRIC_IP='192.168.40.2',FABRIC_NETWORK='192.168.40.0/24',
        HEAD_IFNAME='fabric0',WORKER_IFNAME='fabric9',HEAD_HCA='mlx5_0',WORKER_HCA='mlx5_7')
    env.update(extra)
    return config.load(ROOT,environ=env)


def node_module():
    sys.path.insert(0,str(ROOT/'release/runtime/tools'))
    return launch.module(ROOT/'release/runtime/tools/portable_node.py','public_test_node')


def deployment(dual=False):
    s=settings(**(dict(HEAD_SECONDARY_IP='192.168.40.3',WORKER_SECONDARY_IP='192.168.40.4',
        HEAD_SECONDARY_IFNAME='fabric1',WORKER_SECONDARY_IFNAME='fabric8',
        HEAD_SECONDARY_HCA='mlx5_1',WORKER_SECONDARY_HCA='mlx5_8') if dual else {}))
    nodes=[]
    for i in (0,1):
        root='/srv/head' if i==0 else '/mnt/worker'
        assets=launch.module(ROOT/'release/runtime/tools/engram_assets.py','test_engram_assets')
        nodes.append(dict(ssh=None if i==0 else s['worker'],kit=root+'/kit',model=root+'/model',
            draft=root+'/draft',cache=root+'/cache',runs=root+'/runs',model_receipt=root+'/verified.json',
            engram=assets.reference(Path(root+'/packed'),i),
            image='sha256:'+'a'*64,uid=1001+i,gid=1001+i,drm_card='/dev/dri/card'+str(i),
            drm_gid=44+i,rails=s['rails'][i],**s['rails'][i][0]))
    return dict(format='ds41_two_spark_deployment_v1',run_id='ds41-release-v12345',
        kit_manifest_sha256='a'*64,model_manifest_sha256='b'*64,cache_manifest_sha256='c'*64,
        fabric_network=s['fabric_network'],nodes=nodes,serving=s['serving'],api=s['api'],startup_memory_override=True)


class Configuration(unittest.TestCase):
    def test_tested_defaults(self):
        s=settings();self.assertEqual(s['serving']['max_num_seqs'],6)
        self.assertEqual(s['serving']['kv_cap_mib'],0)
        self.assertEqual(s['api']['port'],8888)
    def test_no_personal_defaults(self):
        with self.assertRaisesRegex(ValueError,'set:'):config.load(ROOT,environ={})
    def test_cli_overrides(self):
        s=settings(); v=config.load(ROOT,{'MAX_NUM_SEQS':4,'API_PORT':9876},s['values'])
        self.assertEqual(v['serving']['max_num_seqs'],4);self.assertEqual(v['api']['port'],9876)
    def test_bad_values(self):
        for key,value in [('WORKER_HOST','-oProxyCommand=bad'),('API_PORT','80'),
            ('GPU_MEMORY_UTILIZATION','nan'),('GPU_MEMORY_UTILIZATION','.93'),('MAX_NUM_SEQS','7'),
            ('MAX_MODEL_LEN','2000000'),('MAX_NUM_BATCHED_TOKENS','1024'),
            ('LONG_PREFILL_TOKEN_THRESHOLD','512'),('HEAD_DRM_CARD','/dev/mem'),
            ('WORKER_FABRIC_IP','192.168.41.2'),('REMOTE_DIR','relative/path')]:
            with self.subTest(key=key),self.assertRaises(ValueError):settings(**{key:value})
    def test_secondary_partial_refused(self):
        with self.assertRaises(ValueError):settings(HEAD_SECONDARY_IP='192.168.40.3')
    def test_util_override_explicit(self):
        with self.assertRaises(ValueError):settings(GPU_MEMORY_UTILIZATION='.90')
        self.assertEqual(settings(GPU_MEMORY_UTILIZATION='.90',ALLOW_STARTUP_MEMORY_SHORTFALL='0')['serving']['gpu_memory_utilization'],.90)
    def test_no_publishing_token_in_configuration(self):
        s=settings(HF_TOKEN_WRITE='not-a-real-secret')
        self.assertNotIn('HF_TOKEN_WRITE',json.dumps(s))


class PortableNodes(unittest.TestCase):
    def test_single_rail_commands(self):
        node=node_module();c=deployment()
        for i in (0,1):
            out=node.docker_command(c,i)
            self.assertEqual(out['env']['NCCL_IB_HCA'],'='+c['nodes'][i]['hca']+':1')
            self.assertEqual(out['env']['NCCL_IB_MERGE_NICS'],'0')
            self.assertIn('--device='+c['nodes'][i]['drm_card']+':/dev/dri/card0',out['command'])
            self.assertIn('--enable-auto-tool-choice',out['cmd'])
            self.assertEqual('--headless' in out['cmd'],i==1)
            self.assertEqual(out['env']['DS41_MAX_NUM_SEQS'],'6')
            self.assertEqual(out['env']['DS41_KV_CAP_MIB'],'0')
            self.assertNotIn('10.100.',json.dumps(out))
            self.assertNotIn('/home/emi',json.dumps(out))
    def test_dual_rail_commands(self):
        node=node_module();out=node.docker_command(deployment(True),1)
        self.assertEqual(out['env']['NCCL_IB_HCA'],'=mlx5_7:1,mlx5_8:1')
        self.assertEqual(out['env']['NCCL_IB_MERGE_NICS'],'1')
    def test_invalid_native_configuration(self):
        node=node_module()
        for update in ('drm','rail','ip'):
            c=deployment()
            if update=='drm':c['nodes'][0]['drm_card']='/dev/mem'
            elif update=='rail':c['nodes'][0]['rails']=[]
            else:c['nodes'][1]['fabric_ip']='192.168.40.1'
            with self.subTest(update=update),self.assertRaises(ValueError):node.validate_config(c)
    def test_no_host_reconfiguration(self):
        text=(ROOT/'release/launch.py').read_text()+(ROOT/'release/runtime/tools/portable_node.py').read_text()
        for forbidden in ('systemd-run','systemctl','iptables','nftables','modprobe','rmmod','ssh-keygen','ssh-copy-id'):
            self.assertNotIn(forbidden,text)
    def test_imports_without_cuda(self):
        self.assertNotIn('torch',sys.modules)

    def test_display_flags_read_directly_when_allowed(self):
        node=node_module()
        with patch.object(Path,'read_text',side_effect=['Y\n','N\n']), \
             patch.object(node,'command',side_effect=AssertionError('No helper needed')):
            self.assertEqual(node.display_flags({'image':'sha256:'+'a'*64}),
                             {'modeset':'Y','fbdev':'N'})

    def test_root_only_display_flags_use_restricted_cpu_helper(self):
        node=node_module()
        with patch.object(Path,'read_text',side_effect=PermissionError), \
             patch.object(node,'command',return_value='Y\nN\n') as command:
            self.assertEqual(node.display_flags({'image':'sha256:'+'a'*64}),
                             {'modeset':'Y','fbdev':'N'})
        argv=command.call_args.args[0]
        for flag in ('--runtime=runc','--pull=never','--network=none','--read-only',
                     '--cap-drop=ALL','--security-opt=no-new-privileges','--memory=32m'):
            self.assertIn(flag,argv)
        for flag in ('--gpus','--privileged','--mount','--volume'):
            self.assertFalse(any(part.startswith(flag) for part in argv))

    def test_bad_display_helper_output_fails_closed(self):
        node=node_module()
        with patch.object(Path,'read_text',side_effect=PermissionError), \
             patch.object(node,'command',return_value='Y\nN\nextra\n'):
            with self.assertRaisesRegex(ValueError,'Invalid NVIDIA'):
                node.display_flags({'image':'sha256:'+'a'*64})


class Lifecycle(unittest.TestCase):
    def test_empty_exec_identity_requires_exact_owned_command(self):
        state=dict(deployment='/srv/run/deployment.json',config=deployment(),
            controller=dict(pid=123,start_ticks='42',uid=1000,argv=[]))
        actual=dict(state['controller'],argv=launch.controller_command(Path(state['deployment']),state['config']))
        with patch.object(launch,'process_identity',return_value=actual):
            self.assertTrue(launch.controller_alive(state))
        for fields in (dict(argv=[]),dict(argv=['unrelated']),dict(start_ticks='43'),dict(uid=1001)):
            with patch.object(launch,'process_identity',return_value=dict(actual,**fields)):
                self.assertFalse(launch.controller_alive(state))

    def test_start_waits_for_complete_exec_identity(self):
        from types import SimpleNamespace
        argv=['python3','controller.py']
        full=dict(pid=123,start_ticks='42',uid=1000,argv=argv)
        child=SimpleNamespace(pid=123,poll=lambda:None)
        with patch.object(launch,'process_identity',side_effect=[dict(full,argv=[]),full]), \
             patch.object(launch.time,'sleep'):
            self.assertEqual(launch.wait_controller_identity(child,argv),full)

    def test_no_state_stop_does_not_touch_other_servers(self):
        with patch.object(launch,'run',side_effect=AssertionError('No external command')),contextlib.redirect_stdout(io.StringIO()):
            launch.stop(None)
    def test_pid_reuse_not_owned(self):
        state={'controller':{'pid':123,'start_ticks':'old'}}
        with patch.object(launch,'process_identity',return_value={'pid':123,'start_ticks':'new'}):
            self.assertFalse(launch.controller_alive(state))
    def test_dry_run_no_side_effects(self):
        s=settings()
        with patch.object(sys,'argv',['launch.py','--dry-run','--restart']),patch.object(launch,'load',return_value=s), \
             patch.object(launch,'run',side_effect=AssertionError('No process/network')), \
             patch.object(launch,'stop',side_effect=AssertionError('No stop')),contextlib.redirect_stdout(io.StringIO()) as output:
            launch.main()
        self.assertFalse(json.loads(output.getvalue())['changed'])
    def test_busy_gpu_prevents_download_or_sync(self):
        with patch.object(launch,'run',return_value=subprocess.CompletedProcess([],0,stdout=b'42\n')) as cmd:
            with self.assertRaisesRegex(ValueError,'GPU busy'):launch.prepare(settings(),{'runtime':{'revision':'x'}})
        self.assertEqual(cmd.call_count,1)
    def test_network_command_is_quoted(self):
        with patch.object(launch.subprocess,'run') as proc:
            launch.run(['python3','/path with spaces/script.py','literal;no-shell'],host='alice@worker')
        argv=proc.call_args.args[0]
        self.assertEqual(argv[-1],"python3 '/path with spaces/script.py' 'literal;no-shell'")


class Downloads(unittest.TestCase):
    def test_traversal_rejected(self):
        for name in ('../escape','/absolute','x/../escape','x\\y','x//y'):
            with self.subTest(name=name),self.assertRaises(ValueError):bootstrap.relative(name)
    def test_extract_refuses_links(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);archive=d/'bad.tar'
            with tarfile.open(archive,'w') as t:
                row=tarfile.TarInfo('link');row.type=tarfile.SYMTYPE;row.linkname='/tmp/out';t.addfile(row)
            with self.assertRaisesRegex(ValueError,'No links'):bootstrap.extract(archive,d/'result')
    def test_existing_corrupt_download_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'asset';p.write_bytes(b'bad')
            with patch.object(bootstrap.urllib.request,'urlopen',side_effect=AssertionError('Offline')):
                with self.assertRaisesRegex(ValueError,'differs'):bootstrap.download('https://example.invalid',p,'a'*64)
    def test_reuse_valid_download_offline(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'asset';p.write_bytes(b'ok')
            with patch.object(bootstrap.urllib.request,'urlopen',side_effect=AssertionError('Offline')):
                bootstrap.download('https://example.invalid',p,hashlib.sha256(b'ok').hexdigest(),2)


class ReleasePayload(unittest.TestCase):
    def test_lock_manifest(self):
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        verify=launch.module(ROOT/'release/runtime/verify.py','bundle_verifier')
        self.assertGreater(verify.verify(ROOT/'release/runtime',lock['kit_manifest_sha256'])['files'],100)
    def test_public_weight_pins(self):
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        for name in ('model','draft'):
            self.assertRegex(lock[name]['revision'],r'^[0-9a-f]{40}$')
            self.assertRegex(lock[name]['manifest_sha256'],r'^[0-9a-f]{64}$')
    def test_payload_has_corresponding_sources_and_license(self):
        for name in ('LICENSE','CREDITS.md','THIRD_PARTY_NOTICES.md','release/runtime/sources/cooperative24.cu',
                     'release/runtime/sources/cooperative24.cuh','release/runtime/sources/display_kv.c'):
            self.assertTrue((ROOT/name).is_file(),name)
        self.assertIn('GNU AFFERO GENERAL PUBLIC LICENSE',(ROOT/'LICENSE').read_text())
    def test_python_syntax(self):
        for path in (ROOT/'release').glob('*.py'):ast.parse(path.read_text(),filename=str(path))


if __name__=='__main__':unittest.main()
