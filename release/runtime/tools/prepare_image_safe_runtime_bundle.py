"""Add the CPU-qualified image bootstrap to frozen v4; never rewrite old kits."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
from verify_runtime_bundle import read_regular, verify
from prepare_runtime_bundle import encoded, write_tar

BASE_SHA = '4891d0ed220108ee7db56cf6a25a2ffa9ee7a9eff0f6ebc077a15cce94cd28ca'
OVERRIDE = 'dd2b90570f6f42a70ce3b2b997e0027d98a6baa75889b441ae5c60c6443173aa'
BOOTSTRAP = '864de0d3fbb81f97cb5793343f7b559ec522c660bfe796ceac4eaf78800b6164'
ENTRYPOINT = '2948fd1c556c9c348193af1e49e0f357a1d5c3afc242fa8f7b2e51982a1e3d31'
PROBE = '4f4949ee954e4958235cd540f2fedb247e1070d0edfb6896e30bae7fa794df6a'
EVIDENCE = {
    'reports/ds41-image-safe-prefix-cpu-v2.json': '68bfeafde2acdfb386b2ddf7621754bc28b9f76763298cb46bf42b222e74fdb7',
    'reports/ds41-image-prefix-bootstrap-cpu-v1.json': '06d3416ee9c3582d5738fed72f3e38aa479b311c5f91ab601b8e20d85cf24988',
    'reports/image-prefix-native-v1/host0/complete.json': '586398df6f104f8394b920dffb781065225d6cf863dd86a836fc096292d94edd',
    'reports/image-prefix-native-v1/host1/complete.json': 'e38cc14c030c5620d43f7f74d52ee5cdbc39c8cdde79d44e735b39b2a651492f',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inspect_inputs():
    base = ROOT/'artifacts/ds41-runtime-inputs-v4'
    verify(base, BASE_SHA)
    old = json.loads(read_regular(base/'bundle-manifest.json'))
    entries = {name:read_regular(base/name) for name in old['files']}
    evidence = {}
    for name, expected in EVIDENCE.items():
        raw = read_regular(ROOT/name)
        if sha(raw) != expected:
            raise ValueError('Completed image-prefix evidence changed: '+name)
        evidence[name] = json.loads(raw)
    unit, loader, *native = evidence.values()
    if (unit['status'] != 'image_safe_prefix_cpu_pass' or len(unit['positive']) != 16
            or len(unit['refusals']) != 15 or unit['implementation_sha256'] != OVERRIDE
            or loader['status'] != 'image_prefix_bootstrap_cpu_pass'
            or len(loader['positive']) != 2 or len(loader['refusals']) != 6
            or loader['bootstrap_sha256'] != BOOTSTRAP or loader['override_sha256'] != OVERRIDE):
        raise ValueError('Incomplete image-safe lookup/bootstrap CPU qualification')
    for host, report in enumerate(native):
        terminal = json.loads(read_regular(ROOT/f'reports/image-prefix-native-v1/host{host}/terminal.json'))
        if (report['status'] != 'native_image_prefix_parent_spawn_cpu_pass'
                or report['probe_sha256'] != PROBE or report['weights_loaded']
                or report['gpu_devices_exposed'] or report['original_image_modified']
                or terminal['state']['Running'] or terminal['state']['OOMKilled']
                or terminal['state']['ExitCode'] != 0):
            raise ValueError('Native CPU probe did not finish cleanly')
        for process in (report['parent'], report['child']):
            if (process['source_sha256'] != OVERRIDE or process['torch_cuda_initialized']
                    or not process['native_registration'] or not process['entrypoint_bootstrap_executed']
                    or process['source_file'] != '/opt/ds41-serving/vision_inputs_override.py'
                    or [row['hit'] for row in process['cases']] != [0, 32512, 1280, 0]):
                raise ValueError('Native parent/spawn did not exercise the reviewed fix')
    provenance = dict(old['source_provenance'])
    staged = ROOT/'artifacts/ds41-image-prefix-probe-v2'
    for name, expected in [('serve.py', ENTRYPOINT), ('spark_image_prefix.py', BOOTSTRAP),
                           ('vision_inputs_override.py', OVERRIDE)]:
        raw = read_regular(staged/name)
        if sha(raw) != expected:
            raise ValueError('Staged native-tested overlay input changed: '+name)
        target = 'serving/'+name
        entries[target] = raw
        provenance[target] = dict(source=str((staged/name).relative_to(ROOT)), anchor_sha256=expected,
            native_cpu_parent_spawn_qualified_both_hosts=True, gpu_qualified=False)
    overlay = {name.removeprefix('serving/'):sha(raw) for name,raw in entries.items()
               if name.startswith('serving/') and name != 'serving/overlay-manifest.json'}
    assert len(overlay) == 12
    entries['serving/overlay-manifest.json'] = encoded(overlay)
    for target, source in (
            ('verify.py', 'probes/verify_runtime_bundle.py'),
            ('README.md', 'docs/runtime-input-bundle-v5.md'),
            ('bootstrap/check_image_prefix_native.py', 'probes/check_image_prefix_native.py'),
            ('tools/prepare_image_safe_runtime_bundle.py', 'scripts/prepare_image_safe_runtime_bundle.py')):
        raw = read_regular(ROOT/source)
        if target.startswith('bootstrap/') and sha(raw) != PROBE:
            raise ValueError('Native probe source changed after qualification')
        entries[target] = raw
        provenance[target] = dict(source=source, currently_hashed=True)
    requirements = json.loads(entries['runtime-requirements.json'])
    inherited = {key:requirements[key] for key in (
        'mxfp8_aot_serving_text_vision_32k_verified', 'mxfp8_aot_serving_1m_verified',
        'mxfp8_aot_serving_1m_reuse_verified')}
    requirements.update(format='ds41_runtime_requirements_v5',
        inherited_v21_serving_qualification=inherited,
        **{key:False for key in inherited},
        image_prefix_override=dict(module='ds41.vllm_vision_inputs',
            file='serving/vision_inputs_override.py', sha256=OVERRIDE,
            bootstrap_file='serving/spark_image_prefix.py', bootstrap_sha256=BOOTSTRAP,
            entrypoint_sha256=ENTRYPOINT, native_cpu_parent_spawn_verified_both_hosts=True,
            gpu_serving_qualified=False),
        portable_release_asset_deployment_verified=False)
    entries['runtime-requirements.json'] = encoded(requirements)
    for generated in ('serving/overlay-manifest.json', 'runtime-requirements.json'):
        provenance[generated] = dict(generated_from_verified_v4=True, image_prefix_evidence_sha256=EVIDENCE)
    manifest = dict(format='ds41_runtime_inputs_v5', standalone_runtime=False,
        clean_rebuild_qualified=False, publication_approved=False,
        base_v4_manifest_sha256=BASE_SHA, image_prefix_evidence_sha256=EVIDENCE,
        files={name:dict(bytes=len(raw),sha256=sha(raw)) for name,raw in sorted(entries.items())},
        source_provenance=provenance,
        missing=['GPU text/vision/cache/1M qualification of the new image-safe overlay',
                 'runtime/cache payload privacy/licensing review before public distribution',
                 'final public model metadata/qualification review'])
    entries['bundle-manifest.json'] = encoded(manifest)
    return entries, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (256*2**20, 256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    entries, manifest = inspect_inputs()
    output = ROOT/'artifacts/ds41-runtime-inputs-v5.tar'
    receipt = ROOT/'reports/ds41-runtime-inputs-v5.json'
    report = dict(status='image_safe_runtime_bundle_planned', files=len(manifest['files']),
        payload_bytes=sum(row['bytes'] for row in manifest['files'].values()),
        manifest_sha256=sha(entries['bundle-manifest.json']), base_manifest_sha256=BASE_SHA,
        image_prefix_evidence_sha256=EVIDENCE, gpu_fix_qualified=False,
        publication_approved=False, original_v4_modified=False)
    if args.execute:
        if output.exists() or receipt.exists() or shutil.disk_usage(ROOT).free < 32*2**30:
            raise ValueError('Preserve prior outputs and32GiB disk reserve')
        write_tar(entries, output)
        after, _ = inspect_inputs()
        if entries != after:
            raise ValueError('Inputs changed during packaging; preserve and inspect archive')
        report.update(status='runtime_input_bundle_prepared', archive=output.name,
            archive_bytes=output.stat().st_size, archive_sha256=sha(output.read_bytes()))
        with receipt.open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
