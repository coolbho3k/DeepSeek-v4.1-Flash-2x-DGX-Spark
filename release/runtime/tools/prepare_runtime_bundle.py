"""Package verified runtime inputs, without Docker, ML imports, weights or uploads.

The bundle is intentionally not advertised as a complete runtime. Original
build/serving receipts anchor its code and binary; it includes no private
operational receipts, calibration inputs, compiled object files or model data.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import resource
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
from verify_runtime_bundle import MAX_TOTAL, read_regular, relative

ANCHORS = {
    'reports/runtime-overlay-v3-build/complete.json':
        '86f6aec8e4940b3316f61106431f1f18bef292ca9dece53c21686be1682c9512',
    'artifacts/ds41-serving-long-v21-overlay/overlay-manifest.json':
        'af96b6221c69495f222b5c2c54ab2bf289d06e825f867f507e0d5028a2c893f0',
    'artifacts/exl3-moe-mul1-build-v1/complete.json':
        '067638fbf1d2c69e2793b27f47e6f7b1c804df2aa7987f4430e0b695b8ba457a',
}
QUALIFIED = {
    'artifacts/ds41-mxfp8-aot-v1/receipt.json':
        '001005ecd14d86380fd4e39d06a2d283321d522c9472b56678674064d7e01381',
    'reports/mxfp8-clean-gpu-v1/complete.json':
        'e28d12004fb5d4d49776e860ba3641c68bc1c52d16b673d684e377a39c2a76e5',
    'reports/ds41-serving-long-v21/aot-runtime.json':
        '45035ba4c8d54c902f715f307ec9375765ebad6a01d0a8568ce945ecb999ad2d',
    'reports/ds41-serving-long-v21-generation-v1.json':
        '252828403a05f0e40dda7780da39780d2b3e1e747666353c6ba86807961d2b50',
    'reports/ds41-serving-long-v21-32k-v1.json':
        '3ef4280c00cbb8a64428170bd455ffb48f74f65eb2307799ec73aed93a44d88c',
    'reports/ds41-runtime-image-identity-v1.json':
        '65b3fd03157cf00eb8f1f7c535a84c3686a4bb4a52da92a9a085c461515fa17d',
    'reports/ds41-download-verifier-v1.json':
        'edd2a2d688ac68bae4201d56bec5cd245c45905cc1e723dd33898ac8e502e747',
}
DISTRIBUTION = {
    'reports/ds41-serving-long-v21-completion-audit-v1.json':
        '309867899067578464312461715a4a05bd29fd8a98baa29cdeb79ddb44fa1026',
    'reports/ds41-portable-serving-cpu-v6.json':
        '8c93e9a16d0137ca763e46bb4f41287fa002235b5924f41acfbf1a0fd35de2d1',
    'reports/ds41-runtime-cache-v1.json':
        '475eb487d5f8f8fd65594778a5c702ce75a8f4ca90c8e75c9e87ef571ac96ca3',
    'reports/ds41-runtime-cache-v1-extraction.json':
        'f4b8e13506fbf136863a04cc5241bf8140b2a5a474052a272f88d32af6cc2f6a',
    'reports/ds41-runtime-image-export-v1.json':
        'd7d233cb104a0e480ec642304c8d7594eb1ed651be85b6751efaf4a30e7a4441',
    'reports/ds41-runtime-image-export-cpu-v1.json':
        'a6e3ba9006606c2fead725aab639f64ad14db568d7b806daacaa61e190bd26e7',
    'reports/ds41-runtime-cache-bundle-cpu-v1.json':
        '07bd4665586639f470baab0dfa3df4f9a96ec92083251975e9c6559d231b223e',
}
BOUND = {
    'reports/ds41-runtime-archive-import-v1.json':
        '0e5a53687214ae30e7a7376d6c65cc429d31c659a2b0dd7e1d09a37fffe26894',
    'reports/ds41-runtime-model-view-v1-full-host0.json':
        '0487f7cf0f8640cdc5c459c7feedf8e3e6bcc23f0315ea2f2062e69de0b67dd8',
    'reports/ds41-runtime-model-view-v1-full-host1.json':
        '5a7b2624b3aa119f873f14d0e43ad73642b15e00faacd0e1d047e4033b367bce',
    'reports/ds41-bound-container-view-v1-host0.json':
        'a2e8965fe7de16b9a6a07d56f1a4405c79e5d56dbb0a87fdf883519b6026429c',
    'reports/ds41-bound-container-view-v1-host1.json':
        '367466b357a3afffc5208124ef10c1fa85a1b0ffbe10a21bda1cb4923778532f',
    'reports/ds41-mapped-release-cpu-v1.json':
        '2a6483ef3921b56024300c6b5cef7b2159a26050d801e0399519af7d10ea9742',
    'reports/ds41-runtime-archive-import-cpu-v1.json':
        '681c47e497d9b5a4bb62998ea997a02c382d169efb0fbab173fc10a5dce5ef69',
}
OVERLAY = ROOT/'artifacts/ds41-serving-long-v21-overlay'
KERNEL = ROOT/'artifacts/exl3-moe-mul1-build-v1'
BAKED = ROOT/'artifacts/ds41-runtime-baked-inputs-v1'
KERNEL_SHA = '66de4aa31e49462fd00fb4b26ebd4e16dfd995b631157e3b0194c27fc1239342'
AOT_SHA = '6abdf60fb353da15d87030427e16a982e7819a6f479563b8acfde81110e78bf4'
IMAGE_IDENTITY_SHA = '03c151b169249d413dc64a365d3fa8f5c104561d1eb3bf2bd30138b9010c3e3b'
NOTICE_SHA = {
    'FLASHINFER-LICENSE': 'f3514f24f2b94d09c409abbf4020cf600e4059098c697110eb91a4a2b235fbb7',
    'CUTLASS-cutlass.h': '3f83df4d3164afea3a865a5e93d7c1aff3cfdd7c604319571a94332b4e9ec8da',
    'FLASHINFER-mxfp8_gemm_cutlass_sm120.cu': '896676c7f8f8238958f3c918c611963a0630729a664f82511ed509e57bb6c962',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def inspect_inputs():
    receipts = {}
    for name, expected in (ANCHORS | QUALIFIED | DISTRIBUTION | BOUND).items():
        raw = read_regular(ROOT/name)
        if sha(raw) != expected:
            raise ValueError('Original runtime evidence changed: '+name)
        receipts[name] = json.loads(raw)
    runtime, overlay, kernel = [receipts[name] for name in ANCHORS]
    if (runtime['status'] != 'built_and_source_manifest_verified'
            or kernel['status'] != 'additive_mul1_kernel_built_cpu_only'
            or kernel['binary_sha256'] != KERNEL_SHA
            or overlay.get('ds41_moe_mul1_v1.so') != KERNEL_SHA or len(overlay) != 10):
        raise ValueError('Unexpected original runtime/kernel identity')
    aot, gpu, mapped, generation, retrieval, images, verifier = [receipts[name] for name in QUALIFIED]
    if (aot['status'] != 'mxfp8_debug_stripped_runtime_unchanged'
            or aot['library_sha256'] != AOT_SHA or aot['source_preserved'] is not True
            or gpu['status'] != 'both_clean_mxfp8_aot_gpu_diagnostics_pass'
            or gpu['library_sha256'] != AOT_SHA or gpu['cases_per_host'] != 36
            or gpu['outputs_identical'] is not True
            or mapped['status'] != 'both_serving_workers_map_qualified_aot_library'
            or mapped['library_sha256'] != AOT_SHA or mapped['jit_fallback_observed'] is not False
            or generation['status'] != 'serial_generation_diagnostics_complete'
            or retrieval['status'] != 'retrieval_pass'
            or images['status'] != 'both_installed_runtime_images_match_public_identity'
            or len(images['actual']) != 2
            or any(row['identity_sha256'] != IMAGE_IDENTITY_SHA for row in images['actual'])
            or verifier['status'] != 'standalone_release_verifier_contracts_pass'):
        raise ValueError('Missing v21 AOT serving or portable-verifier evidence')
    long, portable, cache, extracted, exported, export_test, cache_test = [receipts[name] for name in DISTRIBUTION]
    if (long['status'] != 'v21_same_pair_1m_reuse_and_regression_audited'
            or long['all_29_replies_identical_to_v21_pre1m_and_v20'] is not True
            or portable['status'] != 'portable_serving_cpu_contracts_pass'
            or portable['refusal_cases'] != 69 or portable['actual_public_deployment_tested'] is not False
            or cache['status'] != 'compiler_cache_archive_prepared'
            or extracted['status'] != 'compiler_cache_archive_extracted_verified'
            or cache['archive_sha256'] != extracted['archive_sha256']
            or cache['manifest_sha256'] != extracted['manifest_sha256']
            or cache['files'] != 12957 or extracted['mtime_preserved'] is not True
            or exported['status'] != 'verified_installed_runtime_exported'
            or exported['image']['identity_sha256'] != IMAGE_IDENTITY_SHA
            or export_test['status'] != 'runtime_image_export_cpu_contracts_pass'
            or cache_test['status'] != 'runtime_cache_bundle_cpu_contracts_pass'):
        raise ValueError('Missing qualified1M/controller or transport artifact evidence')
    imported = receipts['reports/ds41-runtime-archive-import-v1.json']
    if (imported['status'] != 'runtime_archive_imported_and_public_identity_verified'
            or imported['archive_sha256'] != exported['sha256']
            or imported['installed']['identity_sha256'] != IMAGE_IDENTITY_SHA):
        raise ValueError('Exact runtime archive import is not verified')
    for index in (0, 1):
        full_name = f'reports/ds41-runtime-model-view-v1-full-host{index}.json'
        full = receipts[full_name]
        view = receipts[f'reports/ds41-bound-container-view-v1-host{index}.json']
        if (full['status'] != 'all_public_release_payloads_sha256_verified_via_bindings'
                or full['files'] != 129 or full['bytes_hashed'] != 426134193164
                or full['full_payload_hash_verified'] is not True
                or view['status'] != 'actual_container_public_bound_view_verified'
                or view['prior_full_receipt_sha256'] != BOUND[full_name]
                or view['files'] != 129 or view['bound_shards'] != 55
                or view['readonly_model_mounts'] != 56
                or view['host_source_fingerprints_identical'] is not True
                or view['source_fingerprints_unchanged_after_probe'] is not True):
            raise ValueError('Complete same-host bound-view verification is required')
    entries, sources = {}, {}

    def add(name, path, expected=None):
        relative(name)
        if name in entries:
            raise ValueError('Duplicate bundle destination: '+name)
        raw = read_regular(path)
        if expected is not None and sha(raw) != expected:
            raise ValueError('Source differs from its qualified receipt: '+str(path))
        entries[name] = raw
        sources[name] = dict(source=path.relative_to(ROOT).as_posix(),
                             anchor_sha256=expected, currently_hashed=True)
        if sum(map(len, entries.values())) > MAX_TOTAL:
            raise ValueError('Input bundle exceeds its small-file budget')

    # Preserve the complete baked Python inventory, not a guessed import subset.
    for name, expected in sorted(runtime['source_sha256'].items()):
        if not (re.fullmatch(r'ds41/[a-z0-9_]+\.py', name)
                or name in ('pyproject.toml', 'docker/Dockerfile.runtime-overlay')):
            raise ValueError('Unexpected baked source category')
        add(name, BAKED/name, expected)
    for name, expected in sorted(overlay.items()):
        add('serving/'+relative(name), OVERLAY/name, expected)
    add('serving/overlay-manifest.json', OVERLAY/'overlay-manifest.json',
        ANCHORS['artifacts/ds41-serving-long-v21-overlay/overlay-manifest.json'])
    # Original inputs allow regenerating the adaptations; generated inputs
    # preserve exactly what the qualified binary was built from, with license.
    for name, expected in sorted(kernel['source_sha256'].items()):
        if not (name.startswith('vendor/exllamav3/') or name.startswith('kernels/exl3_moe_mul1/')
                or name == 'scripts/build_exl3_moe_mul1.py'):
            raise ValueError('Unexpected kernel source category')
        add('kernel-rebuild/'+relative(name), ROOT/name, expected)
    for name, expected in sorted(kernel['generated_sha256'].items()):
        if not (name.startswith('include/') or name in (
                'LICENSE', 'bindings.cpp', 'launch.cu', 'semantics.cuh', 'build_exl3_moe_mul1.py')):
            raise ValueError('Unexpected generated source category')
        add('kernel-generated/'+relative(name), KERNEL/name, expected)
    for source, target in (
            ('probes/generate_mxfp8_graph_cpu.py', 'bootstrap/generate_mxfp8_graph_cpu.py'),
            ('probes/check_mxfp8_graph_contract.py', 'bootstrap/check_mxfp8_graph_contract.py'),
            ('probes/check_seeded_mxfp8_graph.py', 'bootstrap/check_seeded_mxfp8_graph.py'),
            ('probes/prepare_mxfp8_aot.py', 'bootstrap/prepare_mxfp8_aot.py'),
            ('probes/check_mxfp8_elf_identity.py', 'bootstrap/check_mxfp8_elf_identity.py'),
            ('probes/check_clean_mxfp8_gpu.py', 'bootstrap/check_clean_mxfp8_gpu.py'),
            ('probes/verify_runtime_bundle.py', 'verify.py'),
            ('docs/runtime-input-bundle-v4.md', 'README.md'),
            ('docs/portable-deployment.md', 'docs/portable-deployment.md'),
            ('docs/downloaded-release-verification.md', 'docs/downloaded-release-verification.md'),
            ('scripts/verify_downloaded_release.py', 'tools/verify_downloaded_release.py'),
            ('scripts/verify_runtime_image.py', 'tools/verify_runtime_image.py'),
            ('scripts/advise_verified_release_cache.py', 'tools/advise_verified_release_cache.py'),
            ('scripts/prepare_runtime_bundle.py', 'tools/prepare_runtime_bundle.py')):
        add(target, ROOT/source)
    for name in ('portable_node.py','portable_pair.py'):
        add('tools/'+name, ROOT/'serving'/name, portable['source_sha256']['serving/'+name])
    add('deployment.example.json',ROOT/'serving/deployment.example.json')
    add('tools/export_runtime_image.py',ROOT/'scripts/export_runtime_image.py',export_test['exporter_sha256'])
    add('tools/runtime_cache_bundle.py',ROOT/'scripts/runtime_cache_bundle.py',cache_test['source_sha256'])
    add('tools/verify_mapped_release.py',ROOT/'scripts/verify_mapped_release.py',
        '7788d874a68294ef2df0b0a3ba63d39a70953dad4596c6a7536b561934d568ca')
    add('tools/import_runtime_archive.py',ROOT/'scripts/import_runtime_archive.py',
        '052bba3801bdd25aebd5e597e6636cc05e6d88c40875ea95e5ff31c0804fdb48')
    add('auxiliary-cache-manifest.json',ROOT/'artifacts/ds41-runtime-cache-v1/cache-manifest.json',
        cache['manifest_sha256'])
    add('runtime-image-archive.json',ROOT/'artifacts/ds41-runtime-image-v1.tar.gz.json',
        '266d0e6bfa1a57fd34d17fd51781fe39e8d186011ddee79678f25427565ede1b')
    cache_descriptor = dict(format='ds41_auxiliary_cache_archive_v1',file='ds41-runtime-cache-v1.tar',
        bytes=cache['archive_bytes'],sha256=cache['archive_sha256'],
        manifest_sha256=cache['manifest_sha256'],files=cache['files'],
        payload_bytes=cache['payload_bytes'],mtime_preserved=True,
        publication_approved=False,physical_deployment_verified=False)
    entries['auxiliary-cache-archive.json'] = encoded(cache_descriptor)
    if (sha(entries['tools/verify_downloaded_release.py']) != verifier['verifier_sha256']
            or sha(entries['tools/verify_runtime_image.py']) != images['verifier_sha256']
            or sha(entries['bootstrap/check_clean_mxfp8_gpu.py']) != gpu['source_sha256']):
        raise ValueError('Bundled tool differs from its completed qualification')
    add('aot/mxfp8_gemm_cutlass_sm120/mxfp8_gemm_cutlass_sm120.so',
        ROOT/'artifacts/ds41-mxfp8-aot-v1/mxfp8_gemm_cutlass_sm120.so', AOT_SHA)
    add('runtime-image-identity.json', ROOT/'artifacts/ds41-runtime-image-identity-v1.json', IMAGE_IDENTITY_SHA)
    for name, expected in NOTICE_SHA.items():
        add('notices/'+name, ROOT/'artifacts/ds41-runtime-notices-v1'/name, expected)
    # No local paths, container inspection, raw outputs or image environment
    # are exported. These IDs describe already-tested inputs, not a registry.
    entries['runtime-requirements.json'] = encoded(dict(
        format='ds41_runtime_requirements_v4', platform='linux/arm64', cuda_arch='12.1a',
        vllm_commit='e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba',
        exllama_python_commit='6ff3a17ea7f3d0026b273d43239398d57f71b788',
        model_revision='df42c109f1defefcbfcedbe7d905718a12266e40',
        tested_local_image_id=runtime['image_id'], base_image_id=runtime['base_id'],
        public_runtime_image=None, dependency_artifact_lock=False,
        runtime_image_identity_sha256=IMAGE_IDENTITY_SHA,
        image_identity_verified_on_both_hosts=True,
        fused_kernel_sha256=KERNEL_SHA,
        mxfp8_library_sha256=AOT_SHA,
        mxfp8_native_graph_sha256='3aeab12adc90d94d85ca58e94c450a79eb82976d082e5cbbf60ac0c08da031dc',
        mxfp8_seeded_graph_sha256='de1e515a89b8942440260a70afeaab3a0205b12a9d0c93a77cd933bd50f2173c',
        mxfp8_native_empty_cache_component_build_qualified=True,
        mxfp8_aot_serving_text_vision_32k_verified=True,
        mxfp8_aot_serving_1m_verified=True, mxfp8_aot_serving_1m_reuse_verified=True,
        auxiliary_warm_caches_required=True,auxiliary_cache_archive=cache_descriptor,
        runtime_image_archive_file='ds41-runtime-image-v1.tar.gz',
        runtime_image_archive_sha256=exported['sha256'],runtime_image_archive_bytes=exported['bytes'],
        runtime_image_archive_import_verified=True,portable_controller_cpu_tested=True,
        bound_model_full_payload_hashes_verified_both_hosts=True,
        bound_model_cpu_container_views_verified_both_hosts=True,
        portable_release_asset_deployment_verified=False,
        safety=dict(gpu_utilization=.90, tensor_parallel=2, decode_context_parallel=2,
                    max_sequences=1, max_model_len=1048576, max_batched_tokens=1056,
                    kv_cap_bytes_per_rank=1009612800, cpu_memory_gib=8, swap_bytes=0),
        original_evidence_sha256=ANCHORS | QUALIFIED | DISTRIBUTION | BOUND))
    manifest = dict(format='ds41_runtime_inputs_v4', standalone_runtime=False,
        clean_rebuild_qualified=False, publication_approved=False,
        files={name:dict(bytes=len(raw), sha256=sha(raw)) for name, raw in sorted(entries.items())},
        source_provenance=sources,
        missing=['runtime/cache payload privacy/licensing review before public distribution',
                 'physical deployment using portable controller and only release assets',
                 'final public model metadata/qualification review'])
    entries['bundle-manifest.json'] = encoded(manifest)
    return entries, manifest


def write_tar(entries, output):
    # Fixed metadata and uncompressed USTAR make repeat output byte-identical.
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in sorted(entries.items()):
            relative(name)
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o644
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            archive.addfile(info, io.BytesIO(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Fresh artifacts/ds41-runtime-inputs-vN.tar; omit to plan')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (256*2**20, 256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    entries, manifest = inspect_inputs()
    summary = dict(status='runtime_input_bundle_planned', files=len(manifest['files']),
        payload_bytes=sum(row['bytes'] for row in manifest['files'].values()),
        manifest_sha256=sha(entries['bundle-manifest.json']),
        original_evidence_sha256=ANCHORS | QUALIFIED | DISTRIBUTION | BOUND, standalone_runtime=False,
        clean_rebuild_qualified=False, publication_approved=False, uploaded=False)
    if args.output is not None:
        output = args.output.absolute()
        if (output.resolve() != output or output.parent != ROOT/'artifacts'
                or not re.fullmatch(r'ds41-runtime-inputs-v[1-9][0-9]*\.tar', output.name)
                or output.exists()):
            raise ValueError('Use a fresh versioned runtime-input tar under artifacts')
        receipt = ROOT/'reports'/(output.stem+'.json')
        if receipt.exists() or receipt.is_symlink() or shutil.disk_usage(output.parent).free < 32*2**30:
            raise ValueError('Preserve existing outputs and the32GiB disk reserve')
        write_tar(entries, output)
        # Revalidate small original inputs after packaging, without touching
        # candidate weights, caches, inode links or live Docker state.
        after, _ = inspect_inputs()
        if after != entries:
            raise ValueError('Inputs changed during packaging; inspect the preserved tar')
        with output.open('rb') as stream:
            archive_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        summary.update(status='runtime_input_bundle_prepared', archive=output.name,
                       archive_bytes=output.stat().st_size, archive_sha256=archive_sha)
        with receipt.open('x') as stream:
            json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
