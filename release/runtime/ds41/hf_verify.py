"""Read-only, source-anchored checks of a complete HF weight candidate.

Does not use the packer's layout or indexes as the expected tensor inventory.
Receipts are checked against actual shard bytes AND original/selected tensors.
Integrity is not full-model accuracy, serving qualification, or upload approval.
"""
import copy
import fcntl
import json
from pathlib import Path, PurePosixPath
import re
import stat

from .hf_package import PUBLIC_METADATA, metadata_identity, source_catalog
from .quantized_manifest import ROOT, REVISION, check_shape, checksum, digest, read_manifest
from .safetensor_pack import fingerprint, read_slices, tensor_hash, unique_object, verify_shard

CODE_FILES = {'scripts/pack_hf_candidate.py', 'ds41/hf_package.py',
              'ds41/safetensor_pack.py', 'ds41/quantized_manifest.py', 'ds41/coverage_bank.py',
              'ds41/baseline_subset.py', 'ds41/calibration_inputs.py', 'ds41/calibration_inputs_v2.py',
              'scripts/verify_supplemented.py', 'scripts/verify_code_supplemented.py'}
EXTRA_METADATA = {'README.md', 'config.json', 'model.safetensors.index.json',
    'provenance/source-config.json', 'provenance/source-index.json',
    'provenance/selected-experts.json', 'provenance/pack-plan.json',
    'engrams/config.json', 'engrams/model.safetensors.index.json',
    'draft/config.json', 'draft/model.safetensors.index.json'}


def relative_path(name):
    if (not isinstance(name, str) or not name or '\\' in name or '\0' in name
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == '.'):
        raise ValueError('Noncanonical or escaping package path')
    return name


def read_json(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError('Excessively large package JSON metadata')
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    with path.open() as stream:
        return json.load(stream, object_pairs_hook=unique_object, parse_constant=reject_constant)


def file_inventory(package):
    package = Path(package).absolute()
    if package.resolve() != package or not package.is_dir():
        raise ValueError('Package must be a real directory, not a redirected path')
    result = {}
    for path in package.rglob('*'):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Symlinks and nonregular files are not permitted in a package')
        result[relative_path(path.relative_to(package).as_posix())] = fingerprint(path)
    if '.pack.lock' in result and result['.pack.lock'][2] != 0:
        raise ValueError('Unexpected content in the optional packing lock')
    return result


def shard_component(name):
    relative_path(name)
    if re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors', name):
        return 'active'
    if re.fullmatch(r'engrams/engram-layer-(01|14)\.safetensors', name):
        return 'engrams'
    if re.fullmatch(r'draft/model-\d{5}-of-\d{5}\.safetensors', name):
        return 'draft'
    raise ValueError(f'Unexpected package shard path: {name}')


def check_metadata(package, selection, source_hashes, public_metadata, source_config,
                   source_index, expected_readme, implementation_hashes):
    """Cross-check all public metadata; supplied identities are trusted inputs."""
    before = file_inventory(package)
    report = read_json(package / 'package-manifest.json')
    fixed = dict(format='ds41_hf_weight_candidate_v1', status='complete_inventory_unqualified',
        source_revision=REVISION, selected_manifest_sha256=checksum(selection),
        vision_precision='original_BF16_byte_preserved', original_routed_weights_in_active_index=False,
        engrams='unchanged_FP8_SSD_sidecar', draft='unchanged_SSD_sidecar_not_loaded',
        quality_validated=False, tp2_dcp2_serving_validated=False, uploaded=False)
    if (set(report) != set(fixed) | {'pack_plan_sha256', 'shards', 'metadata_sha256', 'component_payload_bytes'}
            or any(report.get(key) != value or type(report.get(key)) is not type(value) for key, value in fixed.items())):
        raise ValueError('Unexpected candidate manifest schema, identity or qualification claim')
    metadata_files = (set(PUBLIC_METADATA) & set(public_metadata)) | EXTRA_METADATA
    if set(report['metadata_sha256']) != metadata_files:
        raise ValueError('Missing or unexpected public metadata inventory')
    for name, sha in report['metadata_sha256'].items():
        if digest(package / relative_path(name)) != sha:
            raise ValueError(f'Changed package metadata: {name}')
    plan = read_json(package / 'provenance/pack-plan.json')
    fixed_plan = dict(format='ds41_hf_candidate_pack_plan_v1', source_revision=REVISION,
        selected_manifest_sha256=checksum(selection), source_shards_sha256=source_hashes,
        source_metadata_sha256=public_metadata, implementation_sha256=implementation_hashes)
    if (set(plan) != set(fixed_plan) | {'shard_payload_target_bytes', 'shards'}
            or any(plan.get(key) != value for key, value in fixed_plan.items())
            or report['pack_plan_sha256'] != checksum(plan)
            or type(plan['shard_payload_target_bytes']) is not int
            or plan['shard_payload_target_bytes'] not in [i * 1024**3 for i in range(1, 9)]):
        raise ValueError('Packing plan differs from the selected source or recipe identity')
    if read_json(package / 'provenance/selected-experts.json') != dict(manifest=selection, manifest_sha256=checksum(selection)):
        raise ValueError('Packaged expert selection is not the independently selected inventory')
    for name in set(PUBLIC_METADATA) & set(public_metadata):
        if digest(package / name) != public_metadata[name]:
            raise ValueError(f'Pinned public metadata was changed: {name}')
    for name in ('provenance/source-config.json', 'engrams/config.json', 'draft/config.json'):
        if (package / name).read_bytes() != source_config:
            raise ValueError('Original config was changed in provenance or an SSD sidecar')
    if (package / 'provenance/source-index.json').read_bytes() != source_index:
        raise ValueError('Original source index was changed')
    expected_config = copy.deepcopy(json.loads(source_config))
    expected_config['quantization_config'] = dict(quant_method='ds41_exl3', codebook='mul1',
        layer_bits=[3] * 40, target_bpw=3.0, scope='deepseek_v41_backbone_routed_experts')
    if read_json(package / 'config.json') != expected_config:
        raise ValueError('Active model config changed beyond the intended routed quantization')
    if (package / 'README.md').read_bytes() != expected_readme:
        raise ValueError('Model card is not the reviewed unqualified candidate template')
    shards, receipts = set(), set()
    for item in report['shards']:
        if set(item) != {'file', 'bytes', 'sha256', 'payload_bytes', 'tensors', 'receipt', 'receipt_sha256'}:
            raise ValueError('Unexpected shard inventory fields')
        name = relative_path(item['file'])
        shard_component(name)
        receipt = 'provenance/shards/' + name.replace('/', '__') + '.json'
        if name in shards or item['receipt'] != receipt:
            raise ValueError('Duplicate shard or noncanonical receipt path')
        shards.add(name)
        receipts.add(receipt)
        if digest(package / receipt) != item['receipt_sha256']:
            raise ValueError('Shard receipt checksum changed')
        if plan['shards'].get(name) != dict(tensors=item['tensors'], payload_bytes=item['payload_bytes']):
            raise ValueError('Shard inventory differs from the pack plan')
    if shards != set(plan['shards']) or not shards:
        raise ValueError('Missing or extra shards relative to the pack plan')
    allowed = metadata_files | shards | receipts | {'package-manifest.json'}
    if set(before) - {'.pack.lock'} != allowed:
        raise ValueError('Package contains missing or unexpected files; do not upload this directory')
    for component, prefix in (('active', ''), ('draft', 'draft/')):
        names = sorted(name for name in shards if shard_component(name) == component)
        expected = [f'{prefix}model-{i:05d}-of-{len(names):05d}.safetensors' for i in range(1, len(names) + 1)]
        if not names or names != expected:
            raise ValueError('Noncontiguous or mislabeled model shard numbering')
    if {name for name in shards if shard_component(name) == 'engrams'} != {
            'engrams/engram-layer-01.safetensors', 'engrams/engram-layer-14.safetensors'}:
        raise ValueError('Both SSD engram layers must be present')
    return report, before


def check_component(package, component, expected, entries, *, payloads=True, progress=None):
    """Check one index against actual headers and independent source slices.

    This primitive also supports bounded test fixtures. The public verifier
    below always supplies the complete40-layer source and selected inventory.
    """
    if component not in ('active', 'engrams', 'draft'):
        raise ValueError('Unknown weight component')
    directory = package if component == 'active' else package / component
    index = read_json(directory / 'model.safetensors.index.json')
    expected = list(expected)
    reference = {tensor.name: tensor for tensor in expected}
    if len(reference) != len(expected) or not reference:
        raise ValueError('Empty or duplicate expected tensor inventory')
    total = sum(tensor.nbytes for tensor in expected)
    if (set(index) != {'metadata', 'weight_map'} or index['metadata'] != {'total_size': total}
            or set(index['weight_map']) != set(reference)):
        raise ValueError('Index differs from the independent complete tensor inventory')
    actual_map, checked_bytes, sources = {}, 0, {}
    for tensor in expected:
        if tensor.path in sources and sources[tensor.path] != tensor.source_fingerprint:
            raise ValueError('Inconsistent source fingerprint in the expected inventory')
        sources[tensor.path] = tensor.source_fingerprint
    for item in entries:
        if shard_component(item['file']) != component:
            raise ValueError('Shard assigned to the wrong component')
        path = package / item['file']
        receipt = read_json(package / relative_path(item['receipt']))
        if (set(receipt) != {'file', 'bytes', 'sha256', 'payload_bytes', 'tensors'}
                or receipt['file'] != path.name
                or receipt['sha256'] != item['sha256'] or receipt['bytes'] != item['bytes']
                or receipt['payload_bytes'] != item['payload_bytes'] or len(receipt['tensors']) != item['tensors']):
            raise ValueError('Receipt does not reconcile with the package manifest')
        actual = verify_shard(path, receipt) if payloads else read_slices(path)
        if (len(actual) != item['tensors'] or sum(t.nbytes for t in actual) != item['payload_bytes']
                or path.stat().st_size != item['bytes']):
            raise ValueError('Actual shard size/count differs from the manifest')
        receipt_tensors = {entry['name']: entry for entry in receipt['tensors']}
        if len(receipt_tensors) != len(actual) or set(receipt_tensors) != {t.name for t in actual}:
            raise ValueError('Duplicate or missing receipt tensor')
        for tensor in actual:
            original = reference.get(tensor.name)
            if (tensor.name in actual_map or original is None
                    or (tensor.dtype, tensor.shape, tensor.nbytes) != (original.dtype, original.shape, original.nbytes)):
                raise ValueError('Missing, duplicated, unexpected or retyped actual tensor')
            entry = receipt_tensors[tensor.name]
            if (set(entry) != {'name', 'dtype', 'shape', 'bytes', 'sha256'}
                    or (entry['dtype'], entry['shape'], entry['bytes']) != (tensor.dtype, list(tensor.shape), tensor.nbytes)
                    or not isinstance(entry['sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', entry['sha256'])):
                raise ValueError('Receipt tensor metadata differs from the actual header')
            if payloads and tensor_hash(original) != receipt_tensors[tensor.name]['sha256']:
                raise ValueError(f'Packed bytes differ from their actual selected/source tensor: {tensor.name}')
            actual_map[tensor.name] = path.name
            checked_bytes += tensor.nbytes
        if progress:
            progress(dict(stage='hf_component_shard_checked', component=component, file=item['file'], payloads_checked=payloads))
    if actual_map != index['weight_map'] or checked_bytes != total:
        raise ValueError('Index maps tensors to absent/wrong shards or omits actual tensors')
    if any(fingerprint(path) != value for path, value in sources.items()):
        raise ValueError('An original/selected tensor source changed during verification')
    return dict(tensors=len(actual_map), payload_bytes=checked_bytes, payloads_checked=payloads)


def verify_package(package, manifest_path, *, preflight_only=False, progress=None):
    """Verify the whole candidate against this checkout's pinned originals."""
    from scripts.pack_hf_candidate import model_card
    package = Path(package).absolute()
    before = file_inventory(package)
    # Shared lock excludes the packer. Uploaded copies can omit its empty lock;
    # immutable file fingerprints are still checked over the entire audit.
    lock = (package / '.pack.lock').open('rb') if '.pack.lock' in before else None
    try:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        stored = read_json(Path(manifest_path))
        check_shape(stored['manifest'])
        selected_fingerprints = {ROOT / item['path']: fingerprint(ROOT / item['path'])
            for layer in stored['manifest']['layers'] for item in layer['artifacts']}
        selection, plans = read_manifest(manifest_path)
        if stored != dict(manifest=selection, manifest_sha256=checksum(selection)):
            raise ValueError('Selected manifest changed or contains unreviewed wrapper fields')
        if set(selection) != {'format', 'source_revision', 'bits', 'quality_status', 'capture_sha256',
                              'corpus_sha256', 'source_verification_sha256', 'layers'}:
            raise ValueError('Unreviewed public selection fields')
        source, source_hashes, groups = source_catalog()
        source_fingerprints = {tensor.path: tensor.source_fingerprint for tensors in groups.values() for tensor in tensors}
        packed = [tensor for plan in plans for artifact in plan.artifacts for tensor in read_slices(artifact.path)]
        if any(t.source_fingerprint != selected_fingerprints[t.path] for t in packed):
            raise ValueError('Selected payload identity changed after manifest verification')
        public_metadata = metadata_identity(source)
        report, frozen = check_metadata(package, selection, source_hashes, public_metadata,
            (source / 'config.json').read_bytes(), (source / 'model.safetensors.index.json').read_bytes(),
            model_card().encode(), {name: digest(ROOT / name) for name in CODE_FILES})
        if before != frozen:
            raise ValueError('Package changed during metadata verification')
        if not preflight_only:
            for name, expected_sha in source_hashes.items():
                if digest(source / name) != expected_sha:
                    raise ValueError(f'Original source differs from its pinned checksum: {name}')
                if progress:
                    progress(dict(stage='hf_original_source_verified', file=name))
        if len(plans) != 40 or len(packed) != 40 * 384 * 12:
            raise ValueError('All40 complete selected EXL3 banks are required')
        expected = dict(active=groups['active_native'] + packed, engrams=groups['engrams'], draft=groups['draft'])
        components = {}
        for component, tensors in expected.items():
            entries = [item for item in report['shards'] if shard_component(item['file']) == component]
            components[component] = check_component(package, component, tensors, entries,
                payloads=not preflight_only, progress=progress)
        if report['component_payload_bytes'] != {name: value['payload_bytes'] for name, value in components.items()}:
            raise ValueError('Component byte accounting differs from the actual complete package')
        manifest_sha = digest(package / 'package-manifest.json')
        if file_inventory(package) != frozen or metadata_identity(source) != public_metadata:
            raise ValueError('Package or pinned metadata changed during the audit')
        if any(fingerprint(path) != value for path, value in {**selected_fingerprints, **source_fingerprints}.items()):
            raise ValueError('Original or selected source payloads changed during the whole-package audit')
        return dict(status='structure_checked_payloads_not_checked' if preflight_only else 'source_anchored_inventory_verified',
            package_manifest_sha256=manifest_sha,
            selected_manifest_sha256=checksum(selection), source_revision=REVISION,
            components=components, original_source_payloads_verified=not preflight_only,
            packed_payloads_compared_to_selected_sources=not preflight_only,
            gpu_used=False, quality_validated=False, tp2_dcp2_serving_validated=False, uploaded=False,
            scope='Complete weight inventory/integrity only; no full-model accuracy, generation or serving claim.')
    finally:
        if lock:
            lock.close()
