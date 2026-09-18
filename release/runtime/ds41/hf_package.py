"""Pinned source/HF storage layout for EXL3 candidates, with SSD sidecars."""
import copy
import hashlib
import json
from pathlib import Path
import re

from .quantized_manifest import ROOT, REVISION, digest
from scripts.run_quant_queue import tensor_inventory
from .safetensor_pack import group_shards, read_slices

EXPERT = re.compile(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$')
TABLE = re.compile(r'layers\.(1|14)\.engram\.embed\.(weight|scale)$')
PUBLIC_METADATA = ('.gitattributes', 'LICENSE', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja')


def metadata_identity(source):
    source = Path(source)
    metadata = json.loads((source / 'hub-metadata.json').read_text())
    if metadata['sha'] != REVISION:
        raise ValueError('Public metadata is not pinned to the selected source revision')
    entries = {entry['rfilename']: entry for entry in metadata['siblings']}
    result = {}
    for name in ('config.json', 'model.safetensors.index.json') + PUBLIC_METADATA:
        if name not in entries:
            if name in ('generation_config.json', 'chat_template.jinja') and not (source / name).exists():
                continue
            raise ValueError(f'Missing pinned public metadata: {name}')
        entry = entries[name]
        data = (source / name).read_bytes()
        expected = entry.get('lfs', {}).get('sha256')
        valid = hashlib.sha256(data).hexdigest() == expected if expected else (
            hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == entry['blobId'])
        if len(data) != entry['size'] or not valid:
            raise ValueError(f'Public metadata differs from its pinned Hub blob: {name}')
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def source_catalog(root=ROOT):
    root = Path(root).resolve()
    source = root / 'source' / REVISION
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    report = json.loads((root / 'reports/source-verification.json').read_text())
    if report['revision'] != REVISION or not report['complete'] or len(report['shards']) != 48:
        raise ValueError('Complete pinned source verification is required')
    hashes = {}
    for shard in report['shards']:
        name = shard['file']
        if (Path(name).name != name or shard['status'] != 'verified'
                or shard['sha256'] != shard['expected_sha256'] or name in hashes):
            raise ValueError('Invalid or incomplete source checksum inventory')
        hashes[name] = shard['sha256']
    if set(index['weight_map'].values()) != set(hashes):
        raise ValueError('Source index and verified shards differ')
    groups = {name: [] for name in ('active_native', 'routed_source', 'engrams', 'draft')}
    tensors = {}
    for name in sorted(hashes):
        path = source / name
        if path.resolve() != path:
            raise ValueError('Source shards must not redirect outside their pinned paths')
        for tensor in read_slices(path):
            if tensor.name in tensors or index['weight_map'].get(tensor.name) != name:
                raise ValueError('Tensor absent, duplicated or in the wrong source shard')
            tensors[tensor.name] = tensor
            if tensor.name.startswith('mtp.'):
                kind = 'draft'
            elif TABLE.fullmatch(tensor.name):
                kind = 'engrams'
            elif EXPERT.fullmatch(tensor.name):
                match = EXPERT.fullmatch(tensor.name)
                if not 0 <= int(match[1]) < 40 or not 0 <= int(match[2]) < 384:
                    raise ValueError('Unexpected backbone expert coordinates')
                kind = 'routed_source'
            else:
                if '.engram.embed.' in tensor.name or '.ffn.experts.' in tensor.name:
                    raise ValueError('Unclassified table or original routed expert tensor')
                kind = 'active_native'
            groups[kind].append(tensor)
    if set(tensors) != set(index['weight_map']) or sum(t.nbytes for t in tensors.values()) != index['metadata']['total_size']:
        raise ValueError('Source tensor/payload inventory does not reconcile')
    expected = {f'layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}'
                for layer in range(40) for expert in range(384) for projection in ('w1', 'w2', 'w3') for suffix in ('weight', 'scale')}
    if {t.name for t in groups['routed_source']} != expected or len(groups['engrams']) != 4:
        raise ValueError('Missing original expert/table tensors')
    vision = [t for t in groups['active_native'] if t.name.startswith(('vision.', 'aligner.'))]
    if not vision or any(t.dtype != 'BF16' for t in vision) or sum(t.nbytes for t in vision) != 970506240:
        raise ValueError('Original BF16 vision/aligner inventory changed')
    if not groups['draft']:
        raise ValueError('Original draft weights must remain preserved in the SSD sidecar')
    return source, hashes, groups


def quantized_config(source):
    original = json.loads((Path(source) / 'config.json').read_text())
    if (original['model_type'] != 'deepseek_v41' or original['text_config']['num_hidden_layers'] != 40
            or original['text_config']['n_routed_experts'] != 384):
        raise ValueError('Unexpected source architecture')
    result = copy.deepcopy(original)
    result['quantization_config'] = dict(quant_method='ds41_exl3', codebook='mul1', layer_bits=[3] * 40,
        target_bpw=3.0, scope='deepseek_v41_backbone_routed_experts')
    return result


def package_layout(groups, plans, max_payload_bytes=4 * 1024**3):
    if [plan.layer for plan in plans] != list(range(40)):
        raise ValueError('Packaging requires every explicitly selected main-model layer')
    packed = []
    for plan in plans:
        plan.validate()
        for artifact in plan.artifacts:
            if (digest(artifact.path) != artifact.sha256
                    or tensor_inventory(artifact.path, artifact.prefix, 3) != artifact.payload_bytes):
                raise ValueError('Selected EXL3 artifact changed before packaging')
            packed.extend(read_slices(artifact.path))
    if len(packed) != 40 * 384 * 12:
        raise ValueError('Incomplete packed expert tensor inventory')
    main = sorted(groups['active_native'] + packed, key=lambda tensor: tensor.name)
    output = {}
    active_shards = group_shards(main, max_payload_bytes)
    for i, tensors in enumerate(active_shards, 1):
        output[f'model-{i:05d}-of-{len(active_shards):05d}.safetensors'] = tensors
    for layer in (1, 14):
        tables = sorted([t for t in groups['engrams'] if t.name.startswith(f'layers.{layer}.')], key=lambda tensor: tensor.name)
        if len(tables) != 2:
            raise ValueError('Missing engram table/scale pair')
        output[f'engrams/engram-layer-{layer:02d}.safetensors'] = tables
    draft_shards = group_shards(sorted(groups['draft'], key=lambda tensor: tensor.name), max_payload_bytes)
    for i, tensors in enumerate(draft_shards, 1):
        output[f'draft/model-{i:05d}-of-{len(draft_shards):05d}.safetensors'] = tensors
    names = [tensor.name for tensors in output.values() for tensor in tensors]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate tensor across package components')
    if any(t.name.startswith('mtp.') or '.engram.embed.' in t.name or EXPERT.fullmatch(t.name)
           for name, tensors in output.items() if '/' not in name for t in tensors):
        raise ValueError('Inactive/original-routed weights leaked into the active HF index')
    return output


def component_indexes(layout):
    result = {}
    for component in ('active', 'engrams', 'draft'):
        selected = {name: tensors for name, tensors in layout.items()
                    if ('active' if '/' not in name else name.split('/')[0]) == component}
        weights = {tensor.name: Path(name).name for name, tensors in selected.items() for tensor in tensors}
        if not weights:
            raise ValueError('Incomplete package component')
        result[component] = dict(metadata=dict(total_size=sum(t.nbytes for tensors in selected.values() for t in tensors)), weight_map=weights)
    return result
