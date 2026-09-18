"""Versioned baseline + first supplement + pinned code-topup routed inputs.

The frozen v1 reader still validates its two original sources. Only the new
calibration-only source is appended here; its portable handoff hashes must
match every activation file. Selection reuses the frozen v1 implementation,
so held-out data/order/seeds remain baseline-only. No capture is launched.
"""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from . import calibration_inputs as v1
from scripts import capture_code_topup as capture_code
from scripts.prepare_calibration_code_topup import shingles

FROZEN_READER_SHA = '4b876f0b45ded1feae6d70aa694371db7f2554f9f7198dd093daf1af071c66e2'


def load_code_source(capture, corpus, layer):
    capture, corpus = Path(capture).absolute(), Path(corpus).absolute()
    capture_code.guard_work(capture)
    if corpus != capture_code.ROOT / capture_code.CORPUS_REL or corpus.resolve() != corpus:
        raise ValueError('Only the designated independently pinned code corpus is supported')
    paths = [capture / 'capture-config.json', corpus / 'records.jsonl',
        corpus / 'manifest.json', capture / 'coverage' / f'{layer:02d}.json',
        capture / 'handoffs' / f'{layer:02d}.json']
    if any(path.is_symlink() for path in paths):
        raise ValueError('Redirected code capture metadata')
    metadata = [(path, v1.digest(path)) for path in paths]
    records, _, expected_identity = capture_code.context()
    identity = json.loads(paths[0].read_bytes())
    manifest = json.loads(paths[2].read_bytes())
    coverage = json.loads(paths[3].read_bytes())
    receipt = json.loads(paths[4].read_bytes())
    saved_records = [json.loads(line) for line in paths[1].read_bytes().splitlines()]
    if (identity != expected_identity or saved_records != records
            or identity['corpus_sha256'] != metadata[1][1]
            or identity['corpus_manifest_sha256'] != metadata[2][1]):
        raise ValueError('Code capture does not match the independently pinned source/corpus/producer')
    names = capture_code.record_names(records)
    if (receipt['format'] != 'ds41_code_capture_handoff_v2'
            or receipt['layer'] != layer or receipt['exact_routes_verified'] is not True
            or receipt['capture_sha256'] != metadata[0][1]
            or receipt['coverage_sha256'] != metadata[3][1]
            or set(receipt['expert_inputs']) != names
            or set(receipt['expert_input_sha256']) != names):
        raise ValueError('Missing/changed portable code-input completion receipt')
    if any(v1.digest(path) != expected for path, expected in metadata):
        raise ValueError('Code metadata changed during indexing')
    return dict(capture=capture, corpus=corpus, identity=identity, manifest=manifest,
        capture_sha256=metadata[0][1], corpus_sha256=metadata[1][1], records=records,
        coverage=coverage, receipt=receipt, metadata=metadata)


def validate_code_exclusions(sources, code):
    if code['identity']['baseline_capture_sha256'] != sources[0]['capture_sha256']:
        raise ValueError('Code capture belongs to a different baseline')
    expected_exclusions = {(source['corpus_sha256'], v1.digest(source['corpus'] / 'manifest.json'))
                           for source in sources}
    declared = {(item['records_sha256'], item['manifest_sha256'])
                for item in code['manifest']['exclusions']}
    if declared != expected_exclusions or len(code['manifest']['exclusions']) != len(sources):
        raise ValueError('Code corpus did not exclude these exact baseline/supplement corpora')
    ids, occupied = set(), set()
    for source in sources:
        for record in source['records']:
            ids.add(record['id'])
            if record['kind'] == 'text':
                occupied.update(shingles(record['tokens']))
    pins = {pin['file']: pin for pin in code['manifest']['sources'] if pin['type'] == 'text'}
    seen_files = set()
    for record in code['records']:
        tokens = record['tokens']
        if (record['split'] != 'calibration' or record['kind'] != 'text'
                or record['id'] in ids or not 0 < len(tokens) <= 2048
                or record['source'] not in pins or record['source'] in seen_files
                or type(record['token_start']) is not int or record['token_start'] < 0
                or record['token_sha256'] != hashlib.sha256(v1.canonical(tokens)).hexdigest()):
            raise ValueError('Invalid, unpinned, duplicate, or held-out code record')
        windows = set(shingles(tokens))
        if windows & occupied:
            raise ValueError('Code record shares a64-token window with an excluded or earlier record')
        occupied.update(windows)
        ids.add(record['id'])
        seen_files.add(record['source'])


class CodeCombinedRoutedInputs(v1.CombinedRoutedInputs):
    def __init__(self, baseline_capture, baseline_corpus, supplement_capture,
                 supplement_corpus, code_capture, code_corpus, layer, *, device='cuda', progress=None):
        if type(layer) is not int or not 0 <= layer < 40:
            raise ValueError('Expected a backbone layer0..39')
        if v1.digest(Path(v1.__file__)) != FROZEN_READER_SHA:
            raise ValueError('Frozen v1 reader changed; review the v2 adapter before reuse')
        # Fail on an absent/incomplete code capture BEFORE reading the large
        # original activation sets. No fallback to the smaller calibration set.
        code = load_code_source(code_capture, code_corpus, layer)
        super().__init__(baseline_capture, baseline_corpus,
            [(supplement_capture, supplement_corpus)], layer, device=device, progress=progress)
        validate_code_exclusions(self.sources, code)
        self._append_code(code, progress)
        self.validate_unchanged()

    def _append_code(self, source, progress):
        source_index = len(self.sources)
        directory = source['capture'] / 'expert-inputs' / f'{self.layer:02d}'
        capture_code.complete_files(directory, source['records'])
        counts = torch.zeros(384, dtype=torch.int64)
        data = self.splits['calibration']
        offsets = data['offsets'].tolist()
        parts, additions = [], []
        for index, record in enumerate(source['records']):
            path = directory / (record['id'] + '.safetensors')
            before = v1.stamp(path)
            file_hash = v1.digest(path)
            if file_hash != source['receipt']['expert_input_sha256'][path.name]:
                raise ValueError(f'Code activation differs from its capture handoff: {path}')
            with safe_open(path, framework='pt', device='cpu') as saved:
                if set(saved.keys()) != {'inputs', 'route_ids', 'route_weights', 'image_mask'}:
                    raise ValueError('Unexpected code capture tensors')
                inputs, routes, weights, mask = (saved.get_tensor(key) for key in
                    ('inputs', 'route_ids', 'route_weights', 'image_mask'))
                rows = len(record['tokens'])
                if (inputs.shape != (rows, 5120) or inputs.dtype != torch.bfloat16
                        or routes.shape != (rows, 6) or routes.dtype != torch.int64
                        or weights.shape != (rows, 6) or weights.dtype != torch.float32
                        or mask.shape != (rows,) or mask.dtype != torch.bool or mask.any()):
                    raise ValueError('Code input tensor layout/length/image mask mismatch')
                ordered = routes.sort(-1).values
                if (not torch.isfinite(weights).all() or (weights < 0).any()
                        or (routes < 0).any() or (routes >= 384).any()
                        or (ordered[:, 1:] == ordered[:, :-1]).any()
                        or any(not torch.isfinite(chunk).all() for chunk in inputs.split(512))):
                    raise ValueError('Invalid code expert routes/weights/activations')
                counts += torch.bincount(routes.flatten(), minlength=384)
                parts.append(routes.clone())
                item = dict(path=path, stamp=before, source=source_index, record=record,
                    rows=rows, sha256=file_hash, row_offset=offsets[-1])
                data['files'].append(item)
                self.files.append(item)
                offsets.append(offsets[-1] + rows)
                additions.append(dict(source=source_index, record=record['id'], split='calibration',
                    rows=rows, sha256=file_hash, bytes=before[2]))
            if v1.stamp(path) != before:
                raise ValueError('Code activation changed during validation')
            if progress and (index % 64 == 0 or index + 1 == len(source['records'])):
                progress(dict(stage='validate_code_combined_inputs', source=source_index,
                    record=index + 1, records=len(source['records']), layer=self.layer))
        actual = dict(calibration=counts.tolist(), heldout=[0] * 384)
        coverage = source['coverage']
        if (coverage['layer'] != self.layer or coverage['records'] != len(source['records'])
                or coverage['routed_counts'] != actual
                or coverage['calibration_image_routed_counts'] != [0] * 384):
            raise ValueError('Code coverage does not match actual saved routes')
        data['routes'] = torch.cat([data['routes'], *parts])
        data['offsets'] = torch.tensor(offsets, dtype=torch.int64)
        self.sources.append(source)
        self.counts_by_source.append(actual)
        self.metadata_files.extend(source['metadata'])
        self.manifest.update(format='ds41_combined_routed_inputs_code_v2',
            selection_order='baseline capture order, first supplement, code-topup-v2; heldout baseline only',
            counts_by_source=self.counts_by_source,
            exact_64_token_code_exclusions_verified=True,
            frozen_v1_reader_sha256=FROZEN_READER_SHA)
        self.manifest['activation_files'].extend(additions)
        self.manifest['sources'].append(dict(capture_sha256=source['capture_sha256'],
            corpus_sha256=source['corpus_sha256'], corpus_manifest_sha256=source['metadata'][2][1],
            coverage_sha256=source['metadata'][3][1], handoff_sha256=source['metadata'][4][1],
            role='calibration_only_code_topup_v2', portable_input_hashes_verified=True))
        self.capture_sha256 = hashlib.sha256(v1.canonical(self.manifest)).hexdigest()
