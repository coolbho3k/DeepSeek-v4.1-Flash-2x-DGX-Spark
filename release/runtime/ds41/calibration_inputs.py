"""Immutable baseline+supplement routed inputs, with bounded activation loads.

Only calibration records may be added. Held-out ordering and seeded selection
remain identical to the frozen baseline RoutedInputs implementation. This
module does not change that implementation or any running capture identity.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import torch
from safetensors import safe_open

from .reference_runtime import REVISION

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CAPTURE_FILES = ('scripts/capture_reference.py', 'ds41/reference_runtime.py', 'ds41/ssd_rows.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp(path):
    state = Path(path).stat()
    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)


def load_source(capture, corpus, supplement):
    capture, corpus = Path(capture).resolve(), Path(corpus).resolve()
    identity_path, records_path = capture / 'capture-config.json', corpus / 'records.jsonl'
    identity = json.loads(identity_path.read_bytes())
    manifest = json.loads((corpus / 'manifest.json').read_bytes())
    corpus_hash = digest(records_path)
    if (identity['source_revision'] != REVISION or manifest['model_revision'] != REVISION
            or identity['corpus_sha256'] != corpus_hash or manifest['records_sha256'] != corpus_hash
            or identity['max_seq_len'] != 2048):
        raise ValueError('Source/corpus/capture identity mismatch')
    expected_method = 'source_layerwise_reference_v1_streaming_topup' if supplement else 'source_layerwise_reference_v1'
    if identity['method'] != expected_method:
        raise ValueError('Only frozen source-model captures can be combined')
    expected_code = set(FROZEN_CAPTURE_FILES) | ({'scripts/capture_topup_stream.py'} if supplement else set())
    if set(identity['implementation_sha256']) != expected_code:
        raise ValueError('Unexpected capture implementation identity')
    for relative, expected in identity['implementation_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen capture implementation changed: {relative}')
    records = [json.loads(line) for line in records_path.read_bytes().splitlines()]
    mapping = {record['id']: record for record in records}
    ids = identity['records']
    if (len(mapping) != len(records) or len(set(ids)) != len(ids)
            or set(ids) != set(mapping) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in ids)):
        raise ValueError('Duplicate, unsafe, missing or partial captured record identities')
    if not records or any(record['split'] not in ('calibration', 'heldout') for record in records):
        raise ValueError('Invalid/empty corpus splits')
    if supplement and any(record['split'] != 'calibration' for record in records):
        raise ValueError('Supplement contains held-out records')
    if not supplement and {record['split'] for record in records} != {'calibration', 'heldout'}:
        raise ValueError('Baseline must retain its calibration and held-out records')
    return dict(capture=capture, corpus=corpus, identity=identity, manifest=manifest,
                capture_sha256=digest(identity_path), corpus_sha256=corpus_hash,
                records=[mapping[name] for name in ids])


def validate_exclusions(sources):
    baseline = sources[0]
    ids, image_hashes, text_hashes, intervals = set(), set(), set(), defaultdict(list)
    for source_index, source in enumerate(sources):
        if source_index:
            if (source['identity'].get('baseline_capture_sha256') != baseline['capture_sha256']
                    or source['manifest'].get('supplements_corpus_sha256') != baseline['corpus_sha256']
                    or any(pin not in baseline['manifest']['sources'] for pin in source['manifest']['sources'])
                    or any(source['identity']['implementation_sha256'][name]
                           != baseline['identity']['implementation_sha256'][name] for name in FROZEN_CAPTURE_FILES)):
                raise ValueError('Supplement does not match the baseline capture/source pins')
        for record in source['records']:
            if record['id'] in ids:
                raise ValueError('Duplicate record identity across captures')
            ids.add(record['id'])
            if record['kind'] == 'text':
                tokens = record['tokens']
                token_hash = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
                if not 0 < len(tokens) <= 2048 or token_hash != record['token_sha256'] or token_hash in text_hashes:
                    raise ValueError('Invalid or duplicate text tokens')
                text_hashes.add(token_hash)
                begin, end = record['token_start'], record['token_start'] + len(tokens)
                ranges = intervals[record['source']]
                if begin < 0 or any(begin < old_end and end > old_begin for old_begin, old_end in ranges):
                    raise ValueError('Overlapping baseline/held-out/supplement text ranges')
                ranges.append((begin, end))
                if not any(pin.get('type') == 'text' and pin.get('file') == record['source'] for pin in source['manifest']['sources']):
                    raise ValueError('Unpinned text source')
            elif record['kind'] == 'image':
                image = (source['corpus'] / record['image']).resolve()
                if not image.is_relative_to(source['corpus']):
                    raise ValueError('Image path escapes the corpus')
                image_hash = digest(image)
                if image_hash != record['image_sha256'] or image_hash in image_hashes:
                    raise ValueError('Invalid/duplicate baseline/held-out/supplement image')
                image_hashes.add(image_hash)
                if not any(pin.get('dataset') == record['source'] and pin.get('revision') == record['revision']
                           and pin.get('file') == record['shard'] for pin in source['manifest']['sources']):
                    raise ValueError('Unpinned image source')
            else:
                raise ValueError('Unknown calibration record kind')


class CombinedRoutedInputs:
    def __init__(self, baseline_capture, baseline_corpus, supplements, layer, *, device='cuda', progress=None):
        if not 0 <= layer < 40 or not supplements:
            raise ValueError('Expected a backbone layer and at least one supplemental capture')
        self.sources = [load_source(baseline_capture, baseline_corpus, False)]
        self.sources += [load_source(capture, corpus, True) for capture, corpus in supplements]
        validate_exclusions(self.sources)
        self.layer, self.device = layer, device
        self.splits = {split: dict(files=[], offsets=[0], route_parts=[]) for split in ('calibration', 'heldout')}
        manifest_sources, manifest_files = [], []
        self.metadata_files, self.files = [], []
        self.counts_by_source = []
        for source_index, source in enumerate(self.sources):
            capture = source['capture']
            coverage_path = capture / 'coverage' / f'{layer:02d}.json'
            coverage = json.loads(coverage_path.read_bytes())  # Missing means not complete; fail closed.
            directory = capture / 'expert-inputs' / f'{layer:02d}'
            expected_names = {record['id'] + '.safetensors' for record in source['records']}
            if {path.name for path in directory.glob('*.safetensors')} != expected_names:
                raise ValueError('Missing/unexpected captured activation files')
            counts = {split: torch.zeros(384, dtype=torch.int64) for split in self.splits}
            image_counts = torch.zeros(384, dtype=torch.int64)
            for record_index, record in enumerate(source['records']):
                path = directory / (record['id'] + '.safetensors')
                before = stamp(path)
                file_hash = digest(path)
                with safe_open(path, framework='pt', device='cpu') as saved:
                    if set(saved.keys()) != {'inputs', 'route_ids', 'route_weights', 'image_mask'}:
                        raise ValueError(f'Unexpected captured tensors: {path}')
                    routes, weights, mask = (saved.get_tensor(key) for key in ('route_ids', 'route_weights', 'image_mask'))
                    inputs = saved.get_tensor('inputs')
                    rows = inputs.shape[0]
                    if (inputs.shape != (rows, 5120) or inputs.dtype != torch.bfloat16
                            or routes.shape != (rows, 6) or routes.dtype != torch.int64
                            or weights.shape != (rows, 6) or weights.dtype not in (torch.float32, torch.bfloat16)
                            or mask.shape != (rows,) or mask.dtype != torch.bool or rows == 0):
                        raise ValueError(f'Invalid captured tensor shapes/dtypes: {path}')
                    if (not torch.isfinite(weights).all() or (weights < 0).any()
                            or (routes < 0).any() or (routes >= 384).any()
                            or (routes.sort(-1).values[:, 1:] == routes.sort(-1).values[:, :-1]).any()):
                        raise ValueError(f'Invalid/duplicate expert routes or weights: {path}')
                    for chunk in inputs.split(512):
                        if not torch.isfinite(chunk).all():
                            raise ValueError(f'Nonfinite captured activation: {path}')
                    if record['kind'] == 'text' and (rows != len(record['tokens']) or mask.any()):
                        raise ValueError('Text capture length/image mask does not match its record')
                    counts[record['split']] += torch.bincount(routes.flatten(), minlength=384)
                    if record['split'] == 'calibration':
                        image_counts += torch.bincount(routes[mask].flatten(), minlength=384)
                    split = self.splits[record['split']]
                    split['route_parts'].append(routes.clone())
                    item = dict(path=path, stamp=before, source=source_index, record=record,
                                rows=rows, sha256=file_hash, row_offset=split['offsets'][-1])
                    split['files'].append(item)
                    split['offsets'].append(split['offsets'][-1] + rows)
                    self.files.append(item)
                    manifest_files.append(dict(source=source_index, record=record['id'], split=record['split'],
                                               rows=rows, sha256=file_hash, bytes=before[2]))
                    del inputs, routes, weights, mask, chunk
                if stamp(path) != before:
                    raise ValueError('Captured file changed during validation')
                if progress and (record_index % 64 == 0 or record_index + 1 == len(source['records'])):
                    progress(dict(stage='validate_combined_inputs', source=source_index,
                                  record=record_index + 1, records=len(source['records']), layer=layer))
            actual = {split: tensor.tolist() for split, tensor in counts.items()}
            if (coverage['layer'] != layer or coverage['records'] != len(source['records'])
                    or coverage['routed_counts'] != actual
                    or coverage['calibration_image_routed_counts'] != image_counts.tolist()):
                raise ValueError('Saved coverage does not match actual captured routes')
            self.counts_by_source.append(actual)
            manifest_sources.append(dict(capture_sha256=source['capture_sha256'], corpus_sha256=source['corpus_sha256'],
                corpus_manifest_sha256=digest(source['corpus'] / 'manifest.json'), coverage_sha256=digest(coverage_path),
                role='baseline' if source_index == 0 else 'calibration_only_supplement'))
            for path in (capture / 'capture-config.json', source['corpus'] / 'records.jsonl', source['corpus'] / 'manifest.json', coverage_path):
                self.metadata_files.append((path, digest(path)))
        for split in self.splits.values():
            split['routes'] = torch.cat(split.pop('route_parts'))
            split['offsets'] = torch.tensor(split['offsets'], dtype=torch.int64)
        self.manifest = dict(format='ds41_combined_routed_inputs_v1', source_revision=REVISION, layer=layer,
                             sources=manifest_sources, activation_files=manifest_files,
                             selection_order='baseline capture record order, then each supplement; heldout baseline only',
                             counts_by_source=self.counts_by_source)
        self.capture_sha256 = hashlib.sha256(canonical(self.manifest)).hexdigest()

    def validate_unchanged(self):
        for path, expected in self.metadata_files:
            if digest(path) != expected:
                raise ValueError(f'Capture/corpus/coverage changed: {path}')
        for item in self.files:
            if stamp(item['path']) != item['stamp']:
                raise ValueError(f'Captured activation changed: {item["path"]}')

    def select(self, expert, split, maximum, seed):
        if split not in self.splits or not 0 <= expert < 384 or maximum < 0:
            raise ValueError('Invalid expert, split or row bound')
        data = self.splits[split]
        row, slot = torch.where(data['routes'] == expert)
        available = len(row)
        if maximum > 0 and len(row) > maximum:
            generator = torch.Generator().manual_seed(seed + 1009 * expert + (split == 'heldout'))
            chosen = torch.randperm(len(row), generator=generator)[:maximum]
            row, slot = row[chosen], slot[chosen]
        x = torch.empty((len(row), 5120), dtype=torch.bfloat16)
        weights = torch.empty((len(row), 1), dtype=torch.float32)
        mask = torch.empty(len(row), dtype=torch.bool)
        file_ids = torch.bucketize(row, data['offsets'][1:], right=True)
        contributions = []
        origins = [None] * len(row)
        for file_id in torch.unique(file_ids).tolist():
            item = data['files'][file_id]
            if stamp(item['path']) != item['stamp']:
                raise ValueError('Captured input changed after indexing')
            chosen = (file_ids == file_id).nonzero().flatten()
            local_rows, local_slots = row[chosen] - item['row_offset'], slot[chosen]
            with safe_open(item['path'], framework='pt', device='cpu') as saved:
                x[chosen] = saved.get_tensor('inputs')[local_rows]
                weights[chosen, 0] = saved.get_tensor('route_weights')[local_rows, local_slots].float()
                mask[chosen] = saved.get_tensor('image_mask')[local_rows]
            if stamp(item['path']) != item['stamp']:
                raise ValueError('Captured input changed during selection')
            record = item['record']
            contributions.append(dict(source=item['source'], record=record['id'], selected=len(chosen),
                                      image_tokens=mask[chosen].sum().item(), domain=record['source']))
            for output_row, input_row, input_slot in zip(chosen.tolist(), local_rows.tolist(), local_slots.tolist()):
                origins[output_row] = (self.sources[item['source']]['capture_sha256'], record['id'], input_row, input_slot)
        selected = dict(inputs=x.to(self.device).half().contiguous(), weights=weights.to(self.device), image_mask=mask.to(self.device))
        if not torch.isfinite(selected['inputs']).all():
            raise ValueError('FP16 EXL3 input overflow')
        meta = dict(available=available, selected=len(row), selected_image_tokens=mask.sum().item(),
                    selected_indices_sha256=hashlib.sha256(torch.stack((row, slot)).numpy().tobytes()).hexdigest(),
                    selected_identities_sha256=hashlib.sha256(canonical(origins)).hexdigest(),
                    available_by_source=[counts[split][expert] for counts in self.counts_by_source],
                    selected_by_source=[sum(c['selected'] for c in contributions if c['source'] == source) for source in range(len(self.sources))],
                    records=contributions)
        return selected, meta
