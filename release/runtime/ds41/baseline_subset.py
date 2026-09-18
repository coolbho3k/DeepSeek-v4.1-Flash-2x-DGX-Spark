"""CPU-only baseline subset provenance/selection checks; no new fitting math.

Allows work on well-covered experts while rare experts await supplementation.
Never lowers the128-row floor or turns an incomplete bank into a model.
"""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .calibration_inputs import canonical, digest, load_source, stamp, validate_exclusions
from .safetensor_pack import read_slices
from scripts import run_quant_queue as baseline

ROOT = Path(__file__).resolve().parents[1]
META_FIELDS = ('available', 'selected', 'selected_image_tokens', 'selected_indices_sha256')


def expert_ids(experts):
    if (not experts or any(type(e) is not int or not 0 <= e < 384 for e in experts)
            or list(experts) != sorted(set(experts))):
        raise ValueError('Expected nonempty, unique, sorted expert IDs in0..383')
    return list(experts)


class BaselineSubsetIndex:
    """Hash/validate one file at a time; retain only CPU routes/image masks."""
    def __init__(self, layer, root=ROOT, progress=None):
        if type(layer) is not int or not 0 <= layer < 40:
            raise ValueError('Expected a backbone layer0..39')
        self.root, self.layer = Path(root).resolve(), layer
        self.config, identity = baseline.preflight(self.root)
        required = dict(seed=41, min_calibration_rows=128, max_calibration_rows=8192, max_heldout_rows=1024)
        if any(self.config[key] != value for key, value in required.items()):
            raise ValueError('Only the unchanged seed41/8192/1024/128 baseline is supported')
        source = load_source(self.root / baseline.CAPTURE, self.root / baseline.CORPUS, False)
        validate_exclusions([source])
        if source['identity'] != identity or source['capture_sha256'] != self.config['capture_sha256']:
            raise ValueError('Baseline reader and fitter identity differ')
        coverage_path = source['capture'] / 'coverage' / f'{layer:02d}.json'
        coverage = json.loads(coverage_path.read_bytes())
        directory = source['capture'] / 'expert-inputs' / f'{layer:02d}'
        if {p.name for p in directory.iterdir()} != {r['id'] + '.safetensors' for r in source['records']}:
            raise ValueError('Incomplete or unexpected baseline input files')
        metadata = [source['capture'] / 'capture-config.json', source['corpus'] / 'records.jsonl',
                    source['corpus'] / 'manifest.json', coverage_path,
                    self.root / 'reports/source-verification.json']
        metadata += [self.root / name for name in self.config['implementation_sha256']]
        metadata += [self.root / name for name in ('ds41/baseline_subset.py', 'ds41/calibration_inputs.py',
                                                  'ds41/safetensor_pack.py', 'scripts/run_quant_queue.py')]
        self.metadata = {str(p.relative_to(self.root)): digest(p) for p in metadata}
        self.files, self.splits = [], {key: {'routes': [], 'image_mask': []} for key in ('calibration', 'heldout')}
        counts = {key: torch.zeros(384, dtype=torch.int64) for key in self.splits}
        image_counts = torch.zeros(384, dtype=torch.int64)
        for number, record in enumerate(source['records']):
            path = directory / (record['id'] + '.safetensors')
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Captured input escapes its designated directory')
            before, sha = stamp(path), digest(path)
            slices = {item.name: item for item in read_slices(path)}
            if set(slices) != {'inputs', 'route_ids', 'route_weights', 'image_mask'}:
                raise ValueError('Unexpected captured tensors')
            with safe_open(path, framework='pt', device='cpu') as saved:
                inputs, routes, weights, mask = (saved.get_tensor(key) for key in
                                                 ('inputs', 'route_ids', 'route_weights', 'image_mask'))
                rows = inputs.shape[0]
                if (inputs.shape != (rows, 5120) or inputs.dtype != torch.bfloat16 or rows <= 0
                        or routes.shape != (rows, 6) or routes.dtype != torch.int64
                        or weights.shape != (rows, 6) or weights.dtype not in (torch.float32, torch.bfloat16)
                        or mask.shape != (rows,) or mask.dtype != torch.bool):
                    raise ValueError('Wrong captured shape/dtype')
                if (not torch.isfinite(weights).all() or (weights < 0).any()
                        or (routes < 0).any() or (routes >= 384).any()
                        or (routes.sort(-1).values[:, 1:] == routes.sort(-1).values[:, :-1]).any()):
                    raise ValueError('Invalid/duplicate routes or nonfinite/negative weights')
                for chunk in inputs.split(512):
                    if not torch.isfinite(chunk).all() or not torch.isfinite(chunk.half()).all():
                        raise ValueError('Nonfinite activation or FP16 EXL3 overflow')
                if record['kind'] == 'text' and (rows != len(record['tokens']) or mask.any()):
                    raise ValueError('Text row count/image mask differs from its pinned record')
                split = record['split']
                counts[split] += torch.bincount(routes.flatten(), minlength=384)
                if split == 'calibration':
                    image_counts += torch.bincount(routes[mask].flatten(), minlength=384)
                self.splits[split]['routes'].append(routes.clone())
                self.splits[split]['image_mask'].append(mask.clone())
                del inputs, routes, weights, mask, chunk
            if stamp(path) != before:
                raise ValueError('Captured input changed while indexing')
            self.files.append(dict(path=str(path.relative_to(self.root)), record=record['id'],
                split=split, rows=rows, bytes=before[2], sha256=sha, stamp=before))
            if progress and (number % 64 == 0 or number + 1 == len(source['records'])):
                progress(dict(stage='baseline_subset_input_validation', layer=layer,
                              records=number + 1, total_records=len(source['records'])))
        self.counts = {key: value.tolist() for key, value in counts.items()}
        if (coverage['layer'] != layer or coverage['records'] != len(source['records'])
                or coverage['routed_counts'] != self.counts
                or coverage['calibration_image_routed_counts'] != image_counts.tolist()):
            raise ValueError('Actual routes differ from source coverage')
        for data in self.splits.values():
            for key in data:
                data[key] = torch.cat(data[key])
        self.manifest = dict(format='ds41_baseline_subset_inputs_v1', layer=layer,
            source_revision=self.config['source_revision'], capture_sha256=self.config['capture_sha256'],
            metadata_sha256=self.metadata, counts=self.counts,
            files=[{k: v for k, v in item.items() if k != 'stamp'} for item in self.files],
            indexer_sha256=digest(Path(__file__)),
            selection_policy='Frozen baseline record order/seed/caps; only calibration counts determine eligibility')
        self.validate_unchanged()

    def validate_unchanged(self):
        for relative, expected in self.metadata.items():
            if digest(self.root / relative) != expected:
                raise ValueError('Baseline metadata/code changed: ' + relative)
        for item in self.files:
            if stamp(self.root / item['path']) != item['stamp']:
                raise ValueError('Baseline activation changed after validation')

    def selection_metadata(self, expert, split, maximum, seed):
        expert_ids([expert])
        if split not in self.splits or type(maximum) is not int or maximum < 0 or type(seed) is not int:
            raise ValueError('Invalid selection split/row cap/seed')
        data = self.splits[split]
        row, slot = torch.where(data['routes'] == expert)
        available = len(row)
        if maximum > 0 and available > maximum:
            generator = torch.Generator().manual_seed(seed + 1009 * expert + (split == 'heldout'))
            chosen = torch.randperm(available, generator=generator)[:maximum]
            row, slot = row[chosen], slot[chosen]
        return dict(available=available, selected=len(row),
            selected_image_tokens=data['image_mask'][row].sum().item(),
            selected_indices_sha256=hashlib.sha256(torch.stack((row, slot)).numpy().tobytes()).hexdigest())

    def ready_experts(self):
        return [e for e, count in enumerate(self.counts['calibration']) if count >= 128]


def plan_subset(inputs):
    experts = inputs.ready_experts()
    if not experts or len(experts) == 384:
        raise ValueError('Subset completion is only for genuinely deferred baseline layers')
    inputs.validate_unchanged()
    return dict(format='ds41_baseline_subset_plan_v1', layer=inputs.layer, bits=3,
        experts=experts, deferred_experts=[e for e in range(384) if e not in experts],
        config={**inputs.config, 'layer': inputs.layer}, input_manifest=inputs.manifest,
        complete_layer=False, quality_qualified=False,
        scope='Ready baseline experts only. No supplemental inputs, changed fitting math or complete-bank/model claim.')


def verify_subset(layer, experts, root=ROOT, progress=None):
    experts = expert_ids(experts)
    inputs = BaselineSubsetIndex(layer, root, progress)
    if any(expert not in inputs.ready_experts() for expert in experts):
        raise ValueError('Subset includes an expert below the unchanged128-row calibration floor')
    root = inputs.root
    expected_config = {**inputs.config, 'layer': layer}
    artifacts = []
    for expert in experts:
        path = root / baseline.OUTPUT / f'layer-{layer:02d}' / f'expert-{expert:03d}-3bit.safetensors'
        report_path = path.with_suffix('.json')
        if (path.is_symlink() or report_path.is_symlink()
                or not path.resolve().is_relative_to(root / baseline.OUTPUT)
                or not report_path.resolve().is_relative_to(root / baseline.OUTPUT)):
            raise ValueError('Subset artifact escapes its designated baseline bank')
        before = (stamp(path), stamp(report_path))
        report = json.loads(report_path.read_bytes())
        prefix = f'layers.{layer}.ffn.experts.{expert}'
        if (report['config'] != expected_config or report['expert'] != prefix or report['bits'] != 3
                or report['codebook'] != 'mul1' or set(report['matrices']) != {'w1', 'w2', 'w3'}
                or any(m['uncalibrated_fallback'] is not False for m in report['matrices'].values())
                or report['artifact_sha256'] != digest(path)
                or report['stored_bytes'] != baseline.tensor_inventory(path, prefix, 3)):
            raise ValueError('Changed/unqualified baseline subset artifact')
        for split, cap in (('calibration', 8192), ('heldout', 1024)):
            if report[split] != inputs.selection_metadata(expert, split, cap, 41):
                raise ValueError('Baseline subset sample differs from exact frozen selection')
        with safe_open(path, framework='pt', device='cpu') as saved:
            for projection in ('w1', 'w2', 'w3'):
                for suffix in ('suh', 'svh'):
                    if not torch.isfinite(saved.get_tensor(f'{prefix}.{projection}.{suffix}')).all():
                        raise ValueError('Nonfinite EXL3 scale')
        if (stamp(path), stamp(report_path)) != before:
            raise ValueError('Subset candidate changed during verification')
        artifacts.append(dict(expert=expert, artifact_sha256=report['artifact_sha256'],
                              report_sha256=digest(report_path), exact_frozen_selection_replayed=True))
    inputs.validate_unchanged()
    return dict(status='verified', format='ds41_baseline_subset_verification_v1', layer=layer, bits=3,
        experts=artifacts, config=expected_config, input_manifest=inputs.manifest,
        complete_layer=experts == list(range(384)), quality_qualified=False,
        scope='Full baseline input hashes/schema/finiteness/routes, exact sample replay and requested packed artifacts only. Not a complete model, supplemental bank assembly or quality qualification.')
