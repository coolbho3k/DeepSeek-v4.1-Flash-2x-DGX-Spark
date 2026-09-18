"""Explicit complete EXL3 model inventories, without importing torch.

Manifests select evaluation candidates, never certify quality. Layer sources
are explicit; missing baseline banks never silently fall back to supplements.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from scripts import run_quant_queue as baseline

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'df42c109f1defefcbfcedbe7d905718a12266e40'
FORMAT = 'ds41_full_quantized_evaluation_manifest_v1'
KINDS = {'baseline': baseline.OUTPUT, 'supplemented': Path('calibration/candidates-topup-v1'),
         'code_supplemented': Path('calibration/candidates-code-topup-v2')}
COVERAGE = 'coverage_assembled'
LAYER_KINDS = (*KINDS, COVERAGE)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ExpertArtifact:
    expert: int
    prefix: str
    path: Path
    sha256: str
    report_sha256: str
    payload_bytes: int


@dataclass(frozen=True)
class LayerPlan:
    layer: int
    artifacts: tuple[ExpertArtifact, ...]
    inventory_sha256: str
    source_revision: str = REVISION

    def validate(self):
        if self.source_revision != REVISION or type(self.layer) is not int or not 0 <= self.layer < 40:
            raise ValueError('Expected a main-model layer of the pinned source revision')
        if [item.expert for item in self.artifacts] != list(range(384)):
            raise ValueError('A quantized layer requires all384 unique ordered experts')
        for item in self.artifacts:
            if (type(item.expert) is not int or item.prefix != f'layers.{self.layer}.ffn.experts.{item.expert}'
                    or type(item.payload_bytes) is not int or item.payload_bytes <= 0 or item.path.suffix != '.safetensors'
                    or any(not re.fullmatch(r'[a-f0-9]{64}', value)
                           for value in (item.sha256, item.report_sha256, self.inventory_sha256))):
                raise ValueError('Invalid selected expert identity, prefix or checksum')


def baseline_layer_plan(layer, root=ROOT, *, verify_hashes=False):
    """Read a complete verified inventory; never rewrite its report."""
    root = Path(root).resolve()
    config, _ = baseline.preflight(root)
    if any(config[key] != value for key, value in dict(source_revision=REVISION, seed=41,
            min_calibration_rows=128, max_calibration_rows=8192, max_heldout_rows=1024).items()):
        raise ValueError('Evaluation selection requires the frozen seed/caps and128-row calibration floor')
    path = root / 'reports' / f'quant-layer-{layer:02d}-3bit.json'
    inventory = json.loads(path.read_text())
    expected_config = {**config, 'layer': layer}
    if (inventory['status'] != 'verified' or inventory['bits'] != 3 or inventory['layer'] != layer
            or inventory['experts'] != 384 or inventory['config'] != expected_config
            or [a['expert'] for a in inventory['artifacts']] != list(range(384))):
        raise ValueError('Missing or incompatible complete3-bit baseline inventory')
    artifacts = []
    for entry in inventory['artifacts']:
        expert = entry['expert']
        prefix = f'layers.{layer}.ffn.experts.{expert}'
        artifact = root / baseline.OUTPUT / f'layer-{layer:02d}' / f'expert-{expert:03d}-3bit.safetensors'
        report_path = artifact.with_suffix('.json')
        report = json.loads(report_path.read_text())
        if (digest(report_path) != entry['report_sha256'] or report['config'] != expected_config
                or report['expert'] != prefix or report['bits'] != 3 or report['codebook'] != 'mul1'
                or report['artifact_sha256'] != entry['artifact_sha256']
                or set(report['matrices']) != {'w1', 'w2', 'w3'}
                or any(m['uncalibrated_fallback'] for m in report['matrices'].values())
                or report['calibration']['selected'] < config['min_calibration_rows']):
            raise ValueError(f'Changed or unqualified expert report: {prefix}')
        size = baseline.tensor_inventory(artifact, prefix, 3)
        if size != report['stored_bytes'] or (verify_hashes and digest(artifact) != entry['artifact_sha256']):
            raise ValueError('Packed payload size/checksum differs from the quantizer report')
        artifacts.append(ExpertArtifact(expert, prefix, artifact, entry['artifact_sha256'], entry['report_sha256'], size))
    if sum(a.payload_bytes for a in artifacts) != inventory['packed_payload_bytes']:
        raise ValueError('Layer payload accounting differs from its verified inventory')
    plan = LayerPlan(layer, tuple(artifacts), digest(path))
    plan.validate()
    return plan


def supplemented_layer_plan(layer, root=ROOT, *, check_inputs=True):
    from scripts import verify_supplemented
    root = Path(root).resolve()
    if root != ROOT:
        raise ValueError('Supplemental verification currently requires this recipe checkout')
    output = root / KINDS['supplemented']
    result = verify_supplemented.verify(output, layer, list(range(384)), check_inputs=check_inputs)
    # A validation work counter is not part of the selected weight identity.
    result.pop('input_files_sha256_checked')
    result['input_manifest_sha256'] = digest(output / 'input-manifests' / f'layer-{layer:02d}.json')
    result['config_sha256'] = digest(output / 'configs' / f'layer-{layer:02d}.json')
    artifacts = []
    for entry in result['experts']:
        expert = entry['expert']
        path = output / f'layer-{layer:02d}' / f'expert-{expert:03d}-3bit.safetensors'
        prefix = f'layers.{layer}.ffn.experts.{expert}'
        artifacts.append(ExpertArtifact(expert, prefix, path, entry['artifact_sha256'], entry['report_sha256'],
                                        baseline.tensor_inventory(path, prefix, 3)))
    plan = LayerPlan(layer, tuple(artifacts), checksum(result))
    plan.validate()
    return plan


def code_supplemented_layer_plan(layer, root=ROOT):
    """Explicit whole-bank v2 selection; always revalidate all three sources.

There is intentionally no check_inputs=False path for this newer reader.
Integrity/coverage verification never promotes a candidate's quality status.
"""
    from scripts import verify_code_supplemented
    root = Path(root).resolve()
    if root != ROOT:
        raise ValueError('Code-supplemented verification requires this recipe checkout')
    output = root / KINDS['code_supplemented']
    result = verify_code_supplemented.verify(output, layer, list(range(384)))
    if (result['status'] != 'verified' or result['layer'] != layer or result['bits'] != 3
            or result['complete_layer'] is not True or result['release_qualified'] is not False
            or result['exact_selection_replay'] is not True
            or result['source_roles'] != ['baseline', 'calibration_only_supplement', 'calibration_only_code_topup_v2']
            or [item['expert'] for item in result['experts']] != list(range(384))
            or any(item['exact_frozen_selection_replayed'] is not True for item in result['experts'])
            or result['verifier_sha256'] != digest(Path(verify_code_supplemented.__file__))):
        raise ValueError('Incomplete or incompatible three-source whole-bank verification')
    # Verification work counters are not part of a selected weight identity.
    result.pop('input_files_sha256_checked')
    result['input_manifest_sha256'] = digest(output / 'input-manifests' / f'layer-{layer:02d}.json')
    result['config_sha256'] = digest(output / 'configs' / f'layer-{layer:02d}.json')
    artifacts = []
    for entry in result['experts']:
        expert = entry['expert']
        path = output / f'layer-{layer:02d}' / f'expert-{expert:03d}-3bit.safetensors'
        prefix = f'layers.{layer}.ffn.experts.{expert}'
        artifacts.append(ExpertArtifact(expert, prefix, path, entry['artifact_sha256'], entry['report_sha256'],
                                        baseline.tensor_inventory(path, prefix, 3)))
    plan = LayerPlan(layer, tuple(artifacts), checksum(result))
    plan.validate()
    return plan


def coverage_spec(value):
    """Canonical source partition plus the explicitly selected bank receipt hash."""
    from .coverage_bank import selection
    if not isinstance(value, dict) or set(value) != {'selection', 'inventory_sha256'}:
        raise ValueError('Coverage selection requires an explicit partition and verified inventory hash')
    chosen = value['selection']
    if not isinstance(chosen, dict) or set(chosen) != set(KINDS):
        raise ValueError('Coverage selection requires all three named source roles')
    expected = selection(chosen['supplemented'], chosen['code_supplemented'])
    if (type(chosen['baseline']) is not list or any(type(e) is not int for e in chosen['baseline'])
            or chosen != expected or not isinstance(value['inventory_sha256'], str)
            or not re.fullmatch(r'[a-f0-9]{64}', value['inventory_sha256'])):
        raise ValueError('Invalid coverage partition or inventory checksum')
    return dict(selection=expected, inventory_sha256=value['inventory_sha256'])


def coverage_layer_plan(layer, selected, root=ROOT):
    """Recheck every selected source; never skip input verification on reread."""
    from . import coverage_bank
    root = Path(root).resolve()
    selected = coverage_spec(selected)
    chosen = selected['selection']
    receipt, plan = coverage_bank.verify_bank(layer, chosen['supplemented'], chosen['code_supplemented'], root)
    if (receipt['format'] != 'ds41_explicit_coverage_bank_v1' or receipt['layer'] != layer
            or receipt['bits'] != 3 or receipt['source_revision'] != REVISION
            or receipt['complete_layer'] is not True or receipt['quality_status'] != 'evaluation_candidate_not_qualified'
            or receipt['selection'] != chosen or plan.layer != layer
            or checksum(receipt) != selected['inventory_sha256'] or plan.inventory_sha256 != checksum(receipt)
            or receipt['implementation_sha256'] != {name: digest(root / name) for name in coverage_bank.IMPLEMENTATION}
            or set(receipt['source_verification_sha256']) != {kind for kind in KINDS if chosen[kind]}
            or any(not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value)
                   for value in receipt['source_verification_sha256'].values())):
        raise ValueError('Coverage bank differs from the explicitly selected complete unqualified receipt')
    entry = layer_entry(plan, COVERAGE, root)
    artifacts = [dict(expert=item['expert'], kind=item['source_kind'], path=item['path'],
        sha256=item['sha256'], report_sha256=item['report_sha256'], payload_bytes=item['payload_bytes'])
        for item in entry['artifacts']]
    if entry['source_selection'] != chosen or receipt['artifacts'] != artifacts:
        raise ValueError('Coverage receipt and actual selected expert paths/hashes differ')
    return plan


def layer_entry(plan, kind, root=ROOT):
    plan.validate()
    if kind not in LAYER_KINDS:
        raise ValueError('Unknown explicitly selected candidate source')
    root = Path(root).resolve()
    artifacts = []
    chosen = {source: [] for source in KINDS}
    for item in plan.artifacts:
        if kind == COVERAGE:
            matches = [source for source, directory in KINDS.items()
                       if item.path == root / directory / f'layer-{plan.layer:02d}' / f'expert-{item.expert:03d}-3bit.safetensors']
            if len(matches) != 1:
                raise ValueError('Mixed artifact does not name exactly one canonical source')
            source = matches[0]
        else:
            source = kind
        expected = KINDS[source] / f'layer-{plan.layer:02d}' / f'expert-{item.expert:03d}-3bit.safetensors'
        if item.path.resolve() != root / expected:
            raise ValueError('Selected artifact escapes or differs from its designated candidate bank')
        artifacts.append(dict(expert=item.expert, prefix=item.prefix, path=str(expected),
            sha256=item.sha256, report_sha256=item.report_sha256, payload_bytes=item.payload_bytes))
        if kind == COVERAGE:
            chosen[source].append(item.expert)
            artifacts[-1]['source_kind'] = source
    entry = dict(layer=plan.layer, kind=kind, inventory_sha256=plan.inventory_sha256, artifacts=artifacts)
    if kind == COVERAGE:
        entry['source_selection'] = coverage_spec(dict(selection=chosen, inventory_sha256=plan.inventory_sha256))['selection']
    return entry


def check_shape(manifest, root=ROOT):
    root = Path(root).resolve()
    if (manifest['format'] != FORMAT or manifest['source_revision'] != REVISION
            or manifest['bits'] != 3 or manifest['quality_status'] != 'evaluation_candidate_not_qualified'
            or [entry['layer'] for entry in manifest['layers']] != list(range(40))):
        raise ValueError('Expected all40 ordered layers of an explicitly unqualified3-bit evaluation candidate')
    for entry in manifest['layers']:
        if entry['kind'] not in LAYER_KINDS:
            raise ValueError('Unknown candidate source; no automatic fallback')
        if entry['kind'] == COVERAGE:
            coverage_spec(dict(selection=entry['source_selection'], inventory_sha256=entry['inventory_sha256']))
        items = tuple(ExpertArtifact(item['expert'], item['prefix'], root / item['path'], item['sha256'],
                                    item['report_sha256'], item['payload_bytes']) for item in entry['artifacts'])
        plan = LayerPlan(entry['layer'], items, entry['inventory_sha256'])
        if layer_entry(plan, entry['kind'], root) != entry:
            raise ValueError('Noncanonical selected bank or artifact paths')


def selected_plans(kinds, root=ROOT, *, verify_hashes=True, check_inputs=True, coverage_banks=None):
    if len(kinds) != 40 or any(kind not in LAYER_KINDS for kind in kinds):
        raise ValueError('Choose exactly40 explicit layer sources')
    coverage_banks = {} if coverage_banks is None else coverage_banks
    if (not isinstance(coverage_banks, dict) or any(type(layer) is not int for layer in coverage_banks)
            or set(coverage_banks) != {layer for layer, kind in enumerate(kinds) if kind == COVERAGE}):
        raise ValueError('Coverage-bank specifications must match exactly the explicit mixed layers')
    coverage_banks = {layer: coverage_spec(value) for layer, value in coverage_banks.items()}
    root = Path(root).resolve()
    # Inventory existence first: no expensive reads or output if any bank is missing.
    missing = []
    for layer, kind in enumerate(kinds):
        sources = ([next(source for source, experts in coverage_banks[layer]['selection'].items() if expert in experts)
                    for expert in range(384)] if kind == COVERAGE else [kind] * 384)
        absent = [expert for expert in range(384)
                  if any(not (root / KINDS[sources[expert]] / f'layer-{layer:02d}' / f'expert-{expert:03d}-3bit{suffix}').is_file()
                         for suffix in ('.safetensors', '.json'))]
        inventory_missing = kind == 'baseline' and not (root / 'reports' / f'quant-layer-{layer:02d}-3bit.json').is_file()
        if absent or inventory_missing:
            missing.append(dict(layer=layer, kind=kind, missing_experts=len(absent),
                                missing_verified_baseline_inventory=inventory_missing))
    if missing:
        raise ValueError({'incomplete_selected_banks': missing})
    plans = []
    for layer, kind in enumerate(kinds):
        if kind == 'baseline':
            plan = baseline_layer_plan(layer, root, verify_hashes=verify_hashes)
        elif kind == 'supplemented':
            plan = supplemented_layer_plan(layer, root, check_inputs=check_inputs)
        elif kind == 'code_supplemented':
            plan = code_supplemented_layer_plan(layer, root)
        else:
            plan = coverage_layer_plan(layer, coverage_banks[layer], root)
        plans.append(plan)
    return plans


def make_manifest(kinds, root=ROOT, *, coverage_banks=None):
    root = Path(root).resolve()
    plans = selected_plans(kinds, root, coverage_banks=coverage_banks)
    config, identity = baseline.preflight(root)
    manifest = dict(format=FORMAT, source_revision=REVISION, bits=3,
        quality_status='evaluation_candidate_not_qualified', capture_sha256=config['capture_sha256'],
        corpus_sha256=identity['corpus_sha256'],
        source_verification_sha256=digest(root / 'reports/source-verification.json'),
        layers=[layer_entry(plan, kind, root) for plan, kind in zip(plans, kinds)])
    check_shape(manifest, root)
    return manifest


def read_manifest(path, root=ROOT):
    stored = json.loads(Path(path).read_text())
    manifest = stored['manifest']
    if stored['manifest_sha256'] != checksum(manifest):
        raise ValueError('Quantized manifest checksum changed')
    check_shape(manifest, root)
    config, identity = baseline.preflight(Path(root))
    if (manifest['capture_sha256'] != config['capture_sha256'] or manifest['corpus_sha256'] != identity['corpus_sha256']
            or manifest['source_verification_sha256'] != digest(Path(root) / 'reports/source-verification.json')):
        raise ValueError('Quantized manifest source/capture provenance changed')
    coverage_banks = {entry['layer']: dict(selection=entry['source_selection'], inventory_sha256=entry['inventory_sha256'])
                      for entry in manifest['layers'] if entry['kind'] == COVERAGE}
    plans = selected_plans([entry['kind'] for entry in manifest['layers']], root, check_inputs=False,
                           coverage_banks=coverage_banks)
    for plan, entry in zip(plans, manifest['layers']):
        if layer_entry(plan, entry['kind'], root) != entry:
            raise ValueError('Selected candidate bank changed since manifest creation')
    return manifest, plans
