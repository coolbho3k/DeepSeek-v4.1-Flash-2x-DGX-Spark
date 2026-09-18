"""Explicit full384 assembly from baseline subsets and rare-expert supplements.

Read-only verification returns existing expert paths; no weights are copied,
refitted, uploaded or promoted. Full-model manifest integration is separate.
"""
import json
from pathlib import Path

from .quantized_manifest import ROOT, REVISION, KINDS, ExpertArtifact, LayerPlan, checksum, digest
from .safetensor_pack import fingerprint
from scripts import run_quant_queue as baseline

SOURCES = ('baseline', 'supplemented', 'code_supplemented')
IMPLEMENTATION = (
    'ds41/coverage_bank.py', 'ds41/baseline_subset.py', 'ds41/calibration_inputs.py',
    'ds41/calibration_inputs_v2.py', 'scripts/verify_supplemented.py',
    'scripts/verify_code_supplemented.py', 'scripts/run_quant_queue.py',
    'ds41/safetensor_pack.py', 'ds41/quantized_manifest.py')


def selection(supplemented, code_supplemented):
    for experts in (supplemented, code_supplemented):
        if (any(type(e) is not int or not 0 <= e < 384 for e in experts)
                or list(experts) != sorted(set(experts))):
            raise ValueError('Supplement expert lists must be unique sorted IDs0..383')
    if set(supplemented) & set(code_supplemented):
        raise ValueError('One expert cannot select two supplemental sources')
    extra = set(supplemented) | set(code_supplemented)
    if not extra or len(extra) == 384:
        raise ValueError('Coverage assembly requires a baseline subset plus explicit supplemental experts')
    return dict(baseline=[e for e in range(384) if e not in extra],
                supplemented=list(supplemented), code_supplemented=list(code_supplemented))


def expert_path(root, layer, expert, kind):
    if kind not in SOURCES:
        raise ValueError('Unknown explicitly selected source')
    path = root / KINDS[kind] / f'layer-{layer:02d}' / f'expert-{expert:03d}-3bit.safetensors'
    for candidate in (path, path.with_suffix('.json')):
        if candidate.is_symlink() or candidate.resolve() != candidate:
            raise ValueError('Selected source path is redirected')
    return path


def verify_baseline(layer, experts, root):
    from .baseline_subset import verify_subset
    return verify_subset(layer, experts, root)


def verify_first(layer, experts, root):
    """Strengthen the frozen v1 verifier with exact replay and finite scales."""
    import torch
    from safetensors import safe_open
    from .calibration_inputs import CombinedRoutedInputs
    from scripts import verify_supplemented
    if root != ROOT:
        raise ValueError('Supplement replay requires this designated checkout')
    output = root / KINDS['supplemented']
    metadata = {path: digest(path) for path in (
        output / 'configs' / f'layer-{layer:02d}.json',
        output / 'input-manifests' / f'layer-{layer:02d}.json', Path(verify_supplemented.__file__))}
    result = verify_supplemented.verify(output, layer, experts, check_inputs=True)
    inputs = CombinedRoutedInputs(root / baseline.CAPTURE, root / baseline.CORPUS,
        [(root / 'calibration/capture-topup-v1', root / 'calibration/corpus-topup-v1')], layer, device='cpu')
    if (inputs.manifest != json.loads((output / 'input-manifests' / f'layer-{layer:02d}.json').read_bytes())
            or inputs.capture_sha256 != result['combined_capture_sha256']):
        raise ValueError('Actual supplemental input manifest changed')
    for entry in result['experts']:
        path = expert_path(root, layer, entry['expert'], 'supplemented')
        report_path = path.with_suffix('.json')
        before = (fingerprint(path), fingerprint(report_path))
        if digest(report_path) != entry['report_sha256'] or digest(path) != entry['artifact_sha256']:
            raise ValueError('Supplement changed between integrity check and exact replay')
        report = json.loads(report_path.read_bytes())
        if any(matrix['uncalibrated_fallback'] is not False for matrix in report['matrices'].values()):
            raise ValueError('Supplement must use explicitly calibrated matrices')
        for split, cap in (('calibration', 8192), ('heldout', 1024)):
            selected, expected = inputs.select(entry['expert'], split, cap, 41)
            del selected
            if report[split] != expected:
                raise ValueError('Supplemental sample differs from the actual frozen selector')
        with safe_open(path, framework='pt', device='cpu') as saved:
            for projection in ('w1', 'w2', 'w3'):
                for suffix in ('suh', 'svh'):
                    if not torch.isfinite(saved.get_tensor(f'layers.{layer}.ffn.experts.{entry["expert"]}.{projection}.{suffix}')).all():
                        raise ValueError('Nonfinite supplemental EXL3 scale')
        if (fingerprint(path), fingerprint(report_path)) != before:
            raise ValueError('Supplement changed during exact replay')
        entry['exact_frozen_selection_replayed'] = True
    inputs.validate_unchanged()
    if any(digest(path) != expected for path, expected in metadata.items()):
        raise ValueError('Supplemental configuration/manifest/verifier changed during replay')
    result.update(exact_selection_replay=True, release_qualified=False,
                  verifier_sha256=digest(Path(verify_supplemented.__file__)))
    return result


def verify_code(layer, experts, root):
    from scripts import verify_code_supplemented
    if root != ROOT:
        raise ValueError('Code supplemental replay requires this designated checkout')
    result = verify_code_supplemented.verify(root / KINDS['code_supplemented'], layer, experts)
    if (result['source_roles'] != ['baseline', 'calibration_only_supplement', 'calibration_only_code_topup_v2']
            or result['verifier_sha256'] != digest(Path(verify_code_supplemented.__file__))):
        raise ValueError('Unreviewed code-supplement provenance')
    return result


def validate_result(result, kind, layer, experts):
    if (result['status'] != 'verified' or result['layer'] != layer or result['bits'] != 3
            or [e['expert'] for e in result['experts']] != experts
            or any(e.get('exact_frozen_selection_replayed') is not True for e in result['experts'])):
        raise ValueError('Incomplete or incompatible requested-source verification')
    if kind == 'baseline':
        if (result['format'] != 'ds41_baseline_subset_verification_v1'
                or result['quality_qualified'] is not False or result['complete_layer'] is not False
                or result['config']['source_revision'] != REVISION):
            raise ValueError('Invalid baseline subset provenance/qualification')
    elif result['exact_selection_replay'] is not True or result['release_qualified'] is not False:
        raise ValueError('Supplemental verification must retain exact replay and unqualified status')


def verify_bank(layer, supplemented, code_supplemented, root=ROOT):
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError('Expected a backbone layer0..39')
    root = Path(root).resolve()
    chosen = selection(supplemented, code_supplemented)
    paths = {(kind, expert): expert_path(root, layer, expert, kind)
             for kind in SOURCES for expert in chosen[kind]}
    missing = [(kind, expert) for (kind, expert), path in paths.items()
               if not path.is_file() or not path.with_suffix('.json').is_file()]
    if missing:
        raise ValueError({'incomplete_explicit_coverage_bank': missing})
    # No expensive input scan and no output until every explicitly selected
    # artifact exists. Nothing silently falls back to another source directory.
    before = {str(p): fingerprint(p) for path in paths.values() for p in (path, path.with_suffix('.json'))}
    implementation = {relative: digest(root / relative) for relative in IMPLEMENTATION}
    results, artifacts, origins = {}, {}, {}
    for kind, verify in (('baseline', verify_baseline), ('supplemented', verify_first), ('code_supplemented', verify_code)):
        if not chosen[kind]:
            continue
        result = verify(layer, chosen[kind], root)
        validate_result(result, kind, layer, chosen[kind])
        if kind == 'baseline':
            counts = result['input_manifest']['counts']['calibration']
            if len(counts) != 384 or any(type(n) is not int or n < 0 for n in counts):
                raise ValueError('Invalid baseline route counts')
            deficient = [e for e, count in enumerate(counts) if count < 128]
            if sorted(chosen['supplemented'] + chosen['code_supplemented']) != deficient:
                raise ValueError('Coverage assembly must replace exactly the baseline calibration-deficient experts')
        # Validation work counters are not identities; source results remain
        # fully checked on every call, not trusted merely from an old receipt.
        result.pop('input_files_sha256_checked', None)
        results[kind] = checksum(result)
        for entry in result['experts']:
            expert = entry['expert']
            path = paths[kind, expert]
            if digest(path.with_suffix('.json')) != entry['report_sha256']:
                raise ValueError('Selected report changed during bank verification')
            prefix = f'layers.{layer}.ffn.experts.{expert}'
            artifacts[expert] = ExpertArtifact(expert, prefix, path, entry['artifact_sha256'],
                entry['report_sha256'], baseline.tensor_inventory(path, prefix, 3))
            origins[expert] = kind
    if sorted(artifacts) != list(range(384)):
        raise ValueError('Coverage bank does not contain all384 experts exactly once')
    receipt = dict(format='ds41_explicit_coverage_bank_v1', layer=layer, bits=3, source_revision=REVISION,
        selection=chosen, source_verification_sha256=results, implementation_sha256=implementation,
        complete_layer=True, quality_status='evaluation_candidate_not_qualified',
        artifacts=[dict(expert=e, kind=origins[e], path=str(artifacts[e].path.relative_to(root)),
            sha256=artifacts[e].sha256, report_sha256=artifacts[e].report_sha256,
            payload_bytes=artifacts[e].payload_bytes) for e in range(384)],
        scope='All384 explicitly selected calibrated3-bit experts; no refit/copy and no heldout-based selection. Integrity only, not a complete model or quality qualification.')
    if any(fingerprint(Path(path)) != value for path, value in before.items()):
        raise ValueError('A selected artifact changed during assembly')
    if any(digest(root / relative) != expected for relative, expected in implementation.items()):
        raise ValueError('Coverage assembly implementation changed during verification')
    plan = LayerPlan(layer, tuple(artifacts[e] for e in range(384)), checksum(receipt))
    plan.validate()
    return receipt, plan


def read_bank(path, root=ROOT):
    stored = json.loads(Path(path).read_bytes())
    receipt = stored['bank']
    if stored['bank_sha256'] != checksum(receipt):
        raise ValueError('Coverage bank receipt checksum changed')
    chosen = receipt['selection']
    if chosen != selection(chosen['supplemented'], chosen['code_supplemented']):
        raise ValueError('Invalid explicit source partition')
    current, plan = verify_bank(receipt['layer'], chosen['supplemented'], chosen['code_supplemented'], root)
    if current != receipt:
        raise ValueError('Selected coverage bank changed since its receipt was created')
    return receipt, plan
