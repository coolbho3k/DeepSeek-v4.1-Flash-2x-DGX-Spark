"""Checked full-trajectory lineage and teacher-forced comparison accounting."""
import json
import math
from pathlib import Path

import torch

from .quantized_capture import expected_receipt, validate_complete
from .quantized_manifest import checksum, digest

NUMERIC = ('reference_nll', 'candidate_nll', 'reference_to_candidate_kl')
BOOLEAN = ('top1_agreement', 'reference_target_top1', 'candidate_target_top1')


def verify_trajectory(output, config, prepared, plans):
    """Check every committed payload/hash link, not just the final directory."""
    output = Path(output)
    validate_complete(output, config)
    config_sha = checksum(config)
    parents = {item['entry']['id']: item['initial_sha256'] for item in prepared}
    for layer, plan in enumerate(plans):
        committed = json.loads((output / 'completed' / f'{layer:02d}.json').read_text())
        for item, entry in zip(prepared, committed['records']):
            name = item['entry']['id']
            path = output / 'states' / f'{layer + 1:02d}' / (name + '.safetensors')
            report_path = path.with_suffix('.json')
            stored = json.loads(report_path.read_text())
            receipt = stored['receipt']
            expected = expected_receipt(config_sha, layer, plan, item, parents[name])
            if (checksum(receipt) != stored['receipt_sha256'] or set(receipt) != set(expected) | {'state_sha256'}
                    or {key: receipt.get(key) for key in expected} != expected
                    or entry['record'] != name or entry['receipt_sha256'] != digest(report_path)
                    or entry['state_sha256'] != receipt['state_sha256'] or digest(path) != receipt['state_sha256']):
                raise ValueError('Quantized forward trajectory or committed checkpoint changed')
            parents[name] = receipt['state_sha256']
    if len(plans) != 40 or len(parents) != 64:
        raise ValueError('Incomplete40-layer/64-record quantized trajectory')
    summary = json.loads((output / 'summary.json').read_text())
    if (summary['status'] != 'quantized_full_model_states_captured' or summary['config_sha256'] != config_sha
            or summary['layers'] != 40 or summary['records'] != 64
            or summary['quality_validated'] is not False
            or summary['original_routed_source_weights_read'] != 0
            or summary['final_states'] != [dict(record=name, state_sha256=sha) for name, sha in parents.items()]):
        raise ValueError('Quantized final-state completion does not match the verified trajectory')
    return parents


def summarize(values):
    lengths = {len(values[name]) for name in NUMERIC + BOOLEAN}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError('Empty or inconsistent teacher-forced comparison columns')
    for name in NUMERIC:
        floor = -1e-5 if name == 'reference_to_candidate_kl' else 0
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < floor for value in values[name]):
            raise ValueError('Invalid NLL/KL value (small FP32 KL roundoff is retained, not clamped)')
    if any(type(value) is not bool for name in BOOLEAN for value in values[name]):
        raise ValueError('Teacher-forced top1 columns must be boolean')
    count = len(values['reference_nll'])
    means = {name: math.fsum(values[name]) / count for name in NUMERIC}
    return dict(tokens=count, reference_mean_nll=means['reference_nll'], candidate_mean_nll=means['candidate_nll'],
        mean_delta_nll=math.fsum(b - a for a, b in zip(values['reference_nll'], values['candidate_nll'])) / count,
        mean_reference_to_candidate_kl=means['reference_to_candidate_kl'],
        reference_perplexity=math.exp(means['reference_nll']) if means['reference_nll'] < 700 else None,
        candidate_perplexity=math.exp(means['candidate_nll']) if means['candidate_nll'] < 700 else None,
        **{name + '_fraction': sum(values[name]) / count for name in BOOLEAN})


def checked_groups(result, tokens, masks, entry):
    positions = torch.stack(list(masks.values())).any(0).nonzero().flatten()
    if (result['prediction_positions'] != positions.tolist() or result['target_ids'] != tokens[positions + 1].tolist()
            or set(result['masks']) != set(masks)
            or any(len(result[name]) != len(positions) for name in NUMERIC + BOOLEAN)):
        raise ValueError('Comparison targets, shifts, groups or lengths changed')
    summarize({name: result[name] for name in NUMERIC + BOOLEAN})
    groups = {}
    for group, native_mask in masks.items():
        mask = result['masks'][group]
        if any(type(value) is not bool for value in mask) or mask != native_mask[positions].tolist():
            raise ValueError('Comparison mask differs from verified assistant/token boundaries')
        values = {name: [value for value, keep in zip(result[name], mask) if keep] for name in NUMERIC + BOOLEAN}
        if len(values['reference_nll']) != entry['target_counts'][group]:
            raise ValueError('Comparison target count differs from the frozen record')
        summarize(values)
        groups[group] = values
    return groups
