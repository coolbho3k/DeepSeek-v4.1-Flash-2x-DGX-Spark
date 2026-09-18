"""Versioned, layerwise held-out forward states for explicitly selected EXL3.

Only initial embeddings come from source capture. Every subsequent hidden,
HC and shared-attention state comes from this quantized forward trajectory.
No calibration fitting, quality promotion, or source-state substitution.
"""
from contextlib import contextmanager
import fcntl
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .heldout_records import ROOT, digest, tensor_digest
from .quantized_manifest import checksum
from .quantized_reference import load_quantized_block

BASE_FIELDS = {'h', 'pre', 'tokens', 'types', 'hashes'}
FULL_METHOD = 'quantized_full_model_heldout_states_v1'
PROBE_METHOD = 'quantized_prefix_probe_states_v1'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Committed capture metadata changed: {path}')
    else:
        atomic_json(path, value)


def validate_state(state, runtime, boundary, tokens, types, hashes_sha256):
    args = runtime.args
    length = len(tokens)
    shared = set()
    if boundary > args.kv_source_layers[0]:
        shared = {'shared.compress_kv', 'shared.index_k', 'shared.topk_idxs'}
    if boundary > args.candidate_source_layer:
        shared.add('shared.candidates')
    if set(state) != BASE_FIELDS | shared:
        raise ValueError('Incomplete or unexpected quantized cross-layer state')
    h, pre, hashes = (state[name] for name in ('h', 'pre', 'hashes'))
    if (h.dtype != torch.bfloat16 or h.shape != (1, length, args.hc_mult, args.dim)
            or pre.dtype != torch.float32 or pre.shape != h.shape[:3]
            or hashes.dtype != torch.int64
            or hashes.shape != (1, length, len(args.engram_layer_ids), args.engram_n_heads * (args.engram_max_ngram_size - 1))):
        raise ValueError('Invalid native hidden/HC/hash state layout')
    if (not torch.equal(state['tokens'], tokens) or state['tokens'].dtype != torch.int64
            or not torch.equal(state['types'], types) or state['types'].dtype != torch.int64
            or tensor_digest(hashes) != hashes_sha256):
        raise ValueError('Quantized forward changed tokens, image types or engram hashes')
    if shared:
        ratio = args.compress_ratios[boundary - 1]
        expected = {'shared.compress_kv': (torch.bfloat16, (1, args.max_seq_len // ratio, args.head_dim)),
                    'shared.index_k': (torch.bfloat16, (1, args.max_seq_len // ratio, args.index_head_dim)),
                    'shared.topk_idxs': (torch.int32, (1, length, min(args.index_topk, length // ratio)))}
        if 'shared.candidates' in shared:
            expected['shared.candidates'] = (torch.bool, (1, length, length // ratio))
        for name, (dtype, shape) in expected.items():
            if state[name].dtype != dtype or state[name].shape != shape:
                raise ValueError(f'Invalid native shared attention layout: {name}')
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite quantized forward state')


@contextmanager
def refuse_source_routed_weights(runtime):
    original = runtime.weights.get
    def checked(key, device='cpu'):
        if '.ffn.experts.' in key:
            raise ValueError('Quantized evaluation must never read original routed expert weights')
        return original(key, device)
    runtime.weights.get = checked
    try:
        yield
    finally:
        runtime.weights.get = original


def implementation_paths(corpus):
    own = ('ds41/quantized_capture.py', 'ds41/quantized_manifest.py', 'ds41/quantized_reference.py',
           'scripts/capture_quantized_heldout.py', 'scripts/run_quant_queue.py', 'scripts/verify_supplemented.py',
           'scripts/verify_code_supplemented.py', 'scripts/quantize_code_supplemented.py',
           'ds41/calibration_inputs.py', 'ds41/calibration_inputs_v2.py', 'scripts/capture_code_topup.py',
           'ds41/coverage_bank.py', 'ds41/baseline_subset.py', 'ds41/safetensor_pack.py',
           'ds41/heldout_records.py', 'ds41/reference_runtime.py', 'ds41/exl3_moe.py', 'ds41/ssd_rows.py')
    native = ('inference/model.py', 'inference/config.json', 'inference/image_processor.py',
              'encoding/encoding.py', 'tokenizer.json', 'model.safetensors.index.json')
    return [corpus.root / name for name in own] + [corpus.runtime.source / name for name in native]


def check_code(config, root=ROOT):
    if any(digest(Path(root) / name) != sha for name, sha in config['implementation_sha256'].items()):
        raise ValueError('Quantized capture implementation changed during the run')


def prepare_config(corpus, plans, manifest_sha256, probe_records=None):
    if not plans or [plan.layer for plan in plans] != list(range(len(plans))):
        raise ValueError('Quantized capture requires a contiguous prefix starting at layer0')
    for plan in plans:
        plan.validate()
    if probe_records is None:
        if len(plans) != 40:
            raise ValueError('Full-model capture requires all40 explicitly selected expert banks')
        records, method = corpus.records, FULL_METHOD
    else:
        records, method = list(probe_records), PROBE_METHOD
        if not 0 < len(plans) < 40:
            raise ValueError('A prefix probe must not be presented as a full-model capture')
    if (not records or len({r['id'] for r in records}) != len(records)
            or any(record not in corpus.records for record in records)):
        raise ValueError('Only unique frozen held-out records may enter quantized evaluation')
    prepared = []
    for record in records:
        tokens, types, masks, entry = corpus.reconstruct(record)
        initial = corpus.capture / 'states/00' / (record['id'] + '.safetensors')
        sha = digest(initial)
        state = load_file(initial)
        hashes_sha = tensor_digest(state['hashes'])
        validate_state(state, corpus.runtime, 0, tokens, types, hashes_sha)
        if digest(initial) != sha:
            raise ValueError('Initial source embedding state changed while being read')
        prepared.append(dict(record=record, tokens=tokens, types=types, masks=masks, entry=entry,
            initial_path=initial, initial_sha256=sha, hashes_sha256=hashes_sha))
    config = dict(method=method, layers=len(plans), manifest_sha256=manifest_sha256,
        capture_sha256=digest(corpus.identity_path), source_revision=corpus.identity['source_revision'],
        corpus_sha256=corpus.identity['corpus_sha256'],
        source_verification_sha256=digest(corpus.root / 'reports/source-verification.json'),
        layer_inventory_sha256=[plan.inventory_sha256 for plan in plans],
        records=[dict(record=item['entry'], initial_state_sha256=item['initial_sha256'],
                      engram_hashes_sha256=item['hashes_sha256']) for item in prepared],
        implementation_sha256={str(path.relative_to(corpus.root)): digest(path) for path in implementation_paths(corpus)},
        torch_version=torch.__version__, arithmetic='Frozen EXL3 eager MoE; routed FP32 accumulation -> BF16 + native BF16 shared; TF32 disabled',
        scope='Quantized forward states only; teacher-forced quality and real TP2/DCP2 serving require separate evaluation.')
    return config, prepared


def expected_receipt(config_sha, layer, plan, item, parent_sha):
    return dict(config_sha256=config_sha, layer=layer, record=item['entry']['id'],
        layer_inventory_sha256=plan.inventory_sha256, parent_state_sha256=parent_sha,
        initial_state_sha256=item['initial_sha256'], engram_hashes_sha256=item['hashes_sha256'])


def read_commit(path, expected, runtime, boundary, item):
    report_path = path.with_suffix('.json')
    if path.exists() != report_path.exists():
        raise ValueError(f'Incomplete checkpoint transaction; inspect before resuming: {path}')
    if not path.exists():
        return None
    stored = json.loads(report_path.read_text())
    receipt = stored['receipt']
    if (checksum(receipt) != stored['receipt_sha256']
            or {key: receipt.get(key) for key in expected} != expected
            or set(receipt) != set(expected) | {'state_sha256'}
            or digest(path) != receipt['state_sha256']):
        raise ValueError('Quantized checkpoint lineage, configuration or payload changed')
    state = load_file(path)
    validate_state(state, runtime, boundary, item['tokens'], item['types'], item['hashes_sha256'])
    if digest(path) != receipt['state_sha256']:
        raise ValueError('Quantized checkpoint changed during validation')
    return receipt


@torch.inference_mode()
def forward_state(runtime, block, state):
    runtime.restore_shared({key.removeprefix('shared.'): value for key, value in state.items() if key.startswith('shared.')})
    h, pre = state['h'].cuda(), state['pre'].cuda()
    image_mask = state['types'].cuda().unsqueeze(0) >= 0
    with torch.device('cuda'):
        if block.engram is not None:
            h = block.engram(h, state['hashes'][:, :, block.engram.layer_hash_index].cuda(), ~image_mask)
        h, pre = block(h, 0, pre, image_mask)
    result = {name: state[name] for name in ('tokens', 'types', 'hashes')}
    result.update(h=h.cpu().contiguous(), pre=pre.cpu().contiguous())
    result.update({'shared.' + name: value for name, value in runtime.stash_shared(runtime.ref).items()})
    return result


def validate_complete(output, config):
    """Structural completion gate; readers still verify each state receipt/hash."""
    if config['method'] != FULL_METHOD or config['layers'] != 40 or len(config['records']) != 64:
        raise ValueError('Prefix/subset captures cannot be scored as a complete model')
    config_sha = checksum(config)
    names = [item['record']['id'] for item in config['records']]
    for layer in range(40):
        path = Path(output) / 'completed' / f'{layer:02d}.json'
        result = json.loads(path.read_text())
        if (result['config_sha256'] != config_sha or result['layer'] != layer
                or result['layer_inventory_sha256'] != config['layer_inventory_sha256'][layer]
                or [item['record'] for item in result['records']] != names):
            raise ValueError('Missing or incompatible complete quantized layer')
        directory = Path(output) / 'states' / f'{layer + 1:02d}'
        expected_files = {name + suffix for name in names for suffix in ('.json', '.safetensors')}
        if {path.name for path in directory.iterdir()} != expected_files:
            raise ValueError('Incomplete or unexpected quantized state inventory')


@torch.inference_mode()
def run_capture(corpus, plans, output, manifest_sha256, *, probe_records=None, observe=None):
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('Disable TF32 before quantized state capture')
    if observe is not None and probe_records is None:
        raise ValueError('Forward observers are restricted to explicitly labeled prefix probes')
    config, prepared = prepare_config(corpus, plans, manifest_sha256, probe_records)
    output = Path(output).resolve()
    if probe_records is None:
        if output.parent != corpus.root / 'reports' or not output.name.startswith('heldout-quantized-states-'):
            raise ValueError('Full capture needs its own reports/heldout-quantized-states-* directory')
    elif output.parent != Path('/tmp') or not output.name.startswith('ds41-quantized-capture-probe-'):
        raise ValueError('Prefix probes require a separate task-specific temporary directory')
    output.mkdir(parents=True, exist_ok=True)
    runtime, config_sha = corpus.runtime, checksum(config)
    with (output / 'capture.lock').open('a+') as lock, refuse_source_routed_weights(runtime):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_immutable(output / 'config.json', config)
        parents = {item['entry']['id']: (item['initial_path'], item['initial_sha256']) for item in prepared}
        for layer, plan in enumerate(plans):
            check_code(config, corpus.root)
            block, inventory = None, []
            try:
                for item in prepared:
                    name = item['entry']['id']
                    parent, parent_sha = parents[name]
                    if digest(parent) != parent_sha:
                        raise ValueError('Quantized parent state changed between layers')
                    path = output / 'states' / f'{layer + 1:02d}' / (name + '.safetensors')
                    expected = expected_receipt(config_sha, layer, plan, item, parent_sha)
                    receipt = read_commit(path, expected, runtime, layer + 1, item)
                    if receipt is None:
                        if block is None:
                            block = load_quantized_block(runtime, plan)
                        state = load_file(parent)
                        validate_state(state, runtime, layer, item['tokens'], item['types'], item['hashes_sha256'])
                        result = forward_state(runtime, block, state)
                        validate_state(result, runtime, layer + 1, item['tokens'], item['types'], item['hashes_sha256'])
                        if digest(parent) != parent_sha:
                            raise ValueError('Parent state changed during quantized forward')
                        if observe is not None:
                            observe(layer, item, block, state, result)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix('.partial.safetensors')
                        if temporary.exists():
                            raise ValueError('Unresolved partial state; inspect before resuming')
                        save_file({name: value.contiguous() for name, value in result.items()}, temporary)
                        temporary.replace(path)
                        receipt = dict(**expected, state_sha256=digest(path))
                        save_immutable(path.with_suffix('.json'), dict(receipt=receipt, receipt_sha256=checksum(receipt)))
                        del state, result
                        stage = 'quantized_heldout_forward'
                    else:
                        stage = 'quantized_heldout_resumed'
                    parents[name] = (path, receipt['state_sha256'])
                    inventory.append(dict(record=name, state_sha256=receipt['state_sha256'], receipt_sha256=digest(path.with_suffix('.json'))))
                    print(json.dumps(dict(stage=stage, layer=layer, record=name)), flush=True)
                check_code(config, corpus.root)
                save_immutable(output / 'completed' / f'{layer:02d}.json', dict(config_sha256=config_sha, layer=layer,
                    layer_inventory_sha256=plan.inventory_sha256, records=inventory))
            finally:
                runtime.close_tables()
                del block
                runtime.ref.shared_attn = runtime.ref.SharedAttentionRuntime()
                gc.collect()
                torch.cuda.empty_cache()
        if probe_records is None:
            validate_complete(output, config)
        summary = dict(status='quantized_full_model_states_captured' if probe_records is None else 'prefix_probe_only',
            config_sha256=config_sha, layers=len(plans), records=len(prepared),
            final_states=[dict(record=name, state_sha256=value[1]) for name, value in parents.items()],
            quality_validated=False, original_routed_source_weights_read=0)
        save_immutable(output / 'summary.json', summary)
        return summary
