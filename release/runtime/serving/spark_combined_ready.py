# SPDX-License-Identifier: AGPL-3.0-only
"""Successful-only inventory AFTER native target/draft capture and warmup.

Unlike the existing post-load receipt, this proves that the actual runner has
native full graphs with live validation owners. It does not claim throughput,
draft acceptance, model quality or that a user request has replayed a graph.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/cache/ds41-combined-ready')
MAX_BYTES = 32 * 2**10
import importlib.util
_profile_spec = importlib.util.spec_from_file_location("ds41_inspection_profile", Path(__file__).parent/"ds41/launch_profile.py")
_profile = importlib.util.module_from_spec(_profile_spec)
_profile_spec.loader.exec_module(_profile)
PROFILE = _profile.from_environment()
INITIAL_UTILIZATION = PROFILE["gpu_memory_utilization"]


def validate_vocabulary_receipt(descriptor, receipt, rank):
    enabled = descriptor.get('ssd_input_vocabulary', False)
    if type(enabled) is not bool:
        raise ValueError('Explicit input-vocabulary mode required')
    if not enabled:
        if receipt is not None:
            raise ValueError('Resident input vocabulary falsely reports offload')
        return
    if (type(receipt) is not dict or receipt.get('rank') != rank
            or receipt.get('parameter_bytes') != 0 or receipt.get('table_buffers') != 0
            or receipt.get('dtype') != 'torch.bfloat16'
            or receipt.get('io_threads') != 16 or receipt.get('source') != 'unchanged_checkpoint'
            or not 0 < receipt.get('staging_bytes', 0) <= 3*2**20
            or type(receipt.get('graph_owners')) is not int or not 0 <= receipt['graph_owners'] <= 18
            or receipt.get('healthy_callback') is not True):
        raise ValueError('Missing exact lossless input-vocabulary inventory')


def vocabulary_inventory(runner, descriptor, rank):
    if not descriptor.get('ssd_input_vocabulary', False):
        return None
    from ds41.native_vocab_stage import NativeVocabStage, _LIVE_STAGES
    from vllm.models.deepseek_v4_1.nvidia import model as native
    target = runner.get_model()
    language = target.get_language_model() if hasattr(target, 'get_language_model') else target
    embed = language.model.embed_tokens
    if (type(embed) is not native.VocabParallelEmbedding
            or not getattr(type(embed), '_ds41_lossless_input_vocabulary', False)
            or list(embed.parameters()) or list(embed.buffers()) or hasattr(embed, 'weight')
            or type(embed.stage) is not NativeVocabStage):
        raise ValueError('The input vocabulary still owns a resident table or changed constructor')
    stage = embed.stage
    if (_LIVE_STAGES != {stage} or stage.failed or stage.closed or not stage.native.store
            or stage.native.rank != rank or stage.native.threads != 16):
        raise ValueError('Input-vocabulary callback identity or health changed')
    managers = [runner.cudagraph_manager]
    if runner.speculator is not None:
        managers.append(runner.speculator.query_cudagraph_manager)
    retained = 0
    for manager in managers:
        for owner in manager._ds41_graph_resources.owners.values():
            if owner.vocab_stages and owner.vocab_stages != [stage]:
                raise ValueError('A model graph retains another vocabulary stage')
            retained += int(stage in owner.vocab_stages)
    # Native multimodal target prepares embeddings OUTSIDE its full graph;
    # zero owners is valid there. Native draft may capture the shared stage.
    if retained != stage.graphs:
        raise ValueError('Input-vocabulary native graph retention count differs')
    result = dict(rank=rank, parameter_bytes=0, table_buffers=0, dtype=str(stage.host_rows.dtype),
        io_threads=stage.native.threads, source='unchanged_checkpoint',
        staging_bytes=stage.staging_bytes, graph_owners=retained, healthy_callback=True)
    validate_vocabulary_receipt(descriptor, result, rank)
    return result


def graph_inventory(manager, required_tokens):
    resources = vars(manager).get('_ds41_graph_resources') if manager is not None else None
    if (resources is None or resources.poisoned or not manager.graphs
            or set(resources.graphs) != set(manager.graphs)
            or set(resources.owners) != set(manager.graphs)):
        raise ValueError('Full model graph manager has no complete matching owner inventory')
    rows = []
    for key, graph in manager.graphs.items():
        owner = resources.owners[key]
        if resources.graphs[key] is not graph or owner.failed:
            raise ValueError('Model graph ownership changed or contains a failed execution')
        if getattr(key.cg_mode, 'name', str(key.cg_mode)) != 'FULL':
            raise ValueError('Expected native full-model capture descriptors')
        rows.append(dict(tokens=key.num_tokens, requests=key.num_reqs))
    if required_tokens not in {row['tokens'] for row in rows}:
        raise ValueError('Missing the actual decode/verification graph shape')
    return sorted(rows, key=lambda row: (row['tokens'], row['requests'] or 0))


def record_ready_worker(worker):
    import torch
    import spark_combined_miaai as combined
    import spark_backend_attestation as loaded
    from ds41.combined_config import validate_worker
    from ds41.vllm_v2_cache import validate_groups
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head

    validate_worker(worker)
    descriptor = combined.register()
    enabled=descriptor['dspark_enabled']
    runner = worker.model_runner
    speculator = runner.speculator
    draft_graphs=[]
    if enabled:
        if type(speculator) is not DSparkSpeculator or type(speculator.model) is not DSparkDeepseekV4ForCausalLM:
            raise ValueError('Native DSpark is not the loaded speculator/model')
        draft = speculator.model
        target = runner.get_model()
        language = target.get_language_model() if hasattr(target, 'get_language_model') else target
        if (draft.model.embed_tokens is not language.model.embed_tokens
                or draft.lm_head is not get_target_lm_head(target, language)
                or draft.quant_config.get_name() != 'deepseek_v4_fp8'
                or len(draft.model.layers) != 3):
            raise ValueError('Draft quantization, layer count or shared vocabulary differs')
        draft_graphs=graph_inventory(speculator.query_cudagraph_manager,3)
    elif speculator is not None or worker.vllm_config.speculative_config is not None:
        raise ValueError('Target-only serving must not load a speculative model')
    from ds41.draft_exl3_serving import inventory as draft_inventory
    packed_draft = draft_inventory(draft) if enabled else None
    validate_groups(runner.kv_cache_config, dspark=enabled)
    target_graphs = graph_inventory(runner.cudagraph_manager, 4 if enabled else 1)
    if enabled:
        for requests in range(1, PROFILE['max_num_seqs']+1):
            if (not any(r['tokens']==4*requests and r['requests']==requests for r in target_graphs)
                    or not any(r['tokens']==3*requests and r['requests']==requests for r in draft_graphs)):
                raise ValueError(f'Missing target/draft graphs for {requests} requests')
    vocabulary = vocabulary_inventory(runner, descriptor, worker.rank)
    allocated, reserved = torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0)
    limit = int(torch.cuda.get_device_properties(0).total_memory * INITIAL_UTILIZATION)
    if not 0 < allocated <= reserved <= limit:
        raise ValueError('Post-capture allocator usage exceeds the frozen worker ceiling')
    result = dict(format='ds41_combined_ready_v1', rank=worker.rank,
        process=loaded.process_identity(os.getpid()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        combined_descriptor=descriptor, target_full_graphs=target_graphs,
        input_vocabulary=vocabulary,
        draft_full_graphs=draft_graphs, shared_vocabulary_identity=True if enabled else None,
        draft_quantization='deepseek_v4_fp8' if enabled else None,
        draft_expert_dtype='exl3_3bit_mul1' if enabled else None, draft_exl3=packed_draft,
        draft_layers=3 if enabled else 0, native_cache_groups_verified=True, gpu_utilization=INITIAL_UTILIZATION,
        torch_allocated_bytes=allocated, torch_reserved_bytes=reserved,
        allocator_limit_bytes=limit, request_graph_replay_measured=False,
        throughput_measured=False)
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_BYTES or ROOT.resolve() != ROOT:
        raise ValueError('Oversized or redirected combined ready receipt')
    ROOT.mkdir(exist_ok=False)
    with (ROOT / 'worker.json').open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(stage='ds41_combined_target_and_draft_captured', rank=worker.rank,
        target_graphs=len(target_graphs), draft_graphs=len(draft_graphs),
        torch_allocated_bytes=allocated, torch_reserved_bytes=reserved)), flush=True)


def inspect_receipt(rank, expected_sha, process, *, process_identity, root=ROOT):
    if (rank not in (0, 1) or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected_sha
            or root.resolve() != root or not root.is_dir()
            or sorted(p.name for p in root.iterdir()) != ['worker.json']):
        raise ValueError('Missing, changed or redirected combined ready inventory')
    path = root / 'worker.json'
    if path.resolve() != path or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Invalid combined ready receipt')
    result = json.loads(path.read_bytes())
    validate_vocabulary_receipt(result.get('combined_descriptor', {}),
        result.get('input_vocabulary'), rank)
    enabled=result.get('combined_descriptor',{}).get('dspark_enabled')
    if type(enabled) is not bool:
        raise ValueError('Explicit speculative mode required in ready inventory')
    if enabled:
        if (result.get('shared_vocabulary_identity') is not True
                or result.get('draft_quantization')!='deepseek_v4_fp8'
                or result.get('draft_expert_dtype')!='exl3_3bit_mul1'
                or result.get('draft_exl3',{}).get('actual_parameter_bytes')!=2562494976 or result.get('draft_layers')!=3
                or 3 not in {r['tokens'] for r in result.get('draft_full_graphs',[])}):
            raise ValueError('Missing native draft readiness proof')
    elif (result.get('shared_vocabulary_identity') is not None
            or result.get('draft_quantization') is not None or result.get('draft_expert_dtype') is not None
            or result.get('draft_layers')!=0 or result.get('draft_full_graphs')!=[]):
        raise ValueError('Target-only inventory falsely reports a draft')
    if (result.get('format') != 'ds41_combined_ready_v1' or result.get('rank') != rank
            or result.get('source_sha256') != expected_sha or result.get('process') != process
            or process_identity(process['pid']) != process
            or result.get('native_cache_groups_verified') is not True
            or result.get('gpu_utilization') != INITIAL_UTILIZATION
            or (4 if enabled else 1) not in {row['tokens'] for row in result.get('target_full_graphs', [])}
            or not 0 < result['torch_allocated_bytes'] <= result['torch_reserved_bytes'] <= result['allocator_limit_bytes']):
        raise ValueError('Combined ready inventory does not match the current loaded worker')
    return result
