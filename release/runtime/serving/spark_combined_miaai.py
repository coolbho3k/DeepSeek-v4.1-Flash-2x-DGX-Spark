# SPDX-License-Identifier: AGPL-3.0-only
# MiaAI-Lab attribution and original MIT notices are retained under
# ../vendor/miaai-serving-stack-agpl. Canonical target weights are unchanged.
"""Prepare the combined V2/graphs/1536-row/96-thread serving stack.

Startup only, before ANY FP4/plugin/native-hook registration. This constructs
and validates all Python changes before installing them together. No model,
CUDA context, GPU buffer, SSD stage or graph is created by registration.
DSpark is a separate admission condition, not falsely reported as enabled.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
from functools import partial
from types import FunctionType

PINS = {
    'ds41.cooperative_contract': '7992e34e9ea867d1f8816b48e426ea95a7526356272ca790f65bdea01d6195ee',
    'ds41.cooperative_routes': 'ae91ace2d4310f9409ade1367b341cadd90bc28b152e355831144f9b09dc7be7',
    'ds41.cooperative_moe': '22edf8a5aed2194ac0fd2050a9fd27ffea48dd82de021c40f68fd17109ec6bc5',
    'ds41.dcp_communication': '7aa4a5e6d978f4be72db26e65619e75c7c09e75a218426afbf4a8aab1d56ce20',
    'spark_dcp_communication': '66607252ec7a96458fcd52bed36fd1841ac75409a51e1bb0cf3509a665321117',
    'spark_sparse_slots': '67bd8fed269ead5fce990d75a396fcab5490885af4e55d5b4026fbe19191831a',
    'ds41.dcp_candidates': 'af608345e8d492299508ed6f4cf6ab58b6baaf1da7149c2a681127000520c726',
    'spark_topk': 'e74afeaf1ae8d3edd8388a2c36fbdfdcf9ab0464d22f4b186af81d4d08820521',
    'ds41.vllm_exl3': 'd23c1e0df03cc097cad69314e53cef161cf599bba0b7b726d082cf144ffdfaa3',
    'ds41.vllm_plugin': '410f34999eae5fea656654cb86a3e089b24a38cd108ca56e142c14e317e4d8e5',
    'ds41.dcp_sparse_slots': 'acbd5dce12e3a988697268c946f7c1a178cc38a3dc738dbb5a94287b7cc43edb',
    'ds41.fused_sparse_attention': '22839ccef76d501a9191b193a522429dcfe98ae9977f31f51be04455ea51d9e9',
    'ds41.vllm_dcp_cache': '37b914ef4d81fae4b61a891736240f49bebb9e95dbc0c013b6e3d4eb98fd16ce',
    'ds41.dcp_key_order': '4269e3e21a58001721a328fccccd9abd30d3acc0d84511100898baf5775431a3',
    'spark_kv_cap': '953b4acc4584eccde51ee974aa04698695076fa96d1457bdb3e072b41bc2c94b',
    'spark_fused_moe': 'c5b573e624be1d4d92ebfcf39e1968130a57e85c3c8c1dde7c9982120639b3de',
    'spark_fused_moe_async': '5000f91b5b8690a4b08a6220e254035914f8675bb9210787e2b005261c304816',
    'spark_b12x_decode_graph': '5b47f0733788c53a3d4b6c1539d5b1564d032f150159c30053cb764fce6fc7fb',
    'spark_grouped_prefill': 'e09cde421ec879a9e6e34aca2cdd6219f437d78d9cf6512e6dccf4a4f57ae839',
    'miaai_engram': 'c9b751ec4ee4251acc26dece3c7794408bb305a45ea32cf666a784e49033aacb',
    'spark_native_engram': 'e407bc2281c481f3de875b616580ef3cf4ef00236591ea0b090b62a104ceb9d5',
    'spark_indexer_k_math': '5ee4f01443af118a6bc50393a967860a30980f7314e38207800a3fbf9840ac57',
    'spark_packed_wo_a': '5bf809bed67f28e2b94b345d039e5e808cb8fcb556d49dc1edfecefe1bbf4fe5',
}
KERNEL_BATCH = {'online_decode_attention': True, 'length_aware_radix_topk': True}
DESCRIPTOR = dict(kernel_batch=KERNEL_BATCH, implementation='combined_miaai_v2_graph_prefill_v1',
    license='AGPL-3.0-only', upstream_commit='8404ac7d389c418300d0bee960d52313247930e1', draft_experts='exl3_3bit_mul1',
    full_model_graphs=True, graph_validation='device_masked_flags_checked_before_output',
    maximum_prefill_tokens=3072, io_threads=96, target_quantization_unchanged=True,
    staged_single_token_experts=True,
    fused_route_preparation=True,parallel_route_prefix=True,
    fused_dense_input_quant=True,direct_final_attention_output=True,
    mhc_decode_prenorm=True,
    verification_fastpath_rows=[2,3,4],
    prefill_zero_copy_output_packing=True,
    draft_context_discarded_queries_elided=True,
    ssd_input_vocabulary=False,
    image_pixels_unchanged=True, vision_dtype='bfloat16', dspark_enabled=False,
    gpu_utilization_ceiling=0.925, native_kv_admission_preserved=True,
    dcp_collective_tokens=512, synchronous_cache_bounds_checks=True,
    nccl_memory_settings=dict(NCCL_BUFFSIZE='1048576', NCCL_LL128_BUFFSIZE='262144',
                              NCCL_PROTO='^LL128', NCCL_MAX_NCHANNELS='8'))
_lock = threading.Lock()
_installed = None
_MISSING = object()


def _shared_compile(function, replacements, extra_globals, compile_fn):
    """Retain the real module's register-state globals, not a copied dict.

    Pure native forward rewrites may use copied globals; register() methods
    cannot, because their idempotence state must remain visible to observers.
    Extra bindings are returned for the same atomic startup commit.
    """
    compiled = compile_fn(function, replacements, extra_globals)
    if compiled.__code__.co_freevars:
        raise RuntimeError('Combined rewrites must use explicit, inspectable globals')
    result = FunctionType(compiled.__code__, function.__globals__, function.__name__,
                          function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__qualname__ = function.__qualname__
    result.__ds41_patch_source__ = compiled.__ds41_patch_source__
    return result


def register():
    global _installed
    if os.environ.get('DS41_ENABLE_COMBINED_MIAAI') != '1':
        raise ValueError('Explicit combined-candidate startup selection required')
    vocabulary_selection = os.environ.get('DS41_ENABLE_SSD_VOCAB', '0')
    if vocabulary_selection not in ('0', '1'):
        raise ValueError('SSD input-vocabulary selection must be explicitly0 or1')
    vocabulary_enabled = vocabulary_selection == '1'
    cooperative_selection = os.environ.get('DS41_ENABLE_COOPERATIVE_MOE', '0')
    if cooperative_selection not in ('0', '1'):
        raise ValueError('Cooperative MoE selection must be0 or1')
    cooperative_enabled = cooperative_selection == '1'
    from ds41 import combined_config as config
    config.configure_transport()
    config.collective_chunk_size()
    with _lock:
        if _installed is not None:
            if DESCRIPTOR.get('cooperative_moe', False) is not cooperative_enabled:
                raise RuntimeError('Cooperative MoE selection changed after startup')
            if DESCRIPTOR['ssd_input_vocabulary'] is not vocabulary_enabled:
                raise RuntimeError('Input-vocabulary selection changed after startup')
            if any(getattr(owner, name, _MISSING) is not new for owner, name, _, new in _installed):
                raise RuntimeError('Combined candidate binding changed after startup')
            return dict(DESCRIPTOR)
        modules = {name: importlib.import_module(name) for name in PINS}
        for name, module in modules.items():
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != PINS[name]:
                raise RuntimeError('Unreviewed combined-stack source: ' + name)
        from ds41 import vllm_dcp as arithmetic, vllm_dcp_runtime as runtime
        from ds41 import vllm_prefill_workspace as workspace, vllm_fp4_main as fp4
        from ds41 import fp4_main_kv, fp4_rope_store, combined_config as config
        from ds41 import vllm_v2_cache as cache_v2, vllm_owned_graphs as graphs
        from ds41 import graph_validation as validation, dcp_indexer_graph as indexer
        from ds41 import combined_dspark as dspark
        from ds41 import online_decode_attention as batch_attention
        from ds41 import length_aware_topk_native as batch_topk
        from ds41 import dcp_topk_graph as topk_graph
        from ds41 import dcp_candidates_graph as candidates_graph
        from ds41 import dcp_head_exchange as head_exchange
        from ds41 import online_sparse_attention as online_attention
        from ds41 import staged_route_prepare as route_prepare
        from ds41 import dense_fused_input as dense_fused
        from ds41 import mhc_decode_prenorm as mhc_prenorm
        from vllm.utils import deep_gemm
        dspark_enabled = os.environ.get('DS41_ENABLE_DSPARK', '0') == '1'
        if os.environ.get('DS41_ENABLE_DSPARK', '0') not in ('0', '1'):
            raise ValueError('DSpark selection must be explicitly0 or1')
        cache = modules['ds41.vllm_dcp_cache']
        plugin = modules['ds41.vllm_plugin']
        engram = modules['miaai_engram']
        native_engram = modules['spark_native_engram']
        moe = modules['spark_fused_moe']
        grouped = modules['spark_grouped_prefill']
        if (runtime._installed or fp4._installed or workspace._original is not None
                or plugin._registered or native_engram._installed is not None
                or engram._LIVE_STAGES or grouped._installed is not None
                or moe._dispatcher is not None):
            raise RuntimeError('Combined candidate must precede all model/kernel registration')
        if os.environ.get('DSV41_IO_THREADS', '96') != '96':
            raise ValueError('Combined Engram requires96 I/O threads')
        changes = []

        def validate_moe_capture(experts, ids):
            import torch
            if not torch.cuda.is_current_stream_capturing():
                return
            validation.require_capture_owner()
            if type(moe._dispatcher) is not grouped.GroupedDispatcher:
                raise RuntimeError('Only the device-only grouped dispatcher may enter a model graph')
            if experts and ids.numel():
                validation.current_owner().retain_moe(moe._dispatcher, experts,
                    needs_fat=ids.numel() > 128 and not (cooperative_enabled and modules['ds41.cooperative_contract'].selected_shape((ids.shape[0],5120),ids.shape)))

        def put(owner, name, value):
            changes.append((owner, name, getattr(owner, name, _MISSING), value))

        put(deep_gemm,'tf32_hc_prenorm_gemm',
            mhc_prenorm.wrap(deep_gemm.tf32_hc_prenorm_gemm,tile_n=4,warps=8))

        # Bind the actual imported helper references used by make_patches().
        # Keep the legacy module itself untouched for packed eager prefill.
        put(arithmetic,'select_candidate_blocks',candidates_graph.select_candidate_blocks)
        put(arithmetic,'apply_candidate_mask',candidates_graph.apply_candidate_mask)
        # Only the independent query-row batch grows. All per-row arithmetic,
        # FP32 exchanges, sparse ordering and synchronous bounds checks stay.
        put(fp4, '_collective_chunk_size', config.collective_chunk_size)
        put(modules['spark_sparse_slots'], 'DESCRIPTOR', dict(
            modules['spark_sparse_slots'].DESCRIPTOR, maximum_rows=512))
        put(modules['spark_dcp_communication'], 'DESCRIPTOR', dict(
            modules['spark_dcp_communication'].DESCRIPTOR, maximum_rows=512,
            maximum_packed_send_bytes=512*32*513*4))
        communication = modules['ds41.dcp_communication']
        put(communication, 'pack_result', head_exchange.pack_result)
        put(communication, 'merge_packed', head_exchange.merge_packed)
        put(communication, 'forward_replacements', head_exchange.forward_replacements)

        def compile_shared(owner, name, rules, extras=None):
            extras = extras or {}
            original = getattr(owner, name)
            replacement = _shared_compile(original, rules, extras, arithmetic._compile)
            module = importlib.import_module(original.__module__)
            for key, value in extras.items():
                put(module, key, value)
            put(owner, name, replacement)

        dcp_hook = modules['spark_dcp_communication']
        compile_shared(dcp_hook, 'register', [
            ('all_packed = group.all_gather(_ds41_pack_result(partial, lse), dim=0)',
             'all_packed = group.all_gather(_ds41_pack_result(partial, lse, rank), dim=0)'),
        ])

        native = json.loads((Path(__file__).parent / 'combined-native.json').read_bytes())
        if (native.get('status') != 'combined_moe_built_cpu_only'
                or native.get('contract_version') != 2 or native.get('max_rows') != 2048
                or native.get('graph_capture_supported') is not True):
            raise RuntimeError('Combined graph/2048-row native launcher is required')
        put(moe, 'BINARY_SHA256', native['binary_sha256'])
        staged=native.get('staged_decode_candidate')
        if (not isinstance(staged,dict) or staged.get('cooperative_launch') is not False
                or staged.get('fp32_mma_accumulation_preserved') is not True
                or staged.get('original_epilogues_preserved') is not True
                or staged.get('parallel_route_prefix') is not True):
            raise RuntimeError('Staged candidate must retain original FP32/epilogue contracts')
        small=native.get('staged_small_candidate')
        if small is not None and (small.get('rows') != [2,3,4]
                or small.get('extra_gpu_allocation_bytes') != 0
                or small.get('original_precision_epilogues_preserved') is not True
                or small.get('original_one_token_source_preserved') is not True):
            raise RuntimeError('Small-batch staged candidate changed its precision/workspace contract')
        grouped_candidate=native.get('staged_grouped_candidate')
        if grouped_candidate is not None and (small is None or grouped_candidate.get('rows') != [2,3,4]
                or grouped_candidate.get('maximum_group_rows') != 4
                or grouped_candidate.get('extra_gpu_allocation_bytes') != 0
                or grouped_candidate.get('original_precision_epilogues_preserved') is not True
                or grouped_candidate.get('original_small_source_preserved') is not True):
            raise RuntimeError('Grouped staged candidate changed its precision/workspace contract')
        from ds41.staged_decode import wrap_module
        compile_shared(moe, 'load_kernel', [
            ('module.contract_version() != 1',
             '(module.contract_version() != 2 or module.max_rows() != 2048 or not module.graph_capture_supported())'),
            ('return module','return _ds41_wrap_staged_module(module)'),
        ], {'_ds41_wrap_staged_module':wrap_module})
        async_moe=modules['spark_fused_moe_async']
        route_rules = route_prepare.forward_replacements()
        route_globals = {'_ds41_prepare_routes':route_prepare.prepare}
        if cooperative_enabled:
            from ds41 import cooperative_moe, cooperative_contract
            route_rules += cooperative_contract.forward_replacements()
            route_globals.update(_ds41_coop_eligible=cooperative_contract.selected_shape,
                _ds41_coop_call=cooperative_moe.configure(Path(__file__).parent))
        compile_shared(async_moe.AsyncSmallDispatcher,'__call__',route_rules,route_globals)
        dense_hook=modules['spark_b12x_decode_graph']
        compile_shared(dense_hook,'_install',[
            ('native_apply = backend.apply_weights',
             'native_apply = _ds41_wrap_dense_apply(backend.apply_weights)'),
        ],{'_ds41_wrap_dense_apply':dense_fused.wrap_apply})

        put(arithmetic, 'validate_config', config.validate_config)
        put(workspace, 'row_bound', config.row_bound)
        put(cache, 'row_bound', config.row_bound)
        put(cache, 'validate_groups', partial(cache_v2.validate_groups, dspark=dspark_enabled))
        put(cache, 'make_cache_patches', partial(cache_v2.make_cache_patches, dspark=dspark_enabled))
        put(arithmetic, 'paged_logits', indexer.paged_logits)
        if modules['spark_topk']._installed is not None:
            raise RuntimeError('Install combined top-k before indexer registration')
        put(modules['spark_topk'], 'decode_topk',
            batch_topk.wrap(topk_graph.decode_topk, Path(__file__).parent/'topk-native')
            if KERNEL_BATCH['length_aware_radix_topk'] else topk_graph.decode_topk)
        put(modules['spark_kv_cap'], 'capped_budgets', config.capped_budgets)
        # Preserve all loader/format/SSD guards; replace only eager selection
        # with the explicit whole-stack validator above.
        compile_shared(plugin, 'register', [
            ('if not current.model_config.enforce_eager:\n            raise ValueError("SSD engrams currently require --enforce-eager")',
             '_ds41_validate_combined(current)'),
        ], {'_ds41_validate_combined': config.validate_config})
        adapter = modules['ds41.vllm_exl3']
        compile_shared(adapter.DS41EXL3MoEMethod, 'create_weights', [
            ('if not current.model_config.enforce_eager or current.parallel_config.enable_expert_parallel:\n        raise ValueError("Initial DS41 EXL3 backend requires eager TP without expert parallel")',
             '_ds41_validate_combined(current)'),
        ], {'_ds41_validate_combined': config.validate_config})
        # All addresses were already masked in these kernels. Eager callers
        # still see the original errors; native graph owners coalesce readback.
        attention = modules['ds41.fused_sparse_attention']
        put(attention, '_attention', online_attention.prefill)
        compile_shared(attention, 'packed_sparse_attention_with_lse', [
            ('not 0 <= query.shape[0] <= 64', 'not 0 <= query.shape[0] <= 512'),
            ('[0..64 tokens, 32/64 heads, 512]', '[0..512 tokens, 32/64 heads, 512]'),
            ('output = torch.empty(query.shape, device=query.device, dtype=torch.float32)\n'
             '    lse = torch.empty(query.shape[:2], device=query.device, dtype=torch.float32)',
             'output, lse = _ds41_attention_outputs(query, split_k)'),
            ("if torch.cuda.is_current_stream_capturing():\n        raise ValueError('Bounds-checked eager attention cannot run inside graph capture')",
             '_ds41_require_capture_owner()'),
            ("if error.item():\n        raise ValueError('Sparse slot exceeds allocated packed cache')",
             "_ds41_check_flags(error, ((1, 'Sparse slot exceeds allocated packed cache'),))"),
        ], {'_ds41_require_capture_owner': validation.require_capture_owner,
            '_ds41_check_flags': validation.check_flags,
            '_ds41_attention_outputs': head_exchange.allocate_attention_outputs})
        sparse = modules['ds41.dcp_sparse_slots']
        compile_shared(sparse, 'sparse_global_to_local_slots', [
            ("flags=errors.cpu().tolist()\n    if any(value&1 for value in flags):\n        raise ValueError('Invalid request index')\n    if any(value&2 for value in flags):\n        raise ValueError('Sparse candidate exceeds allocated block table')\n    if any(value&4 for value in flags):\n        raise ValueError('Unallocated sparse candidate page')",
             "_ds41_check_flags(errors, ((1, 'Invalid request index'), (2, 'Sparse candidate exceeds allocated block table'), (4, 'Unallocated sparse candidate page')))"),
        ], {'_ds41_check_flags': validation.check_flags})
        compile_shared(engram.NativeStage, '__init__', [
            ("not in ('32', '64')", "not in ('96',)"),
            # Eager staging and native model graphs share this fence. CUDA
            # requires an external event node when capture waits on work
            # recorded outside that graph; ordinary events fail isolation.
            ('self.event = torch.cuda.Event()',
             'self.event = torch.cuda.Event(external=True)'),
        ])
        compile_shared(moe.Workspace, '__init__', [
            ('self.ready = torch.cuda.Event()',
             'self.ready = torch.cuda.Event(external=True)'),
        ])
        compile_shared(engram.NativeStage, 'lookup', [
            ("raise RuntimeError('Use NativeStage.capture for managed callback graph lifetime')",
             '_ds41_require_capture_owner()\n        return _ds41_current_owner().enqueue_stage(self, indices, out)'),
        ], {'_ds41_require_capture_owner': validation.require_capture_owner,
            '_ds41_current_owner': validation.current_owner})
        compile_shared(native_engram, 'register', [
            ("('DSV41_IO_THREADS','32')", "('DSV41_IO_THREADS','96')"),
        ])
        descriptor = dict(native_engram.DESCRIPTOR, io_threads=96, full_model_graph_capture_enabled=True)
        put(native_engram, 'DESCRIPTOR', descriptor)
        for module, name, expected, value in (
            (modules['ds41.dcp_communication'], 'MAX_ROWS', 64, 512),
            (sparse, 'MAX_ROWS', 64, 512),
            (fp4_main_kv, 'MAX_WRITE_ROWS', 1056, config.MAX_TOKENS),
            (fp4_rope_store, 'MAX_WRITE_ROWS', 1056, config.MAX_TOKENS),
            (engram, 'MAX_TOKENS', 1056, config.MAX_TOKENS),
            (grouped, 'MAX_ROWS', 1056 * 6, 2048 * 6),
            (grouped, 'MAX_SEGMENTS', 483, 576),
            (modules['spark_indexer_k_math'], 'MAX_ROWS', 1056, config.MAX_TOKENS),
            (modules['spark_indexer_k_math'], 'MAX_TEMPORARY_BYTES', 1056 * 256, config.MAX_TOKENS * 256),
        ):
            if getattr(module, name) != expected:
                raise RuntimeError(f'Unexpected pre-existing row bound: {module.__name__}.{name}')
            put(module, name, value)
        scratch_bytes = (2048 * 6) * (2 * (5120 + 5120 + 1152) + 8 + 4 + 4) + 576 * 12
        compile_shared(grouped.FatWorkspace, '__init__', [
            ('self.bytes!=144466596', f'self.bytes!={scratch_bytes}'),
        ])
        compile_shared(grouped.GroupedDispatcher, '__call__', [
            ('base.validate_inputs(experts,x,ids,weights,chunk_tokens)', "if len(x) > 2048:\n        if len(x) > _ds41_prefill_limit or torch.cuda.is_current_stream_capturing():\n            raise ValueError('Large MoE must be bounded eager prefill')\n        if ids.shape[0] != len(x) or weights.shape != ids.shape:\n            raise ValueError('MoE batch routing shape mismatch')\n        return torch.cat([self(experts,x[start:start+2048],ids[start:start+2048],\n            weights[start:start+2048],chunk_tokens) for start in range(0,len(x),2048)],dim=0)\n    base.validate_inputs(experts,x,ids,weights,chunk_tokens)"),
            ('return super().__call__(experts,x,ids,weights,chunk_tokens)',
             'return AsyncSmallDispatcher.__call__(self,experts,x,ids,weights,chunk_tokens)'),
            # Repeated route IDs were valid in the existing contract. A
            #2048x6 all-one-expert case needs192 segments, not at most128.
            ('fat.seg_rows,fat.seg_lengths,MAX_SEGMENTS,128)',
             'fat.seg_rows,fat.seg_lengths,MAX_SEGMENTS,256)'),
        ], {'_ds41_prefill_limit':config.MAX_TOKENS})
        compile_shared(moe, 'validate_inputs', [
            ('len(x) <= 1056', 'len(x) <= 2048'),
            ("if torch.cuda.is_current_stream_capturing():\n        raise ValueError('DS41 MUL1 shared workspace requires eager execution')",
             '_ds41_validate_moe_capture(experts, ids)'),
        ], {'_ds41_validate_moe_capture': validate_moe_capture})
        order = modules['ds41.dcp_key_order']
        compile_shared(order, 'canonicalize_selected_keys_', [
            ('indices.shape[0] <= 1056', f'indices.shape[0] <= {config.MAX_TOKENS}'),
        ])
        # arithmetic imported this function by value before preparation.
        put(arithmetic, 'canonicalize_selected_keys_', changes[-1][3])
        packed = modules['spark_packed_wo_a']
        compile_shared(packed, 'grouped_projection', [
            ('len(x) > 1056', f'len(x) > {config.MAX_TOKENS}'),
            ("if torch.cuda.is_current_stream_capturing():\n        raise RuntimeError('Packed wo_a graph capture is not qualified')",
             '_ds41_require_capture_owner()'),
        ], {'_ds41_require_capture_owner': validation.require_capture_owner})
        for owner, name, replacement in graphs.make_graph_patches():
            put(owner, name, replacement)
        if dspark_enabled:
            for owner, name, replacement in dspark.make_patches():
                put(owner, name, replacement)
        if vocabulary_enabled:
            from ds41 import combined_vocab as vocabulary
            for owner, name, replacement in vocabulary.make_patches():
                put(owner, name, replacement)
        put(importlib.import_module(__name__), 'DESCRIPTOR', dict(DESCRIPTOR,
            dspark_enabled=dspark_enabled, dspark_full_model_qualified=False,
            serving_profile=dict(config.PROFILE), maximum_prefill_tokens=config.MAX_TOKENS,
            native_moe_subcall_maximum_tokens=2048,
            cooperative_moe=cooperative_enabled,
            cooperative_upstream_commit='b9c49e90bdcc6f1e0192feb57214df11b67d36aa',
            cooperative_rows=list(range(1,25)) if cooperative_enabled else [],
            cooperative_small_rows_keep_staged=False,
            cooperative_additional_persistent_scratch_bytes=0,
            cooperative_quality_qualified=False, cooperative_performance_measured=False,
            staged_verification_rows=[2,3,4] if small is not None else [],
            grouped_verification_rows=[2,3,4] if grouped_candidate is not None else [],
            ssd_input_vocabulary=vocabulary_enabled))
        keys = [(id(owner), name) for owner, name, _, _ in changes]
        if len(keys) != len(set(keys)):
            raise RuntimeError('Duplicate combined-stack replacement target')
        applied = []
        try:
            for index, (owner, name, old, new) in enumerate(changes):
                if owner is attention and name == 'packed_sparse_attention_with_lse' and KERNEL_BATCH['online_decode_attention']:
                    new = batch_attention.wrap(new)
                    changes[index] = (owner, name, old, new)
                if getattr(owner, name, _MISSING) is not old:
                    raise RuntimeError('Combined startup target changed while preparing')
                setattr(owner, name, new)
                applied.append((owner, name, old, new))
        except BaseException:
            for owner, name, old, _ in reversed(applied):
                if old is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, old)
            raise
        _installed = tuple(changes)
        return dict(DESCRIPTOR, python_bindings=len(changes), grouped_scratch_bytes=scratch_bytes)
