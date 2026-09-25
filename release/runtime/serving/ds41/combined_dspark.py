# SPDX-License-Identifier: AGPL-3.0-only
# DS41 integration for the attributed MiaAI serving campaign. Pinned native
# vLLM implementations retain their Apache-2.0 notices and draft mathematics.
"""Scoped DSpark loading, image scheduling and replicated draft-cache slots.

Preparation is CPU-only and does not install anything. The combined startup
transaction commits these hooks with target graphs and per-group V2 caches.
This is implementation, not evidence of full-model memory fit or performance.
"""
from contextvars import ContextVar
import hashlib
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

UPSTREAM = {
    'vllm.config.speculative':
        '537622adc72d954a8ffe9076bc09c231d008f0cc89e6786d0d40300e70775a5d',
    'vllm.models.deepseek_v4_1.nvidia.dspark':
        '37158c568791bb74607e9a8b3ac8e49be6864d98aa04628cdbf31f1dcfc0fc18',
    'vllm.v1.worker.gpu.spec_decode.dspark.utils':
        '5be27eecf7cd08c49ac45977d7615625592801afac391ebf5381726c2dd4db03',
    'vllm.v1.worker.gpu.spec_decode.dspark.speculator':
        'ed020e35bccf6281382f24132717e2f48575acc35d9f26acba2e930cb8f2e8a7',
    'vllm.v1.worker.gpu.spec_decode.dflash.speculator':
        'c14815574ba63baccafada080305379f426441d0a54b1bf9c7c51529066193e0',
    'vllm.v1.worker.gpu.spec_decode.dflash.cudagraph':
        '2d4afe57efb13586decbd42943873467a3fe618e2baf3441e344e8a864721ac6',
}
STREAMING_SHA256 = '3e8e04c7fdeadc2027a4bcad3a8f74a2078214d9990eed2d4a220a691e96ed8c'
VISION_SHA256 = 'dd2b90570f6f42a70ce3b2b997e0027d98a6baa75889b441ae5c60c6443173aa'
_draft_scope = ContextVar('ds41_combined_draft_load', default=None)
NATIVE_DRAFT_QUANTIZATION = 'deepseek_v4_fp8'
DENSE_GRAPH_CORE_SHA256 = '89bb94bf73ebb2859399abc61350794b9cffe205c79d5329b3498e927a60acd2'


def validate_speculation(config):
    spec = config.speculative_config
    if spec is None:
        if os.environ.get('DS41_ENABLE_DSPARK') == '1':
            raise ValueError('DSpark startup selection requires the native speculative config')
        return False
    if (os.environ.get('DS41_ENABLE_DSPARK') != '1'
            or getattr(spec, 'method', None) != 'dspark'
            or getattr(spec, 'num_speculative_tokens', None) != 3
            or getattr(spec, 'enforce_eager', None) is True
            or getattr(spec, 'enable_adaptive_verification', False)
            or getattr(spec, 'kv_cache_dtype', None) not in (None, 'fp8_ds_mla')):
        raise ValueError('Combined speculation requires explicit native DSpark, three drafts and fixed verification')
    draft = getattr(spec, 'draft_model_config', None)
    hf = getattr(draft, 'hf_config', None)
    expected = dict(hidden_size=5120, vocab_size=129280, num_hidden_layers=40,
                    num_nextn_predict_layers=3, dspark_n_routed_experts=128,
                    dspark_num_experts_per_tok=3, dspark_markov_rank=256)
    if (getattr(draft, 'quantization', None) != NATIVE_DRAFT_QUANTIZATION
            or getattr(draft, 'enforce_eager', False)
            or any(getattr(hf, name, None) != value for name, value in expected.items())
            or list(getattr(hf, 'dspark_target_layer_ids', ())) != [37, 38, 39]
            or not getattr(hf, 'sample_from_anchor', True)
            or getattr(hf, 'draft_vocab_size', None) not in (None, 129280)):
        raise ValueError('Use the unchanged full-vocabulary V4.1 FP8/FP4 native draft checkpoint')
    return True


def draft_group_cp(speculator, gid):
    """The native draft input kernel must not shard replicated SWA positions."""
    tables = speculator.block_tables
    owners = getattr(tables, '_ds41_group_owners', None)
    if (owners is None or type(gid) is not int or not 0 <= gid < len(owners)
            or gid not in speculator.draft_kv_cache_group_ids or owners[gid] != 1
            or tables.cp_size != 2 or tables.cp_rank not in (0, 1)
            or tables.cp_interleave != 1 or tables.kernel_block_sizes[gid] != 32):
        raise ValueError('DSpark requires the registered replicated32-token SWA cache group')
    return 0, 1


def make_arch_updater(original):
    def update_arch(spec):
        target = getattr(spec, 'target_model_config', None)
        if (getattr(spec, 'method', None) == 'dspark'
                and getattr(target, 'quantization', None) == 'ds41_exl3'):
            if os.environ.get('DS41_ENABLE_DSPARK') != '1' or spec.quantization != 'fp8':
                raise ValueError('EXL3 target requires explicitly selected native FP8/FP4 draft quantization')
            # Native DSpark's validator unconditionally inherits the target
            # quantizer immediately before update_arch_(). Respect the
            # explicit choice's native V4.1 resolution, not the generic fp8
            # alias. Generic Fp8Config allocates FP8 experts and ignores the
            # checkpoint's packed FP4 experts and32x32 MXFP8 dense layout.
            # No target config or quantized parameter is altered.
            spec.draft_model_config.quantization = NATIVE_DRAFT_QUANTIZATION
        return original(spec)
    return update_arch


def validate_draft_quantizer(config, quantizer):
    """Validate the actual native object before the first draft allocation."""
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
    validate_speculation(config)
    if (type(quantizer) is not DeepseekV4FP8Config
            or quantizer.get_name() != NATIVE_DRAFT_QUANTIZATION
            or quantizer.expert_dtype != 'fp4'
            or quantizer.weight_block_size != [32, 32]
            or not quantizer.is_scale_e8m0):
        raise ValueError('Draft requires native packed MXFP4 experts and32x32 MXFP8 linears')
    return True


def compile_native_loader(original):
    from .vllm_dcp import _compile
    return _compile(original, [
        ('draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)',
         'draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)\n'
         '    _ds41_validate_draft_quantizer(vllm_config, draft_vllm_config.quant_config)'),
    ], {'_ds41_validate_draft_quantizer': validate_draft_quantizer})


def compile_draft_dense_graph_constructor(original):
    """Admit the exact native15360->5120 auxiliary-state projection.

    The legacy target-only adapter accepts widths up to8192. Native DSpark
    concatenates three5120-wide target states; this is not a new quantizer
    or a wider launch/memory budget. Keep all row, dtype, inference, output,
    pool ownership, poison, nested-capture and allocator checks unchanged.
    The original on-disk adapter remains byte-identical for legacy kits.
    """
    from .vllm_dcp import _compile
    return _compile(original, [
        ('or not 1 <= example.shape[1] <= 8192',
         'or not (1 <= example.shape[1] <= 8192 or example.shape[1] == 15360)'),
    ], {})


def make_weight_loader(original, stream):
    from .draft_exl3_serving import make_weight_loader as packed_loader
    return packed_loader(original, stream, _draft_scope)


def make_image_scheduler(original):
    def schedule(scheduler, request, num_computed_tokens, num_new_tokens,
                 encoder_compute_budget, shift_computed_tokens=0):
        if scheduler.vllm_config.speculative_config is not None:
            validate_speculation(scheduler.vllm_config)
            if shift_computed_tokens not in (0, 1):
                raise ValueError('Unreviewed DSpark encoder lookahead')
            # DFlash/DSpark consumes target aux states and a next-token ID,
            # not a future image embedding (supports_mm_inputs=False).
            # Remove ONLY encoder lookahead; token/KV reservation and native
            # whole-image attention, resizing and prefix rollback are intact.
            shift_computed_tokens = 0
        return original(scheduler, request, num_computed_tokens, num_new_tokens,
                        encoder_compute_budget, shift_computed_tokens)
    return schedule


def make_patches():
    import torch
    from torch import nn
    from .vllm_dcp import _compile
    from . import vllm_vision_inputs as vision
    from vllm.v1.worker.gpu import cudagraph_utils as graphs
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head
    import spark_dense_decode_graph as dense_graph

    if hashlib.sha256(Path(dense_graph.__file__).read_bytes()).hexdigest() != DENSE_GRAPH_CORE_SHA256:
        raise RuntimeError('Unreviewed target dense-graph core')

    modules = {name: importlib.import_module(name) for name in UPSTREAM}
    for name, module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM[name]:
            raise RuntimeError('Unreviewed native DSpark source: ' + name)
    stream = importlib.import_module('streaming_loader')
    if hashlib.sha256(Path(stream.__file__).read_bytes()).hexdigest() != STREAMING_SHA256:
        raise RuntimeError('Unreviewed metadata-first draft iterator')
    if hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest() != VISION_SHA256:
        raise RuntimeError('Unreviewed whole-image scheduler')
    if vision._registered or _draft_scope.get() is not None:
        raise RuntimeError('Prepare DSpark before image registration or model loading')
    native = modules['vllm.models.deepseek_v4_1.nvidia.dspark']
    utils = modules['vllm.v1.worker.gpu.spec_decode.dspark.utils']
    dspark = modules['vllm.v1.worker.gpu.spec_decode.dspark.speculator']
    dflash = modules['vllm.v1.worker.gpu.spec_decode.dflash.speculator']
    draft_graphs = modules['vllm.v1.worker.gpu.spec_decode.dflash.cudagraph']
    # Its capture override delegates to super().capture, and run_fullgraph is
    # inherited. The existing owned V2 patches cover BOTH graph managers.
    if (draft_graphs.DFlashCudaGraphManager.__bases__ != (graphs.CudaGraphManager,)
            or 'run_fullgraph' in vars(draft_graphs.DFlashCudaGraphManager)
            or 'propose' in vars(dspark.DSparkSpeculator)
            or dspark.load_dspark_model is not utils.load_dspark_model
            or native.DSparkDeepseekV4ForCausalLM.has_own_embed_tokens
            or native.DSparkDeepseekV4ForCausalLM.has_own_lm_head):
        raise RuntimeError('Native DSpark sharing/graph/proposal contract changed')

    class DeferredSharedTable(nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError('Native draft loader did not alias the target vocabulary table')

    def deferred_table(kind, vocab_size, hidden_size, *, prefix):
        scope = _draft_scope.get()
        expected_prefix = 'model.embed_tokens' if kind == 'embed' else 'lm_head'
        if (scope is None or vocab_size != scope.vocab or hidden_size != scope.hidden
                or prefix != expected_prefix or kind in scope.deferred):
            raise RuntimeError('Unreviewed or unscoped shared draft vocabulary allocation')
        scope.deferred.add(kind)
        # Parameter-free: native draft postprocessing must not touch shared
        # target parameters for a second time before its utility aliases them.
        return DeferredSharedTable()

    original_load = compile_native_loader(utils.load_dspark_model)

    def load_dspark_model(target, config):
        validate_speculation(config)
        if _draft_scope.get() is not None:
            raise RuntimeError('Nested draft loading is unsupported')
        language = target.get_language_model() if hasattr(target, 'get_language_model') else target
        embed = getattr(language.model, 'embed_tokens', None)
        head = get_target_lm_head(target, language)
        vocab = config.model_config.get_vocab_size()
        if embed is None or head is None or vocab != 129280:
            raise ValueError('Native DSpark requires the full target embedding and output head')
        scope = SimpleNamespace(vocab=vocab, hidden=5120, deferred=set())
        token = _draft_scope.set(scope)
        try:
            result = original_load(target, config)
            if (scope.deferred != {'embed', 'head'} or result.model.embed_tokens is not embed
                    or result.lm_head is not head
                    or any(isinstance(m, DeferredSharedTable) for m in result.modules())):
                raise RuntimeError('Native DSpark failed to share the exact target vocabulary tables')
            return result
        finally:
            _draft_scope.reset(token)

    inner_init = _compile(native.DSparkDeepseekV4Model.__init__, [
        ('super().__init__()', 'nn.Module.__init__(self)'),
        ('VocabParallelEmbedding(', '_ds41_deferred_embed('),
    ], {'_ds41_deferred_embed': lambda *a, **k: deferred_table('embed', *a, **k)})
    outer_init = _compile(native.DSparkDeepseekV4ForCausalLM.__init__, [
        ('super().__init__()', 'nn.Module.__init__(self)'),
        ('ParallelLMHead(', '_ds41_deferred_head('),
    ], {'_ds41_deferred_head': lambda *a, **k: deferred_table('head', *a, **k)})
    # Context has no queries. Keep the native KV writer byte-for-byte, but
    # dispatch its smallest supported padding specialization with zero live
    # query heads. Other cache formats retain their original launch shape.
    # The context writer calls the native group-64 op directly; with group-32
    # SWA pages it must use the same writer as target attention, or every draft
    # layer reads its context with the wrong scales.
    from . import swa_kv
    context_edits = [
        ('(n_ctx, attn.n_local_heads, attn.head_dim),',
         '(n_ctx, 0 if cache_dtype == torch.uint8 else attn.n_local_heads, attn.head_dim),'),
        ('            attn.padded_heads,', '            8,'),
    ]
    context_globals = {}
    if swa_kv.GROUP_SIZE == 32:
        context_edits.append(('torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(',
                              '_ds41_swa32_insert('))
        context_globals['_ds41_swa32_insert'] = swa_kv.load_native()
    context_insert = _compile(native._insert_context_kv, context_edits, context_globals)
    context_insert._ds41_original_context_insert = native._insert_context_kv
    proposal = _compile(dflash.DFlashSpeculator.propose, [
        ('for i, gid in enumerate(self.draft_kv_cache_group_ids):',
         'for i, gid in enumerate(self.draft_kv_cache_group_ids):\n        draft_cp_rank, draft_cp_size = _ds41_draft_group_cp(self, gid)'),
        ('self.block_tables.cp_rank,\n            self.block_tables.cp_size,',
         'draft_cp_rank,\n            draft_cp_size,'),
    ], {'_ds41_draft_group_cp': draft_group_cp})
    # _compile intentionally removes decorators; retain native inference mode.
    proposal = torch.inference_mode()(proposal)
    image_config = _compile(vision.validate_config, [
        ('scheduler.max_num_seqs != 1', 'not 1 <= scheduler.max_num_seqs <= 6'),
        ("if current.speculative_config is not None:\n        raise ValueError('Initial native vision serving has no validated speculative decoding path')",
         '_ds41_validate_speculation(current)'),
    ], {'_ds41_validate_speculation': validate_speculation})
    from .draft_exl3_serving import make_quantizer_patches
    from .ngram_draft import wrap as _ds41_ngram_wrap
    return [
        *make_quantizer_patches(_draft_scope),
        (dense_graph.DenseDecodeGraph, '__init__',
         compile_draft_dense_graph_constructor(dense_graph.DenseDecodeGraph.__init__)),
        (modules['vllm.config.speculative'].SpeculativeConfig, 'update_arch_',
         make_arch_updater(modules['vllm.config.speculative'].SpeculativeConfig.update_arch_)),
        (native.DSparkDeepseekV4Model, '__init__', inner_init),
        (native.DSparkDeepseekV4ForCausalLM, '__init__', outer_init),
        (native, '_insert_context_kv', context_insert),
        (native.DSparkDeepseekV4ForCausalLM, 'load_weights',
         make_weight_loader(native.DSparkDeepseekV4ForCausalLM.load_weights, stream)),
        (utils, 'load_dspark_model', load_dspark_model),
        (dspark, 'load_dspark_model', load_dspark_model),
        (dspark.DSparkSpeculator, 'propose', _ds41_ngram_wrap(proposal)),
        (vision, 'validate_config', image_config),
        (vision, 'schedule_whole_images', make_image_scheduler(vision.schedule_whole_images)),
    ]
