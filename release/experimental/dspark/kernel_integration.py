# SPDX-License-Identifier: AGPL-3.0-only
"""Startup-only integration of component-qualified drafter kernel candidates.

No live mutation, draft vocabulary restriction, precision reduction, graph
owner, workspace owner, sampling rule or expert weight changes.
"""
import hashlib
import importlib
from pathlib import Path

CONTEXT_SOURCE='37158c568791bb74607e9a8b3ac8e49be6864d98aa04628cdbf31f1dcfc0fc18'
CONTEXT=(
    'qr_kv, _ = attn.fused_wqa_wkv(main_x)\n        kv = qr_kv[..., attn.q_lora_rank :]',
    'kv = _ds41_context_kv(attn, main_x)',
)
SAMPLING=(
    'logits_i = base_logits[:, i] + bias\n        draft_sampled_i = self._sample_logits(\n'
    '            logits_i, idx_map[:, i], sample_pos[:, i], i\n        )',
    'draft_sampled_i = _ds41_sample_bias(\n'
    '            self, base_logits[:, i], bias, idx_map[:, i], sample_pos[:, i], i\n        )',
)


def sample_bias(speculator,base,bias,mapping,position,step):
    from .markov_sampling import sample
    # Preserve native greedy/no-cache and reduced-vocabulary behavior exactly.
    if speculator.draft_logits is None or speculator._d2t_scatter_index is not None:
        return speculator._sample_logits(base+bias,mapping,position,step)
    return sample(base,bias,mapping,speculator.temperature,speculator.seeds,position-1,
        speculator.draft_logits,speculator._step_cols[step],use_fp64=speculator.use_fp64_gumbel)


def make_patches():
    from . import features
    from ds41.vllm_dcp import _compile
    import torch
    patches=[]
    if features.KV_ONLY:
        from .kv_projection import context_kv
        native=importlib.import_module('vllm.models.deepseek_v4_1.nvidia.dspark')
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=CONTEXT_SOURCE:
            raise ValueError('Changed native DSpark context projection source')
        cls=native.DSparkDeepseekV4Model
        method=_compile(cls.precompute_and_store_context_kv,[CONTEXT],{'_ds41_context_kv':context_kv})
        patches.append((cls,'precompute_and_store_context_kv',torch.inference_mode()(method)))
    if features.MARKOV_ADD:
        from ds41.combined_dspark import UPSTREAM
        name='vllm.v1.worker.gpu.spec_decode.dspark.speculator'
        module=importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=UPSTREAM[name]:
            raise ValueError('Changed native sequential sampler source')
        cls=module.DSparkSpeculator
        method=_compile(cls._sample_sequential,[SAMPLING],{'_ds41_sample_bias':sample_bias})
        patches.append((cls,'_sample_sequential',method))
    return patches


def selected_draft_shape(x_shape,ids_shape):
    return (len(x_shape)==2 and 5<=x_shape[0]<=30 and x_shape[1]==5120
        and tuple(ids_shape)==(x_shape[0],3))


def wrap_moe(target_call,root):
    from .draft_top3 import configure
    from .features import TOP3_SHA256
    root=Path(root)
    draft_call=configure(root/'dspark_draft_top3.so',TOP3_SHA256)
    def call(work,bank,x,ids,weights):
        if selected_draft_shape(x.shape,ids.shape):
            return draft_call(work,bank,x,ids,weights)
        return target_call(work,bank,x,ids,weights)
    return call
