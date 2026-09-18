# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in startup-only, lossless input-vocabulary offload preparation.

make_patches constructs replacements but never installs them. The combined
startup transaction selects the constructor and metadata iterator together,
before any loading. Full serving qualification remains independently required.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path

PINS = {
    'vllm.models.deepseek_v4_1.nvidia.model':
        '1f1419b164f62fa9067f63b31d65409023fd0d73c158cf37528ceee76de268b7',
    'streaming_loader': '3e8e04c7fdeadc2027a4bcad3a8f74a2078214d9990eed2d4a220a691e96ed8c',
}


def validate_request(num_embeddings, embedding_dim, params_dtype, org_num_embeddings,
                     padding_size, prefix, disable_tp, quant_method):
    if (type(num_embeddings) is not int or num_embeddings != 129280
            or type(embedding_dim) is not int or embedding_dim != 5120
            or params_dtype is not None or org_num_embeddings is not None
            or padding_size != 64 or prefix != 'language_model.model.embed_tokens'
            or disable_tp is not False or quant_method is not None):
        raise ValueError('Only the native full-vocabulary BF16 target input embedding is supported')


def wrap_iterator(original):
    if hasattr(original, '_ds41_vocabulary_skip'):
        raise RuntimeError('Vocabulary metadata skip already prepared')
    def iterator(files, skip_weight=None):
        # The predicate runs before mapped-name sorting and get_tensor. Keep
        # the old expert filtering and every vision/draft/output-head tensor.
        return original(files, lambda name: name == 'embed.weight'
                        or (skip_weight is not None and skip_weight(name)))
    iterator._ds41_vocabulary_skip = original
    return iterator


def make_patches(library=None):
    if os.environ.get('DS41_ENABLE_SSD_VOCAB') != '1':
        raise ValueError('Explicit SSD input-vocabulary startup selection required')
    import torch
    from torch import nn
    from vllm.config import get_current_vllm_config
    from vllm.distributed import (get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce)
    from vllm.model_executor.layers import vocab_parallel_embedding as vocabulary
    from .combined_config import validate_config, MAX_TOKENS
    from .native_vocab_stage import NativeVocabStage, _LIVE_STAGES, BINARY_SHA

    modules = {name:importlib.import_module(name) for name in PINS}
    for name, module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != PINS[name]:
            raise RuntimeError('Unreviewed vocabulary constructor/iterator source: '+name)
    native, stream = (modules[name] for name in PINS)
    if (native.VocabParallelEmbedding is not vocabulary.VocabParallelEmbedding
            or stream._registered or _LIVE_STAGES):
        raise RuntimeError('Prepare vocabulary hooks before model/iterator loading')
    library = Path(library) if library is not None else Path(__file__).resolve().parents[1]/'libds41_vocab_rows.so'
    if hashlib.sha256(library.read_bytes()).hexdigest() != BINARY_SHA:
        raise ValueError('Unqualified native vocabulary callback binary')

    class SSDVocabEmbedding(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, params_dtype=None,
                     org_num_embeddings=None, padding_size=64, quant_config=None,
                     prefix='', *, disable_tp=False, quant_method=None):
            super().__init__()
            validate_request(num_embeddings, embedding_dim, params_dtype, org_num_embeddings,
                             padding_size, prefix, disable_tp, quant_method)
            config = get_current_vllm_config()
            validate_config(config)
            if (config.model_config.dtype != torch.bfloat16 or config.lora_config is not None
                    or get_tensor_model_parallel_world_size() != 2
                    or quant_config is None or quant_config.get_name() != 'ds41_exl3'):
                raise ValueError('Preserve the canonical BF16, no-LoRA, TP2 target vocabulary')
            root = Path(config.model_config.model)
            index = json.loads((root/'model.safetensors.index.json').read_bytes())
            name = index['weight_map']['embed.weight']
            if Path(name).name != name or not name.endswith('.safetensors'):
                raise ValueError('Input embedding must be in one mounted canonical shard')
            self.num_embeddings = self.org_vocab_size = self.num_embeddings_padded = num_embeddings
            self.embedding_dim = embedding_dim
            self.tp_size = 2
            self.tp_rank = get_tensor_model_parallel_rank()
            self.num_embeddings_per_partition = num_embeddings//2
            self.stage = NativeVocabStage(root/name, library, rank=self.tp_rank)
            # No weight Parameter, placeholder tensor, or table-sized buffer.
            # Native DSpark aliases this exact module; its output head is untouched.

        def forward(self, input_):
            if (input_.dtype not in (torch.int32, torch.int64) or input_.ndim not in (1, 2)
                    or input_.device != self.stage.device or input_.numel() > MAX_TOKENS):
                raise ValueError('Input embedding requires bounded CUDA token IDs')
            flat = input_.reshape(-1).contiguous()
            out = torch.empty((flat.numel(), self.embedding_dim), device=flat.device, dtype=torch.bfloat16)
            self.stage.lookup(flat, out)
            # Same BF16 collective as native VocabParallelEmbedding.forward.
            return tensor_model_parallel_all_reduce(out).view(*input_.shape, self.embedding_dim)

        def close(self):
            self.stage.close()

    SSDVocabEmbedding._ds41_lossless_input_vocabulary = True
    return [(native, 'VocabParallelEmbedding', SSDVocabEmbedding),
            (stream, 'ordered_nonengram_weights', wrap_iterator(stream.ordered_nonengram_weights))]
