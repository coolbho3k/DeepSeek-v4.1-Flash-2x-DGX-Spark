"""Bounded host-staged engram lookup with vLLM's complete-head TP partition."""
import re

import torch
from safetensors import safe_open

ENGRAM_TABLE = re.compile(r"(?:^|\.)engram\.(?:embed|embed_tokens)\.(?:weight|scale|weight_scale_inv)$")


def nonengram_weights(files, skip_weight=None):
    """Skip table keys before get_tensor: filtering yielded tensors is too late."""
    for path in sorted(files):
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if ENGRAM_TABLE.search(name) or (skip_weight and skip_weight(name)):
                    continue
                yield name, shard.get_tensor(name)


class SSDHeadEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, dim, head_sizes, tp_size, tp_rank, reader,
                 gather=None, chunk_tokens=256):
        super().__init__()
        if not head_sizes or any(size <= 0 for size in head_sizes) or sum(head_sizes) > num_embeddings:
            raise ValueError("Invalid engram head partition")
        if tp_size < 1 or not 0 <= tp_rank < tp_size or chunk_tokens < 1:
            raise ValueError("Invalid rank or staging chunk")
        self.num_embeddings, self.dim = num_embeddings, dim
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = (self.n_hash_cols + tp_size - 1) // tp_size
        self.head_start = tp_rank * self.part_n_hash_cols
        self.head_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        self.vocab_start_idx = sum(head_sizes[:self.head_start])
        self.vocab_end_idx = sum(head_sizes[:self.head_start + self.part_n_hash_cols])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.tp_size, self.reader, self.gather = tp_size, reader, gather
        self.chunk_tokens = chunk_tokens
        # No table-shaped Parameters, pinned allocations, or UVA views exist.

    def lookup(self, indices, out, background=False):
        if indices.ndim != 2 or indices.shape[1] != self.n_hash_cols:
            raise ValueError("Expected [tokens, all_hash_heads] indices")
        if tuple(out.shape) != (len(indices), self.part_n_hash_cols, self.dim) or out.dtype != torch.bfloat16:
            raise ValueError("Invalid engram output shape or dtype")
        if indices.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SSD lookup currently requires --enforce-eager; graph capture is not supported")
        for start in range(0, len(indices), self.chunk_tokens):
            end = min(start + self.chunk_tokens, len(indices))
            local = indices[start:end, self.head_start:self.head_end].to(device="cpu", dtype=torch.long)
            owned = (local >= self.vocab_start_idx) & (local < self.vocab_end_idx)
            staged = torch.zeros((end - start, self.part_n_hash_cols, self.dim), device="cpu", dtype=torch.bfloat16)
            if owned.any():
                values = self.reader.lookup(local[owned], device="cpu")
                staged[:, :local.shape[1]][owned] = values
            out[start:end].copy_(staged)

    def forward(self, indices):
        out = torch.empty((len(indices), self.part_n_hash_cols, self.dim), device=indices.device, dtype=torch.bfloat16)
        self.lookup(indices, out)
        if self.tp_size > 1:
            if self.gather is None:
                raise RuntimeError("A TP collective is required for forward")
            out = self.gather(out, dim=1)[:, :self.n_hash_cols]
        return out

    def close(self):
        self.reader.close()
