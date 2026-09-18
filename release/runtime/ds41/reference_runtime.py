"""Layerwise access to pinned reference numerics, with bounded SSD engrams."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

from .ssd_rows import EngramRows

REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"


class SourceWeights:
    def __init__(self, source):
        self.source = Path(source)
        self.index = json.loads((self.source / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, key, device="cpu"):
        path = self.source / self.index[key]
        with safe_open(path, framework="pt", device=str(device)) as shard:
            return shard.get_tensor(key)

    def dequantize(self, key, device="cuda"):
        weight = self.get(key + ".weight", device)
        if weight.dtype == torch.float8_e4m3fn:
            scale = self.get(key + ".scale", device).float()
            weight = weight.float() * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)[:weight.shape[0], :weight.shape[1]]
        elif weight.dtype == torch.int8:
            scale = self.get(key + ".scale", device).float()
            table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=device, dtype=torch.float32)
            packed = weight.view(torch.uint8).long()
            weight = torch.stack((table[packed & 15], table[packed >> 4]), -1).flatten(1)
            weight *= scale.repeat_interleave(32, 1)
        return weight.float()


class ReferenceRuntime:
    def __init__(self, source, max_seq_len=2048):
        self.source = Path(source).resolve()
        # The published reference uses sibling imports; keep its module names
        # isolated to this process and never mix revisions within a process.
        sys.path.insert(0, str(self.source / "inference"))
        sys.path.insert(0, str(self.source / "encoding"))
        self.ref = importlib.import_module("model")
        if Path(self.ref.__file__).resolve() != self.source / "inference/model.py":
            raise RuntimeError("A different reference model is already imported")
        self.encoding = importlib.import_module("encoding")
        self.image_processor = importlib.import_module("image_processor")
        cfg = json.loads((self.source / "inference/config.json").read_text())
        self.args = self.ref.ModelArgs(**cfg, max_batch_size=1, max_seq_len=max_seq_len)
        self.weights = SourceWeights(self.source)
        self._ssd_readers = []
        original = self.ref.sparse_attn
        if not getattr(original, "ds41_head_tiled", False):
            def tiled(q, kv, sink, indices, scale):
                return torch.cat([original(q[:, :, i:i + 16].contiguous(), kv,
                                  sink[i:i + 16].contiguous(), indices, scale)
                                  for i in range(0, q.shape[2], 16)], dim=2)
            tiled.ds41_head_tiled = True
            self.ref.sparse_attn = tiled

    @contextmanager
    def _ssd_embedding(self):
        runtime = self
        original = self.ref.ParallelEngramEmbedding

        class SSDReferenceEmbedding(torch.nn.Module):
            def __init__(self, num_embeddings, dim):
                super().__init__()
                offset = runtime.args.engram_num_embeddings.index(num_embeddings)
                layer = runtime.args.engram_layer_ids[offset]
                self.reader = EngramRows(runtime.source, layer)
                assert self.reader.weight.row_bytes == dim
                runtime._ssd_readers.append(self.reader)

            def forward(self, indices):
                return self.reader.lookup(indices, device=indices.device)

        self.ref.ParallelEngramEmbedding = SSDReferenceEmbedding
        try:
            yield
        finally:
            self.ref.ParallelEngramEmbedding = original

    @torch.inference_mode()
    def load_parameters(self, module, prefix):
        loaded = set()
        for name, parameter in module.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            if name.endswith("attn.wo_a.weight") or name == "wo_a.weight":
                value = self.weights.dequantize(key.removesuffix(".weight")).bfloat16()
                loaded.add(key.removesuffix(".weight") + ".scale")
            else:
                value = self.weights.get(key, "cuda")
            if parameter.dtype == torch.float4_e2m1fn_x2:
                assert value.dtype == torch.int8, key
                value = value.view(torch.float4_e2m1fn_x2)
            if parameter.shape != value.shape:
                raise ValueError((key, parameter.shape, value.shape))
            parameter.copy_(value)
            loaded.add(key)
        return loaded

    def load_block(self, layer):
        self.close_tables()
        with torch.device("cuda"), self.ref.set_dtype(torch.bfloat16), self._ssd_embedding():
            layout = self.ref.EngramLayout.from_args(self.args)
            block = self.ref.Block(layer, self.args, layout)
        loaded = self.load_parameters(block, f"layers.{layer}")
        expected = {key for key in self.weights.index if key.startswith(f"layers.{layer}.") and ".engram.embed." not in key}
        actual = loaded
        if expected != actual:
            raise ValueError({"unloaded": sorted(expected - actual), "unexpected": sorted(actual - expected)})
        return block

    @staticmethod
    def stash_shared(reference):
        return {name: tensor.detach().cpu().contiguous() for name, tensor in vars(reference.shared_attn).items() if tensor is not None}

    def restore_shared(self, state):
        self.ref.shared_attn = self.ref.SharedAttentionRuntime()
        for name, tensor in state.items():
            if name not in vars(self.ref.shared_attn):
                raise ValueError(f"Unknown shared attention state: {name}")
            setattr(self.ref.shared_attn, name, tensor.cuda())

    def close_tables(self):
        for reader in self._ssd_readers:
            reader.close()
        self._ssd_readers.clear()
