# SPDX-License-Identifier: AGPL-3.0-only
"""Startup-only packed/UVA/deferred integration of MiaAI-derived Engram.

No hashing, ID generation, quantization, all-gather, image mask, or Engram gate
math changes. Only retrieval layout, staging copies and completion scheduling.
Candidate preparation pins policy and the installed vLLM source explicitly.
"""
import ctypes as C
import hashlib
from pathlib import Path
import re

from .overlap import DeferredRows, RetrievalStream
from .policy import LAYOUT, MAPPED, OVERLAP, UPSTREAM_SHA

_retrieval = None
_installed = None


def create_stage(embedding, library):
    global _retrieval
    import miaai_engram as core
    filename = embedding.reader.weight.path.name
    found = re.fullmatch(r'engram-layer-(01|14)\.safetensors', filename)
    if not found or embedding.n_hash_cols != 24 or embedding.part_n_hash_cols != 12:
        raise ValueError('Packed candidate requires the original two layers and TP2 complete heads')
    layer = int(found[1])
    rank = embedding.head_start//12
    if rank not in (0,1):
        raise ValueError('Invalid complete-head rank')
    path = None if LAYOUT=='original' else Path('/opt/ds41-engram-packed')/f'engram-layer-{layer:02}-rank{rank}-{LAYOUT}.bin'
    stage = core.NativeStage(embedding, library, packed=path,
        packed_layer=None if path is None else layer, mapped=MAPPED)
    stage._ds41_engram_layer = layer
    if OVERLAP:
        if _retrieval is None:
            _retrieval = RetrievalStream(stage.device)
        stage._ds41_deferred_rows = DeferredRows(stage, _retrieval)
    return stage


def close_stage(stage):
    if OVERLAP:
        stage._ds41_deferred_rows.close()
    else:
        stage.close()


def install():
    global _installed
    if not OVERLAP:
        return
    from vllm.models.deepseek_v4_1.common import engram
    cls = engram.Engram
    if hashlib.sha256(Path(engram.__file__).read_bytes()).hexdigest() != UPSTREAM_SHA:
        raise ValueError('Unqualified upstream Engram implementation')
    if _installed is not None:
        if cls.prepare_embeddings is not _installed[0] or cls.embed is not _installed[1]:
            raise RuntimeError('Deferred Engram methods changed after installation')
        return
    original_prepare, original_embed = cls.prepare_embeddings, cls.embed
    for method,name in ((original_prepare,'prepare_embeddings'),(original_embed,'embed')):
        if (method.__module__ != engram.__name__ or method.__qualname__ != 'Engram.'+name
                or Path(method.__code__.co_filename).resolve() != Path(engram.__file__).resolve()):
            raise ValueError('Engram was already patched by another implementation')

    def prepare_embeddings(self, hash_ids):
        self.embed_tokens._ds41_native_stage._ds41_deferred_rows.prepare(
            hash_ids, self.staged_rows[:hash_ids.shape[0]])

    def embed(self, hash_ids):
        self.embed_tokens._ds41_native_stage._ds41_deferred_rows.consume()
        return original_embed(self, hash_ids)

    cls.prepare_embeddings, cls.embed = prepare_embeddings, embed
    _installed = prepare_embeddings, embed


def audit(stages):
    install()  # Idempotently verify exact installed methods.
    if {s._ds41_engram_layer for s in stages} != {1,14}:
        raise ValueError('Missing packed Engram table')
    layout_code = ('original','dense','page15').index(LAYOUT)
    for stage in stages:
        profile = (C.c_uint64*4)()
        stage.lib.ds41_row_store_profile(stage.store,profile)
        if profile[3] != layout_code or stage.mapped != MAPPED:
            raise ValueError('Live packed layout or mapped mode differs from the pinned policy')
        for host,device in ((stage.host_w,stage.dev_w),(stage.host_s,stage.dev_s)):
            if (host.data_ptr()==device.data_ptr()) != MAPPED:
                raise ValueError('Staging alias/copy ownership differs from mapped policy')
        if OVERLAP:
            deferred = stage._ds41_deferred_rows
            if deferred.stage is not stage or deferred.retrieval is not _retrieval or deferred.prepared:
                raise ValueError('Deferred staging ownership or consumption failed')
    return dict(layout=LAYOUT,gpu_readable_host=MAPPED,deferred_retrieval=OVERLAP,
                row_bytes_unchanged=True,original_quantization_unchanged=True)
