# SPDX-License-Identifier: AGPL-3.0-only
# Integration of MiaAI-Lab's native row-store/callback path; see
# ../vendor/miaai-dsv41-agpl/LICENSE, LICENSE.MIT and UPSTREAM.json.
"""Startup-only replacement of SSD embedding storage lookup, not its hasher."""
import hashlib
import os
from pathlib import Path

BASE_SHA = 'c4b79cfd9063a8b79c524e1176a3185d7d58173d2600fb43733e0d09ee8d8e98'
CORE_SHA = 'c9b751ec4ee4251acc26dece3c7794408bb305a45ea32cf666a784e49033aacb'
DESCRIPTOR = dict(implementation='miaai_parallel_native_engram_v1', license='AGPL-3.0-only',
    core_sha256=CORE_SHA, maximum_chunk_tokens=256, maximum_local_heads=144,
    staging_bytes_per_layer_ceiling=19759104, cache_bytes_per_layer_ceiling=67108864,
    io_threads=32, resident_tables=False, resident_scales=False,
    native_image_hasher_unchanged=True, unowned_and_dead_ids_zero=True,
    callback_stream_ordering=True, full_model_graph_capture_enabled=False)
_installed = None


def register(library_path=None):
    global _installed
    mode = os.environ.get('DS41_ENABLE_NATIVE_ENGRAM', '0')
    if mode not in ('0','1'):
        raise ValueError('Native Engram mode must be exactly 0 or 1')
    if mode == '0':
        if _installed is not None:
            raise RuntimeError('Native Engram mode cannot change after startup')
        return
    for key, value in (('OFFLOAD_MODE','ssd'),('DSV41_RESIDENT_SCALES','0'),('DSV41_IO_THREADS','32')):
        if os.environ.get(key, value) != value:
            raise ValueError('Native Engram candidate requires '+key+'='+value)
        os.environ[key] = value
    from ds41 import ssd_embedding as base
    import miaai_engram as core
    for module, expected in ((base,BASE_SHA),(core,CORE_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Native Engram source differs from qualification')
    library = (Path(__file__).resolve().parent/'miaai-row-store-v1.so'
               if library_path is None else Path(library_path).resolve())
    if hashlib.sha256(library.read_bytes()).hexdigest() != core.BINARY_SHA:
        raise RuntimeError('Native Engram binary differs from qualification')
    cls = base.SSDHeadEmbedding
    if _installed is not None:
        if (_installed['class'] is not cls or _installed['library'] != library
                or any(getattr(cls,name) is not replacement
                       for name,replacement in _installed['replacements'].items())):
            raise RuntimeError('Native Engram binding changed after registration')
        return
    original = {name:getattr(cls,name) for name in ('__init__','lookup','close')}
    for name, fn in original.items():
        if (fn.__module__ != base.__name__ or fn.__qualname__ != 'SSDHeadEmbedding.'+name
                or Path(fn.__code__.co_filename).resolve() != Path(base.__file__).resolve()):
            raise RuntimeError('Unexpected preexisting SSD embedding patch')

    def initialize(self, *args, **kwargs):
        if len(core._LIVE_STAGES) >= 2:
            raise RuntimeError('At most the two original Engram tables may be staged')
        original['__init__'](self, *args, **kwargs)
        self._ds41_native_stage = core.NativeStage(self, library)

    def lookup(self, indices, out, background=False):
        self._ds41_native_stage.lookup(indices, out, background=background)

    def close(self):
        self._ds41_native_stage.close()
        original['close'](self)

    replacements = dict(__init__=initialize, lookup=lookup, close=close)
    for name,replacement in replacements.items():
        setattr(cls,name,replacement)
    _installed = dict(original=original,replacements=replacements,library=library,**{'class':cls})
