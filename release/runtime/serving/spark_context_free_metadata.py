# SPDX-License-Identifier: AGPL-3.0-only
"""Avoid an unnecessary CUDA context in CPU parents during FlashInfer import.

The installed Apache-2.0 FlashInfer source remains unchanged on disk. Only
two pinned import-time hardware queries are replaced; kernel bodies and their
hardware values are preserved. No Torch monkeypatch or persistent fake device.
"""
import ctypes
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

MODULE='flashinfer.gdn_kernels.gdn_decode_bf16_state'
SOURCE_SHA='c4268cd8dfb14648c1212ad789e79da8fd63112eb0cf3facdaa2f28d36c5a844'
KDA_MODULE='flashinfer.kda_kernels'
KDA_SHA='f41af6dd16b47797703ccb179444a7ee142d794e1fd45fda2737dbadd471278c'
B12X_MODULE='b12x._lib.gating'
B12X_SHA='5bc885d507a66d61df73fa5dcacea201499dc91b727a0cffa74227fb3613e1b4'
_finder=None
_proof=None


def metadata():
    """cuDeviceGetAttribute needs the driver, not a primary context."""
    global _proof
    driver=ctypes.CDLL('libcuda.so.1')
    driver.cuInit.argtypes=[ctypes.c_uint];driver.cuInit.restype=ctypes.c_int
    driver.cuCtxGetCurrent.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxGetCurrent.restype=ctypes.c_int
    driver.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    driver.cuDeviceGet.restype=ctypes.c_int
    driver.cuDeviceGetAttribute.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
    driver.cuDeviceGetAttribute.restype=ctypes.c_int
    def check(code):
        if code:raise RuntimeError(f'CUDA driver metadata query failed: {code}')
    check(driver.cuInit(0))
    before=ctypes.c_void_p();check(driver.cuCtxGetCurrent(ctypes.byref(before)))
    device=ctypes.c_int();check(driver.cuDeviceGet(ctypes.byref(device),0))
    values=[]
    for attribute in (16,75,76):
        value=ctypes.c_int()
        check(driver.cuDeviceGetAttribute(ctypes.byref(value),attribute,device.value))
        values.append(value.value)
    after=ctypes.c_void_p();check(driver.cuCtxGetCurrent(ctypes.byref(after)))
    if before.value!=after.value or values[0]<=0 or values[1:]!=[12,1]:
        raise RuntimeError('Context-free metadata requires the qualified GB10 device')
    _proof=dict(stage='ds41_context_free_flashinfer_metadata',sm_count=values[0],
        capability=values[1:],context_before=before.value,context_after=after.value,
        original_source_sha256=SOURCE_SHA,kernel_bodies_unchanged=True,
        disk_files_modified=False,cuda_allocations=False)
    print(json.dumps(_proof),flush=True)
    return values[0],values[1]


def capability():
    if _proof is None:metadata()
    return tuple(_proof['capability'])


def transformed(raw,module=MODULE):
    if module==B12X_MODULE:
        anchor='return torch.cuda.get_device_capability(dev)'
        if hashlib.sha256(raw).hexdigest()!=B12X_SHA or raw.decode().count(anchor)!=1:
            raise RuntimeError('Unreviewed B12X support-query source')
        # Before CUDA initialization, this single-GPU deployment can obtain
        # device0 metadata without a context. Preserve native behavior for
        # initialized CUDA, other devices, and every original gating check.
        return raw.decode().replace(anchor,
            "return (__import__('spark_context_free_metadata').capability() "
            "if not torch.cuda.is_initialized() and dev.index in (None,0) "
            "else torch.cuda.get_device_capability(dev))")
    if module==KDA_MODULE:
        anchor='_torch.cuda.get_device_capability(0)'
        if hashlib.sha256(raw).hexdigest()!=KDA_SHA or raw.decode().count(anchor)!=1:
            raise RuntimeError('Unreviewed FlashInfer KDA import source')
        return raw.decode().replace(anchor,"__import__('spark_context_free_metadata').capability()")
    if module!=MODULE:raise RuntimeError('Unexpected metadata replacement module')
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise RuntimeError('Unreviewed FlashInfer import source')
    source=raw.decode()
    first='NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count'
    second='_GPU_MAJOR, _ = torch.cuda.get_device_capability(0)'
    if source.count(first)!=1 or source.count(second)!=1:
        raise RuntimeError('FlashInfer hardware query anchors changed')
    return source.replace(first,
        "NUM_SMS, _GPU_MAJOR = __import__('spark_context_free_metadata').metadata()").replace(second,
        '# _GPU_MAJOR came from the context-free driver query above.')


class Loader(importlib.abc.Loader):
    def __init__(self,path,module):self.path,self.module=path,module
    def create_module(self,spec):return None
    def exec_module(self,module):
        path=Path(self.path)
        if path.resolve()!=path or not path.is_file() or path.stat().st_size>512*2**10:
            raise RuntimeError('Canonical bounded FlashInfer source required')
        exec(compile(transformed(path.read_bytes(),self.module),str(path),'exec'),module.__dict__)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in (MODULE,KDA_MODULE,B12X_MODULE):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or not spec.origin or not isinstance(spec.loader,importlib.machinery.SourceFileLoader):
            raise RuntimeError('Native FlashInfer source loader required')
        return importlib.util.spec_from_file_location(fullname,spec.origin,
            loader=Loader(spec.origin,fullname),submodule_search_locations=spec.submodule_search_locations)


def register():
    global _finder
    if _finder is not None:
        if _finder not in sys.meta_path:raise RuntimeError('Metadata finder was removed')
        return
    if any(n in sys.modules for n in (MODULE,KDA_MODULE,B12X_MODULE)):raise RuntimeError('Register before backend import')
    _finder=Finder();sys.meta_path.insert(0,_finder)
