# SPDX-License-Identifier: AGPL-3.0-only
# Callback/fence architecture adapted from native_vocab_stage.py and the
# attributed MiaAI integration. Original notices remain under vendor/.
"""Unselected lossless FP4 draft stage; GPU/native-kernel qualification pending.

One shared9-slot bank across all3draft layers. Consumer execution stays under
the stage's stream fence and lock; never return its mutable bank as a result.
Original routing weights are not changed. CPU maps IDs to loaded local slots.
"""
import ctypes as C
import threading
import torch

from .native_draft_records import NativeDraftRecords, Work, RECORD_BYTES, SLOTS
from .draft_expert_records import COMPONENTS
from . import graph_validation as validation

BINARY_SHA = 'f4bf4aa832b223f7cde10cdd18808a02a2f9020f723dfe39d7a6969dacb0e361'
_LIVE_STAGES = set()
ERRORS = ((1,'Invalid native draft route or callback descriptor'),
          (2,'Draft expert bank changed during serving'),
          (4,'Draft expert O_DIRECT read failed or was short'),
          (8,'Draft expert native store is poisoned'))


class NativeDraftStage:
    def __init__(self,bank,manifest,library,*,expected_manifest_sha,rank,device=None):
        if torch.cuda.is_current_stream_capturing() or _LIVE_STAGES:
            raise RuntimeError('Create one shared draft bank before graph capture')
        self.device = torch.device('cuda',torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.type!='cuda' or self.device.index!=torch.cuda.current_device():
            raise ValueError('Draft stage requires current visible CUDA device')
        from miaai_engram import load_cudart
        self.cuda=load_cudart()
        self.cuda.cudaMemcpy2DAsync.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,
                                            C.c_size_t,C.c_size_t,C.c_int,C.c_void_p]
        self.cuda.cudaMemcpy2DAsync.restype=C.c_int
        self.native=NativeDraftRecords(bank,manifest,library,expected_manifest_sha,BINARY_SHA,rank=rank,threads=3)
        self.lock=threading.Lock()
        self.closed=self.failed=self.pending=False
        self.graphs=0
        try:
            self.host_allocation=torch.empty(SLOTS*RECORD_BYTES+4095,dtype=torch.uint8,device='cpu',pin_memory=True)
            offset=(-self.host_allocation.data_ptr())%4096
            self.host_records=self.host_allocation[offset:offset+SLOTS*RECORD_BYTES]
            assert self.host_records.data_ptr()%4096==0
            self.ids=torch.empty(SLOTS,dtype=torch.int64,device='cpu',pin_memory=True)
            self.host_mapped=torch.empty(SLOTS,dtype=torch.int32,device='cpu',pin_memory=True)
            self.host_status=torch.zeros(1,dtype=torch.int32,device='cpu',pin_memory=True)
            self.mapped=torch.empty(SLOTS,dtype=torch.int32,device=self.device)
            self.device_errors=torch.zeros(1,dtype=torch.int32,device=self.device)
            self.buffers={name:torch.empty((SLOTS,*shape),dtype=torch.uint8,device=self.device)
                          for name,shape,_,_ in COMPONENTS}
            self.works={(layer,count):Work(self.native.store,self.ids.data_ptr(),
                self.host_records.data_ptr(),self.host_mapped.data_ptr(),self.host_status.data_ptr(),layer,count)
                for layer in range(3) for count in (0,3,6,9)}
            self.event=torch.cuda.Event(external=True)
            self.pinned_bytes=self.host_allocation.numel()+self.ids.numel()*8+self.host_mapped.numel()*4+4
            self.gpu_bytes=sum(v.numel() for v in self.buffers.values())+self.mapped.numel()*4+4
            assert self.pinned_bytes+self.gpu_bytes < 162*2**20
            _LIVE_STAGES.add(self)
        except BaseException:
            self.native.close()
            raise

    def _validate(self,ids,layer):
        if self.closed or self.failed or not self.native.store:
            raise RuntimeError('Draft stage is failed or closed')
        if (type(layer) is not int or layer not in range(3) or ids.device!=self.device
                or ids.dtype not in (torch.int32,torch.int64) or ids.ndim!=2
                or ids.shape[1]!=3 or not 0<=ids.shape[0]<=3 or not ids.is_contiguous()):
            raise ValueError('Draft stage supports0..3query rows with3routes each')

    def _wait(self):
        stream=torch.cuda.current_stream(self.device)
        if self.pending:
            stream.wait_event(self.event)
        return stream

    def _record(self,stream):
        self.event.record(stream)
        self.pending=True

    def _enqueue(self,ids,layer):
        count=ids.numel()
        work=self.works[(layer,count)]
        stream=torch.cuda.current_stream(self.device).cuda_stream
        self.ids[:count].copy_(ids.reshape(-1).to(torch.int64),non_blocking=True)
        error=self.cuda.cudaLaunchHostFunc(stream,C.cast(self.native.lib.ds41_draft_record_lookup,C.c_void_p),C.addressof(work))
        if error:
            raise RuntimeError(f'Draft record callback enqueue failed:{error}')
        # Four pitched transfers transpose record-major staging into separate
        # contiguous native GPU weight/scale tensors without reinterpreting bits.
        for name,_,offset,size in COMPONENTS:
            error=self.cuda.cudaMemcpy2DAsync(self.buffers[name].data_ptr(),size,
                self.host_records.data_ptr()+offset,RECORD_BYTES,size,SLOTS,1,stream)
            if error:
                raise RuntimeError(f'Draft record component copy enqueue failed:{error}')
        self.mapped[:count].copy_(self.host_mapped[:count],non_blocking=True)
        self.device_errors.copy_(self.host_status,non_blocking=True)
        validation.check_flags(self.device_errors,ERRORS)
        return self.buffers,self.mapped[:count].view_as(ids)

    def apply(self,ids,layer,consume):
        self._validate(ids,layer)
        if not callable(consume):
            raise ValueError('Draft bank requires an in-fence consumer')
        if torch.cuda.is_current_stream_capturing():
            validation.require_capture_owner()
            buffers,mapped=validation.current_owner().enqueue_draft_stage(self,ids,layer)
            return consume(buffers,mapped)
        with self.lock:
            stream=self._wait()
            try:
                buffers,mapped=self._enqueue(ids,layer)
                result=consume(buffers,mapped)
                self._record(stream)
                return result
            except BaseException:
                self.failed=True
                # Queued raw callbacks retain descriptors and pinned buffers.
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.failed or self.graphs:
                raise RuntimeError('Retain failed or graph-owned draft callback resources')
            if self.pending:
                self.event.synchronize()
            self.native.close()
            self.closed=True
            _LIVE_STAGES.remove(self)
