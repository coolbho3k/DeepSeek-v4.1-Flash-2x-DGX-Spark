# SPDX-License-Identifier: AGPL-3.0-only
"""Defer Engram's completion dependency until its layer consumes the rows.

Local scheduling on top of MiaAI-derived NativeStage. One ordered I/O stream
serves both layers, so layer14's retrieval cannot overtake layer1's. Native
stage locks/events still own callbacks and pinned-buffer reuse. Model graph
capture/replay remains owned by the existing GraphOwner, not another manager.
"""
import torch


class RetrievalStream:
    def __init__(self, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Create retrieval stream before graph capture')
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)


class DeferredRows:
    def __init__(self, stage, retrieval):
        if stage.device != retrieval.device or torch.cuda.is_current_stream_capturing():
            raise ValueError('Current-device deferred staging must be initialized before capture')
        self.stage, self.retrieval = stage, retrieval
        self.ready = torch.cuda.Event()
        self.prepared = False
        self.closed = False

    def prepare(self, indices, out):
        if self.closed or self.prepared:
            raise RuntimeError('Previous deferred rows were not consumed')
        main = torch.cuda.current_stream(self.stage.device)
        stream = self.retrieval.stream
        stream.wait_stream(main)  # Hash IDs and previous row consumers precede reuse.
        try:
            with torch.cuda.stream(stream):
                if torch.cuda.is_current_stream_capturing():
                    from ds41.graph_validation import require_capture_owner, current_owner
                    require_capture_owner()
                    current_owner().enqueue_stage(self.stage, indices, out)
                else:
                    self.stage.lookup(indices, out)
                self.ready.record(stream)
            self.prepared = True
        except BaseException:
            # A caller must not recover and free raw callback storage after an
            # incomplete fork/join or partially enqueued operation.
            self.stage.failed = True
            raise

    def consume(self):
        if self.closed or not self.prepared:
            raise RuntimeError('Deferred rows must be prepared exactly once before consumption')
        torch.cuda.current_stream(self.stage.device).wait_event(self.ready)
        self.prepared = False

    def close(self):
        if self.prepared:
            raise RuntimeError('Join deferred retrieval before close')
        self.stage.close()
        self.closed = True
