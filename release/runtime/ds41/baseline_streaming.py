"""Bounded baseline activations with the frozen combined reader's gather.

Only I/O/storage changes: the exact baseline route order, seeded indices,
FP16 inputs, FP32 weights, masks and four-field selection metadata are retained.
No edits to the frozen capture, fitter, subset index or supplemental reader.
"""
import json

import torch

from .baseline_subset import BaselineSubsetIndex, META_FIELDS, expert_ids
from .calibration_inputs import CombinedRoutedInputs
from scripts import run_quant_queue as baseline


class BaselineStreamingInputs(CombinedRoutedInputs):
    def __init__(self, index, *, device='cuda'):
        if not isinstance(index, BaselineSubsetIndex):
            raise ValueError('Streaming inputs require the fully validated baseline index')
        index.validate_unchanged()
        self.index, self.device, self.layer = index, device, index.layer
        self.capture_sha256 = index.config['capture_sha256']
        self.sources = [{'capture_sha256': self.capture_sha256}]
        self.counts_by_source = [index.counts]
        records = {record['id']: record for record in map(json.loads,
            (index.root / baseline.CORPUS / 'records.jsonl').read_bytes().splitlines())}
        self.splits = {}
        for split, tensors in index.splits.items():
            files, offsets = [], [0]
            for item in index.files:
                if item['split'] != split:
                    continue
                files.append(dict(path=index.root / item['path'], stamp=item['stamp'], source=0,
                    record=records[item['record']], rows=item['rows'], sha256=item['sha256'], row_offset=offsets[-1]))
                offsets.append(offsets[-1] + item['rows'])
            if offsets[-1] != len(tensors['routes']):
                raise ValueError('Baseline file offsets differ from validated route order')
            self.splits[split] = dict(files=files, offsets=torch.tensor(offsets, dtype=torch.int64),
                                     routes=tensors['routes'])
        self.validate_unchanged()

    def validate_unchanged(self):
        self.index.validate_unchanged()

    def select(self, expert, split, maximum, seed):
        expert_ids([expert])
        cap = {'calibration': 8192, 'heldout': 1024}.get(split)
        if (cap is None or type(maximum) is not int or not 0 < maximum <= cap or type(seed) is not int):
            raise ValueError('Require a bounded baseline split/row cap and integer seed')
        self.validate_unchanged()
        # Reuse the already frozen, tested per-file gather without adding a
        # supplement or changing the identity/order of baseline activations.
        selected, rich_meta = super().select(expert, split, maximum, seed)
        metadata = {key: rich_meta[key] for key in META_FIELDS}
        if metadata != self.index.selection_metadata(expert, split, maximum, seed):
            raise ValueError('Streaming gather differs from exact frozen baseline selection')
        self.validate_unchanged()
        return selected, metadata

    def retained_index_bytes(self):
        return sum(tensor.numel() * tensor.element_size()
                   for data in self.index.splits.values() for tensor in data.values())
