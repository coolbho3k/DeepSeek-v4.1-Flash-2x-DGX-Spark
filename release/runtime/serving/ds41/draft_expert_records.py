# SPDX-License-Identifier: AGPL-3.0-only
# TP layout follows the Apache-2.0 vLLM RoutedExperts _load_w13/_load_w2
# contract. This original CPU byte packer does not change FP4 values/scales.
"""Lossless TP2 draft expert records, before native DeepGEMM scale conversion.

An unselected offline layout, not a draft quantizer or a serving backend.
Each4096-aligned record holds w13, w2, w13_scale, w2_scale. Original UE8M0
scale bytes are retained; native GPU layout conversion is still required.
"""
import os
from pathlib import Path
import re

from .safetensor_pack import read_slices, stat_fingerprint

RECORD_BYTES = 9_400_320
COMPONENTS = (
    ('w13_weight', (2304, 2560), 0, 5_898_240),
    ('w2_weight', (5120, 576), 5_898_240, 2_949_120),
    ('w13_scale', (2304, 160), 8_847_360, 368_640),
    ('w2_scale', (5120, 36), 9_216_000, 184_320),
)
SOURCE_LAYOUT = {
    ('w1', 'weight'): ('I8', (2304, 2560)),
    ('w3', 'weight'): ('I8', (2304, 2560)),
    ('w2', 'weight'): ('I8', (5120, 1152)),
    ('w1', 'scale'): ('F8_E8M0', (2304, 160)),
    ('w3', 'scale'): ('F8_E8M0', (2304, 160)),
    ('w2', 'scale'): ('F8_E8M0', (5120, 72)),
}


def tp_half(payload, rows, columns, axis, rank):
    """Split a row-major byte matrix on the native expert TP axis."""
    if (type(rank) is not int or rank not in (0, 1) or axis not in (0, 1)
            or type(rows) is not int or type(columns) is not int
            or rows <= 0 or columns <= 0 or len(payload) != rows*columns
            or (rows, columns)[axis] % 2):
        raise ValueError('Invalid TP2 byte matrix')
    if axis == 0:
        size = len(payload)//2
        return payload[rank*size:(rank+1)*size]
    width = columns//2
    result = bytearray(rows*width)
    for row in range(rows):
        start = row*columns + rank*width
        result[row*width:(row+1)*width] = payload[start:start+width]
    return result


class DraftExpertRecords:
    """Bounded source reads; caller decides whether/how records are persisted."""

    def __init__(self, paths):
        self.tensors = {}
        self.identities = {}
        self.fds = {}
        try:
            for path in paths:
                path = Path(path).absolute()
                if path.resolve() != path or not path.is_file():
                    raise ValueError('Require canonical regular checkpoint files')
                for tensor in read_slices(path):
                    match = re.fullmatch(r'mtp\.([012])\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)', tensor.name)
                    if not match:
                        if '.experts.' in tensor.name:
                            raise ValueError('Unrecognized draft expert tensor')
                        continue
                    layer, expert = map(int, match.group(1, 2))
                    part = match.group(3, 4)
                    if not 0<=expert<128 or (tensor.dtype, tensor.shape)!=SOURCE_LAYOUT[part]:
                        raise ValueError('Draft expert shape/dtype changed')
                    key = (layer, expert, *part)
                    if key in self.tensors:
                        raise ValueError('Duplicate draft expert tensor')
                    self.tensors[key] = tensor
                    self.identities[path] = tensor.source_fingerprint
            expected = {(layer, expert, *part) for layer in range(3)
                        for expert in range(128) for part in SOURCE_LAYOUT}
            if set(self.tensors)!=expected:
                raise ValueError('Incomplete384-expert draft inventory')
            for path, identity in self.identities.items():
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                self.fds[path] = fd
                if stat_fingerprint(os.fstat(fd)) != identity:
                    raise ValueError('Checkpoint changed before record access')
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

    def verify_sources(self):
        if set(self.fds)!=set(self.identities):
            raise ValueError('Closed/incomplete expert record reader')
        for path, identity in self.identities.items():
            if (stat_fingerprint(os.fstat(self.fds[path])) != identity
                    or stat_fingerprint(path.stat()) != identity):
                raise ValueError('Checkpoint changed during record access')

    def source_bytes(self, layer, expert, matrix, kind):
        self.verify_sources()
        tensor = self.tensors[(layer, expert, matrix, kind)]
        data = os.pread(self.fds[tensor.path], tensor.nbytes, tensor.offset)
        if len(data)!=tensor.nbytes:
            raise OSError('Short draft tensor read')
        self.verify_sources()
        return data

    def pack(self, layer, expert, rank):
        if (type(layer) is not int or layer not in range(3) or type(expert) is not int
                or expert not in range(128) or type(rank) is not int or rank not in (0,1)):
            raise ValueError('Invalid draft layer/expert/TP rank')
        result = bytearray(RECORD_BYTES)
        for name, shape, offset, size in COMPONENTS:
            kind = 'scale' if name.endswith('scale') else 'weight'
            if name.startswith('w13'):
                position = offset
                for matrix in ('w1', 'w3'):
                    source_shape = SOURCE_LAYOUT[(matrix, kind)][1]
                    data = self.source_bytes(layer, expert, matrix, kind)
                    selected = tp_half(data, *source_shape, 0, rank)
                    result[position:position+len(selected)] = selected
                    position += len(selected)
                assert position == offset+size
            else:
                data = self.source_bytes(layer, expert, 'w2', kind)
                selected = tp_half(data, *SOURCE_LAYOUT[('w2', kind)][1], 1, rank)
                assert len(selected)==size
                result[offset:offset+size] = selected
        self.verify_sources()
        assert len(result)==RECORD_BYTES and RECORD_BYTES%4096==0
        return result

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
