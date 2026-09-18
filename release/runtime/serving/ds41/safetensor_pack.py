"""Bounded, byte-preserving safetensors repacking; no tensor/GPU allocation."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct

WIDTHS = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1,
          'I16': 2, 'U16': 2, 'BF16': 2, 'F16': 2, 'I32': 4, 'U32': 4, 'F32': 4,
          'I64': 8, 'U64': 8, 'F64': 8}


def fingerprint(path):
    return stat_fingerprint(Path(path).stat())


def stat_fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field in safetensors header')
        result[key] = value
    return result


@dataclass(frozen=True)
class TensorSlice:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int
    nbytes: int
    source_fingerprint: tuple

    def validate(self):
        if (not isinstance(self.name, str) or not self.name or self.name == '__metadata__' or '\0' in self.name
                or self.dtype not in WIDTHS or any(type(dim) is not int or dim < 0 for dim in self.shape)
                or type(self.offset) is not int or self.offset < 8
                or type(self.nbytes) is not int or self.nbytes != math.prod(self.shape) * WIDTHS[self.dtype]
                or self.offset + self.nbytes > self.source_fingerprint[2]):
            raise ValueError('Invalid raw tensor slice')


def read_slices(path):
    path = Path(path).resolve()
    before = fingerprint(path)
    with path.open('rb') as stream:
        encoded = stream.read(8)
        if len(encoded) != 8:
            raise ValueError('Truncated safetensors header')
        size, = struct.unpack('<Q', encoded)
        if not 0 < size < 64 * 1024**2 or size + 8 > before[2]:
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(size), object_pairs_hook=unique_object)
    metadata = header.get('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError('Safetensors metadata must contain string values')
    result, intervals = [], []
    for name, item in header.items():
        if name == '__metadata__':
            continue
        if set(item) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('Unknown tensor header fields')
        begin, end = item['data_offsets']
        if type(begin) is not int or type(end) is not int or begin < 0 or end < begin:
            raise ValueError('Invalid tensor byte range')
        tensor = TensorSlice(name, item['dtype'], tuple(item['shape']), path, 8 + size + begin, end - begin, before)
        tensor.validate()
        intervals.append((begin, end))
        result.append(tensor)
    position = 0
    for begin, end in sorted(intervals):
        if begin != position:
            raise ValueError('Overlapping or unaccounted tensor payload bytes')
        position = end
    if size + 8 + position != before[2] or fingerprint(path) != before:
        raise ValueError('Trailing/truncated payload or source changed while reading its header')
    return result


def group_shards(tensors, max_payload_bytes=4 * 1024**3):
    if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
        raise ValueError('Shard payload target must be positive')
    seen, result, current, size = set(), [], [], 0
    for tensor in tensors:
        tensor.validate()
        if tensor.name in seen:
            raise ValueError('Duplicate selected tensor name')
        seen.add(tensor.name)
        if current and size + tensor.nbytes > max_payload_bytes:
            result.append(current)
            current, size = [], 0
        current.append(tensor)
        size += tensor.nbytes
    if current:
        result.append(current)
    return result


def encoded_header(tensors):
    header, position = {'__metadata__': {'format': 'pt'}}, 0
    for tensor in tensors:
        tensor.validate()
        if tensor.name in header:
            raise ValueError('Duplicate selected tensor')
        header[tensor.name] = dict(dtype=tensor.dtype, shape=list(tensor.shape), data_offsets=[position, position + tensor.nbytes])
        position += tensor.nbytes
    encoded = json.dumps(header, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded) % 8)
    if not tensors or len(encoded) >= 64 * 1024**2:
        raise ValueError('Empty or excessively large output shard header')
    return struct.pack('<Q', len(encoded)) + encoded


def write_shard(tensors, path, buffer_bytes=8 * 1024**2):
    """Stream unchanged payload bytes, hashing each tensor and the complete file."""
    tensors, path = list(tensors), Path(path)
    if type(buffer_bytes) is not int or not 4096 <= buffer_bytes <= 64 * 1024**2:
        raise ValueError('Use a bounded4KiB..64MiB copy buffer')
    header = encoded_header(tensors)
    sources = {tensor.path: tensor.source_fingerprint for tensor in tensors}
    if any(fingerprint(source) != expected for source, expected in sources.items()):
        raise ValueError('Source changed since the packing plan was built')
    if path.exists() or path.resolve() in sources:
        raise ValueError('Refusing to overwrite a shard or any source file')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial.safetensors')
    file_hash, inventory = hashlib.sha256(), []
    buffer = bytearray(buffer_bytes)
    with partial.open('xb', buffering=0) as output:
        if output.write(header) != len(header):
            raise OSError('Short safetensors header write')
        file_hash.update(header)
        for tensor in tensors:
            value_hash = hashlib.sha256()
            with tensor.path.open('rb', buffering=0) as source:
                if stat_fingerprint(os.fstat(source.fileno())) != tensor.source_fingerprint:
                    raise ValueError('Source identity changed before tensor copy')
                source.seek(tensor.offset)
                remaining = tensor.nbytes
                while remaining:
                    view = memoryview(buffer)[:min(remaining, buffer_bytes)]
                    count = source.readinto(view)
                    if not count:
                        raise ValueError('Truncated tensor while packing')
                    piece = view[:count]
                    if output.write(piece) != count:
                        raise OSError('Short safetensors write')
                    file_hash.update(piece)
                    value_hash.update(piece)
                    remaining -= count
                if stat_fingerprint(os.fstat(source.fileno())) != tensor.source_fingerprint:
                    raise ValueError('Source changed during tensor copy')
            inventory.append(dict(name=tensor.name, dtype=tensor.dtype, shape=list(tensor.shape),
                                  bytes=tensor.nbytes, sha256=value_hash.hexdigest()))
        os.fsync(output.fileno())
    if any(fingerprint(source) != expected for source, expected in sources.items()):
        raise ValueError('A packing source changed before shard commit')
    # No clobber on a racing target; the caller also holds an output-directory lock.
    os.link(partial, path)
    partial.unlink()
    return dict(file=path.name, bytes=path.stat().st_size, sha256=file_hash.hexdigest(),
                payload_bytes=sum(tensor.nbytes for tensor in tensors), tensors=inventory)


def tensor_hash(tensor, buffer_bytes=8 * 1024**2):
    value = hashlib.sha256()
    with tensor.path.open('rb') as source:
        source.seek(tensor.offset)
        remaining = tensor.nbytes
        while remaining:
            chunk = source.read(min(buffer_bytes, remaining))
            if not chunk:
                raise ValueError('Truncated tensor while verifying')
            value.update(chunk)
            remaining -= len(chunk)
    return value.hexdigest()


def verify_shard(path, receipt):
    path = Path(path)
    before = fingerprint(path)
    slices = read_slices(path)
    if (not slices or receipt['file'] != path.name or receipt['bytes'] != before[2]
            or receipt['payload_bytes'] != sum(tensor.nbytes for tensor in slices)
            or len(slices) != len(receipt['tensors'])):
        raise ValueError('Packed shard checksum, size or inventory changed')
    file_hash = hashlib.sha256()
    with path.open('rb') as stream:
        file_hash.update(stream.read(slices[0].offset))
        for tensor, entry in zip(slices, receipt['tensors']):
            if stream.tell() != tensor.offset:
                raise ValueError('Noncanonical packed tensor order')
            value_hash, remaining = hashlib.sha256(), tensor.nbytes
            while remaining:
                chunk = stream.read(min(8 * 1024**2, remaining))
                if not chunk:
                    raise ValueError('Truncated packed tensor')
                file_hash.update(chunk)
                value_hash.update(chunk)
                remaining -= len(chunk)
            if entry != dict(name=tensor.name, dtype=tensor.dtype, shape=list(tensor.shape), bytes=tensor.nbytes, sha256=value_hash.hexdigest()):
                raise ValueError('Packed tensor payload or metadata changed')
    if file_hash.hexdigest() != receipt['sha256'] or fingerprint(path) != before:
        raise ValueError('Packed shard changed during verification')
    return slices
