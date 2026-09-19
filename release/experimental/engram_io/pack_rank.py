# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare lossless, rank-owned experimental shards without altering weights.

Dense row format is MiaAI-Lab's; page15 geometry and this orchestration are
local AGPLv3 additions. Run in a memory-limited CPU container, one per Spark.
No resume, overwrites, network, credentials, or serving lifecycle operations.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

from packing import PAGE, PER_PAGE, ROW, pack, partitions, table


def upstream_partitions(config, path):
    """Execute only the reviewed pure layout AST, without importing vLLM/CUDA."""
    raw = Path(path).read_bytes()
    parsed = ast.parse(raw)
    names = {'_is_prime', 'find_next_prime', 'EngramLayout'}
    selected = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in names]
    if {n.name for n in selected} != names or len(selected) != 3:
        raise ValueError('Changed upstream layout definitions')
    namespace = {'np': np, 'torch': SimpleNamespace(tensor=np.asarray)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    layout = namespace['EngramLayout'](SimpleNamespace(**(config.get('text_config') or config)))
    result = {}
    for layer, groups in zip(layout.layer_ids, layout.primes, strict=True):
        sizes = [p for group in groups for p in group]
        split = sum(sizes[:len(sizes)//2])
        result[layer] = dict(head_sizes=sizes, ranges=((0, split), (split, sum(sizes))))
    return result, hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--layouts', nargs='+', choices=('dense', 'page15'), default=['page15', 'dense'])
    p.add_argument('--upstream-layout', type=Path,
        default=Path('/opt/ds41-venv/lib/python3.12/site-packages/vllm/models/deepseek_v4_1/common/engram.py'))
    a = p.parse_args()
    if len(set(a.layouts)) != len(a.layouts):
        raise ValueError('Duplicate output layout')
    raw = a.config.read_bytes()
    layout = partitions(json.loads(raw))
    upstream, upstream_sha = upstream_partitions(json.loads(raw), a.upstream_layout)
    if layout != upstream:
        raise ValueError('Independent partitions differ from installed vLLM')
    root = a.output.absolute()
    if root.resolve() != root or not root.is_dir() or any(root.iterdir()):
        raise ValueError('Use a fresh empty output directory')
    required = 0
    sources = {}
    for layer, row in layout.items():
        source = a.sources / f'engram-layer-{layer:02}.safetensors'
        sources[layer] = table(source, layer)
        if sources[layer]['rows'] != sum(row['head_sizes']):
            raise ValueError('Config rows differ from checkpoint')
        lo, hi = row['ranges'][a.rank]
        for variant in a.layouts:
            payload = ((hi-lo+PER_PAGE-1)//PER_PAGE*PAGE if variant == 'page15'
                       else ((hi-lo)*ROW+PAGE-1)//PAGE*PAGE)
            required += PAGE + payload
    if shutil.disk_usage(root).free < required + 16*2**30:
        raise ValueError('Insufficient disk room plus 16 GiB reserve')
    record = dict(status='running', rank=a.rank, config_sha256=hashlib.sha256(raw).hexdigest(),
                  upstream_layout_sha256=upstream_sha, independent_partition_match=True,
                  partitions=layout, expected_bytes=required, started_at=time.time(), shards=[])
    def save():
        # Generated experiment report only; never a serving/source artifact.
        temp = root / 'rank-manifest.json.next'
        with temp.open('x') as f:
            json.dump(record, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, root / 'rank-manifest.json')
    save()
    print(json.dumps(dict(stage='layout_verified', rank=a.rank, expected_gib=required/2**30)), flush=True)
    try:
        for variant in a.layouts:
            for layer, row in layout.items():
                lo, hi = row['ranges'][a.rank]
                target = root / f'engram-layer-{layer:02}-rank{a.rank}-{variant}.bin'
                receipt = pack(sources[layer]['path'], target, layer, lo, hi, variant)
                record['shards'].append(dict(path=str(target), **receipt)); save()
                print(json.dumps(dict(stage='shard_complete', path=str(target),
                                      seconds=receipt['elapsed_seconds'])), flush=True)
        record.update(status='complete', finished_at=time.time())
    except BaseException as error:
        record.update(status='failed', finished_at=time.time(), error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
