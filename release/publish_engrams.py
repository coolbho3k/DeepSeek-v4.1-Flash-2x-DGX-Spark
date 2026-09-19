# SPDX-License-Identifier: AGPL-3.0-only
"""Maintainer-only additive upload. Never overwrite the canonical Engrams.

Upload each rank to its own staging branch, then publish all four files and
the manifest in ONE main-branch commit. Git is updated only after that commit
is anonymously verified. Credentials are consumed for authentication only;
never printed, persisted, or passed in process arguments.
"""
import argparse
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import resource
import sys
import threading
import time

PREFIX = 'engram-page15-v1'
REPO = 'coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw'


class BoundedFile(io.BufferedReader):
    def __init__(self, raw, offset=0, length=None):
        super().__init__(raw)
        self.origin = offset
        self.length = os.fstat(self.fileno()).st_size-offset if length is None else length
        super().seek(offset)

    def tell(self):
        return super().tell()-self.origin

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.tell()+offset if whence == 1 else self.length+offset if whence == 2 else -1
        if not 0 <= position <= self.length:
            raise ValueError('Seek outside bounded upload range')
        return super().seek(self.origin+position)-self.origin

    def read(self, size=-1):
        if size < 0 or size > 64 * 2**20:
            raise ValueError('Unbounded upload read refused')
        result = super().read(min(size, self.length-self.tell()))
        os.posix_fadvise(self.fileno(), self.origin+max(0, self.tell()-len(result)),
                        len(result), os.POSIX_FADV_DONTNEED)
        return result


def fingerprint(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--base-revision', required=True)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--rank-manifest', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--token-stdin', action='store_true')
    p.add_argument('--publish', action='store_true', required=True)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (768*2**20, 768*2**20))
    token = (sys.stdin.buffer.readline(4096).decode().strip() if a.token_stdin
             else os.environ.pop('HF_TOKEN_WRITE', ''))
    if not token.startswith('hf_'):
        raise ValueError('HF_TOKEN_WRITE authentication is required')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    logging.disable(logging.CRITICAL)
    def guard():
        while True:
            available = next(int(x.split()[1])*1024 for x in
                Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
            if available < 1024*2**20:
                print('Upload stopped to preserve serving headroom.', flush=True)
                os._exit(75)
            time.sleep(2)
    threading.Thread(target=guard, daemon=True).start()
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)
    if api.whoami().get('name') != 'coolbho3k':
        raise ValueError('Unexpected publisher identity')
    manifest = json.loads(a.manifest.read_bytes())
    proof = json.loads(a.rank_manifest.read_bytes())
    if (manifest['repo_id'] != REPO or manifest['format'] != 'ds41_engram_page15_release_v1'
            or proof['rank'] != a.rank or proof['status'] != 'complete'
            or not proof['independent_partition_match']):
        raise ValueError('Invalid publication inventory')
    branch = PREFIX + '-rank' + str(a.rank)
    api.create_branch(REPO, branch=branch, revision=a.base_revision, exist_ok=True)
    info = api.model_info(REPO, revision=branch, files_metadata=True)
    existing = {s.rfilename: s for s in info.siblings}
    operations, streams, sources, parts = [], [], [], {}
    try:
        for name, row in manifest['files'].items():
            if row['rank'] != a.rank:
                continue
            source = next(s for s in proof['shards'] if s['layer'] == row['layer'] and s['layout'] == 'page15')
            path = Path(source['path'])
            if (path.resolve() != path or fingerprint(path) != source['packed_fingerprint']
                    or not source['complete_byte_readback'] or source['packed_sha256'] != row['sha256']
                    or path.stat().st_size != row['bytes'] or name != PREFIX+'/'+path.name):
                raise ValueError('Packed source changed after verification')
            sources.append((path, fingerprint(path)))
            parts[name] = []
            for number, offset in enumerate(range(0, row['bytes'], 8*2**30)):
                part_name = name+f'.part-{number:05d}'
                length = min(8*2**30, row['bytes']-offset)
                print(json.dumps(dict(stage='hashing_part', file=part_name, bytes=length)), flush=True)
                stream = BoundedFile(io.FileIO(path, 'r'), offset, length)
                streams.append(stream)
                op = CommitOperationAdd(path_in_repo=part_name, path_or_fileobj=stream)
                part = dict(path=part_name, offset=offset, bytes=length, sha256=op.upload_info.sha256.hex())
                if op.upload_info.size != length:
                    raise ValueError('Bounded upload range length differs')
                parts[name].append(part)
                if part_name in existing:
                    remote = existing[part_name]
                    if remote.size != length or not remote.lfs or remote.lfs.sha256 != part['sha256']:
                        raise ValueError('Staging part already contains different bytes')
                else:
                    operations.append(op)
        if len(sources) != 2:
            raise ValueError('Expected exactly two rank-owned tables')
        print(json.dumps(dict(stage='uploading', rank=a.rank, files=len(operations))), flush=True)
        if operations:
            commit = api.create_commit(REPO, revision=branch, parent_commit=info.sha,
                operations=operations, num_threads=1,
                commit_message='Stage lossless page15 Engrams for rank '+str(a.rank))
            revision = commit.oid
        else:
            revision = info.sha
        if any(fingerprint(path) != before for path, before in sources):
            raise ValueError('Source changed during upload; do not promote')
        result = dict(status='rank_staged_not_promoted', repo=REPO, branch=branch,
                      revision=revision, rank=a.rank,
                      manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(), parts=parts)
        with a.receipt.open('x') as out:
            out.write(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # SDK exceptions can contain signed URLs; never echo their messages.
        print('Publication failed: '+type(error).__name__+'; no server action performed.', flush=True)
        raise SystemExit(1)
