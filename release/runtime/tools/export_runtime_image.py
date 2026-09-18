"""Export one verified installed runtime, offline and without starting containers.

Default is read-only planning. Execute only with idle GPU and ample shared RAM.
The image is streamed once through gzip; no uncompressed duplicate is created.
Failed output/journal are preserved, and a saved attempt is never retried.
This produces a transport artifact, NOT publication/privacy/license approval.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import verify_runtime_image as images

GIB = 2**30
DISK_RESERVE = 32*GIB
MEMORY_RESERVE = 48*GIB
CHUNK = 2**20


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def fresh_path(path):
    if (path.resolve() != path or len(path.parts) < 3 or not path.parent.is_dir()
            or path.exists() or path.is_symlink()):
        raise ValueError('Use a fresh explicit file in an existing unredirected directory')


def resources(directory):
    mem = {line.split(':')[0]:int(line.split()[1])*1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.split(':')[0] in ('MemTotal','MemAvailable')}
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True,timeout=20).strip()
    return dict(memory=mem, gpu_processes=pids, disk_free=shutil.disk_usage(directory).free)


def validate_resources(sample):
    if sample['gpu_processes']:
        raise ValueError('GPU is busy: stop no jobs and do not export image layers')
    if sample['memory']['MemAvailable'] < MEMORY_RESERVE:
        raise ValueError('Image export requires at least48GiB available shared RAM')
    if sample['disk_free'] < DISK_RESERVE + 16*2**20:
        raise ValueError('Image export must preserve32GiB free disk')


class BoundedWriter:
    """Hash the compressed bytes, checking disk reserve before every write."""
    def __init__(self, stream, budget, free):
        self.stream, self.budget, self.free = stream, budget, free
        self.count = 0
        self.digest = hashlib.sha256()

    def write(self, raw):
        if self.count + len(raw) > self.budget or self.free() < DISK_RESERVE + len(raw) + CHUNK:
            raise ValueError('Archive reached its bounded disk allowance; preserve partial output')
        count = self.stream.write(raw)
        if count != len(raw):
            raise OSError('Short archive write')
        self.count += count
        self.digest.update(raw)
        return count

    def flush(self):
        self.stream.flush()


def stream_gzip(source, target, budget, free, observe, progress, monotonic=time.monotonic):
    writer = BoundedWriter(target,budget,free)
    raw_bytes = 0
    previous = monotonic()
    with gzip.GzipFile(filename='',mode='wb',compresslevel=1,mtime=0,fileobj=writer) as compressed:
        while True:
            data = source.read(CHUNK)
            if not data:
                break
            raw_bytes += len(data)
            if raw_bytes > 128*GIB:
                raise ValueError('Unexpected image stream larger than128GiB')
            compressed.write(data)
            current = monotonic()
            if current - previous >= 10:
                sample = observe()
                validate_resources(sample)
                progress(dict(time=now(),uncompressed_bytes=raw_bytes,
                    compressed_bytes=writer.count,available_bytes=sample['memory']['MemAvailable'],
                    disk_free_bytes=sample['disk_free']))
                previous = current
    writer.flush()
    return dict(bytes=writer.count,sha256=writer.digest.hexdigest(),uncompressed_bytes=raw_bytes)


def export(image, expected, identity_sha, output, receipt):
    fresh_path(output)
    fresh_path(receipt)
    partial = output.with_name(output.name+'.partial')
    journal = receipt.with_suffix('.journal')
    descriptor = output.with_name(output.name+'.json')
    for path in (partial,journal,descriptor):
        fresh_path(path)
    if len({output,receipt,partial,journal,descriptor}) != 5:
        raise ValueError('Use separate archive, metadata and private receipt paths')
    if not output.name.endswith('.tar.gz') or receipt.suffix != '.json':
        raise ValueError('Use an explicit .tar.gz archive and separate .json receipt')
    sample = resources(output.parent)
    validate_resources(sample)
    checked = images.verify(images.inspect(image),expected)
    budget = min(64*GIB,sample['disk_free']-DISK_RESERVE-16*2**20)
    journal.mkdir()
    exclusive(journal/'attempt.json',dict(time=now(),image=image,identity_sha256=identity_sha,
        output=str(output),archive_budget_bytes=budget,initial=sample,
        automatic_retries=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    process = None
    started = time.monotonic()
    try:
        with (journal/'docker-save.stderr').open('xb') as errors, partial.open('xb') as target:
            process = subprocess.Popen(['docker','image','save',image],stdout=subprocess.PIPE,stderr=errors)
            exclusive(journal/'process.json',dict(pid=process.pid,time=now()))
            with (journal/'progress.jsonl').open('x') as log:
                def progress(row):
                    line = json.dumps(row,sort_keys=True)
                    log.write(line+'\n')
                    log.flush()
                    print(line,flush=True)
                result = stream_gzip(process.stdout,target,budget,
                    lambda:shutil.disk_usage(output.parent).free,
                    lambda:resources(output.parent),progress)
            process.stdout.close()
            code = process.wait(timeout=60)
            if code != 0:
                raise ValueError('docker image save failed; inspect preserved private journal')
            target.flush()
            os.fsync(target.fileno())
        validate_resources(resources(output.parent))
        images.verify(images.inspect(image),expected)
        # Exclusive destination check; output remains private until separately reviewed.
        if output.exists() or output.is_symlink():
            raise ValueError('Archive destination appeared during export')
        partial.rename(output)
        public = dict(format='ds41_runtime_image_archive_v1',file=output.name,**result,
            runtime_image_identity_sha256=identity_sha,platform='linux/arm64',
            publication_approved=False,archive_payload_reviewed=False,
            imported_archive_verified=False)
        exclusive(descriptor,public)
        result = dict(status='verified_installed_runtime_exported',time=now(),**result,
            elapsed_seconds=time.monotonic()-started,image=checked,
            containers_started=False,image_pulled=False,source_image_changed=False,
            output=str(output),descriptor=str(descriptor),publication_approved=False,
            imported_archive_verified=False)
        exclusive(receipt,result)
        return result
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()  # Only our docker-save client, never the daemon or serving jobs.
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        exclusive(journal/'failed.json',dict(time=now(),error=f'{type(error).__name__}: {error}',
            source_image_changed=False,partial_preserved=partial.exists(),automatic_retries=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--identity-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    expected = images.read_identity(args.identity.absolute(),args.identity_sha256)
    output, receipt = args.output.absolute(), args.receipt.absolute()
    fresh_path(output)
    fresh_path(receipt)
    if args.execute:
        result = export(args.image,expected,args.identity_sha256,output,receipt)
    else:
        result = dict(status='runtime_image_export_plan',image=images.verify(images.inspect(args.image),expected),
            output=str(output),receipt=str(receipt),disk_reserve_bytes=DISK_RESERVE,
            memory_reserve_bytes=MEMORY_RESERVE,execution_requires_idle_gpu=True,
            containers_started=False,image_layers_read=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
