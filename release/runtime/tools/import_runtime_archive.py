"""Import the reviewed untagged DS41 runtime archive once on another idle host.

Pinned archive identity also pins its observed tag-free Docker manifest.
Streams to docker load without storing a second archive or starting containers.
Default only plans. A saved attempt is never automatically dispatched again.
Public archive/privacy/license approval is separate from transport verification.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify_runtime_image as images

ARCHIVE_SHA = 'dd6b4604be3985ad2413e40123d3a74415c4e63b38d0caeff2e54b47eb7fe5a3'
ARCHIVE_BYTES = 21625698481
IDENTITY_SHA = '03c151b169249d413dc64a365d3fa8f5c104561d1eb3bf2bd30138b9010c3e3b'
IMPORTED_ID = 'sha256:314893911de4009c4e989408d17d48ec0f050bcd972e9683adf02e6cc38a8e58'
GIB = 2**30
REMOTE_CODE = '''import json,pathlib,shutil,subprocess,sys
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemTotal','MemAvailable')}
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
print(json.dumps(dict(memory=mem,gpu_processes=pids,disk_free=shutil.disk_usage(sys.argv[1]).free)))'''


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def ssh_args(remote,command):
    if not re.fullmatch(r'(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9.-]*',remote):
        raise ValueError('Use a plain SSH endpoint, not options or shell text')
    return ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',remote,shlex.join(command)]


def observe(remote,directory):
    if not directory.startswith('/') or '..' in Path(directory).parts:
        raise ValueError('Use an explicit existing remote filesystem check path')
    output = subprocess.check_output(ssh_args(remote,['python3','-c',REMOTE_CODE,directory]),text=True,timeout=30)
    return json.loads(output)


def resources(sample,initial=False):
    if sample['gpu_processes'] or sample['memory']['MemAvailable'] < 48*GIB:
        raise ValueError('Runtime import requires idle GPU and48GiB available shared RAM')
    if sample['disk_free'] < (160 if initial else 32)*GIB:
        raise ValueError('Runtime import requires128GiB staging allowance plus32GiB disk reserve initially')


def stamp(path):
    if path.resolve() != path or not path.is_file():
        raise ValueError('Use an unredirected regular runtime archive')
    info = path.stat()
    return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_nlink]


def verify_archive(path):
    before = stamp(path)
    if before[3] != ARCHIVE_BYTES:
        raise ValueError('Unexpected archive size')
    fd = os.open(path,os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    if digest != ARCHIVE_SHA or stamp(path) != before:
        raise ValueError('Runtime archive differs from the reviewed untagged transport')
    return before


def execute(archive,identity,remote,disk_path,output):
    journal = output.with_suffix('.journal')
    for path in (output,journal):
        if path.resolve() != path or path.exists() or path.is_symlink() or not path.parent.is_dir():
            raise ValueError('Preserve existing attempts; use fresh private output paths')
    expected = images.read_identity(identity,IDENTITY_SHA)
    sample = observe(remote,disk_path)
    resources(sample,initial=True)
    # The local model must also be absent during the full21.6GB read.
    jobs = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
    if jobs:
        raise ValueError('Local GPU is busy; leave serving jobs unchanged')
    before = verify_archive(archive)
    journal.mkdir()
    write(journal/'attempt.json',dict(time=now(),remote=remote,archive=str(archive),
        archive_sha256=ARCHIVE_SHA,archive_stamp=before,initial=sample,
        automatic_retries=False,containers_started=False,tag_updates=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    process = None
    started = time.monotonic()
    try:
        with archive.open('rb') as source,(journal/'docker-load.log').open('xb') as log:
            process = subprocess.Popen(ssh_args(remote,['docker','image','load']),
                stdin=source,stdout=log,stderr=subprocess.STDOUT)
            write(journal/'process.json',dict(pid=process.pid,time=now()))
            with (journal/'watch.jsonl').open('x') as watch:
                while process.poll() is None:
                    try:
                        sample = observe(remote,disk_path)
                        resources(sample)
                        row = dict(time=now(),stage='same_import_watch',**sample)
                    except (subprocess.SubprocessError,OSError,json.JSONDecodeError) as error:
                        row = dict(time=now(),stage='observation_retry_no_import_restart',error=str(error))
                    line = json.dumps(row)
                    watch.write(line+'\n');watch.flush();print(line,flush=True)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
            if process.returncode != 0:
                raise ValueError('Import did not acknowledge completion; inspect the same saved attempt')
        if stamp(archive) != before:
            raise ValueError('Source archive changed during transport')
        checked = images.verify(images.inspect(IMPORTED_ID,remote),expected)
        resources(observe(remote,disk_path))
        result = dict(status='runtime_archive_imported_and_public_identity_verified',time=now(),
            archive_sha256=ARCHIVE_SHA,archive_bytes=ARCHIVE_BYTES,remote=remote,
            elapsed_seconds=time.monotonic()-started,installed=checked,
            source_archive_unchanged=True,containers_started=False,tag_updates=False,
            automatic_retries=False,publication_approved=False,fresh_os_install_tested=False)
        write(output,result)
        return result
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()  # Only our SSH client, never unrelated jobs or Docker daemon.
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=30)
        write(journal/'failed.json',dict(time=now(),error=f'{type(error).__name__}: {error}',
            automatic_retries=False,remote_import_state_requires_inspection=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--identity',required=True,type=Path)
    parser.add_argument('--ssh',required=True)
    parser.add_argument('--remote-disk-path',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = execute(args.archive.absolute(),args.identity.absolute(),args.ssh,args.remote_disk_path,args.output.absolute())
    else:
        result = dict(status='runtime_archive_import_plan',command=ssh_args(args.ssh,['docker','image','load']),
            archive_sha256=ARCHIVE_SHA,archive_bytes=ARCHIVE_BYTES,expected_image_id=IMPORTED_ID,
            image_layers_read=False,containers_started=False,tag_updates=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
