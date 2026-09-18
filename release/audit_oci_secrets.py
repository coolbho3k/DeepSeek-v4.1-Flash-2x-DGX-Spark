# SPDX-License-Identifier: AGPL-3.0-only
"""Audit every OCI layer and nested release archives without exposing secrets.

Includes superseded/deleted content, image metadata, known environment/GHCR
credentials and provider/private-key patterns. Reports contain only paths,
categories and pattern hashes, never matched values or credential context.
Existing upstream fixture exceptions apply only to immutable base layers.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tarfile
import tempfile
import time
import zipfile

from repack_oci import TOKEN, KEY, SENSITIVE, safe_name

ROOT = Path(__file__).resolve().parents[1]
EXTRA = re.compile(rb'(?:AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|'
                   rb'xox[baprs]-[0-9A-Za-z-]{20,}|sk-(?:proj-|svcacct-)[0-9A-Za-z_-]{40,})')
ENCRYPTED = re.compile(rb'-----BEGIN (?:ENCRYPTED |DSA )PRIVATE KEY-----')
PRIVATE_PATH = re.compile(r'(^|/)(?:\.bash_history|\.zsh_history|\.npmrc|\.pypirc|credentials\.json|application_default_credentials\.json|\.env)$')


def credentials():
    values = {v.encode() for k, v in os.environ.items()
              if re.search(r'(TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY)', k, re.I) and len(v) >= 12}
    config = Path.home() / '.docker/config.json'
    if config.exists():
        data = json.loads(config.read_bytes())
        for host, row in data.get('auths', {}).items():
            if host.rstrip('/').removeprefix('https://') != 'ghcr.io':
                continue
            if row.get('auth'):
                encoded = row['auth'].encode()
                decoded = base64.b64decode(encoded, validate=True)
                values.update((encoded, decoded, decoded.split(b':', 1)[-1]))
            if row.get('identitytoken'):
                values.add(row['identitytoken'].encode())
        helper = data.get('credHelpers', {}).get('ghcr.io', data.get('credsStore'))
        if helper:
            if not re.fullmatch('[A-Za-z0-9._-]+', helper):
                raise ValueError('Invalid Docker credential helper')
            result = subprocess.run(['docker-credential-' + helper, 'get'], input=b'ghcr.io\n',
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            values.add(json.loads(result.stdout)['Secret'].encode())
    return tuple(value for value in values if len(value) >= 12)


class Audit:
    def __init__(self, known=(), check_memory=True):
        self.known = known
        self.findings = []
        self.entries = self.bytes = self.archives = 0
        self.check_memory = check_memory
        self.last_check = 0.
        self.minimum_available = 2**63

    def guard(self):
        if not self.check_memory or time.monotonic() - self.last_check < 2:
            return
        self.last_check = time.monotonic()
        available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))
        self.minimum_available = min(self.minimum_available, available)
        if available < 1536 * 2**20:
            raise RuntimeError('Audit stopped to preserve serving headroom')

    def name(self, value):
        raw = value.encode(errors='replace')
        for secret in self.known:
            raw = raw.replace(secret, b'<redacted>')
        return EXTRA.sub(b'<redacted>', TOKEN.sub(b'<redacted>', raw)).decode(errors='replace')

    def add(self, path, reason, **metadata):
        row = dict(file=self.name(path), reason=reason, **metadata)
        if row not in self.findings:
            self.findings.append(row)

    def stream(self, source, path):
        audit = self

        class Reader:
            def __init__(self):
                self.tail = b''
                self.sha = hashlib.sha256()

            def read(self, size):
                if size < 0:
                    raise ValueError('Unbounded audit read refused')
                audit.guard()
                block = source.read(min(size, 2**20))
                self.sha.update(block)
                audit.bytes += len(block)
                if audit.bytes > 128 * 2**30:
                    raise ValueError('Expanded-content audit budget exceeded')
                data = self.tail + block
                for secret in audit.known:
                    if secret in data:
                        audit.add(path, 'known_current_credential')
                for pattern, reason in ((TOKEN, 'token_like'), (KEY, 'private_key_header'),
                                        (EXTRA, 'additional_provider_token'), (ENCRYPTED, 'private_key_header')):
                    for match in pattern.finditer(data):
                        if block and match.end() == len(data) and len(match.group()) < 512:
                            continue
                        audit.add(path, reason, match_sha256=hashlib.sha256(match.group()).hexdigest())
                self.tail = data[-max([512] + [len(v) for v in audit.known]):]
                return block

        return Reader()

    def file(self, source, path, depth=0):
        reader = self.stream(source, path)
        lower = path.lower()
        if depth > 5:
            raise ValueError('Nested archive depth exceeded')
        if lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.xz', '.tar.bz2')):
            try:
                with tarfile.open(fileobj=reader, mode='r|*') as archive:
                    self.archives += 1
                    for member in archive:
                        name = safe_name(member.name)
                        self.entries += 1
                        combined = path + '!' + name
                        self.path_check(combined, member)
                        if member.isfile():
                            self.file(archive.extractfile(member), combined, depth + 1)
                        archive.members.clear()
            except (tarfile.ReadError, EOFError):
                self.add(path, 'archive_unreadable')
        elif lower.endswith(('.zip', '.whl')):
            with tempfile.TemporaryFile(dir=ROOT / 'artifacts') as temporary:
                while block := reader.read(2**20):
                    temporary.write(block)
                    if temporary.tell() > 2**30:
                        raise ValueError('Zip spool budget exceeded')
                temporary.seek(0)
                try:
                    with zipfile.ZipFile(temporary) as archive:
                        self.archives += 1
                        for member in archive.infolist():
                            if not member.is_dir():
                                name = path + '!' + safe_name(member.filename)
                                with archive.open(member) as stream:
                                    self.file(stream, name, depth + 1)
                except (zipfile.BadZipFile, RuntimeError):
                    self.add(path, 'archive_unreadable')
                os.posix_fadvise(temporary.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        while reader.read(2**20):
            pass
        return reader.sha.hexdigest()

    def path_check(self, path, member):
        # Include archive-internal filenames in path checks.
        last = path.rsplit('!', 1)[-1]
        if SENSITIVE.search(last) or PRIVATE_PATH.search(last):
            self.add(path, 'sensitive_path', kind='directory' if member.isdir() else 'file', bytes=member.size)


def run(layout, output):
    if output.exists():
        raise ValueError('Use a fresh audit output')
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
    index = json.loads((layout / 'index.json').read_bytes())
    descriptor = index['manifests'][0]
    blobs = layout / 'blobs/sha256'
    manifest = json.loads((blobs / descriptor['digest'].split(':')[1]).read_bytes())
    base = ROOT / 'artifacts/ds41-ghcr-oci-v1'
    base_index = json.loads((base / 'index.json').read_bytes())['manifests'][0]
    if base_index['digest'] != 'sha256:8cc05f677a94367d56a058c0bd93742b51b7e1596953e61bcd9fabcc0ffc9fde':
        raise ValueError('Unreviewed base')
    base_manifest = json.loads((base / 'blobs/sha256' / base_index['digest'].split(':')[1]).read_bytes())
    base_layers = {row['digest'] for row in base_manifest['layers']}
    prior = json.loads((base / 'audit.json').read_bytes())['findings']
    signature = lambda row: (row['file'], row['reason'], row.get('match_sha256'), row.get('kind'), row.get('bytes'))
    known_findings = {signature(row) for row in prior}
    audit = Audit(credentials())
    started = time.monotonic()
    results = []
    for item in [descriptor, manifest['config']] + manifest['layers']:
        path = blobs / item['digest'].split(':')[1]
        if path.is_symlink() or path.stat().st_size != item['size']:
            raise ValueError('Blob size/path mismatch')
        audit.findings = []
        with path.open('rb') as raw:
            if 'layer' in item['mediaType']:
                reader = audit.stream(raw, 'blob:' + item['digest'])
                with gzip.GzipFile(fileobj=reader, mode='rb') as uncompressed:
                    with tarfile.open(fileobj=uncompressed, mode='r|') as archive:
                        for member in archive:
                            name = safe_name(member.name)
                            audit.entries += 1
                            audit.path_check(name, member)
                            if member.isfile():
                                audit.file(archive.extractfile(member), name)
                            archive.members.clear()
                while reader.read(2**20):
                    pass
                actual = reader.sha.hexdigest()
            else:
                actual = audit.file(raw, 'metadata:' + item['digest'])
            os.posix_fadvise(raw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if actual != item['digest'].split(':')[1]:
            raise ValueError('Blob hash mismatch')
        findings = audit.findings
        reviewed = [row for row in findings if item['digest'] in base_layers and signature(row) in known_findings]
        unresolved = [row for row in findings if row not in reviewed]
        results.append(dict(digest=item['digest'], bytes=item['size'], base_layer=item['digest'] in base_layers,
                            reviewed_upstream_findings=reviewed, unresolved_findings=unresolved))
        print(json.dumps(dict(stage='blob_audited', blobs=len(results), entries=audit.entries,
            unresolved=sum(len(r['unresolved_findings']) for r in results), elapsed_s=round(time.monotonic()-started))), flush=True)
    result = dict(status='passed' if not any(r['unresolved_findings'] for r in results) else 'review_required',
        manifest_digest=descriptor['digest'], layers=len(manifest['layers']), all_layer_blobs_rehashed=True,
        superseded_files_scanned=True, nested_archives_scanned=audit.archives, entries=audit.entries,
        expanded_and_compressed_bytes_scanned=audit.bytes, known_credentials_checked=len(audit.known),
        known_credential_findings=sum(f['reason'] == 'known_current_credential'
            for r in results for f in r['unresolved_findings']),
        minimum_mem_available_bytes=audit.minimum_available, elapsed_s=time.monotonic()-started,
        results=results, limitations='Pattern and current-credential audit, not a comprehensive security certification.',
        server_touched=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.layout.resolve(), args.output)
