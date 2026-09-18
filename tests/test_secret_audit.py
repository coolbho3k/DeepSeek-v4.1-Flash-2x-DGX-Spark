# SPDX-License-Identifier: AGPL-3.0-only
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'release'))
from audit_oci_secrets import Audit


class SecretAudit(unittest.TestCase):
    def test_no_credentials_required(self):
        audit = Audit(check_memory=False)
        audit.file(io.BytesIO(b'harmless'), 'source.py')
        self.assertEqual(audit.findings, [])

    def test_chunk_boundary_and_no_value_disclosure(self):
        token = b'hf_' + b'q' * 30
        audit = Audit((token,), check_memory=False)
        reader = audit.stream(io.BytesIO(b'prefix ' + token + b' suffix'), 'fixture')
        while reader.read(7):
            pass
        self.assertEqual({row['reason'] for row in audit.findings}, {'token_like', 'known_current_credential'})
        self.assertNotIn(token.decode(), json.dumps(audit.findings))

    def test_nested_compressed_source_scanned(self):
        secret = b'known-' + b'credential-' * 5
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            member = tarfile.TarInfo('source.txt')
            member.size = len(secret)
            archive.addfile(member, io.BytesIO(secret))
        output.seek(0)
        audit = Audit((secret,), check_memory=False)
        audit.file(output, 'runtime-source.tar.gz')
        self.assertEqual(audit.archives, 1)
        self.assertIn('known_current_credential', [row['reason'] for row in audit.findings])
        self.assertNotIn(secret.decode(), json.dumps(audit.findings))

    def test_sensitive_archive_path(self):
        member = tarfile.TarInfo('root/.docker/config.json')
        audit = Audit(check_memory=False)
        audit.path_check('release.tar!root/.docker/config.json', member)
        self.assertEqual(audit.findings[0]['reason'], 'sensitive_path')

    def test_known_secret_in_path_is_redacted(self):
        secret = b'credential-' * 3
        audit = Audit((secret,), check_memory=False)
        audit.add('prefix/' + secret.decode(), 'test')
        self.assertNotIn(secret.decode(), json.dumps(audit.findings))

    def test_unbounded_read_refused(self):
        audit = Audit(check_memory=False)
        with self.assertRaisesRegex(ValueError, 'Unbounded'):
            audit.stream(io.BytesIO(), 'empty').read(-1)
