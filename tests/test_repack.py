# SPDX-License-Identifier: AGPL-3.0-only
"""Small offline OCI repack checks; no Docker, credentials or GPU use."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'release'))
import repack_oci


class Repack(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(repack_oci.safe_name('./usr/lib/'),'usr/lib')
        for name in ('/etc/passwd','../bad','x/../../bad','nul\0'):
            with self.assertRaises(ValueError):repack_oci.safe_name(name)

    def test_credential_audit_never_records_match(self):
        fake=b'hf_'+b'x'*30
        findings=[];stream=repack_oci.ScannedReader(io.BytesIO(b'prefix '+fake+b' suffix'),'fixture',findings)
        while stream.read(13):pass
        self.assertEqual(len(findings),1)
        self.assertEqual(findings[0]['match_sha256'],hashlib.sha256(fake).hexdigest())
        self.assertNotIn(fake.decode(),json.dumps(findings))

    def test_reader_must_be_bounded(self):
        stream=repack_oci.ScannedReader(io.BytesIO(b'hello'),'fixture',[])
        with self.assertRaises(ValueError):stream.read()
        with self.assertRaises(ValueError):stream.read(2**20+1)

    def test_layers_roundtrip_and_digests(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            builder=repack_oci.Layers(Path(d),limit=65536)
            original={f'path/{i}':bytes([i])*24576 for i in range(4)}
            for name,data in original.items():
                member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o640;member.uid=1001
                builder.add(member,io.BytesIO(data))
            builder.finish();restored={}
            self.assertGreater(len(builder.manifests),1)
            for descriptor,diffid in zip(builder.manifests,builder.diffids):
                raw=(Path(d)/'blobs/sha256'/descriptor['digest'].split(':')[1]).read_bytes()
                self.assertEqual('sha256:'+hashlib.sha256(raw).hexdigest(),descriptor['digest'])
                plain=gzip.decompress(raw)
                self.assertEqual('sha256:'+hashlib.sha256(plain).hexdigest(),diffid)
                with tarfile.open(fileobj=io.BytesIO(plain)) as archive:
                    for member in archive:
                        self.assertEqual(member.mode,0o640);self.assertEqual(member.uid,1001)
                        restored[member.name]=archive.extractfile(member).read()
            self.assertEqual(restored,original)

    def test_oversized_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            builder=repack_oci.Layers(Path(d),limit=65536)
            member=tarfile.TarInfo('huge');member.size=65536
            with self.assertRaises(ValueError):builder.add(member,io.BytesIO(b''))


if __name__=='__main__':unittest.main()
