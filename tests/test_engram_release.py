# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only atomic download/compatibility tests. No GPU, SSH or large files."""
import copy
import hashlib
import importlib.util
import io
import json
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tested_engram_assets', ROOT/'release/runtime/tools/engram_assets.py')
assets = importlib.util.module_from_spec(spec); spec.loader.exec_module(assets)


def digest(raw): return hashlib.sha256(raw).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data); self.status=status; self.headers=headers or {}


class PackedDownloads(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.path=self.root/'table.bin'; self.data=b'first-part'+b'second-part'
        self.row=dict(bytes=len(self.data),sha256=digest(self.data),parts=[
            dict(path='part0',offset=0,bytes=10,sha256=digest(self.data[:10])),
            dict(path='part1',offset=10,bytes=11,sha256=digest(self.data[10:]))])
        self.requests=[]
    def tearDown(self): self.tmp.cleanup()
    def opener(self, request, timeout):
        self.requests.append(request)
        part=self.row['parts'][int(request.full_url[-1])]
        data=self.data[part['offset']:part['offset']+part['bytes']]
        header=request.headers.get('Range')
        if header:
            start=int(header.split('=')[1].split('-')[0])
            return Response(data[start:],206,{'Content-Range':f'bytes {start}-{len(data)-1}/{len(data)}'})
        return Response(data)
    def download(self): assets.download_table('https://example.invalid/',self.row,self.path,self.opener)
    def test_exact_assembly_without_part_copies(self):
        self.download(); self.assertEqual(self.path.read_bytes(),self.data)
        self.assertEqual(list(self.root.iterdir()),[self.path])
    def test_resume_within_second_part(self):
        self.path.with_name('table.bin.download-part').write_bytes(self.data[:14])
        self.download(); self.assertEqual(self.path.read_bytes(),self.data)
        self.assertEqual(len(self.requests),1);self.assertEqual(self.requests[0].headers['Range'],'bytes=4-')
    def test_reuse_complete_file_offline(self):
        self.path.write_bytes(self.data)
        with patch.object(self,'opener',side_effect=AssertionError('No network')): self.download()
        self.assertEqual(self.path.read_bytes(),self.data)
    def test_corrupt_complete_prefix_is_not_published(self):
        self.path.with_name('table.bin.download-part').write_bytes(b'X'+self.data[1:14])
        with self.assertRaisesRegex(ValueError,'checksum'): self.download()
        self.assertFalse(self.path.exists()); self.assertFalse(self.requests)
    def test_corrupt_download_not_published(self):
        with patch.object(self,'opener',return_value=Response(b'x'*10)):
            with self.assertRaisesRegex(ValueError,'checksum'): self.download()
        self.assertFalse(self.path.exists())
    def test_wrong_full_digest_not_published(self):
        self.row['sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'Assembled'): self.download()
        self.assertFalse(self.path.exists())
    def test_ignored_resume_range_refused(self):
        self.path.with_name('table.bin.download-part').write_bytes(self.data[:14])
        with patch.object(self,'opener',return_value=Response(self.data[10:])):
            with self.assertRaisesRegex(ValueError,'resume range'): self.download()
        self.assertFalse(self.path.exists())
    def test_truncated_response_is_resumable(self):
        with patch.object(self,'opener',return_value=Response(self.data[:4])):
            with self.assertRaisesRegex(ValueError,'Interrupted'): self.download()
        self.assertFalse(self.path.exists()); self.download()
        self.assertEqual(self.path.read_bytes(),self.data)
    def test_symlinks_refused(self):
        victim=self.root/'victim'; victim.write_bytes(b'untouched')
        self.path.with_name('table.bin.download-part').symlink_to(victim)
        with self.assertRaisesRegex(ValueError,'Redirected'): self.download()
        self.assertEqual(victim.read_bytes(),b'untouched')
    def test_rank_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError,'rank'): assets.check_reference(assets.reference(self.root,0),1)
    def test_mounts_only_own_tables_readonly(self):
        for rank in (0,1):
            mounts=assets.mounts(assets.reference(self.root,rank),rank)
            self.assertEqual(len(mounts),2)
            for source,target,readonly in mounts:
                self.assertTrue(readonly);self.assertIn(f'rank{rank}',source)
                self.assertTrue(target.startswith('/opt/ds41-engram-packed/'))
    def test_atomic_pointer_survives_failure(self):
        path=self.root/'prepared.json';path.write_text('{"old":true}')
        with patch.object(assets.os,'replace',side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError): assets.atomic_json(path,dict(new=True))
        self.assertEqual(json.loads(path.read_text()),{'old':True})
        self.assertEqual(list(self.root.iterdir()),[path])


class ReleaseCompatibility(unittest.TestCase):
    def test_publication_reader_and_transport_pins_agree(self):
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        raw=(ROOT/'release/engram-release-manifest.json').read_bytes()
        manifest=assets.parse_manifest(raw)
        self.assertEqual(digest(raw),lock['engram']['manifest_sha256'])
        # Packed from the original weights; the pinned (refit) weights keep those tables byte for byte.
        self.assertEqual(manifest['source_model']['repo'],lock['model']['repo'])
        self.assertEqual(manifest['source_model']['revision'],'5571a3c9ee09f9495e9118b54e022ac591b85373')
        model=json.loads((ROOT/'release/model-release-manifest.json').read_bytes())
        self.assertEqual({n:model['files'][n] for n in ('engrams/engram-layer-01.safetensors','engrams/engram-layer-14.safetensors')},
            {'engrams/engram-layer-01.safetensors':{'bytes':101377628616,'sha256':'9ecd9bc00c8c045104d639dfdd8d6de4fe0457ce05ad8885baab062d059b0c9e'},
             'engrams/engram-layer-14.safetensors':{'bytes':101380404312,'sha256':'0c33910e5a59c2787d1bc1891ca5961248e1db94e7919c562afbf4cedba2f5b2'}})
        self.assertEqual(manifest['repo_id'],lock['engram']['repo'])
        self.assertNotEqual(lock['model']['revision'],lock['engram']['revision'])
        self.assertEqual(sum(len(row['parts']) for row in manifest['files'].values()),28)
        for layer in (1,14):
            rows=sorted((row for row in manifest['files'].values() if row['layer']==layer),key=lambda row:row['rank'])
            self.assertEqual(rows[0]['lo'],0)
            self.assertEqual(rows[0]['hi'],rows[1]['lo'])
            self.assertEqual(rows[1]['hi'],rows[0]['total_rows'])
            self.assertEqual(rows[0]['total_rows'],rows[1]['total_rows'])
    def test_model_and_draft_pins(self):
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        self.assertEqual(lock['model']['revision'],'650cae2c13aaaec303871a35301503570889c0be')
        self.assertEqual(lock['draft']['revision'],'f74b8c9b7d4448e2deeaf23be7d03557f2878504')
        self.assertEqual(digest((ROOT/'release/model-release-manifest.json').read_bytes()),lock['model']['manifest_sha256'])
    def test_native_payload_matches_serving_trial(self):
        binary=ROOT/'release/runtime/serving/miaai-row-store-v1.so'
        self.assertEqual(digest(binary.read_bytes()),'b66c3eac86ed189277cb456c5d3b866ed494eed346e99449504cb2c7aa8b1f71')
        self.assertEqual((ROOT/'release/runtime/serving/ds41/engram_io/integration.py').read_bytes(),
                         (ROOT/'release/experimental/engram_io/integration.py').read_bytes())
        self.assertEqual((ROOT/'release/runtime/serving/ds41/engram_io/overlap.py').read_bytes(),
                         (ROOT/'release/experimental/engram_io/overlap.py').read_bytes())


class RankPreparation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.cache=Path(self.tmp.name)
        self.manifest=dict(format='ds41_engram_page15_release_v1',repo_id='example/fixture',
            layout='page15',reader_abi=2,tensor_parallel_size=2,row_bytes=264,
            page_bytes=4096,rows_per_page=15,lossless=True,files={})
        self.payloads={};self.requests=[]
        for rank in (0,1):
            for layer in (1,14):
                name=f'engram-page15-v1/engram-layer-{layer:02}-rank{rank}-page15.bin'
                lo,hi=rank*15,(rank+1)*15
                raw=struct.pack('<8Q',assets.MAGIC,layer,lo,hi,30,264,15,4096).ljust(4096,b'\0')
                raw+=bytes([rank+layer])*4096
                row=dict(layer=layer,rank=rank,lo=lo,hi=hi,total_rows=30,bytes=len(raw),sha256=digest(raw),parts=[])
                for i,offset in enumerate((0,4096)):
                    part=name+f'.part-{i:05d}'
                    data=raw[offset:offset+4096];self.payloads[part]=data
                    row['parts'].append(dict(path=part,offset=offset,bytes=len(data),sha256=digest(data)))
                self.manifest['files'][name]=row
        self.raw=assets.encoded(self.manifest)
        self.digest=digest(self.raw)
        self.spec=dict(repo='example/fixture',revision='a'*40,manifest_path='engram-page15-v1/manifest.json',manifest_sha256=self.digest)
        self.pin=patch.object(assets,'MANIFEST_SHA',self.digest);self.pin.start()
        self.original_download=assets.download_table
    def tearDown(self):self.pin.stop();self.tmp.cleanup()
    def opener(self,request,timeout):
        name=request.full_url.split('/'+'a'*40+'/',1)[1]
        self.requests.append(name)
        return Response(self.payloads[name])
    def metadata(self,url,path,expected,size=None):
        self.assertEqual(expected,self.digest);path.write_bytes(self.raw)
    def prepare(self,rank):
        def download(base,row,path):return self.original_download(base,row,path,self.opener)
        with patch.object(assets,'download_table',side_effect=download):
            return assets.prepare(self.spec,self.cache,rank,self.metadata)
    def test_each_rank_only_downloads_own_tables(self):
        for rank in (0,1):
            self.requests=[];ref=self.prepare(rank)
            self.assertEqual(len(self.requests),4)
            self.assertTrue(all(f'rank{rank}' in name for name in self.requests))
            self.assertEqual(assets.validate(ref,rank)['files'],2)
            self.assertEqual(len(list(Path(ref['root']).iterdir())),4)
    def test_verified_reuse_does_not_read_payloads_or_network(self):
        ref=self.prepare(0)
        with patch.object(assets,'download_table',side_effect=AssertionError('No payload pass')):
            with patch.object(self,'metadata',side_effect=AssertionError('No network')):
                self.assertEqual(assets.prepare(self.spec,self.cache,0,self.metadata),ref)
    def test_one_table_failure_never_publishes_receipt_or_pointer(self):
        pointer=self.cache/'prepared.json';pointer.write_text('{"old":true}')
        actual=self.opener
        def fail_second(request,timeout):
            if 'layer-14' in request.full_url:raise TimeoutError('simulated interruption')
            return actual(request,timeout)
        with patch.object(self,'opener',side_effect=fail_second):
            with self.assertRaises(TimeoutError):self.prepare(0)
        rank_root=self.cache/'engram-assets'/self.digest/'rank0'
        self.assertFalse((rank_root/'verified.json').exists())
        self.assertEqual(json.loads(pointer.read_text()),{'old':True})
        self.requests=[];ref=self.prepare(0)
        self.assertTrue(all('layer-14' in name for name in self.requests))
        self.assertEqual(assets.validate(ref,0)['files'],2)
    def test_changed_file_refused_on_reuse(self):
        ref=self.prepare(0);path=next(Path(ref['root']).glob('*.bin'))
        with path.open('r+b') as out:out.seek(5000);out.write(b'changed')
        with self.assertRaisesRegex(ValueError,'changed after'):assets.validate(ref,0)
    def test_wrong_manifest_and_rank_receipt_refused(self):
        ref=self.prepare(1);root=Path(ref['root'])
        with self.assertRaisesRegex(ValueError,'rank'):assets.validate(ref,0)
        receipt=json.loads((root/'verified.json').read_bytes());receipt['rank']=0
        (root/'verified.json').write_bytes(assets.encoded(receipt))
        with self.assertRaisesRegex(ValueError,'changed after'):assets.validate(ref,1)
        (root/'manifest.json').write_bytes(self.raw+b'\n')
        with self.assertRaisesRegex(ValueError,'inventory differs'):assets.validate(ref,1)
    def test_pinned_inventory_still_requires_complete_ordered_parts(self):
        for alteration in ('missing','offset','name','oversized'):
            manifest=copy.deepcopy(self.manifest)
            row=next(iter(manifest['files'].values()))
            if alteration=='missing':row['parts'].pop()
            elif alteration=='offset':row['parts'][1]['offset']+=1
            elif alteration=='name':row['parts'][0]['path']='../escape.bin'
            else:row['parts'][0]['bytes']=8*2**30+1
            raw=assets.encoded(manifest)
            with patch.object(assets,'MANIFEST_SHA',digest(raw)):
                with self.assertRaises(ValueError):assets.parse_manifest(raw)
    def test_publication_and_runtime_use_same_inventory_parser(self):
        self.assertEqual(assets.parse_manifest(self.raw),self.manifest)
        with patch.object(assets,'MANIFEST_SHA',digest(self.raw+b' '*65536)):
            with self.assertRaisesRegex(ValueError,'Oversized'):
                assets.parse_manifest(self.raw+b' '*65536)


class HeaderStandIns(unittest.TestCase):
    """Sparse stand-ins replace the original tables; the pins match the canonical manifest."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.cache=Path(self.tmp.name).resolve()
        self.model=self.cache/'model'; (self.model/'engrams').mkdir(parents=True)
        for name in assets.SOURCE_METADATA: (self.model/'engrams'/name).write_text('{"meta":"'+name+'"}')
        self.headers={name:b'H'*16+name.encode() for name in assets.SOURCE_TABLES}
        self.tables={name:dict(bytes=1<<30,header_bytes=len(raw),header_sha256=digest(raw))
                     for name,raw in self.headers.items()}
        self.spec=dict(repo='owner/repo',revision='a'*40,manifest_sha256=assets.MODEL_MANIFEST_SHA)
        self.requests=[]
    def tearDown(self): self.tmp.cleanup()
    def opener(self, request, timeout):
        self.requests.append(request)
        name=request.full_url.rsplit('/',1)[1]; raw=self.headers[name]
        self.assertEqual(request.headers['Range'],f'bytes=0-{len(raw)-1}')
        return Response(raw,206,{'Content-Range':f'bytes 0-{len(raw)-1}/{1<<30}'})
    def prepare(self, opener=None):
        with patch.object(assets,'SOURCE_TABLES',self.tables):
            return assets.prepare_source(self.spec,self.model,self.cache,opener or self.opener)
    def test_stand_ins_are_sparse_with_exact_header_and_size(self):
        root=Path(self.prepare())
        self.assertEqual(len(self.requests),2)
        for name,raw in self.headers.items():
            info=(root/name).stat()
            self.assertEqual(info.st_size,1<<30)
            self.assertLess(info.st_blocks*512,1<<20)
            self.assertEqual((root/name).open('rb').read(len(raw)),raw)
        for name in assets.SOURCE_METADATA:
            self.assertEqual((root/name).read_bytes(),(self.model/'engrams'/name).read_bytes())
    def test_verified_reuse_needs_no_network(self):
        first=self.prepare()
        self.assertEqual(self.prepare(lambda *a,**k:self.fail('network')),first)
    def test_wrong_header_or_range_refused_before_receipt(self):
        def wrong(request,timeout):
            return Response(b'X'*len(self.headers[request.full_url.rsplit('/',1)[1]]),206,
                            {'Content-Range':f'bytes 0-{len(self.headers[request.full_url.rsplit("/",1)[1]])-1}/{1<<30}'})
        with self.assertRaisesRegex(ValueError,'differs from the pin'):self.prepare(wrong)
        with self.assertRaisesRegex(ValueError,'range'):self.prepare(lambda request,timeout:Response(b'',200))
        self.assertFalse((self.cache/'engram-source'/assets.MODEL_MANIFEST_SHA/'verified.json').exists())
    def test_changed_or_extra_stand_in_refused(self):
        root=Path(self.prepare()); name=next(iter(self.tables))
        with patch.object(assets,'SOURCE_TABLES',self.tables):
            (root/'extra').write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'inventory'):assets.validate_source(root)
            (root/'extra').unlink()
            with (root/name).open('r+b') as out:out.write(b'Z')
            with self.assertRaisesRegex(ValueError,'differs'):assets.validate_source(root)
    def test_other_model_manifest_refused(self):
        self.spec['manifest_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'another model manifest'):self.prepare()
    def test_reference_mounts_stand_ins_over_original_table_directory(self):
        root=self.cache/'packed'
        ref=dict(assets.reference(root,0),source=self.prepare())
        self.assertEqual(assets.check_reference(ref,0),root)
        mounts=assets.mounts(ref,0)
        self.assertEqual(mounts[-1],(ref['source'],'/model/engrams',True))
        self.assertEqual(len(assets.mounts(assets.reference(root,0),0)),2)
        for bad in ('relative/path','/a/../b/c','/a,b/c'):
            with self.assertRaises(ValueError):assets.check_reference(dict(ref,source=bad),0)
    def test_pins_match_canonical_model_manifest(self):
        lock=json.loads((ROOT/'recipe-lock.json').read_bytes())
        manifest=json.loads((ROOT/'release/model-release-manifest.json').read_bytes())
        self.assertEqual(lock['model']['manifest_sha256'],assets.MODEL_MANIFEST_SHA)
        self.assertEqual({n for n in manifest['files'] if n.startswith('engrams/engram-layer-')},set(assets.SOURCE_NAMES))
        for name,row in assets.SOURCE_TABLES.items():
            self.assertEqual(manifest['files']['engrams/'+name]['bytes'],row['bytes'])
            self.assertEqual(row['header_bytes'],264)
        for name in assets.SOURCE_METADATA:
            self.assertIn('engrams/'+name,manifest['files'])


if __name__=='__main__': unittest.main()
