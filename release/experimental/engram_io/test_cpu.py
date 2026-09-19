# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import packing
from native import Reader

LIB=Path(os.environ.get('DS41_ENGRAM_TEST_LIBRARY','/results/librow_store.so'))


def fixture(path,rows=1009):
    rng=np.random.default_rng(413)
    w=rng.integers(0,256,(rows,256),dtype=np.uint8)
    s=rng.integers(0,256,(rows,8),dtype=np.uint8)
    head={'layers.1.engram.embed.scale':dict(dtype='F8_E8M0',shape=[rows,8],data_offsets=[0,rows*8]),
          'layers.1.engram.embed.weight':dict(dtype='F8_E4M3',shape=[rows,256],data_offsets=[rows*8,rows*264])}
    raw=json.dumps(head,separators=(',',':')).encode()
    raw=raw.ljust((len(raw)+7)//8*8,b' ')
    with path.open('xb') as f:f.write(struct.pack('<Q',len(raw)));f.write(raw);f.write(s.tobytes());f.write(w.tobytes())
    return w,s


class PackedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='engram-fixture-',dir='/results')
        self.root=Path(self.tmp.name)
        self.source=self.root/'source.safetensors'
        self.weights,self.scales=fixture(self.source)
        self.info=packing.table(self.source,1)
        self.lo,self.hi=7,1002
    def tearDown(self):self.tmp.cleanup()

    def packed(self,layout):
        path=self.root/(layout+'.bin')
        receipt=packing.pack(self.source,path,1,self.lo,self.hi,layout,chunk_rows=120)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),receipt['packed_sha256'])
        self.assertEqual(hashlib.sha256(self.weights[self.lo:self.hi].tobytes()).hexdigest(),receipt['owned_weight_sha256'])
        self.assertEqual(hashlib.sha256(self.scales[self.lo:self.hi].tobytes()).hexdigest(),receipt['owned_scale_sha256'])
        self.assertTrue(receipt['complete_byte_readback'])
        self.assertEqual(path.stat().st_size%4096,0)
        return path

    def test_every_row_original_dense_page15(self):
        for layout in ('original','dense','page15'):
            path=None if layout=='original' else self.packed(layout)
            r=Reader(LIB,self.info,self.lo,self.hi,path,budget=0)
            try:
                ids=np.arange(-1,self.info['rows'])
                gotw,gots=r.lookup(ids)
                wantw=np.zeros_like(gotw);wants=np.zeros_like(gots)
                mask=(ids>=self.lo)&(ids<self.hi)
                wantw[mask]=self.weights[ids[mask]];wants[mask]=self.scales[ids[mask]]
                np.testing.assert_array_equal(gotw,wantw);np.testing.assert_array_equal(gots,wants)
                stats=r.stats();owned=self.hi-self.lo
                self.assertEqual(stats['misses'],owned)
                self.assertEqual(stats['reads'],owned*(2 if layout=='original' else 1))
                self.assertEqual(stats['layout'],('original','dense','page15').index(layout))
                if layout=='page15':self.assertEqual(stats['requested_io_bytes'],owned*4096)
                self.assertEqual(stats['scale_bytes'],0)
            finally:r.close()

    def test_cache_duplicates_dead_ids(self):
        path=self.packed('page15')
        r=Reader(LIB,self.info,self.lo,self.hi,path,budget=2**20)
        try:
            ids=np.array([7,8,9,18,21,22,400,700,-1,0,1008]*12)
            a=r.lookup(ids);before=r.stats();b=r.lookup(ids);after=r.stats()
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            self.assertEqual(before['misses'],after['misses'])
            self.assertEqual(before['reads'],after['reads'])
            self.assertLessEqual(after['cache_bytes'],2**20)
            r.clear();self.assertEqual(r.stats()['reads'],0)
            c=r.lookup(ids)
            for x,y in zip(a,c):np.testing.assert_array_equal(x,y)
        finally:r.close()

    def test_partial_page_and_chunk_tail(self):
        for count in (1,14,15,16,119,120,121):
            out=self.root/f'tail-{count}.bin'
            packing.pack(self.source,out,1,self.lo,self.lo+count,'page15',chunk_rows=120)
            data=out.read_bytes()
            self.assertEqual(len(data),4096+((count+14)//15)*4096)
            for i in range(count):
                pos=packing.packed_offset(self.lo+i,self.lo,'page15')
                self.assertEqual(data[pos:pos+256],self.weights[self.lo+i].tobytes())
                self.assertEqual(data[pos+256:pos+264],self.scales[self.lo+i].tobytes())
            for p in range(4096,len(data),4096):self.assertEqual(data[p+3960:p+4096],bytes(136))

    def test_invalid_header_does_not_silently_fallback(self):
        path=self.packed('page15')
        with path.open('r+b') as f:f.write(bytes(8))
        with self.assertRaisesRegex(ValueError,'did not attach'):
            Reader(LIB,self.info,self.lo,self.hi,path)

    def test_wrong_partition_refused(self):
        path=self.packed('dense')
        with self.assertRaisesRegex(ValueError,'did not attach'):
            Reader(LIB,self.info,self.lo+1,self.hi,path)

    def test_no_overwrite(self):
        path=self.packed('page15');before=path.read_bytes()
        with self.assertRaises(ValueError):packing.pack(self.source,path,1,self.lo,self.hi,'dense')
        self.assertEqual(before,path.read_bytes())

    def test_invalid_ranges(self):
        for lo,hi in ((-1,1),(1,1),(2,1),(0,1010)):
            with self.assertRaises(ValueError):
                packing.pack(self.source,self.root/'invalid',1,lo,hi,'page15')

    def test_page_geometry_never_crosses(self):
        for i in range(10000):self.assertLessEqual(packing.packed_offset(i,0,'page15')%4096+264,4096)

    def test_source_symlink_rejected(self):
        link=self.root/'redirect';link.symlink_to(self.source)
        with self.assertRaises(ValueError):packing.table(link,1)

    def test_short_source_header_rejected(self):
        path=self.root/'empty-model-view-placeholder'
        path.touch()
        with self.assertRaisesRegex(ValueError,'Truncated'):
            packing.table(path,1)

    def test_wrong_page_geometry_rejected(self):
        for field,value in ((48,14),(56,8192),(40,263)):
            path=self.root/f'bad-field-{field}.bin'
            packing.pack(self.source,path,1,self.lo,self.hi,'page15',chunk_rows=120)
            with path.open('r+b') as f:f.seek(field);f.write(struct.pack('<Q',value))
            self.assert_native_rejected(path, 'did not attach' if field==40 else
                                        'invalid page-contained packed geometry')

    def assert_native_rejected(self,path,message):
        # Native malformed-storage failures deliberately terminate the worker;
        # test that contract in a child, not inside the unittest runner.
        code='import json,sys; from native import Reader; Reader(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])'
        result=subprocess.run([sys.executable,'-B','-c',code,str(LIB),json.dumps(self.info),
            str(self.lo),str(self.hi),str(path)],capture_output=True,text=True,timeout=20)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(message,result.stderr)

    def test_truncated_packed_file_rejected(self):
        for layout in ('dense','page15'):
            path=self.packed(layout)
            with path.open('r+b') as f:f.truncate(path.stat().st_size-4096)
            self.assert_native_rejected(path,'packed Engram shard is truncated')


if __name__=='__main__':unittest.main()
