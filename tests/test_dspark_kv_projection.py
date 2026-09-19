# SPDX-License-Identifier: AGPL-3.0-only
from dataclasses import dataclass, replace
import unittest
from test_dspark_contracts import module

kv=module('kv_projection')


@dataclass
class Tensor:
    shape: tuple
    storage: int
    slices: tuple=()
    def narrow(self,dimension,start,length):
        shape=list(self.shape)
        if not 0<=start<start+length<=shape[dimension]:raise ValueError('slice')
        shape[dimension]=length
        return Tensor(tuple(shape),self.storage,self.slices+((dimension,start,length),))
    def untyped_storage(self):return self
    def data_ptr(self):return self.storage


@dataclass(frozen=True)
class Rows:
    values: Tensor
    scale_rows: Tensor
    scale_mma: Tensor
    values_tiled: object=None


@dataclass(frozen=True)
class Weight:
    weight: Rows
    in_features: int=5120
    padded_in_features: int=5120
    out_features: int=1792


class KVView(unittest.TestCase):
    def source(self):
        return Weight(Rows(Tensor((1792,5120),1),Tensor((1,1792,160),2),Tensor((32,4,14,4,40,1),3)))
    def test_view_selects_aligned_kv_rows_without_new_storage(self):
        parent=self.source();view=kv.slice_kv_weight(parent)
        self.assertEqual(view.out_features,512)
        for name,expected in (('values',(0,1280,512)),('scale_rows',(1,1280,512)),('scale_mma',(2,10,4))):
            old,new=getattr(parent.weight,name),getattr(view.weight,name)
            self.assertEqual(old.storage,new.storage)
            self.assertEqual(new.slices,(expected,))
            self.assertEqual(old.slices,())
    def test_unknown_layout_or_secondary_packed_copy_is_rejected(self):
        parent=self.source()
        for changed in (replace(parent,out_features=512),replace(parent,in_features=4096),
                        replace(parent,weight=replace(parent.weight,values_tiled=object()))):
            with self.assertRaises(ValueError):kv.slice_kv_weight(changed)


if __name__=='__main__':unittest.main()
