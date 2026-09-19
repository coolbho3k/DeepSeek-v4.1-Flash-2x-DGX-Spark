# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from test_dspark_contracts import module

draft=module('draft_top3')


class Top3(unittest.TestCase):
    def test_admits_only_bounded_draft_shapes(self):
        for rows in range(1,31):self.assertTrue(draft.eligible((rows,5120),(rows,3),128))
        for x,ids,experts in (((31,5120),(31,3),128),((3,5120),(3,6),128),
                              ((3,5120),(3,3),384),((3,4096),(3,3),128)):
            self.assertFalse(draft.eligible(x,ids,experts))
    def test_metadata_never_overlaps_live_activation_storage(self):
        for rows in range(1,31):
            start,end=draft.metadata_interval(rows)
            self.assertGreaterEqual(start,rows*3*5120*2)
            self.assertEqual(end-start,rows*3*8)
            self.assertLessEqual(end,6*128*5120*2)
        for rows in (0,31,True):
            with self.assertRaises(ValueError):draft.metadata_interval(rows)


if __name__=='__main__':unittest.main()
