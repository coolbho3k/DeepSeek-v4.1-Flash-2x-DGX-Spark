import hashlib
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[1]
TRANSFORM=runpy.run_path(str(ROOT/'release/experimental/moe_critical_path/prefill_rows.py'))
SOURCE=ROOT/'release/runtime/vendor/miaai-grouped-prefill-ds41-v2/include/quant/exl3_fat_moe.cu'


class PrefillRows(unittest.TestCase):
    def test_exact_parent_and_reversible_guard(self):
        original=SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),TRANSFORM['PARENT_SHA'])
        candidate=TRANSFORM['transform'](original).decode()
        self.assertEqual(candidate.count('if (mb * 16 >= rows) break;'),1)
        self.assertEqual(candidate.replace(TRANSFORM['REPLACEMENT'],TRANSFORM['ANCHOR']).encode(),original)

    def test_changed_parent_rejected(self):
        with self.assertRaises(ValueError):TRANSFORM['transform'](SOURCE.read_bytes()+b'\n')

    def test_live_rows_covered_for_every_segment_size(self):
        for rows in range(1,65):
            blocks=[mb for mb in range(4) if mb*16<rows]
            self.assertEqual(blocks,list(range((rows+15)//16)))
            self.assertGreaterEqual(len(blocks)*16,rows)
            self.assertLess(len(blocks)*16-rows,16)


if __name__=='__main__':unittest.main()
