# SPDX-License-Identifier: AGPL-3.0-only
import ast
from pathlib import Path
import unittest
from release.experimental.moe_critical_path.geometry import transform

ROOT=Path(__file__).resolve().parents[1]


class GeometryExperiment(unittest.TestCase):
    def setUp(self):
        self.source=(ROOT/'release/runtime/serving/ds41/cooperative_moe.py').read_bytes()
    def test_default_is_byte_exact(self):
        self.assertEqual(transform(self.source,1),self.source)
    def test_prepare_and_launch_choose_same_geometry(self):
        for geometry in (0,1,2):
            code=transform(self.source,geometry).decode();ast.parse(code)
            self.assertIn(f'goal50_coop_info(3, {geometry}, info)',code)
            self.assertIn(f'len(bank.keys), 10., {geometry}, 0,',code)
            self.assertEqual(len(code.splitlines()),len(self.source.splitlines()))
    def test_unreviewed_parent_rejected(self):
        with self.assertRaises(ValueError):transform(self.source+b'\n',2)
    def test_no_arbitrary_geometry(self):
        for geometry in (-1,3,True,2.0,'2'):
            with self.assertRaises(ValueError):transform(self.source,geometry)


if __name__=='__main__':unittest.main()
