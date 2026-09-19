# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from test_dspark_contracts import module

rebase=module('rebase_public')


class PublicRebase(unittest.TestCase):
    def test_serving_must_be_identical_despite_portable_launcher_changes(self):
        candidate={'serving/a.py':b'checked','tools/portable_node.py':b'private'}
        staged={'serving/a.py':b'checked','tools/portable_node.py':b'public'}
        self.assertEqual(rebase.require_serving_parity(staged,candidate),1)
        self.assertEqual(staged['tools/portable_node.py'],b'public')

    def test_reject_changed_missing_or_extra_serving_files(self):
        candidate={'serving/a.py':b'checked'}
        for staged in ({},{'serving/a.py':b'changed'},
                       {'serving/a.py':b'checked','serving/b.py':b'extra'}):
            with self.assertRaises(ValueError):rebase.require_serving_parity(staged,candidate)


if __name__=='__main__':unittest.main()
