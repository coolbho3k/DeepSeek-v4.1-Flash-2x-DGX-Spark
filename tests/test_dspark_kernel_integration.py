# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from test_dspark_contracts import module

integration=module('kernel_integration')


class DraftDispatch(unittest.TestCase):
    def test_select_only_profitable_draft_envelope(self):
        for rows in range(1,38):
            for routes in (1,3,6):
                self.assertEqual(integration.selected_draft_shape((rows,5120),(rows,routes)),
                    5<=rows<=30 and routes==3)
        for x,ids in (((5,4096),(5,3)),((5,5120),(6,3)),((5,5120,1),(5,3))):
            self.assertFalse(integration.selected_draft_shape(x,ids))


if __name__=='__main__':unittest.main()
