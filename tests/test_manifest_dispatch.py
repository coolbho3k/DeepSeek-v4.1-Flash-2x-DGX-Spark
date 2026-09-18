# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_release import node_module, ROOT


class ManifestDispatch(unittest.TestCase):
    def test_uploaded_manifest_uses_strict_public_verifier(self):
        node=node_module()
        digest=json.loads((ROOT/'recipe-lock.json').read_bytes())['model']['manifest_sha256']
        config={'nodes':[{'kit':str(ROOT/'release/runtime'),'model':'/srv/example/model',
                         'model_receipt':'/srv/example/receipt.json'}], 'model_manifest_sha256':digest}
        fake=SimpleNamespace(load_manifest=lambda *_: ({},{}),small_json=lambda *_: ({},b''),
                             check_receipt=lambda *_: {'verified':True})
        with patch.object(node,'module',return_value=fake) as imported:
            self.assertEqual(node.check_model_receipt(config,0,{}),{'verified':True})
        self.assertEqual(imported.call_args.args[2],'tools/verify_public_download.py')


if __name__=='__main__':unittest.main()
