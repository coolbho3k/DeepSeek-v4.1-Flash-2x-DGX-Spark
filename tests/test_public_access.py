# SPDX-License-Identifier: AGPL-3.0-only
import io
from pathlib import Path
import sys
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'release'))
import registry


class PublicAccess(unittest.TestCase):
    def test_private_package_has_actionable_failure(self):
        for status in (401,403,404):
            error=urllib.error.HTTPError('https://ghcr.io/',status,'denied',{},io.BytesIO())
            with self.subTest(status=status),patch.object(registry,'public_manifest',side_effect=error):
                with self.assertRaisesRegex(ValueError,'make the package Public'):
                    registry.require_public({})

    def test_public_manifest_passes_through(self):
        with patch.object(registry,'public_manifest',return_value={'schemaVersion':2}):
            self.assertEqual(registry.require_public({}),{'schemaVersion':2})


if __name__=='__main__':unittest.main()
