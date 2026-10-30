import unittest
from pathlib import Path
import tempfile
import numpy as np

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.assets import load_manifest, AssetError, PromptAssets
from src.maya_fast.embed import EmbeddingTable


class TestMayaFastAssets(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.assets_dir = make_tiny_assets(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_manifest_sha_mismatch_raises(self):
        # corrupt a file
        corrupt_target = self.assets_dir / "embed_bf16.bin"
        with open(corrupt_target, "ab") as f:
            f.write(b"\x00\x00\x00\x00")

        with self.assertRaises(AssetError) as ctx:
            load_manifest(self.assets_dir, verify=True)
        self.assertIn("embed_bf16.bin", str(ctx.exception))

    def test_manifest_missing_file_raises(self):
        (self.assets_dir / "prompt.npz").unlink()
        with self.assertRaises(AssetError) as ctx:
            load_manifest(self.assets_dir, verify=True)
        self.assertIn("prompt.npz", str(ctx.exception))

    def test_bf16_lookup_exact(self):
        # write 3 known float32 values that are exact in bf16, look up -> equal
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            p = Path(td) / "test_embed.bin"
            # 1.0, -2.5, 0.125 are exactly representable in bf16
            vals = np.array([[1.0, -2.5, 0.125], [0.0, 4.0, -0.5]], dtype=np.float32)
            u32 = vals.view(np.uint32)
            bf16 = (u32 >> 16).astype(np.uint16)
            p.write_bytes(bf16.tobytes())

            table = EmbeddingTable(p, rows=2, dim=3)
            res = table.lookup(np.array([0, 1]))
            np.testing.assert_array_equal(res, vals)


if __name__ == "__main__":
    unittest.main()
