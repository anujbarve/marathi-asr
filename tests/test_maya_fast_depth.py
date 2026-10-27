import unittest
from pathlib import Path
import tempfile
import numpy as np

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.depth import Depth


class TestMayaFastDepth(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assets_dir = make_tiny_assets(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_depth_runtime_shapes(self):
        # tiny assets: frame() returns 16 codes in [0, 2048), codes[0] == cb0
        depth_path = self.assets_dir / "depth_frame_wo8.onnx"
        depth = Depth(depth_path, threads=1)

        d = 16
        vocab = 64
        d_mem = 16

        h = np.random.randn(d).astype(np.float32)
        cb0 = 17
        mem = np.random.randn(1, 1, d_mem).astype(np.float32)
        noise = np.random.randn(15, vocab).astype(np.float32)

        codes = depth.frame(h, cb0, mem, noise)

        self.assertEqual(codes.shape, (16,))
        self.assertEqual(codes.dtype, np.int64)
        self.assertEqual(codes[0], cb0)
        self.assertTrue(np.all((codes >= 0) & (codes < 2048)))


if __name__ == "__main__":
    unittest.main()
