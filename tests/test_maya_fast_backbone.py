import unittest
from pathlib import Path
import tempfile
import numpy as np

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.backbone import Backbone


class TestMayaFastBackbone(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assets_dir = make_tiny_assets(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_backbone_step_kv_grows(self):
        # runtime: tiny synthetic 2-layer graph built with onnx.helper in tests/fixtures/tiny_assets.py;
        # past_len goes 5 -> 6 -> 7 and reset() returns it to 0
        backbone_path = self.assets_dir / "backbone_wo8.onnx"
        bb = Backbone(backbone_path, threads=1, manifest_path=self.assets_dir / "manifest.json")
        self.assertEqual(bb.past_len, 0)

        # Prefill 5 tokens
        # d_model = 16 for tiny
        d = 16
        prompt_embeds = np.random.randn(5, d).astype(np.float32)
        h, cb0 = bb.step(prompt_embeds)
        self.assertEqual(bb.past_len, 5)
        self.assertEqual(h.shape, (d,))
        self.assertEqual(cb0.shape, (65,))  # vocab + 1

        # Step 6
        tok1 = np.random.randn(1, d).astype(np.float32)
        h, cb0 = bb.step(tok1)
        self.assertEqual(bb.past_len, 6)

        # Step 7
        tok2 = np.random.randn(1, d).astype(np.float32)
        h, cb0 = bb.step(tok2)
        self.assertEqual(bb.past_len, 7)

        # Reset
        bb.reset()
        self.assertEqual(bb.past_len, 0)


if __name__ == "__main__":
    unittest.main()
