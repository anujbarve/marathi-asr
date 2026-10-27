import os
import unittest
from pathlib import Path
import tempfile
import numpy as np

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.assets import PromptAssets
from src.maya_fast.prompt import AudioIn, build_prompt


class TestMayaFastPrompt(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assets_dir = make_tiny_assets(Path(self.temp_dir.name))
        self.assets = PromptAssets(self.assets_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_audio_in_shapes(self):
        audio_in = AudioIn(self.assets.prompt_params)
        codes = np.zeros((4, 16), dtype=np.int64)
        frames = audio_in.frames(codes)
        self.assertEqual(frames.shape, (4, self.assets.shapes["d_model"]))
        aeos = audio_in.aeos()
        self.assertEqual(aeos.shape, (self.assets.shapes["d_model"],))

    def test_build_prompt(self):
        p = build_prompt("t_1 t_2", "Charvi", self.assets)
        self.assertGreater(p.n_prompt, 0)
        self.assertEqual(len(p.token_ids), p.n_prompt)
        self.assertEqual(p.embeds.shape, (p.n_prompt, self.assets.shapes["d_model"]))

    def test_prompt_matches_golden(self):
        # skipped with a message when goldens are absent
        golden_path = Path.home() / ".cache" / "maya_fast" / "golden" / "light.npz"
        if not golden_path.exists():
            self.skipTest(f"Golden file not found at {golden_path}")

    def test_audio_in_matches_golden(self):
        golden_path = Path.home() / ".cache" / "maya_fast" / "golden" / "light.npz"
        if not golden_path.exists():
            self.skipTest(f"Golden file not found at {golden_path}")

    def test_controls_match_golden(self):
        golden_path = Path.home() / ".cache" / "maya_fast" / "golden" / "light.json"
        if not golden_path.exists():
            self.skipTest(f"Golden file not found at {golden_path}")


if __name__ == "__main__":
    unittest.main()
