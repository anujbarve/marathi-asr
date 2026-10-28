import json
import unittest
from pathlib import Path
import tempfile
import numpy as np

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.assets import PromptAssets, default_assets_dir
from src.maya_fast.prompt import AudioIn, build_prompt

GOLDEN = Path.home() / ".cache" / "maya_fast" / "golden"
TOL = 1e-5


class TestMayaFastPromptTiny(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assets_dir = make_tiny_assets(Path(self.temp_dir.name))
        self.assets = PromptAssets(self.assets_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_audio_in_shapes(self):
        audio_in = self.assets.audio_in
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

    def test_unknown_voice_lists_voices(self):
        with self.assertRaisesRegex(KeyError, "Charvi"):
            build_prompt("t_1", "nobody", self.assets)


class TestMayaFastPromptGolden(unittest.TestCase):
    """Compare the numpy port against the upstream PyTorch goldens (tools/maya_export/golden_light.py)."""

    @classmethod
    def setUpClass(cls):
        assets_dir = default_assets_dir()
        if not (GOLDEN / "light.npz").exists() or not (assets_dir / "manifest.json").exists():
            raise unittest.SkipTest(f"goldens at {GOLDEN} or assets at {assets_dir} not found")
        cls.g = np.load(GOLDEN / "light.npz")
        with open(GOLDEN / "light.json", "r", encoding="utf-8") as f:
            cls.gj = json.load(f)
        cls.assets = PromptAssets(assets_dir)

    def test_prompt_matches_golden(self):
        for s_id, item in self.gj["sentences"].items():
            for voice in self.gj["voices"]:
                key = f"prompt/{s_id}/{voice}"
                n = int(self.g[f"{key}/n_prompt"])
                p = build_prompt(item["text"], voice, self.assets, lang="mr")
                self.assertEqual(p.n_prompt, n, key)
                np.testing.assert_array_equal(p.token_ids, self.g[f"{key}/token_ids"][:n], err_msg=key)
                err = float(np.max(np.abs(p.embeds - self.g[f"{key}/prompt_emb"])))
                self.assertLessEqual(err, TOL, key)

    def test_audio_in_matches_golden(self):
        audio_in = self.assets.audio_in
        frames = audio_in.frames(self.g["audio_in_test_codes"])
        self.assertLessEqual(float(np.max(np.abs(frames - self.g["audio_in_frames"]))), TOL)
        self.assertLessEqual(float(np.max(np.abs(audio_in.aeos() - self.g["aeos"]))), TOL)


if __name__ == "__main__":
    unittest.main()
