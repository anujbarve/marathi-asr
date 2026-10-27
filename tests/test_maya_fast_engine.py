from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest
import numpy as np

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.engine import MayaFast
from src.maya_fast.control import split_text


class TestMayaFastEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assets_dir = make_tiny_assets(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthesize_returns_audio(self):
        # test_synthesize_returns_audio (length = frames * 1920)
        engine = MayaFast(assets_dir=self.assets_dir, threads=1)
        wav, metrics = engine.synthesize("t_1 t_2", voice="Charvi", seed=42)
        self.assertEqual(len(wav), metrics["frames"] * 1920)
        self.assertEqual(wav.dtype, np.float32)
        self.assertIn("audio_s", metrics)
        self.assertIn("wall_s", metrics)
        self.assertIn("rtf", metrics)
        self.assertIn("reasons", metrics)

    def test_same_seed_same_audio(self):
        engine = MayaFast(assets_dir=self.assets_dir, threads=1)
        wav1, _ = engine.synthesize("t_1 t_2", voice="Charvi", seed=123)
        wav2, _ = engine.synthesize("t_1 t_2", voice="Charvi", seed=123)
        np.testing.assert_array_equal(wav1, wav2)

    def test_empty_text_raises(self):
        engine = MayaFast(assets_dir=self.assets_dir, threads=1)
        with self.assertRaises(ValueError):
            engine.synthesize("", voice="Charvi")
        with self.assertRaises(ValueError):
            engine.synthesize("।,.", voice="Charvi")

    def test_cap_stops_generation(self):
        # tiny assets built with an aeos logit bias of -100 -> reasons == ["cap"], frames == cap
        with tempfile.TemporaryDirectory() as td:
            biased_assets_dir = make_tiny_assets(Path(td), aeos_bias=-100.0)
            engine = MayaFast(assets_dir=biased_assets_dir, threads=1)
            wav, metrics = engine.synthesize("t_1", voice="Charvi", seed=0)
            self.assertEqual(metrics["reasons"], ["cap"])

    def test_long_text_is_split_and_kv_bounded(self):
        # 3-sentence text -> len(reasons) == len(split_text(...)), and Backbone.past_len never exceeds the largest piece's prompt + cap
        text = "t_1 t_2. t_1 t_3. t_2 t_3."
        engine = MayaFast(assets_dir=self.assets_dir, threads=1)
        prior = engine.controls.get("text_frames_prior", {})
        expected_chunks = split_text(text, "mr", prior, chunk_s=0.5)

        wav, metrics = engine.synthesize(text, voice="Charvi", seed=0, chunk_s=0.5)
        self.assertEqual(len(metrics["reasons"]), len(expected_chunks))
        self.assertLessEqual(metrics["max_past_len"], metrics["max_chunk_bound"])

    def test_thread_count_does_not_change_output(self):
        # threads 1 and 16 -> identical wav
        engine1 = MayaFast(assets_dir=self.assets_dir, threads=1)
        engine2 = MayaFast(assets_dir=self.assets_dir, threads=16)
        wav1, _ = engine1.synthesize("t_1 t_2", voice="Charvi", seed=7)
        wav2, _ = engine2.synthesize("t_1 t_2", voice="Charvi", seed=7)
        np.testing.assert_array_equal(wav1, wav2)

    def test_unknown_voice_lists_voices(self):
        engine = MayaFast(assets_dir=self.assets_dir, threads=1)
        with self.assertRaises(KeyError) as ctx:
            engine.synthesize("t_1", voice="UnknownVoice")
        self.assertIn("Charvi", str(ctx.exception))

    def test_runtime_has_no_torch(self):
        # ast scan of every file under src/maya_fast/: no torch import
        pkg_dir = Path(__file__).resolve().parent.parent / "src" / "maya_fast"
        for py_file in pkg_dir.glob("**/*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotEqual(alias.name.split(".")[0], "torch", f"torch imported in {py_file}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.assertNotEqual(node.module.split(".")[0], "torch", f"torch imported in {py_file}")


if __name__ == "__main__":
    unittest.main()
