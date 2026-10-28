import json
import unittest
from pathlib import Path
import numpy as np

from src.maya_fast.assets import PromptAssets, default_assets_dir
from src.maya_fast.control import (
    expected_frames,
    frame_cap,
    min_frames,
    split_text,
    StopPolicy,
    sample_cb0,
)


class TestMayaFastControl(unittest.TestCase):
    def test_controls_basic(self):
        prior = {"mr": {"a": 4.0, "b": 0.5, "resid_p99": 10.0}}
        ef, er = expected_frames(prior, "mr", 20)
        self.assertEqual(ef, 14.0)
        self.assertEqual(er, 10.0)

        controls = {
            "guard": {
                "enabled": True,
                "dur_k": 2.2,
                "dur_margin_s": 1.2,
                "hard_cap": 1500,
                "dur_prior_coefs": {"mr": [1.0, 0.05, 0.03, 0.3, 0.3]},
                "dur_prior_default": [1.0, 0.05, 0.03, 0.3, 0.3],
            }
        }
        cap = frame_cap("नमस्कार", "mr", ef, er, controls)
        self.assertGreater(cap, 0)
        self.assertLessEqual(cap, 1500)

        mf = min_frames(ef)
        self.assertGreaterEqual(mf, 4)

        # test split_text
        text = "हा पहिला भाग आहे. हा दुसरा भाग आहे."
        chunks = split_text(text, "mr", prior, chunk_s=1.0)
        self.assertGreater(len(chunks), 1)

    def test_stop_policy_reasons(self):
        # test_stop_policy_reasons: hand-built p vectors trigger each reason at the right t; nothing fires before min_frames
        policy = StopPolicy(min_frames=5, tau=0.35, hazard=0.6)
        aeos_class = 2048
        p = np.zeros(2049, dtype=np.float32)

        # Before min_frames, none should fire even if aeos is 1.0
        p[aeos_class] = 1.0
        for t in range(5):
            self.assertIsNone(policy.check(t, p))

        # At t = 5, argmax aeos triggers aeos_argmax
        self.assertEqual(policy.check(5, p), "aeos_argmax")

        # Test tau trigger when argmax is another token
        policy = StopPolicy(min_frames=5, tau=0.35, hazard=0.6)
        p = np.zeros(2049, dtype=np.float32)
        p[0] = 0.5
        p[aeos_class] = 0.4  # >= 0.35 but argmax is 0
        self.assertEqual(policy.check(5, p), "aeos_tau")

        # Test hazard trigger: accumulated hazard 1 - prod(1 - p_aeos) >= 0.6
        # If p_aeos = 0.2 at each step after min_frames:
        # t=5: 1 - 0.8 = 0.2
        # t=6: 1 - 0.8^2 = 0.36
        # t=7: 1 - 0.8^3 = 0.488
        # t=8: 1 - 0.8^4 = 0.5904
        # t=9: 1 - 0.8^5 = 0.67232 >= 0.6 -> aeos_hazard!
        policy = StopPolicy(min_frames=5, tau=0.35, hazard=0.6)
        p = np.zeros(2049, dtype=np.float32)
        p[0] = 0.7
        p[aeos_class] = 0.2
        for t in range(5, 9):
            self.assertIsNone(policy.check(t, p))
        self.assertEqual(policy.check(9, p), "aeos_hazard")

    def test_sample_cb0(self):
        # aeos zeroed, Gumbel-max over the rest
        aeos_class = 64
        p = np.zeros(65, dtype=np.float32)
        p[aeos_class] = 0.99
        p[10] = 0.01

        noise = np.zeros(65, dtype=np.float32)
        # Even though aeos had 0.99, sample_cb0 masks aeos out, so token 10 must be chosen
        chosen = sample_cb0(p, noise, aeos_class=aeos_class)
        self.assertEqual(chosen, 10)


GOLDEN = Path.home() / ".cache" / "maya_fast" / "golden"


class TestMayaFastControlGolden(unittest.TestCase):
    """Sentence controls (ef, er, cap, minimum, split_text) must equal the upstream values in light.json."""

    @classmethod
    def setUpClass(cls):
        assets_dir = default_assets_dir()
        if not (GOLDEN / "light.json").exists() or not (assets_dir / "manifest.json").exists():
            raise unittest.SkipTest(f"goldens at {GOLDEN} or assets at {assets_dir} not found")
        with open(GOLDEN / "light.json", "r", encoding="utf-8") as f:
            cls.gj = json.load(f)
        cls.controls = PromptAssets(assets_dir).controls

    def test_controls_match_golden(self):
        prior = self.controls["text_frames_prior"]
        for s_id, item in self.gj["sentences"].items():
            text = item["text"]
            ef, er = expected_frames(prior, "mr", len(text))
            self.assertAlmostEqual(ef, item["ef"], places=9, msg=s_id)
            self.assertAlmostEqual(er, item["er"], places=9, msg=s_id)
            self.assertEqual(frame_cap(text, "mr", ef, er, self.controls), item["cap"], s_id)
            self.assertEqual(min_frames(ef), item["minimum"], s_id)
            self.assertEqual(split_text(text, "mr", prior, chunk_s=16.0), item["split_text"], s_id)


if __name__ == "__main__":
    unittest.main()
