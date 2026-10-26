import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalizer import normalize_marathi_text
from src.audio_utils import load_and_resample
from src.models import get_model

class TestMarathiASR(unittest.TestCase):
    def test_devanagari_normalization(self):
        sample = "पोलीस अधीक्षक, 'चंद्रशेखर'! आणि सोळंकी।॥"
        normalized = normalize_marathi_text(sample)
        self.assertEqual(normalized, "पोलीस अधीक्षक चंद्रशेखर आणि सोळंकी")

    def test_audio_loader_and_resampler(self):
        # Test on existing benchmark sample
        audio_path = "benchmarks/audio/clean_sample_0.wav"
        if os.path.exists(audio_path):
            data, duration = load_and_resample(audio_path, target_sr=16000)
            self.assertEqual(data.dtype, np.float32)
            self.assertTrue(duration > 0)
            self.assertEqual(data.ndim, 1)

    def test_indicconformer_instantiation(self):
        engine = get_model("indicconformer-mr-int8")
        self.assertEqual(engine.model_id, "indicconformer-mr-int8")
        self.assertTrue(engine.disk_size_mb > 0)

if __name__ == "__main__":
    unittest.main()
