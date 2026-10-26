import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import IndicConformerONNXASR


class FakeStream:
    def __init__(self, recorder):
        self._recorder = recorder
        self.result = type("Result", (), {"text": " नमस्कार "})()

    def accept_waveform(self, sr, samples):
        self._recorder["sr"] = sr
        self._recorder["samples"] = samples


class FakeRecognizer:
    def __init__(self):
        self.recorded = {}

    def create_stream(self):
        return FakeStream(self.recorded)

    def decode_stream(self, stream):
        pass


class TestIndicConformerTranscribe(unittest.TestCase):
    def test_indicconformer_transcribe_conditions_audio(self):
        engine = IndicConformerONNXASR()
        fake = FakeRecognizer()
        engine._recognizer = fake  # skip load(): no model download
        text = engine.transcribe(np.full(16000, 0.05, dtype=np.float32))
        self.assertEqual(len(fake.recorded["samples"]), 22400)
        self.assertEqual(fake.recorded["sr"], 16000)
        self.assertEqual(text, "नमस्कार")


if __name__ == "__main__":
    unittest.main()
