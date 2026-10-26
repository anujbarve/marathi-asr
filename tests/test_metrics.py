import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import score_transcript


class TestScoreTranscript(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(score_transcript("जे कोणी उंच भागात", "जे कोणी उंच भागात")["wer"], 0.0)

    def test_lenient_forgives_vowel_length(self):
        s = score_transcript("पोलिस अधीक्षक", "पोलीस अधीक्षक")
        self.assertEqual(s["wer"], 0.5)
        self.assertEqual(s["wer_lenient"], 0.0)

    def test_empty_prediction_scores_one(self):
        s = score_transcript("जे कोणी उंच भागात", "")
        self.assertEqual((s["wer"], s["cer"], s["wer_lenient"], s["cer_lenient"]), (1.0, 1.0, 1.0, 1.0))

    def test_empty_reference_scores_zero(self):
        self.assertEqual(score_transcript("।", "काहीतरी")["wer"], 0.0)


if __name__ == "__main__":
    unittest.main()
