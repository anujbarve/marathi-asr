import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalizer import normalize_marathi_text

class TestNormalizer(unittest.TestCase):
    def test_strips_zero_width_joiners(self):
        self.assertEqual(normalize_marathi_text("क्‍ष ‌आणि"), "क्ष आणि")

    def test_folds_nukta(self):
        # U+095B (ज़) and decomposed ज + U+093C both become ज
        self.assertEqual(normalize_marathi_text("ज़ ज़"), "ज ज")

    def test_strict_keeps_vowel_length(self):
        self.assertNotEqual(normalize_marathi_text("पोलिस"), normalize_marathi_text("पोलीस"))

    def test_lenient_folds_vowel_length_and_chandrabindu(self):
        self.assertEqual(normalize_marathi_text("पोलिस", lenient=True), normalize_marathi_text("पोलीस", lenient=True))
        self.assertEqual(normalize_marathi_text("सुर", lenient=True), normalize_marathi_text("सूर", lenient=True))
        self.assertEqual(normalize_marathi_text("हँ", lenient=True), normalize_marathi_text("हं", lenient=True))

if __name__ == "__main__":
    unittest.main()
