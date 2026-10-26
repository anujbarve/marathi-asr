import unittest
import numpy as np
import os
import sys
import tempfile
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_utils import resample, load_and_resample


def tone(freq, sr, seconds=1.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestResample(unittest.TestCase):
    def test_removes_content_above_nyquist(self):
        # 10 kHz is above the 8 kHz Nyquist of 16 kHz output; linear interp aliases it to 6 kHz
        out = resample(tone(10000, 48000), 48000, 16000)
        self.assertLess(np.sqrt(np.mean(out[1000:-1000] ** 2)), 0.01)

    def test_preserves_speech_band_tone(self):
        out = resample(tone(1000, 44100), 44100, 16000)
        self.assertEqual(len(out), 16000)
        self.assertAlmostEqual(np.sqrt(np.mean(out[1000:-1000] ** 2)), 0.5 / np.sqrt(2), delta=0.02)

    def test_upsample_8k(self):
        out = resample(tone(1000, 8000), 8000, 16000)
        self.assertEqual(len(out), 16000)
        self.assertEqual(out.dtype, np.float32)

    def test_16k_passthrough(self):
        x = tone(1000, 16000)
        self.assertTrue(np.array_equal(resample(x, 16000, 16000), x))

    def test_load_and_resample_uses_soxr(self):
        stereo = np.stack([tone(10000, 48000), tone(10000, 48000)], axis=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, stereo, 48000)
            data, duration = load_and_resample(path)
        finally:
            os.remove(path)
        self.assertEqual(data.ndim, 1)
        self.assertEqual(data.dtype, np.float32)
        self.assertAlmostEqual(duration, 1.0, places=3)
        self.assertLess(np.sqrt(np.mean(data[1000:-1000] ** 2)), 0.01)


if __name__ == "__main__":
    unittest.main()
