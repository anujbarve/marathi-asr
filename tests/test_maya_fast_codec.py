import unittest
from pathlib import Path
import tempfile
import numpy as np

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.codec import MimiDecoder


class TestMayaFastCodec(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.assets_dir = make_tiny_assets(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_codec_length(self):
        # tiny assets: F = 5 -> 9600 samples
        codec_path = self.assets_dir / "mimi_decoder.onnx"
        mimi = MimiDecoder(codec_path, threads=1)

        F = 5
        codes = np.random.randint(0, 64, size=(16, F), dtype=np.int64)
        wav = mimi.decode(codes)

        self.assertEqual(wav.shape, (F * 1920,))
        self.assertEqual(wav.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
