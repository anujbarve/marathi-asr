from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort

from .ort_opts import session_options


class MimiDecoder:
    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        self.path = Path(path)
        self.opts = session_options(threads)
        self.session = ort.InferenceSession(str(self.path), self.opts, providers=["CPUExecutionProvider"])

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Input:
            codes: (16, F) int64
        Returns:
            wav: (F * 1920,) float32
        """
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim == 2:
            codes_input = codes[None, :, :]  # (1, 16, F)
        elif codes.ndim == 3:
            codes_input = codes
        else:
            raise ValueError(f"Expected codes to have 2 or 3 dims, got {codes.shape}")

        outputs = self.session.run(None, {"codes": codes_input})
        wav = outputs[0]  # (1, 1, samples)
        wav = np.squeeze(wav)
        return wav.astype(np.float32)
