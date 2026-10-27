from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort

from .ort_opts import session_options


class Depth:
    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        self.path = Path(path)
        self.opts = session_options(threads)
        self.session = ort.InferenceSession(str(self.path), self.opts, providers=["CPUExecutionProvider"])

    def frame(
        self,
        h: np.ndarray,
        cb0: int | np.integer | np.ndarray,
        mem: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        """
        Inputs:
            h: (d_model,) or (1, d_model) float32
            cb0: int or (1,) int64
            mem: (1, R, d_mem) float32
            noise: (15, vocab) float32
        Returns:
            codes: (16,) int64
        """
        if h.ndim == 1:
            h = h[None, :]
        h = h.astype(np.float32)

        if isinstance(cb0, (int, np.integer)):
            cb0 = np.array([cb0], dtype=np.int64)
        else:
            cb0 = np.asarray(cb0, dtype=np.int64)
            if cb0.ndim == 0:
                cb0 = cb0.reshape(1)

        mem = np.asarray(mem, dtype=np.float32)
        if mem.ndim == 2:
            mem = mem[None, :, :]

        noise = np.asarray(noise, dtype=np.float32)

        feed = {
            "h": h,
            "cb0": cb0,
            "mem": mem,
            "noise": noise,
        }

        outputs = self.session.run(None, feed)
        codes = outputs[0]
        if codes.ndim > 1:
            codes = codes.squeeze()
        return codes.astype(np.int64)
