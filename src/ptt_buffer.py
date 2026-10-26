from typing import Optional

import numpy as np


class ClipAssembler:
    """
    Builds push-to-talk clips from a stream of mono chunks.

    A ring buffer always keeps the last pre_roll_s of audio, so a clip can start
    slightly before the key press. After release, post_roll_s of audio is
    collected so a clip can end slightly after the key release. Not thread-safe;
    the caller must serialize access.
    """
    def __init__(self, sample_rate: int = 16000, pre_roll_s: float = 0.3, post_roll_s: float = 0.25):
        self.sample_rate = sample_rate
        self._pre_n = int(pre_roll_s * sample_rate)
        self._post_n = int(post_roll_s * sample_rate)
        self._ring = np.zeros(0, dtype=np.float32)
        self._parts = []
        self._held = False
        self._post_pending = False
        self._post_got = 0

    @property
    def is_recording(self) -> bool:
        return self._held

    def _push_ring(self, chunk: np.ndarray) -> None:
        buf = np.concatenate([self._ring, chunk])
        self._ring = buf[len(buf) - self._pre_n:] if len(buf) > self._pre_n else buf

    def feed(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._push_ring(chunk)
        if self._held:
            self._parts.append(chunk)
            return None
        if not self._post_pending:
            return None
        take = chunk[:self._post_n - self._post_got]
        self._parts.append(take)
        self._post_got += len(take)
        if self._post_got >= self._post_n:
            return self._finish()
        return None

    def start(self) -> Optional[np.ndarray]:
        if self._held:
            return None
        flushed = self._finish() if self._post_pending else None
        # A flushed clip seeds the new clip with nothing; otherwise seed with pre-roll
        self._parts = [] if flushed is not None else [self._ring.copy()]
        self._held = True
        return flushed

    def stop(self) -> None:
        if self._held:
            self._held = False
            self._post_pending = True
            self._post_got = 0

    def _finish(self) -> np.ndarray:
        clip = np.concatenate(self._parts) if self._parts else np.zeros(0, dtype=np.float32)
        self._parts = []
        self._post_pending = False
        self._post_got = 0
        return clip
