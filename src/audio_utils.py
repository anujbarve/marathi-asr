import soundfile as sf
import numpy as np
import soxr

def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resamples a mono float array to target_sr with soxr (anti-aliased, HQ).
    Returns the input unchanged when the rates already match.
    """
    if orig_sr == target_sr:
        return samples
    return soxr.resample(samples, orig_sr, target_sr, quality="HQ").astype(np.float32)

def condition_audio(samples: np.ndarray, sample_rate: int = 16000, pad_s: float = 0.2,
                    target_peak: float = 0.9, max_gain: float = 10.0) -> np.ndarray:
    """
    Prepares a mono clip for the model: removes DC offset, scales the peak to
    target_peak (gain capped at max_gain so near-silence is not blown up), and
    pads pad_s seconds of zeros at both ends. An all-zero clip skips scaling.
    Returns float32, 1-D.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size:
        x = x - np.mean(x, dtype=np.float32)
        peak = float(np.max(np.abs(x)))
        if peak > 0.0:
            x = x * min(target_peak / peak, max_gain)
    pad = np.zeros(int(round(pad_s * sample_rate)), dtype=np.float32)
    return np.concatenate([pad, x.astype(np.float32), pad])

def load_and_resample(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, float]:
    """
    Loads an audio file, converts stereo to mono if needed,
    and resamples to target_sr using soxr if required.
    Returns:
        samples: np.ndarray (float32, 1D array)
        duration_s: float
    """
    data, sr = sf.read(file_path, dtype="float32")
    
    # If multi-channel, average to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample if sample rate differs
    if sr != target_sr:
        data = resample(data, sr, target_sr)
        sr = target_sr
        
    duration_s = len(data) / float(sr)
    return data, duration_s
