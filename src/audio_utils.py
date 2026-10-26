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
