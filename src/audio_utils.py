import soundfile as sf
import numpy as np

def load_and_resample(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, float]:
    """
    Loads an audio file, converts stereo to mono if needed,
    and resamples to target_sr using linear interpolation if required.
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
        num_target_samples = int(round(len(data) * float(target_sr) / sr))
        orig_indices = np.linspace(0, len(data) - 1, len(data))
        target_indices = np.linspace(0, len(data) - 1, num_target_samples)
        data = np.interp(target_indices, orig_indices, data).astype(np.float32)
        sr = target_sr
        
    duration_s = len(data) / float(sr)
    return data, duration_s
