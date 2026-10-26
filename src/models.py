import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

class BaseMarathiASR(ABC):
    """
    Common abstract interface for all Marathi ASR engine candidates.
    """
    
    @abstractmethod
    def load(self) -> None:
        """Loads model into memory."""
        pass
        
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribes single-channel float32 audio."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier of the model."""
        pass

    @property
    @abstractmethod
    def disk_size_mb(self) -> float:
        """Approximate size of model weights on disk in MB."""
        pass


class IndicConformerONNXASR(BaseMarathiASR):
    """
    AI4Bharat IndicConformer Marathi ASR deployed via Sherpa-ONNX (INT8 CTC).
    Lightweight, CPU-optimized, offline speech recognition.
    """
    def __init__(self, num_threads: int = 4):
        self._num_threads = num_threads
        self._recognizer = None
        self._model_path = None
        self._tokens_path = None

    @property
    def model_id(self) -> str:
        return "indicconformer-mr-int8"

    @property
    def disk_size_mb(self) -> float:
        if self._model_path and os.path.exists(self._model_path):
            return round(os.path.getsize(self._model_path) / (1024 * 1024), 2)
        return 188.4  # default known size

    def load(self) -> None:
        if self._recognizer is not None:
            return
            
        from huggingface_hub import hf_hub_download
        import sherpa_onnx
        
        self._model_path = hf_hub_download(
            repo_id="parismitaglobalsolutions/indicconformer-sherpa-onnx",
            filename="mr/model.int8.onnx"
        )
        self._tokens_path = hf_hub_download(
            repo_id="parismitaglobalsolutions/indicconformer-sherpa-onnx",
            filename="tokens.txt"
        )
        
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=self._model_path,
            tokens=self._tokens_path,
            num_threads=self._num_threads,
            provider="cpu"
        )

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        if self._recognizer is None:
            self.load()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_data)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()


class FasterWhisperASR(BaseMarathiASR):
    """
    Faster-Whisper (CTranslate2) implementation for Whisper models.
    Supports tiny, base, small, and quantized variants with fixed Marathi language decoding.
    """
    def __init__(self, size: str = "tiny", compute_type: str = "int8", num_threads: int = 4):
        self.size = size
        self.compute_type = compute_type
        self.num_threads = num_threads
        self._model = None

    @property
    def model_id(self) -> str:
        return f"whisper-{self.size}-{self.compute_type}"

    @property
    def disk_size_mb(self) -> float:
        # Approximate size based on model variant
        sizes = {
            "tiny": 75.0,
            "base": 145.0,
            "small": 460.0
        }
        return sizes.get(self.size, 100.0)

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.size,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.num_threads
        )

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        if self._model is None:
            self.load()
        # Force language to Marathi ('mr') and task to 'transcribe'
        segments, _ = self._model.transcribe(
            audio_data,
            language="mr",
            task="transcribe",
            beam_size=1
        )
        return " ".join(seg.text for seg in segments).strip()


def get_model(model_name: str) -> BaseMarathiASR:
    """
    Factory to return an instantiated ASR model wrapper.
    """
    name = model_name.lower().strip()
    if name in ["indicconformer", "indicconformer-mr", "indicconformer-mr-int8", "indicconformer-int8"]:
        return IndicConformerONNXASR()
    elif name in ["whisper-tiny", "tiny"]:
        return FasterWhisperASR(size="tiny", compute_type="int8")
    elif name in ["whisper-base", "base"]:
        return FasterWhisperASR(size="base", compute_type="int8")
    elif name in ["whisper-small", "small"]:
        return FasterWhisperASR(size="small", compute_type="int8")
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Available: indicconformer-mr-int8, whisper-tiny, whisper-base, whisper-small")
