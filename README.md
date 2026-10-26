# Lightweight Marathi Speech Recognition System (मराठी वाणी ASR)

A high-performance, local-first, Marathi-only Automatic Speech Recognition system optimized for Apple Silicon (macOS) and standard CPUs.

Powered by **AI4Bharat IndicConformer CTC (Quantized INT8)** executed via **ONNX Runtime / Sherpa-ONNX**, delivering over **30x–50x faster-than-real-time** transcription on Apple Silicon CPU with low RAM overhead and high Marathi phonetic fidelity.

---

## 1. Key Performance Highlights (M5 Apple Silicon)

| Metric | IndicConformer MR (INT8 ONNX) | Whisper Tiny INT8 | Whisper Base INT8 | Whisper Small INT8 |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Marathi WER** | **14.0%** (0.0% on Sample 2) | 100.0% (Romanized hallucination) | 106.7% (Romanized hallucination) | 84.2% – 133.3% |
| **Clean Marathi CER** | **3.5%** | 92.1% | 90.4% | 20.7% |
| **Average Latency (8s audio)**| **0.21s** | 4.05s | 6.48s | 21.30s |
| **Real-Time Factor (RTF)** | **0.030 (33x real-time)** | 0.651 | 1.170 | 4.603 |
| **Model Disk Footprint** | **188.4 MB** | 75 MB | 145 MB | 460 MB |
| **Peak RAM Consumption** | **~630 MB** | ~864 MB | ~1,018 MB | ~1,588 MB |
| **Architecture** | Conformer CTC (120M params) | Encoder-Decoder Transformer | Encoder-Decoder Transformer | Encoder-Decoder Transformer |

---

## 2. Installation & Quickstart (macOS / Apple Silicon)

### Prerequisites
- macOS 12+ (Apple Silicon M-series or Intel)
- Python 3.10+ (tested on Python 3.13)
- `ffmpeg` (installed via `brew install ffmpeg`)

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. CLI Usage

### Basic Plain-Text Transcription
```bash
python3 src/cli.py benchmarks/audio/clean_sample_0.wav
```
*Output:*
```text
पोलीस अधीक्षक चंद्रशेखर सोळंकी यांनी सांगितले की आरोपी चेहरा झाकून घेऊन कोर्टात हजर झाला
```

### JSON Output (with Latency, RTF, RAM Metadata)
```bash
python3 src/cli.py benchmarks/audio/clean_sample_0.wav --json
```
*Output:*
```json
{
  "model": "indicconformer-mr-int8",
  "transcript": "पोलीस अधीक्षक चंद्रशेखर सोळंकी यांनी सांगितले की आरोपी चेहरा झाकून घेऊन कोर्टात हजर झाला",
  "raw_transcript": "पोलीस अधीक्षक चंद्रशेखर सोळंकी यांनी सांगितले की आरोपी चेहरा झाकून घेऊन कोर्टात हजर झाला",
  "audio_duration_s": 8.16,
  "inference_latency_s": 0.205,
  "model_load_latency_s": 1.293,
  "real_time_factor": 0.0251,
  "peak_ram_mb": 559.06,
  "language": "mr"
}
```

### With Devanagari Normalization
```bash
python3 src/cli.py benchmarks/audio/clean_sample_0.wav --normalize
```

### Comparing with Whisper Baselines
```bash
python3 src/cli.py benchmarks/audio/clean_sample_0.wav --model whisper-base --json
```

---

## 4. Reproducible Benchmarking

To run the complete benchmark suite comparing all candidate models on clean read, noisy microphone background, and short voice commands:
```bash
python3 benchmarks/run_benchmark.py
```
Results are saved to `benchmarks/benchmark_results.csv`.

---

## 5. Running Tests
```bash
python3 -m unittest tests/test_pipeline.py
```
