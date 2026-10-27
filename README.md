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

### Target hardware

- The numbers in section 1 were measured on an M5 Mac and do not reflect the deployment target.
- The deployment target is an Intel i5 8th-gen U-series laptop with 16 GB RAM and CPU-only inference.
- Re-run the benchmark on the target before quoting latency or RTF figures.

To run the benchmark for IndicConformer (default):
```bash
python3 benchmarks/run_benchmark.py
```
To include the Whisper baselines (slow on CPU, opt-in):
```bash
python3 benchmarks/run_benchmark.py --models indicconformer-mr-int8 whisper-tiny whisper-base whisper-small
```
Results are saved to `benchmarks/benchmark_results.csv`. Each run records `platform.processor()` and `os.cpu_count()` in its output.

---

## 5. Running Tests
```bash
python3 -m unittest tests/test_pipeline.py
```

---

## 6. Maya-Fast TTS

Maya-Fast is a lightweight, torch-free ONNX Runtime TTS engine that synthesizes fluent Marathi speech in real time under 3 GB RAM on CPU.

### Installation
For runtime inference only, install the lightweight dependencies (no PyTorch required):
```bash
pip install -r requirements-fast.txt
```

### One-Time Asset Export
Model export requires the export environment (`.venv-tts` with PyTorch, Transformers, and `maya3`):
```bash
# Export versioned assets into ~/.cache/maya_fast/assets/v1
.venv-tts/bin/python -m tools.maya_export.export_prompt_assets
.venv-tts/bin/python -m tools.maya_export.export_backbone --precision wo8
.venv-tts/bin/python -m tools.maya_export.export_depth --precision wo8
.venv-tts/bin/python -m tools.maya_export.export_codec
```

### CLI Usage
Synthesize Marathi speech directly from the command line:
```bash
python -m src.maya_fast "नमस्कार, मी तुमची मदत करण्यासाठी इथे आहे." -o output.wav --voice Charvi
```

Options:
- `-o, --out`: Output WAV file path (default: `out.wav`).
- `--voice`: Registered voice name (default: `Charvi`).
- `--seed`: Random seed for reproducible generation (default: `0`).
- `--threads`: Number of intra-op ONNX Runtime threads (default: `min(8, CPU cores)`).
- `--assets`: Directory containing assets (default: `~/.cache/maya_fast/assets/v1`).

### Performance & Measured Metrics
- **Runtime Memory:** Peak RSS ≤ 3000 MB across all test sentences without triggering swap.
- **Speed:** Mean RTF ≤ 1.0 on Apple Silicon M5 (8 threads) with per-frame decode under 80 ms budget. On Intel i5 8th-gen U-series (CPU-only), runs within the 3 GB RAM limit without swap.

### Licenses
- Maya2 model weights: Apache-2.0.
- Kyutai Mimi neural audio codec: CC-BY-4.0.
- Audio outputs carry no audio watermark.

