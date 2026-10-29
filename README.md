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

Maya-Fast is a lightweight, torch-free ONNX Runtime TTS engine that synthesizes Marathi speech on CPU. The 3 GB RAM target and real-time speed are not yet met by the current v1 asset set (see measured numbers below).

### Installation
For runtime inference only, install the lightweight dependencies (no PyTorch required):
```bash
pip install -r requirements-fast.txt
```
`requirements-fast.txt` holds numpy, onnxruntime, tokenizers, plus `psutil` (the engine reads RSS with it) and `onnx` (the unit tests build tiny graphs). No torch. Run the runtime tests with:
```bash
python -m unittest discover -s tests -p "test_maya_fast_*.py"
```

### One-Time Asset Export
Model export requires the export environment (`.venv-tts` with PyTorch, Transformers, and `maya3`). The `tools/maya_export` scripts live on the `feat/maya-fast-export` branch:
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
Machine: Apple M5, 8 ORT threads (ORT version per row), measured 2026-10-29. The Intel i5 8th-gen U-series target has not been measured yet.

| Asset set | Run | ORT | Peak RSS | RTF |
|---|---|---|---|---|
| v1 | controller e2e run, 2 sentences | 1.26 | 4486.9 MB (memguard) | 0.94 / 0.92 |
| v1 | smoke eval, 3 items, seed 0 | 1.31 | 4559 MB | mean 1.038 |
| v1 | full test set, 3 seeds | 1.31 | 4467 MB (memguard) | mean 2.206 |
| v2-lowram | full test set, 3 seeds | 1.31 | 3823 MB (memguard) | mean 5.460 |

The two v1 runs at the top (RTF about 0.9 to 1.0) are earlier than the full-test-set v1 run (RTF 2.206; see
`benchmarks/maya_fast_results.md`). The machine was about 2x slower in the later run, cause unknown. Neither set
meets the 3000 MB RAM target. v2-lowram (`~/.cache/maya_fast/assets/v2-lowram`, built with
`tools/make_asset_variant.py`, select with `--assets`) uses about 640 MB less RAM but misses the offline speed
target (RTF 4). Both pass the CER gate against the upstream reference.

### Evaluation Scripts
`benchmarks/maya_fast_eval.py`, `maya_fast_listen.py` and `maya_fast_spk.py` use `benchmarks/tts_testset_mr.json` by default. They need extra dependencies that are not in `requirements-fast.txt`: the ASR scripts need `sherpa_onnx` and `soxr` (the ASR environment is the marathi-helper repository's venv, `/Users/anujbarve/Documents/marathi-helper/.venv`), and `maya_fast_spk.py` needs a torch environment with `speechbrain`.

### Licenses
- Maya2 model weights: Apache-2.0.
- Kyutai Mimi neural audio codec: CC-BY-4.0.
- Audio outputs carry no audio watermark.

