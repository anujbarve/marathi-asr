# Marathi Helper (मराठी वाणी ASR & Maya-Fast TTS)

A complete, high-performance, local-first Marathi speech processing toolkit optimized for Apple Silicon (macOS) and standard CPUs.

- **Lightweight Marathi ASR**: Powered by **AI4Bharat IndicConformer CTC (INT8 ONNX)**, executing at **33x faster-than-real-time** on Apple Silicon CPU with low RAM overhead (~630 MB) and high phonetic accuracy.
- **Maya-Fast ONNX TTS**: A lightweight, torch-free ONNX Runtime text-to-speech engine running entirely on CPU with support for built-in personas, emotion tags (`[laughs]`, `[whisper]`, etc.), and Hindi cross-lingual speech.
- **True In-Context Voice Cloning**: Clone any voice directly from a 4–8 second audio sample or live microphone recording. Features automatic ASR transcription, 192-dim ECAPA speaker embedding extraction, discrete Mimi acoustic codebook conditioning, and native ONNX frame generation.
- **Interactive Listening Test Suite**: Standalone blind A/B evaluation interface ([`listening_test.html`](listening_test.html)) with inline base64 audio and real-time scoring against quality gates.

---

## 1. Key Performance Highlights (Apple Silicon M5)

### ASR: IndicConformer vs. Whisper Baselines

| Metric | IndicConformer MR (INT8 ONNX) | Whisper Tiny INT8 | Whisper Base INT8 | Whisper Small INT8 |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Marathi WER** | **14.0%** (0.0% on Sample 2) | 100.0% (Romanized hallucination) | 106.7% (Romanized hallucination) | 84.2% – 133.3% |
| **Clean Marathi CER** | **3.5%** | 92.1% | 90.4% | 20.7% |
| **Average Latency (8s audio)**| **0.21s** | 4.05s | 6.48s | 21.30s |
| **Real-Time Factor (RTF)** | **0.030 (33x real-time)** | 0.651 | 1.170 | 4.603 |
| **Model Disk Footprint** | **188.4 MB** | 75 MB | 145 MB | 460 MB |
| **Peak RAM Consumption** | **~630 MB** | ~864 MB | ~1,018 MB | ~1,588 MB |
| **Architecture** | Conformer CTC (120M params) | Encoder-Decoder Transformer | Encoder-Decoder Transformer | Encoder-Decoder Transformer |

### TTS: Maya-Fast ONNX Engine & Voice Cloning

| Mode | Input / Text | Peak RSS | RTF | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **TTS (Charvi persona)** | *"नमस्कार"* | ~4,044 MB | 2.83 | Clear Marathi synthesis |
| **TTS + Emotion `[laughs]`** | *"अरे वा! [laughs] हे तर खूपच छान झालं."* | ~3,894 MB | 4.46 | Laugh inflection synthesized |
| **TTS (Hindi cross-lingual)** | *"नमस्ते, आप कैसे हैं?"* | ~4,480 MB | 1.97 | Hindi Devanagari acoustic alignment |
| **True Voice Cloning (Mic)** | 6s user mic recording $\to$ 7.36s target sentence | ~4,269 MB | 5.23 | **100.0% word accuracy** on IndicConformer ASR |

---

## 2. Installation & Quickstart

### Prerequisites
- macOS 12+ (Apple Silicon M-series or Intel) or Linux x86_64
- Python 3.11+ (tested on Python 3.13)
- `ffmpeg` (macOS: `brew install ffmpeg`)

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For **runtime-only TTS inference** without PyTorch:
```bash
pip install -r requirements-fast.txt
```

---

## 3. Automatic Speech Recognition (ASR) CLI

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

### With Devanagari Normalization
```bash
python3 src/cli.py benchmarks/audio/clean_sample_0.wav --normalize
```

### Comparing with Whisper Baselines
```bash
python3 src/cli.py benchmarks/audio/clean_sample_0.wav --model whisper-base --json
```

---

## 4. Maya-Fast Text-to-Speech (TTS)

Maya-Fast executes on CPU using optimized ONNX sessions:
- **Backbone**: Weight-only 8-bit quantized (`wo8`, MatMulNBits int8 block 32).
- **Depth**: FP16 shared-group single graph or `v2-lowram` INT8 depth.
- **Audio Codec**: Kyutai Mimi neural decoder ONNX graph (~1 ms/frame).

### CLI Synthesis
Synthesize Marathi speech directly with standard voices:
```bash
python -m src.maya_fast "नमस्कार, मी तुमची मदत करण्यासाठी इथे आहे." -o output.wav --voice Charvi
```

### Supported Emotion Markers
Inline emotion tags can be embedded in any sentence:
```bash
python -m src.maya_fast "अरे वा! [laughs] हे तर खूप छान झाले आहे." -o laughter.wav --voice Charvi
```
Supported tags: `[laughs]`, `[sighs]`, `[gasp]`, `[whisper]`, `[happy]`.

---

## 5. In-Context Voice Cloning

Clone any voice from a short 4–8 second audio recording.

### Interactive Microphone Recording & Cloning
Launch the interactive CLI to select your microphone, record a sample, automatically transcribe it with IndicConformer, and immediately synthesize in your voice:

```bash
python tools/record_and_clone.py \
  --text "नमस्कार! हा माझा स्वतःचा आवाज आहे आणि मॉडेल आता माझ्या आवाजात बोलत आहे." \
  --play
```

### Cloning from an Existing Audio File
If you already have a recording (`.wav`):
```bash
python tools/record_and_clone.py \
  --sample my_sample_recording.wav \
  --text "तुमच्या पसंतीचे नवीन वाक्य येथे लिहा." \
  --play
```

### Standalone Cloning Script
For headless or scripted automation:
```bash
python tools/clone_voice_fast.py \
  --ref-audio my_sample_recording.wav \
  --ref-text "हॅलो मी काहीतरी मराठीमध्ये बोलत आहे तुम्हाला कळतय का" \
  --text "नमस्कार! हा माझा स्वतःचा आवाज आहे." \
  --out cloned_output.wav
```

---

## 6. Blind A/B Listening Test Web Application

We provide a self-contained web application for evaluating perceptual audio quality between model asset variants (`v1` vs. `v2-lowram`):

```bash
# Open directly in your browser
open listening_test.html
```

Features:
- Self-contained HTML with embedded Base64 audio pairs.
- Randomized blind A/B playback for 10 curated test sentences.
- Interactive voting (Prefer A, Prefer B, or Equal) with real-time percentage scoring against the $\ge 45\%$ acceptance threshold.
- Generated via `tools/build_listen_page.py`.

---

## 7. Testing & Verification

Run the full project test suite (ASR, Maya-Fast assets, backbone, depth, sampling, codec, and memory guard):
```bash
python -m unittest discover tests -p "test_*.py"
```
*Current test status: 83 tests passing (`OK`), zero failures.*

### Memory Safety Guard
All memory-intensive model benchmarks and synthesis runs can be wrapped in `tools/memguard.py` to prevent runaway memory usage or system swap growth:
```bash
python tools/memguard.py --rss-limit-mb 5000 --min-free-mb 1000 -- python -m src.maya_fast "चाचणी वाक्य"
```

---

## 8. Licenses

- **IndicConformer / AI4Bharat**: MIT License.
- **Maya2 Model Weights**: Apache-2.0 License.
- **Kyutai Mimi Audio Codec**: CC-BY-4.0 License.
- Audio generated by this system carries no audio watermarking.
