# Marathi Helper & Maya-Fast Project Status (2026-10-30)

## Overview
All experimental branches have been unified into `main`. The repository provides a complete local Marathi speech stack:
1. **Lightweight Marathi ASR**: IndicConformer CTC INT8 (~188 MB, 33x faster than real-time on Apple Silicon, ~630 MB peak RAM).
2. **Maya-Fast ONNX TTS**: Torch-free ONNX Runtime TTS engine for Marathi/Hindi speech synthesis on CPU.
3. **In-Context Voice Cloning**: Clone any voice directly from a 4-8s microphone recording with automatic ASR transcription, ECAPA speaker embedding, and Mimi acoustic token conditioning.
4. **Interactive Listening Evaluation**: Standalone blind A/B evaluation suite (`listening_test.html`) with interactive scoring against the quality gate.

---

## Current Status & Milestones

### 1. Branch Consolidation
- Merged `feat/maya2-cpu-tts`, `feat/maya-fast-export`, and `feat/maya-fast-runtime` into `main`.
- Integrated codebase from secondary workspace `maya-fast-loader`.
- Pushed and synchronized with remote repository `https://github.com/anujbarve/marathi-asr.git`.

### 2. Voice Cloning Pipeline
- Built [tools/clone_voice_fast.py](file:///Users/anujbarve/Documents/marathi-helper/tools/clone_voice_fast.py):
  - Extracts 192-dim speaker vector with `speechbrain/spkrec-ecapa-voxceleb`.
  - Encodes acoustic tokens $(32, F)$ with `kyutai/mimi`.
  - Builds in-context conditioning prompt via `V5Builder` and `PackerV2`.
  - Generates speech frames via `MayaFast` ONNX Backbone + Depth + Mimi decoder without falling back to built-in personas.
- Built interactive CLI [tools/record_and_clone.py](file:///Users/anujbarve/Documents/marathi-helper/tools/record_and_clone.py):
  - Allows selecting from available audio input devices (microphones).
  - Records a natural 6s sample with peak/RMS normalization.
  - Automatically transcribes the recording using the local `indicconformer-mr-int8` model.
  - Generates cloned voice output and supports immediate playback on macOS (`afplay`).
- **Empirical Validation**:
  - Sample: `my_sample_recording.wav` ("हॅलो मी काहीतरी मराठीमध्ये बोलत आहे तुम्हाला कळतय का")
  - Generated: `my_actual_cloned_voice.wav` ("नमस्कार! हा माझा स्वतःचा आवाज आहे आणि मॉडेल आता माझ्या आवाजात बोलत आहे.")
  - ASR Verification: 100% word-for-word accuracy via IndicConformer.
  - Peak RSS: 4,269 MB (memguard passed, swap growth: 0 MB).

### 3. Emotion Markers & Cross-Lingual Capabilities
- Tested and confirmed native support for emotion tags in synthesis:
  - `[laughs]`, `[sighs]`, `[gasp]`, `[whisper]`, `[happy]`, etc.
  - Tested: *"अरे वा! [laughs] हे तर खूपच छान झालं."* -> Peak RSS 3894 MB, RTF 4.46.
- Cross-lingual Hindi support verified:
  - Tested: *"नमस्ते, आप कैसे हैं?"* -> Peak RSS 4480 MB, RTF 1.97.

### 4. Listening Evaluation Tooling
- Implemented [tools/build_listen_page.py](file:///Users/anujbarve/Documents/marathi-helper/tools/build_listen_page.py).
- Produced [listening_test.html](file:///Users/anujbarve/Documents/marathi-helper/listening_test.html):
  - 10 blind A/B pairs comparing `v1` vs `v2-lowram`.
  - Self-contained HTML with base64 embedded audio.
  - Real-time scoring against the $\ge 45\%$ preference acceptance gate.

### 5. Test Suite
- All 83 unit tests passing (`OK`) with zero regressions across ASR, Maya-Fast assets, backbone, depth, sampling, codec, and memguard.
