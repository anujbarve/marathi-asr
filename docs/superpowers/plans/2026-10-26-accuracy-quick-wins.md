# Accuracy Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise real and measured Marathi transcription accuracy with low-cost changes that stay fast on an Intel i5 8th-gen U-series laptop.

**Architecture:** Fix the measurement first (normalizer, metrics, benchmark references), then fix the audio the model sees: anti-aliased resampling, DC removal, peak normalization, and silence padding before every IndicConformer call. Push-to-talk gets pre-roll and post-roll through a small hardware-free `ClipAssembler`, so it can be unit-tested headless.

**Tech Stack:** Python 3.10+, numpy, soundfile, `soxr` (new), sherpa-onnx, jiwer, unittest.

**Spec:** No spec file. Requirements come from the 2026-10-26 conversation: the accuracy recommendations, re-ranked for the i5 8th-gen U target. They are summarized under Global Constraints.

## Global Constraints

- Target hardware: Intel i5 8th-gen U-series (4 cores / 8 threads, AVX2, no AVX-512/VNNI), 16 GB RAM, integrated GPU only. CPU-only inference; no CUDA, Metal or GPU code paths.
- OS is not decided. All code must be cross-platform (macOS, Windows, Linux): no OS-specific APIs, no shelling out to `ffmpeg`.
- Only one new runtime dependency: `soxr>=0.3.7`. Do not add scipy, torch or librosa.
- Python 3.10+. Tests use `unittest` and run with `.venv/bin/python -m unittest`.
- New tests must not download models, open a microphone, or read `benchmarks/audio/` (it is gitignored). Synthesize audio in-test.
- Keep `IndicConformerONNXASR(num_threads=4)` as the default.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Tap with no speech (silent or near-silent clip):** `condition_audio` must not divide by zero or blow the noise floor up to full scale. Gain is capped at `max_gain=10.0`. Test: `test_near_silent_gain_is_capped`, `test_all_zero_stays_zero` (Task 3).
2. **Key re-pressed during post-roll:** the first clip must be emitted whole, and no samples may be duplicated into the second clip. Test: `test_restart_during_post_roll_flushes_without_duplication` (Task 4).
3. **Mic refuses 16 kHz (common on Windows WASAPI):** `listen.py` must fall back to the device's default rate and resample clips to 16 kHz, not crash. Covered by `resample` tests at 48 kHz and 44.1 kHz (Task 2) plus the manual step in Task 4.
4. **8 kHz phone audio or already-16 kHz files:** upsampling must work, and 16 kHz input must be returned unchanged. Test: `test_upsample_8k`, `test_16k_passthrough` (Task 2).
5. **Model returns empty text:** scoring must give WER/CER 1.0 and not raise. Test: `test_empty_prediction_scores_one` (Task 5).

---

### Task 1: Normalizer handles ZWJ/ZWNJ, nukta, and a lenient mode

**Files:**
- Modify: `src/normalizer.py`
- Test: `tests/test_normalizer.py` (create)

**Interfaces:**
- Produces: `normalize_marathi_text(text: str, lenient: bool = False) -> str`. The existing one-argument calls keep working.

- [ ] **Step 1: Write the failing tests** in `tests/test_normalizer.py` (class `TestNormalizer`):

```python
def test_strips_zero_width_joiners(self):
    self.assertEqual(normalize_marathi_text("क्‍ष ‌आणि"), "क्ष आणि")

def test_folds_nukta(self):
    # U+095B (ज़) and decomposed ज + U+093C both become ज
    self.assertEqual(normalize_marathi_text("ज़ ज़"), "ज ज")

def test_strict_keeps_vowel_length(self):
    self.assertNotEqual(normalize_marathi_text("पोलिस"), normalize_marathi_text("पोलीस"))

def test_lenient_folds_vowel_length_and_chandrabindu(self):
    self.assertEqual(normalize_marathi_text("पोलिस", lenient=True), normalize_marathi_text("पोलीस", lenient=True))
    self.assertEqual(normalize_marathi_text("सुर", lenient=True), normalize_marathi_text("सूर", lenient=True))
    self.assertEqual(normalize_marathi_text("हँ", lenient=True), normalize_marathi_text("हं", lenient=True))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m unittest tests.test_normalizer -v`
Expected: `test_strips_zero_width_joiners`, `test_folds_nukta` and the lenient test FAIL (the lenient test with `TypeError: unexpected keyword argument 'lenient'`).

- [ ] **Step 3: Implement**

Remove `‌` and `‍` before the punctuation step. Fold nukta by NFD-normalizing, dropping `़`, then NFC-normalizing. U+0958–U+095F are excluded from composition, so NFD splits them. When `lenient=True`, also map `ि`(U+093F) to `ी`(U+0940), `ु`(U+0941) to `ू`(U+0942), and `ँ`(U+0901) to `ं`(U+0902). Do not fold `ळ`/`ल`; that is a real Marathi distinction.

- [ ] **Step 4: Run the new and existing tests**

Run: `.venv/bin/python -m unittest tests.test_normalizer tests.test_pipeline.TestMarathiASR.test_devanagari_normalization -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/normalizer.py tests/test_normalizer.py
git commit -m "feat(normalizer): strip ZWJ/ZWNJ, fold nukta, add lenient mode"
```

---

### Task 2: Anti-aliased resampling with soxr

**Files:**
- Modify: `src/audio_utils.py`, `requirements.txt`
- Test: `tests/test_audio_utils.py` (create)

**Interfaces:**
- Produces: `resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray` (float32, 1-D). Returns the input unchanged when `orig_sr == target_sr`.
- `load_and_resample(file_path, target_sr=16000) -> tuple[np.ndarray, float]` keeps its signature and calls `resample`.

- [ ] **Step 1: Add the dependency and install it**

Add `soxr>=0.3.7` to `requirements.txt`. Run `.venv/bin/pip install "soxr>=0.3.7"`.
Expected: `Successfully installed soxr-...`.

- [ ] **Step 2: Write the failing tests** in `tests/test_audio_utils.py` (class `TestResample`). Use a helper `tone(freq, sr, seconds=1.0, amp=0.5)` that returns a float32 sine.

```python
def test_removes_content_above_nyquist(self):
    # 10 kHz is above the 8 kHz Nyquist of 16 kHz output; linear interp aliases it to 6 kHz
    out = resample(tone(10000, 48000), 48000, 16000)
    self.assertLess(np.sqrt(np.mean(out[1000:-1000] ** 2)), 0.01)

def test_preserves_speech_band_tone(self):
    out = resample(tone(1000, 44100), 44100, 16000)
    self.assertEqual(len(out), 16000)
    self.assertAlmostEqual(np.sqrt(np.mean(out[1000:-1000] ** 2)), 0.5 / np.sqrt(2), delta=0.02)

def test_upsample_8k(self):
    out = resample(tone(1000, 8000), 8000, 16000)
    self.assertEqual(len(out), 16000)
    self.assertEqual(out.dtype, np.float32)

def test_16k_passthrough(self):
    x = tone(1000, 16000)
    self.assertTrue(np.array_equal(resample(x, 16000, 16000), x))

def test_load_and_resample_uses_soxr(self):
    # write a 48 kHz stereo 10 kHz tone with soundfile to a tempfile.NamedTemporaryFile(suffix=".wav")
    # expect: mono float32, duration 1.0 s, RMS < 0.01
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `.venv/bin/python -m unittest tests.test_audio_utils -v`
Expected: `ImportError: cannot import name 'resample'`.

- [ ] **Step 4: Implement `resample` and replace the `np.interp` block in `load_and_resample`**

Use `soxr.resample(samples, orig_sr, target_sr, quality="HQ")` and cast the result to float32.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m unittest tests.test_audio_utils tests.test_pipeline -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/audio_utils.py tests/test_audio_utils.py
git commit -m "fix(audio): anti-aliased resampling via soxr"
```

---

### Task 3: Condition audio before every IndicConformer call

**Files:**
- Modify: `src/audio_utils.py`, `src/models.py` (`IndicConformerONNXASR.transcribe`)
- Test: `tests/test_audio_utils.py` (add class `TestConditionAudio`), `tests/test_models.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `condition_audio(samples: np.ndarray, sample_rate: int = 16000, pad_s: float = 0.2, target_peak: float = 0.9, max_gain: float = 10.0) -> np.ndarray` (float32, 1-D). The steps are: subtract the mean, scale the peak to `target_peak` with gain at most `max_gain`, then add `pad_s` of zeros at both ends. An all-zero input skips scaling.

- [ ] **Step 1: Write the failing tests**

```python
# TestConditionAudio
def test_removes_dc_normalizes_and_pads(self):
    x = (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) + 0.05).astype(np.float32)
    out = condition_audio(x)
    self.assertEqual(len(out), 16000 + 2 * 3200)
    self.assertTrue(np.all(out[:3200] == 0) and np.all(out[-3200:] == 0))
    self.assertLess(abs(out[3200:-3200].mean()), 1e-3)
    self.assertAlmostEqual(np.abs(out).max(), 0.9, delta=1e-3)

def test_all_zero_stays_zero(self):
    out = condition_audio(np.zeros(1600, dtype=np.float32))
    self.assertFalse(np.isnan(out).any())
    self.assertEqual(np.abs(out).max(), 0.0)

def test_near_silent_gain_is_capped(self):
    x = (0.001 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    self.assertAlmostEqual(np.abs(condition_audio(x)).max(), 0.01, delta=1e-4)

def test_empty_input_returns_padding_only(self):
    self.assertEqual(len(condition_audio(np.zeros(0, dtype=np.float32))), 6400)
```

In `tests/test_models.py`, `test_indicconformer_transcribe_conditions_audio`: set `engine._recognizer` to a fake whose `create_stream()` returns a stream that records `accept_waveform(sr, samples)` and has `.result.text = " नमस्कार "`. Call `engine.transcribe(np.full(16000, 0.05, dtype=np.float32))`. Assert the recorded length is `22400`, the recorded `sr` is `16000`, and the return value is `"नमस्कार"`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m unittest tests.test_audio_utils tests.test_models -v`
Expected: `ImportError: cannot import name 'condition_audio'`.

- [ ] **Step 3: Implement `condition_audio` and call it as the first line of `IndicConformerONNXASR.transcribe`, after the lazy `load()`**

Leave `FasterWhisperASR` unchanged; Whisper pads internally.

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/audio_utils.py src/models.py tests/test_audio_utils.py tests/test_models.py
git commit -m "feat(asr): DC removal, peak normalization and silence padding before IndicConformer"
```

---

### Task 4: Push-to-talk pre-roll, post-roll and sample-rate fallback

**Files:**
- Create: `src/ptt_buffer.py` (no `sounddevice` or `pynput` imports)
- Modify: `src/listen.py`
- Test: `tests/test_ptt_buffer.py` (create)

**Interfaces:**
- Consumes: `resample(samples, orig_sr, target_sr)` from Task 2.
- Produces: `class ClipAssembler(sample_rate: int = 16000, pre_roll_s: float = 0.3, post_roll_s: float = 0.25)` with:
  - `feed(chunk: np.ndarray) -> Optional[np.ndarray]`: always updates a ring buffer that keeps the last `pre_roll` samples. While held, appends to the clip. During post-roll, appends until exactly `int(post_roll_s * sample_rate)` post-release samples are collected, then returns the finished clip. Otherwise returns `None`.
  - `start() -> Optional[np.ndarray]`: begins a clip seeded with the ring contents. If a post-roll is still pending, finalizes and returns that clip and seeds the new clip with nothing; the ring holds the old clip's samples.
  - `stop() -> None`: begins post-roll.
  - `is_recording: bool` property, true only while held.

- [ ] **Step 1: Write the failing tests** in `tests/test_ptt_buffer.py`. Use the helper `chunk(value, n=160)` for `np.full(n, value, float32)`. The values mark where samples came from.

```python
def test_clip_includes_pre_and_post_roll(self):
    a = ClipAssembler()
    for _ in range(100): self.assertIsNone(a.feed(chunk(1)))
    self.assertIsNone(a.start())
    for _ in range(50): self.assertIsNone(a.feed(chunk(2)))
    a.stop()
    outs = [a.feed(chunk(3)) for _ in range(25)]
    self.assertTrue(all(o is None for o in outs[:-1]))
    clip = outs[-1]
    self.assertEqual(len(clip), 4800 + 8000 + 4000)
    self.assertTrue(np.all(clip[:4800] == 1) and np.all(clip[4800:12800] == 2) and np.all(clip[12800:] == 3))

def test_pre_roll_limited_to_available_history(self):
    # feed 10 chunks, start, feed 1, stop, feed 25: clip starts with exactly 1600 samples of value 1

def test_post_roll_trimmed_exactly_with_odd_chunk_size(self):
    # chunk size 512 throughout: clip length == min(history, 4800) + held_samples + 4000

def test_restart_during_post_roll_flushes_without_duplication(self):
    a = ClipAssembler()
    a.start(); [a.feed(chunk(2)) for _ in range(10)]; a.stop(); [a.feed(chunk(3)) for _ in range(5)]
    first = a.start()
    self.assertEqual(len(first), 1600 + 800)
    [a.feed(chunk(4)) for _ in range(10)]; a.stop()
    second = [a.feed(chunk(5)) for _ in range(25)][-1]
    self.assertTrue(np.all(second[:1600] == 4))
    self.assertEqual(len(second), 1600 + 4000)

def test_is_recording_tracks_hold_only(self):
    # False initially, True after start(), False after stop() even while post-roll pending
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m unittest tests.test_ptt_buffer -v`
Expected: `ModuleNotFoundError: No module named 'src.ptt_buffer'`.

- [ ] **Step 3: Implement `ClipAssembler` in `src/ptt_buffer.py`**

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m unittest tests.test_ptt_buffer -v`
Expected: all PASS.

- [ ] **Step 5: Rewire `PushToTalkRecorder` in `src/listen.py`**

- The recorder owns a `ClipAssembler`. `_audio_callback` calls `feed` under the lock and puts any returned clip on `self.clips`.
- `_on_press` calls `start()` (guarded by `is_recording`, for key auto-repeat) and queues any flushed clip.
- `_on_release` calls `stop()`.
- In `start()`, open `sd.InputStream` at 16000. On `sd.PortAudioError`, reopen at `int(sd.query_devices(kind="input")["default_samplerate"])`. Store the rate as `self.device_sr` and build the `ClipAssembler` with it.
- In `main()`, resample each clip with `resample(clip, recorder.device_sr, SAMPLE_RATE)` before the `MIN_CLIP_S` check. Print the fallback rate once at startup when it differs from 16000.

- [ ] **Step 6: Manual check on a machine with a microphone**

Run: `.venv/bin/python src/listen.py`. Hold Right Option and say "नमस्कार, मी मराठी बोलतो" while pressing and releasing quickly. Expected: the first and last words appear in full. Then tap the key twice within 0.25 s. Expected: two outputs, no crash.

- [ ] **Step 7: Commit**

```bash
git add src/ptt_buffer.py src/listen.py tests/test_ptt_buffer.py
git commit -m "feat(listen): pre/post-roll buffering and mic sample-rate fallback"
```

---

### Task 5: Trustworthy benchmark for the i5 target

**Files:**
- Create: `src/metrics.py`
- Modify: `benchmarks/run_benchmark.py`, `benchmarks/dataset_manifest.json`, `README.md`
- Test: `tests/test_metrics.py` (create)

**Interfaces:**
- Consumes: `normalize_marathi_text(text, lenient=False)` from Task 1.
- Produces: `score_transcript(reference: str, prediction: str) -> dict[str, float]` with keys `wer`, `cer`, `wer_lenient`, `cer_lenient`. It normalizes internally. An empty normalized reference scores 0.0 for all keys. An empty prediction against a non-empty reference scores 1.0 for all keys.

- [ ] **Step 1: Write the failing tests** in `tests/test_metrics.py`:

```python
def test_identical_is_zero(self):
    self.assertEqual(score_transcript("जे कोणी उंच भागात", "जे कोणी उंच भागात")["wer"], 0.0)

def test_lenient_forgives_vowel_length(self):
    s = score_transcript("पोलिस अधीक्षक", "पोलीस अधीक्षक")
    self.assertEqual(s["wer"], 0.5)
    self.assertEqual(s["wer_lenient"], 0.0)

def test_empty_prediction_scores_one(self):
    s = score_transcript("जे कोणी उंच भागात", "")
    self.assertEqual((s["wer"], s["cer"], s["wer_lenient"], s["cer_lenient"]), (1.0, 1.0, 1.0, 1.0))

def test_empty_reference_scores_zero(self):
    self.assertEqual(score_transcript("।", "काहीतरी")["wer"], 0.0)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m unittest tests.test_metrics -v`
Expected: `ModuleNotFoundError: No module named 'src.metrics'`.

- [ ] **Step 3: Implement `score_transcript` with `jiwer.wer` and `jiwer.cer`**

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m unittest tests.test_metrics -v`
Expected: all PASS.

- [ ] **Step 5: Update `benchmarks/run_benchmark.py`**

- Add `argparse` with `--models` (`nargs="+"`, default `["indicconformer-mr-int8"]`). Whisper-small runs at RTF 4.6 even on the M5, so it is too slow for the i5 by default.
- Replace the inline jiwer block with `score_transcript`. Add CSV columns `wer_lenient` and `cer_lenient` after `cer`.
- Print `platform.processor()` and `os.cpu_count()` at the start, so every results file records its hardware.

- [ ] **Step 6: Fix `benchmarks/dataset_manifest.json`**

- In `clean_0` and `noisy_0`, change the reference to `पोलीस अधीक्षक चंद्रशेखर सोळंकी यांनी सांगितले की आरोपी चेहरा झाकून घेऊन कोर्टात हजर झाला`. `पोलीस`, the single-word name `चंद्रशेखर`, and `सोळंकी` are the standard Marathi spellings.
- Delete the three `short_*` entries. They are hard crops that end mid-word, and their references are guesses. Re-cutting them at word boundaries belongs in the follow-up dataset plan.

- [ ] **Step 7: Run the full suite and the benchmark (needs `benchmarks/audio/` locally)**

Run: `.venv/bin/python -m unittest discover -s tests -v`. Expected: all PASS.
Run: `.venv/bin/python benchmarks/run_benchmark.py`. Expected: the CSV has 6 data rows and the new columns. Compare IndicConformer clean-row WER with the old CSV (`clean_0` was 0.2667, `clean_1` 0.1538, `clean_2` 0.0). If any clean row is worse, stop and report it rather than committing. Task 3 changed what the model hears.

- [ ] **Step 8: Update `README.md`**

Add a "Target hardware" section. It should say:
- The section 1 figures were measured on an M5 Mac.
- The deployment target is an i5 8th-gen U-series laptop with 16 GB RAM and CPU-only inference.
- How to re-run with `python benchmarks/run_benchmark.py` on the target.
- Whisper models are opt-in via `--models`.

Do not invent i5 numbers.

- [ ] **Step 9: Commit**

```bash
git add src/metrics.py tests/test_metrics.py benchmarks/run_benchmark.py benchmarks/dataset_manifest.json benchmarks/benchmark_results.csv README.md
git commit -m "fix(benchmark): correct references, drop cropped clips, add lenient metrics"
```

---

## Out of Scope (separate follow-up plans)

- FP32 and RNNT IndicConformer variants, with hotwords: one model plan, benchmarked on the i5.
- KenLM + `pyctcdecode` shallow fusion: needs raw logits through onnxruntime, so it is its own plan.
- RNNoise denoising and Silero VAD.
- A 50–200 clip test set (FLEURS `mr_in`, Common Voice `mr`, IndicVoices) and re-cut short clips.
- OpenVINO execution provider.
- OS-specific fixes (`brew` in README, Wayland `pynput`, PortAudio install), once the OS is known.
