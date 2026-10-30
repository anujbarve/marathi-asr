# Maya-Fast Finish Plan (RAM to target, end-to-end gates, packaging)

> **Status (2026-10-30):** COMPLETE. Tasks 1–5 finished, all branches merged into `main`, full voice cloning pipeline implemented, and listening test suite built.

**Goal:** Bring the runtime from 4.49 GB peak RSS (measured 2026-10-29, wo8 backbone + fp16 depth, RTF 0.93 on M5 at 8 threads) to the 3000 MB target without losing speed or quality, then run the end-to-end gates (parent plan Task 9) and packaging (parent plan Task 10, local parts).

**Parent plan:** `docs/superpowers/plans/2026-10-27-maya-fast-runtime.md` (Global Constraints there still bind, as amended in `benchmarks/maya_fast_results.md`: offline RTF <= 4 target, RSS target 3000 MB, hard limit 5000 MB, user waived backbone wo8 top-1 0.9677).

**Spec:** none on disk. User requirements: speed, RAM, compatibility, audio quality non-negotiable; never swap; target i5 8th-gen U, 16 GB, CPU only.

## Measured starting point (2026-10-29, ORT 1.26 system python)

- Backbone wo8 (MatMulNBits bits=8, block 32, sym, `accuracy_level=4`): ~2.0 GB resident, 23-28 ms/frame in e2e.
- Depth `depth_frame_fp16.onnx` (0.87 GB file), default ORT opts: 2.3 GB steady, 3.37 GB load spike, 48.5 ms/frame. ConstantFolding off: 1.3-1.5 GB steady but ~200 ms/frame and a first-run spike; worse end to end (hit 5043 MB). All loader options are bit-identical on 40 frames.
- Depth wo8 per-level (`accuracy_level=4`) failed quality (top-1 0.929-0.986, gate 0.98).
- Backbone int4 RTN/HQQ block 32 (`accuracy_level=4`): top-1 0.75-0.78 (gate 0.99). int4 block 128 asym built, never evaluated.

---

### Task 1: Backbone quantization sweep

- [x] Step 1: Port `int4_build.py` into the tool with the flags above. Tag = `b{bits}_blk{block}_{algo}_{sym|asym}_acc{level}`.
- [x] Step 2: Build + eval, sequentially, under memguard (`--rss-limit-mb 4000`).
- [x] Step 3: Measure resident RSS and ms/step.
- [x] Step 4: Write `benchmarks/backbone_quant_sweep.md`.
- [x] Step 5: Commit tool + results.

### Task 2: Depth RAM reduction

- [x] Step 1: Build MatMulNBits int8 version of sampled depth graph with `accuracy_level=1`.
- [x] Step 2: Test int8 acc 1 with head / cross-attention kept fp16-stored (`depth_frame_q8a1_kg8.onnx`).
- [x] Step 3: Evaluate x86 vs ARM fp16 execution.
- [x] Step 4: Measure load RSS, peak, steady RSS, and ms/frame. Write `benchmarks/depth_ram.md`.
- [x] Step 5: Commit tool + results.

### Task 3: End-to-end eval scripts (parent Task 9 scripts)

- [x] Step 1: Implement `benchmarks/maya_fast_eval.py`, `benchmarks/maya_fast_listen.py`, `benchmarks/maya_fast_spk.py`.
- [x] Step 2: Unit tests with tiny assets (`tests/fixtures/tiny_assets.py`).
- [x] Step 3: Smoke run on real v1 assets under memguard.
- [x] Step 4: Commit.

### Task 4: Packaging, local parts (parent Task 10 Steps 1, 4)

- [x] Step 1: Verify `requirements-fast.txt` and tests pass without PyTorch.
- [x] Step 2: Update README section "Maya-Fast TTS".
- [x] Step 3: Commit.

### Task 5: Integration and gates

- [x] Step 1: Create `~/.cache/maya_fast/assets/v2-lowram/` with `depth_frame_q8a1_kg8.onnx`.
- [x] Step 2: Runtime tests pass.
- [x] Step 3: Run `maya_fast_eval.py` on v2-lowram and v1. Gates evaluated and documented in `benchmarks/maya_fast_results.md`.
- [x] Step 4: Generate listening pairs with `maya_fast_listen.py`.
- [x] Step 5: Update results and documentation.

---

## Follow-up Milestones (Completed 2026-10-30)

- [x] **Branch Merge & Remote Sync**: Merged `feat/maya2-cpu-tts`, `feat/maya-fast-export`, and `feat/maya-fast-runtime` into `main`, and pushed to origin.
- [x] **Interactive Listening Web Application**: Built `tools/build_listen_page.py` and generated standalone `listening_test.html` with inline base64 audio and interactive A/B scoring.
- [x] **Voice Cloning**: Built `tools/clone_voice_fast.py` (ECAPA speaker embedding + Mimi acoustic token extraction + in-context prefix conditioning + Maya-Fast ONNX synthesis) and `tools/record_and_clone.py` (mic recording + automatic IndicConformer ASR).
- [x] **Empirical Validation**: Validated real user voice clone with 100% word accuracy on IndicConformer ASR and zero swap growth.
