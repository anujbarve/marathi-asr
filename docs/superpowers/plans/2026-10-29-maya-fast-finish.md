# Maya-Fast Finish Plan (RAM to target, end-to-end gates, packaging)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Bring the runtime from 4.49 GB peak RSS (measured 2026-10-29, wo8 backbone + fp16 depth, RTF 0.93 on M5 at 8 threads) to the 3000 MB target without losing speed or quality, then run the end-to-end gates (parent plan Task 9) and packaging (parent plan Task 10, local parts).

**Parent plan:** `docs/superpowers/plans/2026-10-27-maya-fast-runtime.md` (Global Constraints there still bind, as amended in `benchmarks/maya_fast_results.md`: offline RTF <= 4 target, RSS target 3000 MB, hard limit 5000 MB, user waived backbone wo8 top-1 0.9677).

**Spec:** none on disk. User requirements: speed, RAM, compatibility, audio quality non-negotiable; never swap; target i5 8th-gen U, 16 GB, CPU only.

## Measured starting point (2026-10-29, ORT 1.26 system python)

- Backbone wo8 (MatMulNBits bits=8, block 32, sym, `accuracy_level=4`): ~2.0 GB resident, 23-28 ms/frame in e2e.
- Depth `depth_frame_fp16.onnx` (0.87 GB file), default ORT opts: 2.3 GB steady, 3.37 GB load spike, 48.5 ms/frame. ConstantFolding off: 1.3-1.5 GB steady but ~200 ms/frame and a first-run spike; worse end to end (hit 5043 MB). All loader options are bit-identical on 40 frames.
- Depth wo8 per-level (`accuracy_level=4`) failed quality (top-1 0.929-0.986, gate 0.98).
- Backbone int4 RTN/HQQ block 32 (`accuracy_level=4`): top-1 0.75-0.78 (gate 0.99). int4 block 128 asym built, never evaluated.

**Hypothesis to test first:** every quantized build used `accuracy_level=4`, which quantizes activations to int8 inside MatMulNBits. Weight-only error at 8 bits is small; the top-1 losses may come from activation quantization. `accuracy_level=1` (fp32 compute) keeps weights compressed in RAM.

## Global Constraints (this plan)

- Every command that loads model weights runs under `tools/memguard.py` (exclusive lock `~/.cache/maya_fast/model.lock`; one model process at a time across agents). Check `sysctl vm.swapusage` before and after. Swap growth > 64 MB fails the run. Default guard: `--rss-limit-mb 5000 --min-free-mb 5500`.
- Write build outputs only under `~/.cache/maya_fast/backbone_build/<tag>/` or `~/.cache/maya_fast/depth_build/<tag>/`. Never modify `~/.cache/maya_fast/assets/v1/` (only Task 5 creates `assets/v2/`).
- Quality gates (teacher-forced, existing `tools/maya_export/eval_teacher_forced.py`): backbone top-1 >= 0.99 and mean KL <= 0.01 vs fp32; depth per level top-1 >= 0.98 and KL <= 0.02. Never relax a threshold; never report a number not produced by a run.
- Never commit weights, ONNX, audio, goldens.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No pushes.
- Export work uses the export worktree's `.venv-tts` and `PYTHONPATH=<worktree>:<scratchpad>/shim` (see `run_chain2.sh` in the scratchpad for the exact pattern). Scratchpad: `/private/tmp/claude-501/-Users-anujbarve-Documents-marathi-helper/8632c127-d21f-465e-a3e2-41d3cae7fd02/scratchpad`.

---

### Task 1: Backbone quantization sweep

**Worktree / branch:** `<scratchpad>/maya-int4`, branch `int4-export-exp`.
**Files:** Create `tools/maya_export/quant_sweep_backbone.py` (port of `<scratchpad>/int4_build.py` into the repo, with `--bits {4,8}`, `--block`, `--asym`, `--accuracy-level`, `--algo {rtn,hqq}`); results file `benchmarks/backbone_quant_sweep.md`.

- [ ] Step 1: Port `int4_build.py` into the tool with the flags above. Tag = `b{bits}_blk{block}_{algo}_{sym|asym}_acc{level}`.
- [ ] Step 2: Build + eval, sequentially, under memguard (`--rss-limit-mb 4000`): (a) bits 8 block 32 sym acc 1; (b) bits 4 block 128 rtn asym acc 4 (existing build `int4_b128_rtn_asym` may be reused for eval); (c) bits 4 block 32 rtn asym acc 1; (d) bits 4 block 128 rtn asym acc 1. Stop early on a variant only if its build fails twice.
- [ ] Step 3: For every variant that passes the gate, and for the current wo8 baseline, measure backbone-alone resident RSS after load and median ms/step at past_len 32/256 (T=1, 100 steps) with the runtime `Backbone` class from `/Users/anujbarve/Documents/maya-fast-loader/src/maya_fast/backbone.py` (merge the per-layer graphs into a single `backbone_<tag>.onnx` the same way `wo8` was merged).
- [ ] Step 4: Write `benchmarks/backbone_quant_sweep.md`: table of tag, top-1, KL, gate, RSS, ms/step. Recommendation: smallest-RSS variant that passes and is <= 1.3x wo8 ms/step; else best-quality variant with a note.
- [ ] Step 5: Commit tool + results.

### Task 2: Depth RAM reduction

**Worktree / branch:** new worktree `<scratchpad>/maya-depth` on new branch `depth-ram-exp` from `int4-export-exp` at `a484ef0`.
**Files:** Create `tools/maya_export/quant_depth_frame.py`; results `benchmarks/depth_ram.md`.

Goal: depth steady RSS <= 1000 MB and load spike <= 1500 MB, ms/frame <= 80, quality gate pass for all 15 levels (or codes bit-identical to `depth_build/loader_bench/base.codes.npy` on the 40-frame bench, which implies it).

- [ ] Step 1: Build a MatMulNBits int8 (block 32, sym) version of the sampled depth graph with `accuracy_level=1` from the fp32 depth source (`depth_build/fp32` or an fp32 export of `depth_frame`). Gate it with the depth teacher-forced eval (all 15 levels) and with the 40-frame codes bench (`<job tmp>/bench_depth.py`-style: seeds 1234+i).
- [ ] Step 2: If Step 1 fails quality, try int8 acc 1 with the head / cross-attention kept fp16-stored, then report.
- [ ] Step 3: Independently, try an fp16 graph where weights are not converted at load: e.g. pre-cast is avoided by keeping MatMul in fp16 only if CPU EP supports it on x86 (check `onnxruntime` CPU kernels for fp16 MatMul on x86; the i5 target has no fp16 arithmetic). Record result; drop if x86 unsupported.
- [ ] Step 4: For each candidate: load RSS, peak during first 5 frames, steady RSS, median ms/frame (30 runs, 8 threads and 4 threads). Write `benchmarks/depth_ram.md` with recommendation.
- [ ] Step 5: Commit tool + results.

### Task 3: End-to-end eval scripts (parent Task 9 scripts)

**Worktree / branch:** `/Users/anujbarve/Documents/maya-fast-loader`, `feat/maya-fast-runtime`.
**Files:** Create `benchmarks/maya_fast_eval.py`, `benchmarks/maya_fast_listen.py`; `benchmarks/maya_fast_spk.py` only if speechbrain ECAPA is available in some local env (else record that in the report).

- [ ] Step 1: Implement per parent plan Task 9 Interfaces. `maya_fast_eval.py` takes `--assets DIR`, `--threads`, `--seeds`, `--limit N`; writes WAVs to `~/.cache/maya_fast/eval/<run>/` and a JSON + markdown table (CER/WER per category, RTF, peak RSS via `resource.getrusage` max RSS, ms/frame per phase). TTS first, ASR on saved WAVs after TTS process exits (separate process or explicit `del` + gc, so both models never sit in RAM).
- [ ] Step 2: Unit tests with tiny assets (`tests/fixtures/tiny_assets.py`) for the table/aggregation code, no real models.
- [ ] Step 3: Smoke run on real v1 assets, `--limit 3 --seeds 0`, under memguard. Report numbers (not gated here).
- [ ] Step 4: Commit.

### Task 4: Packaging, local parts (parent Task 10 Steps 1, 4)

**Worktree / branch:** `/Users/anujbarve/Documents/maya-fast-loader`, `feat/maya-fast-runtime`. Runs after Task 3 (same worktree).
**Files:** `requirements-fast.txt` (exists; verify contents per parent plan), create `.github/workflows/maya-fast.yml`, modify `README.md`.

- [ ] Step 1: Workflow per parent Task 10. Fresh-venv check in `$CLAUDE_JOB_DIR/tmp/mf` with only `requirements-fast.txt`; `python -m unittest discover -s tests -p "test_maya_fast_*.py"` passes, no torch.
- [ ] Step 2: README section "Maya-Fast TTS" per parent Task 10 Step 4. Numbers labelled with machine and date; placeholders marked TBD for the Task 5 numbers.
- [ ] Step 3: Commit. No push.

### Task 5: Integration and gates

Depends on Tasks 1-4.

- [ ] Step 1: Create `~/.cache/maya_fast/assets/v2/` = v1 files (hard links) with backbone and depth replaced by the Task 1 and Task 2 recommendations; regenerate manifest hashes (finalize step). Engine picks graph file names from the manifest or a constructor arg (no hard-coded `backbone_wo8.onnx` if v2 names differ).
- [ ] Step 2: Runtime tests pass.
- [ ] Step 3: Run `maya_fast_eval.py` on v2 and v1, full test set, seeds {0,1,2}, under memguard. Gates: CER <= v1 CER + 0.01; peak RSS <= 3000 MB; RTF recorded.
- [ ] Step 4: Generate listening pairs (v1 vs v2) with `maya_fast_listen.py`; the user rates later.
- [ ] Step 5: Update `benchmarks/maya_fast_results.md` and README numbers; commit.
