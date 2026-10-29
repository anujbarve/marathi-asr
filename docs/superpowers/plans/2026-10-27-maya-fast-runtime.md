# Maya-Fast Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the PyTorch Maya2 serving path with a torch-free ONNX Runtime engine that speaks Marathi in real time on this Mac, stays under 3 GB RAM, runs unchanged on macOS/Linux/Windows (arm64 and x86-64), and is measurably no worse in audio quality than the upstream PyTorch model.

**Architecture:** Two halves. An offline export toolchain (`tools/maya_export/`, PyTorch, `.venv-tts`) turns the upstream checkpoint into versioned assets, streaming one layer or one module at a time so it never needs the whole model in RAM. A runtime package (`src/maya_fast/`, numpy + onnxruntime + tokenizers only) runs two ONNX calls per audio frame: a fused 28-layer backbone step (weight-only int8, `accuracy_level=4`, KV cache in/out) and a fused depth-frame graph that samples all 15 deeper codebooks in one call (unrolled levels, KV-cached band for levels 9-15, Gumbel-max sampling from noise passed in). Mimi decodes the frames in a third graph. Embeddings are a bf16 memory map, so the 1.6 GB table costs only the rows used. Every exported component is checked against the PyTorch reference before it is quantized, and every quantized component passes teacher-forced gates before the end-to-end gates run.

**Tech Stack:** Runtime: Python 3.10-3.13, `numpy>=2.0`, `onnxruntime==1.31.*` (CPU EP, `com.microsoft.MatMulNBits`), `tokenizers>=0.22,<0.23`, stdlib `wave`. Export and eval: `.venv-tts` (torch 2.8.0, transformers 5.6.2, upstream `maya3` at the SHA pinned in `requirements-tts.txt`), `onnx`, `onnx-ir`, `psutil`. ASR scoring: `src/models.py` `IndicConformerONNXASR`, `src/metrics.py` `score_transcript`.

**Spec:** No spec file. Requirements from the user, 2026-10-27: "we can rewrite the whole architecture but speed, ram, compatibility and audio quality is non negotiable"; never trigger swap (memory `no-swap`). Measured starting point: `benchmarks/maya_fast_e2e.py` and https://github.com/anujbarve/maya-fast README (ORT int8 backbone RTF 5.3-5.6, depth 0.35 s/frame, peak RSS 5.3 GB).

## Global Constraints

- **No swap, enforced.** Every command that loads a model runs under `tools/memguard.py` (Task 1). Swap growth over 64 MB during a run fails the run. One model process at a time.
- **RAM:** runtime peak RSS ≤ 3000 MB for any test-set sentence. Export/eval process limits: light ≤ 6000 MB, backbone export ≤ 4000 MB, depth export ≤ 6000 MB, Mimi export ≤ 2000 MB, heavy reference ≤ 9000 MB with ≥ 10000 MB free before start.
- **Speed (this Mac, Apple M5, 8 threads):** mean RTF ≤ 1.0 over the test set. Per-frame budgets (one frame = 80 ms of audio): backbone step ≤ 35 ms, depth frame ≤ 35 ms, Mimi ≤ 8 ms per frame. i5 8th-gen U: must run correctly within the RAM limit; RTF is measured and reported, not gated.
- **Quality, teacher-forced (proposed thresholds, user confirms at plan review):** backbone cb0 vs fp32 reference: top-1 agreement ≥ 0.99, mean KL ≤ 0.01 nats. Depth per level vs PyTorch fp32: top-1 ≥ 0.98, mean KL ≤ 0.02 nats. fp32 exports vs PyTorch: max abs error ≤ 1e-4 (Mimi wav), max relative error ≤ 1e-4 (backbone hidden, depth logits).
- **Quality, end to end:** over test set × seeds {0, 1, 2}: mean round-trip CER ≤ reference CER + 0.01; mean ECAPA speaker cosine ≥ reference − 0.02; user blind A/B: candidate preferred or tied in ≥ 45% of pairs. Reference = upstream PyTorch bf16 path (Task 3).
- **A missed gate stops the plan.** Report the measured numbers. Never relax a threshold silently; never write a number not produced by a run of the config it describes.
- **Sampling is upstream's:** `q_serve=16`, `temp=0.7`, `top_k=64`, stop `"controlled"` with `tau=0.35`, `hazard=0.6`, `cfg_scale=0`. Gumbel-max replaces `torch.multinomial` (same distribution, different random stream).
- **Runtime has no torch import.** CI fails if `import torch` appears under `src/maya_fast/`.
- **ORT session options everywhere:** `intra_op_num_threads` = requested threads (default `min(8, physical cores)`), `session.intra_op.allow_spinning=0`, `enable_cpu_mem_arena=False`, `ORT_ENABLE_ALL`.
- **Artifacts live outside the repo** in `MAYA_FAST_WORK` (default `~/.cache/maya_fast`). Never commit weights, ONNX files, audio or goldens. Maya2 weights are Apache-2.0, Mimi is CC-BY-4.0: the README states both.
- **Tests:** `unittest`, same style as `tests/test_metrics.py`. Runtime tests run in `.venv`; export tests in `.venv-tts`.
- **Branch:** `feat/maya-fast-runtime` from `main`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Long input** (a paragraph of several sentences): expected to be split with upstream `split_text`, each piece a fresh KV session, audio concatenated, KV length never above one piece's cap. Test: `test_long_text_is_split_and_kv_bounded` (Task 8).
2. **Empty or punctuation-only text:** expected `ValueError("no speakable text")`, not a silent empty WAV or a hang. Test: `test_empty_text_raises` (Task 8).
3. **Model never emits `<aeos>`:** generation must stop at the frame cap with reason `"cap"` and return audio. Test: `test_cap_stops_generation` (Task 8).
4. **Low-end machine settings** (1 thread, or more threads than cores): engine runs and output is identical for the same seed. Test: `test_thread_count_does_not_change_output` (Task 8).
5. **Partial or corrupt asset download:** load fails with an error naming the bad file. Test: `test_manifest_sha_mismatch_raises` (Task 4).

---

### Task 1: Memory guard

**Files:**
- Create: `tools/memguard.py`
- Test: `tests/test_memguard.py`

**Interfaces:**
- Produces: `run_guarded(cmd: list[str], *, rss_limit_mb: int, min_free_mb: int, max_swap_growth_mb: int = 64, poll_s: float = 0.2) -> GuardResult`. `GuardResult` is a dataclass: `returncode: int`, `peak_rss_mb: float`, `swap_before_mb: float`, `swap_after_mb: float`, `killed: bool`, `reason: str | None`. RSS is the sum over the child and its descendants (`psutil`). Refuses to start (`reason="insufficient free memory"`, `returncode=-1`) when `psutil.virtual_memory().available` is below `min_free_mb`. Kills the tree when RSS passes `rss_limit_mb` (`reason="rss limit"`). Sets `reason="swap growth"` when swap used grew more than the limit.
- Produces: CLI `python tools/memguard.py --rss-limit-mb N --min-free-mb M -- <cmd...>`; prints one line `MEMGUARD peak_rss_mb=... swap_delta_mb=... reason=...`; exits non-zero on any violation.

- [ ] **Step 1: Branch.** `git checkout main && git checkout -b feat/maya-fast-runtime`.
- [ ] **Step 2: Write failing tests** `test_kills_on_rss_limit` (child allocates 300 MB with `bytearray`, limit 100 → `killed and reason == "rss limit"`), `test_refuses_when_not_enough_free` (`min_free_mb` = total RAM + 1 → `returncode == -1`), `test_clean_run_reports_peak` (child allocates 50 MB, limit 500 → `returncode == 0`, `peak_rss_mb >= 40`).
- [ ] **Step 3: Run** `.venv/bin/python -m unittest tests/test_memguard.py -v`. Expected: FAIL, module not found.
- [ ] **Step 4: Implement `tools/memguard.py`.**
- [ ] **Step 5: Run the tests.** Expected: 3 PASS.
- [ ] **Step 6: Commit** `feat(tools): add memory guard that blocks swap during model runs`.

---

### Task 2: Test set and light goldens

**Files:**
- Create: `benchmarks/tts_testset_mr.json`
- Create: `tools/maya_export/__init__.py`, `tools/maya_export/common.py`, `tools/maya_export/golden_light.py`
- Test: `tests/export/test_common.py`

**Interfaces:**
- Produces: test set JSON list of `{"id": str, "category": str, "text": str}`; categories `conversational`, `long`, `proper_noun`, `number`, `code_switch`; 5 per category. Includes "नमस्कार, मी तुमची मदत करण्यासाठी इथे आहे." in `conversational`.
- Produces in `common.py`: `work_dir() -> Path` (`MAYA_FAST_WORK` or `~/.cache/maya_fast`), `ckpt_paths() -> dict` (wraps `maya3.hub.resolve("maya-research/maya2-native")`), `iter_ckpt_tensors(skip_prefixes: tuple[str, ...]) -> Iterator[tuple[str, Tensor]]` (one tensor at a time, `model.` prefix stripped), `build_light_model() -> TTSModelV2` (meta-device build with zero backbone decoder layers and per-tensor assignment, moved from `benchmarks/maya_fast_e2e.py` `stub_lm` + `load_weights`).
- Produces: `golden_light.py` writes `work_dir()/golden/light.npz` and `light.json`. For every test-set sentence × every voice in `voices.json`: `token_ids`, `kinds`, `prompt_emb` (fp32, from `S.prompt_from_example` on the light model), `n_prompt`. Per voice: `mem` (`depth.spk_memory` output), `ecapa`. Per sentence: `ef`, `er`, `cap`, `minimum` exactly as `benchmarks/tts_quant_spike.py:174-181` computes them, and `split_text(text, "mr", prior, chunk_s)` with upstream's default `chunk_s`. Also `audio_in_frames` for 8 fixed random code rows (seed 0) and `aeos` (the `audio_in.aeos(1)` vector).

- [ ] **Step 1: Write the test set file.** Validate: `python -c "import json,collections;d=json.load(open('benchmarks/tts_testset_mr.json'));print(collections.Counter(x['category'] for x in d))"` → 5 per category.
- [ ] **Step 2: Write failing test** `test_iter_ckpt_tensors_skips_prefix` (tiny safetensors file in a temp dir with keys `backbone.lm.model.layers.0.w` and `depth.x`; skip `backbone.lm.model.layers.` → only `depth.x` yielded, once).
- [ ] **Step 3: Run** `.venv-tts/bin/python -m unittest tests/export/test_common.py -v` → FAIL.
- [ ] **Step 4: Implement `common.py`.** Then the test → PASS.
- [ ] **Step 5: Implement `golden_light.py` and run it guarded:** `.venv/bin/python tools/memguard.py --rss-limit-mb 6000 --min-free-mb 7000 -- .venv-tts/bin/python -m tools.maya_export.golden_light`. Expected: `MEMGUARD ... reason=None`, both files written.
- [ ] **Step 6: Commit** test set, `tools/maya_export/`, test. Goldens stay in `MAYA_FAST_WORK`.

---

### Task 3: Heavy reference corpus (upstream PyTorch, bf16)

**Files:**
- Create: `tools/maya_export/golden_reference.py`

**Interfaces:**
- Consumes: test set (Task 2), the upstream path in `benchmarks/tts_quant_spike.py` with `--quant none --dtype bf16` (depth fp32, as upstream serve does).
- Produces: per sentence × seed in {0, 1, 2}, voice `Charvi`: `work_dir()/golden/ref/{id}_s{seed}.npz` with `codes` (16, F) int64, `wav` (fp32, 24 kHz), `reason`, `h_frames` (F, 2048) fp32 (the backbone hidden fed to depth for each frame), `cb0_logits` (F, 2049) fp16, `depth_logits` (F, 15, 2048) fp16 (teacher-forced: `depth._run` per level on `h_frames` and the sampled `codes`), `seq_emb` (the full backbone input sequence: prompt + fed-back frames) fp32.

- [ ] **Step 1: Implement** by wrapping `ServeSession._step` and `depth_generate` to record the values above. One process generates the whole corpus (load once).
- [ ] **Step 2: Ask the user to close other apps.** This step needs ≥ 10 GB free.
- [ ] **Step 3: Run guarded:** `.venv/bin/python tools/memguard.py --rss-limit-mb 9000 --min-free-mb 10000 -- .venv-tts/bin/python -m tools.maya_export.golden_reference`. Expected: 75 files, `MEMGUARD reason=None`. If refused for free memory, stop and tell the user. Do not lower the limit.
- [ ] **Step 4: Commit** the script.

---

### Task 4: Prompt assets, manifest, numpy prompt builder and controls

**Files:**
- Create: `tools/maya_export/export_prompt_assets.py`, `tools/maya_export/manifest.py`
- Create: `src/maya_fast/__init__.py`, `src/maya_fast/assets.py`, `src/maya_fast/embed.py`, `src/maya_fast/prompt.py`, `src/maya_fast/control.py`
- Test: `tests/test_maya_fast_assets.py`, `tests/test_maya_fast_prompt.py`, `tests/test_maya_fast_control.py`

**Interfaces:**
- Produces assets in `work_dir()/assets/v1/`: `tokenizer.json` (the extended tokenizer from `extend_tokenizer`, saved fast), `embed_bf16.bin` (raw little-endian uint16, row-major, shape in manifest), `prompt.npz` (`anchor_fc1_w`, `anchor_fc1_b`, `anchor_fc2_w`, `anchor_fc2_b`, `anchor_null`, `aeos`, plus every `audio_in` parameter under `audio_in/<name>`), `voices.json` + `voices.npz` (per voice: mode, prefix data needed by the builder, `ecapa`, `mem`), `controls.json` (`text_frames_prior`, guard and duration-prior settings, sampling constants), `manifest.json` `{"version": 1, "files": {name: sha256}, "shapes": {...}}`.
- Produces: `load_manifest(assets_dir: Path, verify: bool = True) -> dict`; raises `AssetError` whose message contains the file name on a missing file or sha256 mismatch.
- Produces: `EmbeddingTable(path: Path, rows: int, dim: int)` with `lookup(ids: np.ndarray) -> np.ndarray` (float32, `(len(ids), dim)`). bf16 to fp32 is exact: `(memmap[ids].astype(np.uint32) << 16).view(np.float32)`.
- Produces: `AudioIn(params: dict)` with `frames(codes: np.ndarray) -> np.ndarray` (`(N, 16)` int64 → `(N, 2048)` float32) and `aeos() -> np.ndarray` (2048,), ported from `tts/model/audio_io.py:AudioInputEmbed`.
- Produces: `build_prompt(text: str, voice: str, assets: "PromptAssets") -> Prompt`, dataclass `Prompt(token_ids: np.ndarray, embeds: np.ndarray, n_prompt: int)`. Port of the inference path only of `maya3.prompt` `DEFAULT_BUILDER.prefix` / `.example`, `tts.data.packing_v2` `PackerV2` + `collate_v2`, `tts.model.load_s3.truncate_prompt_batch`, and `TTSModelV2.compose_inputs` + `_add_speaker_anchor` + `SpeakerAnchor` (eval mode: no dropout).
- Produces in `control.py`: `expected_frames(prior: dict, lang: str, nchar: int) -> tuple[float, float]`, `frame_cap(text: str, lang: str, ef: float, er: float, controls: dict) -> int`, `min_frames(ef: float) -> int`, `split_text(text: str, lang: str, prior: dict, chunk_s: float) -> list[str]`, and `StopPolicy(min_frames: int, tau: float = 0.35, hazard: float = 0.6)` with `check(t: int, p: np.ndarray) -> str | None` returning `"aeos_argmax"`, `"aeos_tau"`, `"aeos_hazard"` or `None`, mirroring the `"controlled"` branch of `ServeSession.speak`. Also `sample_cb0(p: np.ndarray, noise: np.ndarray) -> int` (aeos zeroed, Gumbel-max over the rest).

- [ ] **Step 1: Write failing tests.** `test_manifest_sha_mismatch_raises` (temp dir, manifest naming `a.bin` with a wrong hash → `AssetError` with `"a.bin"` in the message). `test_bf16_lookup_exact` (write 3 known float32 values that are exact in bf16, look up → equal). `test_prompt_matches_golden` (every sentence × voice in `light.npz`: `token_ids` equal, `max|embeds - prompt_emb| <= 1e-5`; skipped with a message when goldens are absent). `test_audio_in_matches_golden` (≤ 1e-5). `test_controls_match_golden` (`ef`, `er`, `cap`, `minimum`, `split_text` equal to `light.json`). `test_stop_policy_reasons` (hand-built `p` vectors trigger each reason at the right `t`; nothing fires before `min_frames`).
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests/test_maya_fast_assets.py tests/test_maya_fast_prompt.py tests/test_maya_fast_control.py -v` → FAIL.
- [ ] **Step 3: Implement `export_prompt_assets.py` and `manifest.py`.** Run guarded (`--rss-limit-mb 6000 --min-free-mb 7000`).
- [ ] **Step 4: Implement `assets.py`, `embed.py`, `prompt.py`, `control.py`.** No torch import.
- [ ] **Step 5: Run the tests.** Expected: all PASS. A golden mismatch is a port bug: fix the port, never the tolerance.
- [ ] **Step 6: Commit.**

---

### Task 5: Fused backbone graph

**Files:**
- Create: `tools/maya_export/export_backbone.py`, `tools/maya_export/eval_teacher_forced.py`
- Create: `src/maya_fast/ort_opts.py`, `src/maya_fast/backbone.py`
- Test: `tests/export/test_backbone_export.py`, `tests/test_maya_fast_backbone.py`

**Interfaces:**
- Consumes: `LayerKV`, `masks`, `rope` from `benchmarks/onnx_layer_spike.py` (move them into `export_backbone.py`; the spike files stay as they are).
- Produces: `export_backbone.py --precision fp32|wo8`. Per layer: build one `Qwen3DecoderLayer(cfg, layer_idx=0)`, load its tensors, export (opset 17, `dynamo=False`), quantize for `wo8` with `MatMulNBitsQuantizer(bits=8, block_size=32, is_symmetric=True, accuracy_level=4)`, free it. Then merge the 28 layer graphs (prefixed names, `onnx.compose`), append final RMSNorm and `cb0_head` (both fp32), save with external data as `backbone_{precision}.onnx`. Inputs: `embeds (1, T, 2048)`, `cos (1, T, 128)`, `sin (1, T, 128)`, `mask (1, 1, T, P+T)`, `past_k_{i}`, `past_v_{i}` (`(1, 8, P, 128)`, i = 0..27). Outputs: `hidden (1, T, 2048)` (after final norm), `cb0_logits (1, T, 2049)`, `present_k_{i}`, `present_v_{i}`. `fp32` is for evaluation only.
- Produces: `ort_opts.session_options(threads: int) -> ort.SessionOptions` (Global Constraints values).
- Produces: `Backbone(path: Path, threads: int)` with `reset()`, `step(embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (last-position `hidden (2048,)`, `cb0_logits (2049,)`), and `past_len: int`. `cos`/`sin`/`mask` computed in numpy once per step from `past_len`; the rotary table comes from the manifest config (`rope_theta`, `head_dim`).
- Produces: `eval_teacher_forced.py backbone --candidate wo8` → `{"top1": float, "kl_mean": float, "frames": int}`. Reference = fp32 per-layer sessions run one layer at a time over each `seq_emb` from Task 3 (one layer resident). Candidate = fused graph. Compared at the generating positions only.

- [ ] **Step 1: Write failing tests.** `test_fused_matches_layers` (export tests: fp32 fused graph vs the 28 per-layer fp32 graphs chained, on 8 prefill + 4 decode random embeds → max rel error ≤ 1e-4). `test_backbone_step_kv_grows` (runtime: tiny synthetic 2-layer graph built with `onnx.helper` in `tests/fixtures/tiny_assets.py`; `past_len` goes 5 → 6 → 7 and `reset()` returns it to 0).
- [ ] **Step 2: Run both test files** → FAIL.
- [ ] **Step 3: Implement the export, `ort_opts.py`, `backbone.py`, and `tests/fixtures/tiny_assets.py`** (`make_tiny_assets(dir: Path) -> Path`: a complete tiny asset set with the same file names, input/output names and manifest as real assets, random weights, `d=16`, vocab 64, 2 backbone layers; also used by Tasks 6-8).
- [ ] **Step 4: Run the tests** → PASS.
- [ ] **Step 5: Export both precisions guarded** (`--rss-limit-mb 4000 --min-free-mb 5000`).
- [ ] **Step 6: Quality gate.** Run `eval_teacher_forced.py backbone --candidate wo8` guarded (`--rss-limit-mb 4000`). Pass: `top1 >= 0.99` and `kl_mean <= 0.01`. Write the numbers into `benchmarks/maya_fast_results.md`.
- [ ] **Step 7: Speed gate.** Time `Backbone.step` (T = 1) at `past_len` 32, 256 and 768 with 8 threads, 200 steps each, median. Pass: median ≤ 35 ms at 256 and the 768 median ≤ 1.25 × the 32 median. A miss on the second condition means KV concat cost: stop and report for a plan amendment (shared-buffer `GroupQueryAttention`). Record the numbers.
- [ ] **Step 8: Commit.**

---

### Task 6: Depth-frame graph

**Files:**
- Create: `tools/maya_export/depth_frame.py`, `tools/maya_export/export_depth.py`
- Create: `src/maya_fast/sampling.py`, `src/maya_fast/depth.py`
- Test: `tests/export/test_depth_frame.py`, `tests/test_maya_fast_sampling.py`, `tests/test_maya_fast_depth.py`

**Interfaces:**
- Produces: `DepthFrame(depth: DepthV2, q: int = 16, temp: float = 0.7, top_k: int = 64)`, an `nn.Module` that reuses DepthV2's weights. `forward(h, cb0, mem, noise) -> codes`: `h (1, 2048)`, `cb0 (1,)` int64, `mem (1, R, 1024)`, `noise (15, 2048)` float32 Gumbel noise → `codes (16,)` int64 (`cb0` first). Levels 1-15 unrolled. `GROUP_OF_LEVEL` sends levels 1-8 to groups 0-7 (each runs its own full prefix: no reuse possible) and levels 9-15 to group 8. For group 8, cache self-attention K/V inside the graph: level 9 runs its 10-position prefix once, and each later level adds one position. This is exact because the prefix is causal and position-wise after attention (`DepthV2._run_at` docstring). Speaker cross-attention K/V comes from `mem` once per call. Sampling per level: `argmax(where(rank < top_k, logits / temp + noise[level-1], -inf))`.
- Produces: `DepthTF(depth)` with `forward(h, codes, mem) -> logits (15, 2048)`, the teacher-forced twin, eval only.
- Produces: `export_depth.py --precision fp32|wo8` → `depth_frame_{p}.onnx`, `depth_tf_{p}.onnx` (wo8 = same MatMulNBits settings as Task 5; `heads`, `in_proj`, `code_up` included; embeddings and norms stay fp32). Depth weights load via `iter_ckpt_tensors` into a meta-built `DepthV2`.
- Produces: `sampling.gumbel(rng: np.random.Generator, shape: tuple) -> np.ndarray` (float32, `-log(-log(U))` with U clipped to `[1e-10, 1 - 1e-7]`), `sampling.topk_gumbel_argmax(logits, temp, top_k, noise) -> int` (the numpy reference of the graph's rule).
- Produces: `Depth(path: Path, threads: int)` with `frame(h: np.ndarray, cb0: int, mem: np.ndarray, noise: np.ndarray) -> np.ndarray` (16,) int64.
- Produces: `eval_teacher_forced.py depth --candidate wo8` → `{"top1_per_level": list[float], "kl_per_level": list[float]}` against Task 3 `depth_logits`.

- [ ] **Step 1: Write failing tests.** `test_topk_gumbel_matches_multinomial` (numpy: fixed 2048-logit vector, 40 000 draws, chi-square p > 0.01 against `softmax(topk_masked(logits) / 0.7)`). `test_depth_tf_matches_depthv2` (export test: random weights, `DepthTF` logits vs `DepthV2._run` + `heads` per level, max rel error ≤ 1e-5). `test_depth_frame_consistent_with_tf` (feed `DepthFrame`'s own sampled codes to `DepthTF`; each level's argmax of `logits / 0.7 + noise` equals the emitted code). `test_depth_runtime_shapes` (tiny assets: `frame()` returns 16 codes in `[0, 2048)`, `codes[0] == cb0`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement `depth_frame.py`, `export_depth.py`, `sampling.py`, `depth.py`.**
- [ ] **Step 4: Run the tests** → PASS.
- [ ] **Step 5: Export both precisions guarded** (`--rss-limit-mb 6000 --min-free-mb 7000`). fp32 parity on real weights: `depth_tf_fp32` vs Task 3 `depth_logits` for 20 frames, max rel error ≤ 1e-4 (fp16 storage of the golden allows 1e-3 abs; record both).
- [ ] **Step 6: Quality gate.** `eval_teacher_forced.py depth --candidate wo8`. Pass: every level `top1 >= 0.98` and `kl <= 0.02`. On a fail, re-run with `fp32` heads and int8 trunks. If that also fails, ship fp32 depth (2.2 GB, still under the 3000 MB budget with a wo8 backbone; recheck it in Task 9). Record which one ships.
- [ ] **Step 7: Speed gate.** Median `Depth.frame` over 200 calls, 8 threads: ≤ 35 ms. Record.
- [ ] **Step 8: Commit.**

---

### Task 7: Mimi decoder graph

**Files:**
- Create: `tools/maya_export/export_codec.py`
- Create: `src/maya_fast/codec.py`
- Test: `tests/test_maya_fast_codec.py`

**Interfaces:**
- Produces: `mimi_decoder.onnx` from `transformers.MimiModel.decode`: input `codes (1, 16, F)` int64, output `wav (1, 1, F * 1920)` float32, dynamic `F`, fp32.
- Produces: `MimiDecoder(path: Path, threads: int)` with `decode(codes: np.ndarray) -> np.ndarray` (`(16, F)` → `(F * 1920,)` float32).

- [ ] **Step 1: Write failing test** `test_codec_length` (tiny assets: `F = 5` → 9600 samples).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement both files.** **Step 4: Run** → PASS.
- [ ] **Step 5: Export guarded** (`--rss-limit-mb 2000`). Parity on 5 Task 3 `codes`: max abs error vs `MimiModel.decode` ≤ 1e-4. Speed: median decode time ≤ 8 ms per frame at F = 50. Record both.
- [ ] **Step 6: Commit.**

---

### Task 8: Engine

**Files:**
- Create: `src/maya_fast/engine.py`, `src/maya_fast/__main__.py`
- Test: `tests/test_maya_fast_engine.py`

**Interfaces:**
- Consumes: everything from Tasks 4-7.
- Produces: `MayaFast(assets_dir: str | Path | None = None, threads: int | None = None)`. `assets_dir` defaults to `work_dir()/assets/v1` (no torch import: the runtime reads `MAYA_FAST_WORK` itself). `.voices -> list[str]`. `.synthesize(text: str, voice: str = "Charvi", seed: int = 0) -> tuple[np.ndarray, dict]`: 24 kHz float32 wav, metrics with keys `audio_s`, `wall_s`, `rtf`, `frames`, `reasons` (one per piece), `backbone_s`, `depth_s`, `mimi_s`, `peak_rss_mb`. Per piece of `split_text`: `Backbone.reset()`, prompt step, then per frame: `p = softmax(cb0_logits / 0.7)` with top-64 mask, `StopPolicy.check`, `sample_cb0`, `Depth.frame`, `AudioIn.frames`, next `Backbone.step`. After the stop, one `aeos()` step as upstream does (it changes nothing returned, so skip it and say why in a comment). Noise from `np.random.default_rng(seed)`, drawn in a fixed order. Unknown voice → `KeyError` listing `.voices`.
- Produces: CLI `python -m src.maya_fast "<text>" [-o out.wav] [--voice Charvi] [--seed 0] [--threads N]`, writes 16-bit PCM WAV with `wave`, prints `audio_s=... wall_s=... rtf=... peak_rss_mb=...`.

- [ ] **Step 1: Write failing tests** (all on tiny assets): `test_synthesize_returns_audio` (length = `frames * 1920`), `test_same_seed_same_audio`, `test_empty_text_raises` (`""` and `"।,."` → `ValueError("no speakable text")`), `test_cap_stops_generation` (tiny assets built with an aeos logit bias of −100 → `reasons == ["cap"]`, `frames == cap`), `test_long_text_is_split_and_kv_bounded` (3-sentence text → `len(reasons) == len(split_text(...))`, and `Backbone.past_len` never exceeds the largest piece's prompt + cap, checked through a recorded maximum), `test_thread_count_does_not_change_output` (threads 1 and 16 → identical wav), `test_unknown_voice_lists_voices`, `test_runtime_has_no_torch` (`ast` scan of every file under `src/maya_fast/`: no `torch` import).
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests/test_maya_fast_engine.py -v` → FAIL.
- [ ] **Step 3: Implement `engine.py` and `__main__.py`.**
- [ ] **Step 4: Run all runtime tests** `.venv/bin/python -m unittest discover -s tests -p "test_maya_fast_*.py" -v` → PASS.
- [ ] **Step 5: Smoke on real assets, guarded** (`--rss-limit-mb 3000 --min-free-mb 4000`): `python -m src.maya_fast "नमस्कार, मी तुमची मदत करण्यासाठी इथे आहे." -o $MAYA_FAST_WORK/smoke.wav`. Expected: WAV written, `MEMGUARD reason=None`.
- [ ] **Step 6: Commit.**

---

### Task 9: End-to-end gates

**Files:**
- Create: `benchmarks/maya_fast_eval.py`, `benchmarks/maya_fast_spk.py`, `benchmarks/maya_fast_listen.py`
- Modify: `benchmarks/maya_fast_results.md`

**Interfaces:**
- Consumes: `MayaFast` (Task 8), Task 3 reference corpus, `IndicConformerONNXASR`, `src.audio_utils.resample`, `score_transcript`.
- Produces: `maya_fast_eval.py` (`.venv`): for the test set × seeds {0, 1, 2}, synthesize, transcribe (24 kHz → 16 kHz), score; the same for the reference wavs. Writes `work_dir()/eval/` WAVs and a table with per-category and overall mean CER/WER, RTF, peak RSS, per-phase ms/frame. Loads one model at a time: the TTS engine first, then the ASR on saved WAVs.
- Produces: `maya_fast_spk.py` (`.venv-tts`, speechbrain ECAPA from `ckpt_paths()["ecapa_dir"]`): cosine of each output's embedding to the voice's `ecapa`, mean for candidate and reference.
- Produces: `maya_fast_listen.py`: 10 sentences, one blind A/B pair each (reference vs candidate, seed 0, random left/right), files `pair_XX_A.wav` / `_B.wav` in `work_dir()/listen/`, key in `work_dir()/listen/key.json`; `--score answers.json` prints the candidate-preferred-or-tied rate.

- [ ] **Step 1: Run eval guarded** (`--rss-limit-mb 3000 --min-free-mb 4000`), then `maya_fast_spk.py` guarded (`--rss-limit-mb 4000`).
- [ ] **Step 2: Gates.** CER ≤ reference + 0.01; speaker cosine ≥ reference − 0.02; mean RTF ≤ 1.0; every run's peak RSS ≤ 3000 MB; swap growth ≤ 64 MB. Record every number in `benchmarks/maya_fast_results.md` with date, machine and thread count.
- [ ] **Step 3: Listening.** Generate pairs; the user rates them. This step waits for the user; the agent does not rate. Pass: ≥ 45% preferred or tied.
- [ ] **Step 4: Commit** the scripts and results.

---

### Task 10: Compatibility and packaging

**Files:**
- Create: `requirements-fast.txt`, `.github/workflows/maya-fast.yml`
- Modify: `README.md` (new section "Maya-Fast TTS")

**Interfaces:**
- Consumes: `tests/fixtures/tiny_assets.py` (Task 5) so CI needs no weights.
- Produces: `requirements-fast.txt`: `numpy>=2.0`, `onnxruntime==1.31.*`, `tokenizers>=0.22,<0.23`. Nothing else.
- Produces: workflow on push/PR: matrix `os: [ubuntu-latest, ubuntu-24.04-arm, macos-14, macos-13, windows-latest]` × `python: ["3.10", "3.13"]`; installs only `requirements-fast.txt`; runs `python -m unittest discover -s tests -p "test_maya_fast_*.py" -v`.

- [ ] **Step 1: Write the requirements file and workflow.** Check locally in a fresh venv: `python3 -m venv /tmp/mf && /tmp/mf/bin/pip install -r requirements-fast.txt && /tmp/mf/bin/python -m unittest discover -s tests -p "test_maya_fast_*.py"` → PASS with no torch installed.
- [ ] **Step 2: Push the branch** (ask the user first: it publishes to `origin`). All matrix jobs green. A failing OS is a bug to fix, not a job to drop.
- [ ] **Step 3: i5 run (user).** The user copies `assets/v1` to the i5 machine, installs `requirements-fast.txt`, runs `python -m src.maya_fast` there and pastes the output line. Gate: runs, peak RSS ≤ 3000 MB. RTF recorded, not gated.
- [ ] **Step 4: README section:** install (`requirements-fast.txt`), one-time asset export (needs `.venv-tts`), CLI usage, measured RTF/RSS from Task 9 labelled with the machine, licenses (Apache-2.0 weights, CC-BY-4.0 Mimi), no audio watermark.
- [ ] **Step 5: Commit.**

---

## Out of Scope

- Streaming playback (first audio before the piece finishes). The engine returns whole pieces; streaming Mimi decode is a later plan.
- GPU or NPU execution providers (CoreML, DirectML). CPU EP only.
- Retraining, distillation, vocabulary pruning, fewer codebooks (`q_serve < 16` lowers quality and is excluded by the quality constraint).
- Removing the spike scripts in `benchmarks/`. They stay as the research record.

## Self-review

- **Coverage:** speed (Tasks 5-7 per-component gates, Task 9 RTF), RAM (memguard on every model run, 3000 MB runtime gate), compatibility (no-torch runtime, CI matrix, i5 run), quality (fp32 parity, then teacher-forced gates, then CER, speaker similarity and blind listening), no-swap (Task 1, enforced in every run step).
- **Names checked:** `run_guarded`, `work_dir`, `iter_ckpt_tensors`, `build_light_model`, `load_manifest`, `AssetError`, `EmbeddingTable`, `AudioIn`, `build_prompt`, `Prompt`, `StopPolicy`, `sample_cb0`, `session_options`, `Backbone`, `Depth`, `DepthFrame`, `DepthTF`, `MimiDecoder`, `MayaFast`, `make_tiny_assets` are used with the same signatures wherever they appear.
- **Known risks, gated instead of guessed:** backbone ≤ 35 ms/step is not yet measured fused (unfused, it measured 47-73 ms); depth wo8 accuracy is not yet measured; the prompt port (Task 4) is the largest hand-port and is pinned by golden parity on every voice.
