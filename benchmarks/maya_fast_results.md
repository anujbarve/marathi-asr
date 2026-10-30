# maya-fast results

Machine: Apple M-series Mac, 8 ORT threads, CPU EP.

## Task 5: backbone (wo8, MatMulNBits int8, block 32, accuracy_level 4)

Artifact: `backbone_wo8.onnx` + `backbone_wo8.data` (1.6 GB), exported 2026-10-28 by `tools/maya_export/export_backbone.py`.

Speed gate (`Backbone.step`, T = 1, 200 steps each, median):

| past_len | median ms | gate |
|---|---|---|
| 32 | 48.22 | - |
| 256 | 58.00 | <= 35 ms: FAIL |
| 768 | 82.35 | <= 1.25 x 32-median (60.3 ms): FAIL |

Peak RSS during the run: 2945.8 MB (guard 4000 MB).

Status: speed gate FAILED. Quality gate not run (needs `eval_teacher_forced.py`, not written).
Per plan, stop and amend: the 32-token median already exceeds 35 ms, so the cost is weight bandwidth
(1.6 GB of int8 weights read per token), not KV concat alone. The 768/32 ratio also misses (KV copy).
Options for amendment: bits=4 weights (about half the bytes), or a revised speed gate.

### Speed gate, amended (user accepted the measured numbers)

Revised gate: report the speed numbers above; no pass/fail. Measured: 48.2 / 58.0 / 82.4 ms at past_len 32 / 256 / 768.

### Quality gate: teacher-forced, wo8 vs fp32 per-layer reference

`tools/maya_export/eval_teacher_forced.py backbone --candidate wo8` over the 75-file reference corpus
(4272 generating positions). Peak RSS 2566.7 MB.

| metric | value | gate | result |
|---|---|---|---|
| top1 agreement (cb0 argmax) | 0.9677 | >= 0.99 | FAIL |
| mean KL (softmax(logits/0.7), nats) | 0.00336 | <= 0.01 | pass |

Status: quality gate FAILED on top-1. Not relaxed. Next: find the source of top-1 misses (int8 weights vs. near-tie logits), then amend or change the quantization.

### Waiver

User accepted the backbone wo8 quality gate miss (top1 0.9677 < 0.99; KL 0.00336 passes) and asked to move on. Gate recorded as failed; waived by the user for proceeding to Task 6. Top-1 misses not investigated.

## Task 6: depth (per-level graphs, 15 levels, q_serve 16)

Artifacts: `depth_build/{fp32,wo8,wo8_fph}/level_XX.onnx` (+ `.data`). fp32 = 4.0 GB total. Built by
`tools/maya_export/export_depth.py`; one graph per level, each with its own trunk group and shared cross-attention.

Parity, fp32 graphs vs upstream PyTorch depth (`tools/maya_export/depth_check.py`, 20 golden frames, conv_01_s0):
worst rel 1.6e-6, worst abs 4.0e-5 (gate rel <= 1e-4). PASS.

Quality gate, teacher-forced vs stored reference (75 files, 4272 frames), per level (gate top1 >= 0.98, KL <= 0.02):

| candidate | levels passing | notes |
|---|---|---|
| fp32 | 15 / 15 (min top1 0.9965, max KL 5.7e-6) | PASS, ships per plan fallback |
| wo8 (int8 trunks + int8 head) | 1 / 15 (top1 0.929 to 0.986) | FAIL |
| wo8_fph (int8 trunks, fp32 head) | 2 / 15 (top1 0.929 to 0.989) | FAIL |

Speed gate (depth chain, F = 1, 8 threads, median of 20): 215.3 ms for the 15 levels. Gate 35 ms: FAIL.
Peak RSS when all 15 level sessions are resident: 3743.8 MB. Over the 3000 MB runtime budget.

Status: quality gate passes with fp32; speed gate and memory budget FAIL. The per-level design re-reads
the group-8 and cross-attention weights for each of levels 9..15 (about 4 GB read per frame). The plan's
single DepthFrame graph with shared group-8 KV is the design that reduces this; not built yet.

### Amended targets (user decision)

Speed: offline target, mean RTF <= 4 on this Mac. Real-time (RTF <= 1) is dropped. The 35 ms per-frame backbone and depth gates are superseded by this RTF target.
Memory: unchanged, runtime peak RSS <= 3000 MB. Depth moves to fp16 storage (single shared-group graph) to meet it. Needs its own quality gate.

## Task 6 revised: single depth graph, fp16 storage (depth_all.onnx, 833 MB)

Quality gate, teacher-forced vs stored reference (75 files, 4272 frames), levels 1..15, gate top1 >= 0.98, KL <= 0.02:
all 15 levels PASS (min top1 0.9965, max KL 5.7e-6). Same numbers as the fp32 per-level chain to four decimals.

Speed, F = 1, 8 threads, median of 30: 154.7 ms (default ORT optimisation). With ConstantFolding disabled: 262.6 ms, RSS 2.29 GB.
Default kept.

Memory, backbone wo8 + depth fp16 + Mimi loaded together (no inference): peak RSS 2671.6 MB. Under the 4 GB limit; inference adds KV and activations (not yet measured).

Offline RTF estimate (amended target <= 4): backbone ~60-80 ms/token + depth 155 ms + Mimi ~1 ms per 80 ms frame, about 2.9 RTF. Not measured end to end; the engine is not wired to these graphs yet.

### RAM limit change (user decision, 2026-10-28)

Hard RSS limit raised from 4000 MB to 5000 MB. Target stays 3000 MB. Previous 4000 MB runs are unaffected; new guard runs use 5000 MB.

## 2026-10-29: end-to-end gates, asset sets v1 and v2-lowram (finish plan, Task 5)

Machine: Apple M5 (10 cores), CPU EP. TTS + ASR via `benchmarks/maya_fast_eval.py` run with the ASR venv
(ORT 1.31.0). Full test set `benchmarks/tts_testset_mr.json` (25 sentences, 5 categories), voice Charvi,
seeds 0 1 2 (75 utterances per set), default threads (8). Every run under
`tools/memguard.py --rss-limit-mb 5000 --min-free-mb 5500`; swap delta 0 MB (v1 full: -8 MB). Reference =
upstream PyTorch corpus `golden/ref` (same ids and seeds), scored by the same ASR.

Asset sets:
- **v1**: `assets/v1` (backbone `backbone_wo8.onnx`, depth `depth_frame_fp16.onnx`). Unchanged.
- **v2-lowram**: `assets/v2-lowram`, hard links of v1 with the depth graph swapped for
  `depth_frame_q8a1_kg8.onnx` (int8 block 32, accuracy_level 1, group 8 fp16; teacher-forced gate pass, min
  top-1 0.9806). Manifest `graphs: {"depth": "depth_frame_q8a1_kg8.onnx"}`, built by
  `tools/make_asset_variant.py`. Backbone unchanged (no quantized backbone passed its gate).

| set | threads / seeds | CER (ref) | CER - ref | peak RSS MB (memguard / ru_maxrss) | RTF mean (max) | backbone / depth / mimi ms per frame |
|---|---|---|---|---|---|---|
| v1 | 8 / 0,1,2 | 0.137 (0.147) | -0.010 | 4466.5 / 4500 | 2.206 (2.704) | 53.4 / 117.2 / 2.8 |
| v2-lowram | 8 / 0,1,2 | 0.122 (0.147) | -0.025 | 3822.8 / 3884 | 5.460 (6.265) | 38.5 / 394.2 / 2.4 |
| v1 | 4 / 0 | 0.133 (0.145) | -0.012 | 4459.4 / 4458 | 2.072 (2.316) | 50.9 / 110.0 / 3.0 |
| v2-lowram | 4 / 0 | 0.119 (0.145) | -0.026 | 3812.8 / 3812 | 6.123 (6.518) | 37.9 / 447.4 / 2.5 |

Speaker similarity (`benchmarks/maya_fast_spk.py`, ECAPA, target = reference-corpus centroid, full runs, n=75):
v1 0.8110, v2-lowram 0.8069, reference 0.8202. Candidate - reference: v1 -0.0092, v2-lowram -0.0134 (gate >= -0.02).

### Gates

| gate | v1 | v2-lowram |
|---|---|---|
| CER <= reference + 0.01 | PASS (0.137 vs 0.157) | PASS (0.122 vs 0.157) |
| plan gate: v2 CER <= v1 CER + 0.01 | - | PASS (0.12161 <= 0.13657 + 0.01 = 0.14657) |
| peak RSS <= 3000 MB target | FAIL (4466.5) | FAIL (3822.8) |
| peak RSS <= 5000 MB hard limit | PASS | PASS |
| mean RTF <= 4 (offline target) | PASS (2.206) | FAIL (5.460) |
| mean RTF <= 1 (real time) | FAIL | FAIL |
| speaker cosine >= reference - 0.02 | PASS (-0.0092) | PASS (-0.0134) |

The plan gate "v2 CER <= v1 CER + 0.01" uses the exact values from `~/.cache/maya_fast/eval/t5-v1-full/results.json`
(v1 0.13657) and `t5-v2lowram-full/results.json` (v2-lowram 0.12161). A plain "v2" set was not built: no backbone or
depth replacement passed its gate other than the RAM-motivated depth swap (v2-lowram), so a speed-motivated v2
would equal v1.

Neither set meets all gates. v1 meets quality and speed but not the RAM target; v2-lowram saves about 640 MB
peak and meets quality, but misses both the RAM target and the RTF target. Lower CER than the reference is
ASR noise on 75 utterances, not a quality gain.

### Speed note

RTF on this run is about 2x the 2026-10-29 smoke run (v1 RTF 1.038, 25.6 / 54.8 ms per frame). Not the ORT
version: a TTS-only rerun of the first 3 items, seed 0, with ORT 1.26 (system python) gave v1 RTF 2.22 / 2.08
/ 2.08 (53.4 / 108.3 ms per frame) and v2-lowram 5.79 / 5.64 / 5.57 (35.2 / 412.8 ms per frame), the same as
ORT 1.31. Machine on AC, no thermal warning recorded; the cause of the slowdown versus the smoke run is not
known. In e2e the q8a1_kg8 depth costs 394-447 ms per frame, not the 245 ms of the Task 2 isolated bench.

Listening pairs (unrated): `~/.cache/maya_fast/listen/v2-lowram/` and `~/.cache/maya_fast/listen/v1/`
(10 pairs each vs reference, seed 0, key in `key.json`).

Run dirs: `~/.cache/maya_fast/eval/{t5-v1-full,t5-v2lowram-full,t5-v1-t4-s0,t5-v2lowram-t4-s0,t5-ort126-v1-l3,t5-ort126-v2lowram-l3}`.

### Controller e2e run (ORT 1.26)

Controller end-to-end run, 2026-10-29, v1 assets, 2 sentences, 8 threads, ORT 1.26: memguard peak RSS 4486.9 MB,
RTF 0.94 / 0.92 (cited from the controller's run; no separate run dir). This is the source of the README "about 0.9" v1 row.
