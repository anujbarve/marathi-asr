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
