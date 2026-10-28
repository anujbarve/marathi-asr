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
