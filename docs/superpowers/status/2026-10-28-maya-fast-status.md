# maya-fast status (2026-10-28)

Resume point. Full numbers: `benchmarks/maya_fast_results.md`. Plan: `docs/superpowers/plans/2026-10-27-maya-fast-runtime.md`.

## Targets as amended (user decisions)
- Speed: offline, mean RTF <= 4 on this Mac (real-time dropped).
- RAM: runtime peak RSS <= 3000 MB target; hard limit 4000 MB (user: RAM non-negotiable).

## Done
- Task 1 memory guard (`tools/memguard.py`).
- Task 2 test set, light golden (bit-identical with bf16 storage, peak 697 MB).
- Task 3 reference corpus: 75 files (`~/.cache/maya_fast/golden/ref`), streamed backbone (`tools/maya_export/golden_reference.py`).
- Task 4 prompt assets: token-exact (500 prompts), embeddings max err 7.5e-9.
- Task 7 Mimi decoder ONNX: parity 1.5e-6, 1.0 ms/frame.
- Task 5 backbone: fp32 + wo8 fused graphs; merge parity test passes.
  - wo8 speed (48 / 58 / 82 ms per token): gate waived by user.
  - wo8 top-1 0.9677 vs gate 0.99: waived by user.
- Task 6 depth:
  - fp32 per-level chain: quality pass (top-1 >= 0.9965), 215 ms/frame, too slow.
  - fp16 single graph (`depth_all.onnx`, 833 MB): quality pass all 15 levels.
  - fp16 sampled graph (`depth_frame_fp16.onnx`): in-graph Gumbel top-k sampling; consistent with teacher-forced rule (0 of 600 disagree).
  - Runtime engine wired to the sampled graph; runtime tests 25/25 pass.

## Blocker: end-to-end RAM
- Backbone wo8 alone: ~2.0 GB resident (prepacked).
- Depth fp16 session alone: 2.1 GB (ConstantFolding disabled); 3.4 GB default (ORT folds fp16 Casts).
- Both loaded: ~4.1 GB before Mimi and inference. Over the 4 GB limit.
- Tried, no fix: ORT arena on/off, prepacking off, threads 4 vs 8, memory-pattern off.
- First end-to-end run stopped by guard at 4.16 GB.

## Next (pick up here)
1. Backbone 4-bit weights (MatMulNBits bits=4, block 32/128): export, quality gate vs fp32 reference (`eval_teacher_forced.py backbone`), RSS check.
2. Depth loader: ORT holds a second copy of external weights at load (1.6 GB at load for an 0.87 GB file). Options: different weight layout, or fp16 without Cast nodes.
3. Re-measure combined RSS; target <= 3 GB, must be <= 4 GB.
4. End-to-end synthesis on one sentence, RTF and peak RSS.
5. Listening check (plan Task 9 step 3) — user.
6. Task 10: push done for branches; README, requirements, i5 run — user.

## Artifacts (not in repo)
- `~/.cache/maya_fast/assets/v1/`: runtime assets, manifest with 11 files and runtime shapes.
- `~/.cache/maya_fast/golden/`, `ref/`: goldens and reference corpus.
- `~/.cache/maya_fast/depth_build/fp16/`, `backbone_build/`: build intermediates.

## Branches
- `feat/maya-fast-runtime` (runtime, benchmarks, plan, this doc).
- `feat/maya-fast-export` (export tools, tests).
