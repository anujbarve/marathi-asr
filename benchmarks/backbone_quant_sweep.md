# Backbone quantization sweep (teacher-forced gate vs fp32)

Gate: top-1 >= 0.99 and mean KL <= 0.01 (75 files, 4272 frames). Tool: `tools/maya_export/quant_sweep_backbone.py`.
Machine: Apple M5 (arm64), 8 ORT threads (Backbone default min(8, cores)). Gate: ORT 1.31.1 (`.venv-tts`).
Speed/RSS: runtime `maya_fast.backbone.Backbone`, ORT 1.26 system python (8 threads), T=1 steps, 100 steps after a prefill of
`past_len` tokens, median ms/step. RSS = `ps` resident size right after load (before any step) and after the 100+100 steps.

| tag | top-1 | KL | gate | RSS after load (MB) | RSS end (MB) | ms/step past32 | ms/step past256 |
|---|---|---|---|---|---|---|---|
| wo8 baseline (b8 blk32 sym, acc4, assets/v1) | 0.9677 | 0.00336 | FAIL | 1958.5 | 2326.6 | 38.27 | 45.38 |
| b8_blk32_rtn_sym_acc1 | 0.9810 | 0.00170 | FAIL | 97.3 (mmap, lazy) | 2213.4 | 882.02 | 768.37 |
| b4_blk128_rtn_asym (existing build, see note) | 0.7205 | 0.3796 | FAIL | not measured | | | |
| b4_blk32_rtn_asym_acc1 | 0.7821 | 0.2211 | FAIL | 770.4 | 1631.6 | 52.58 | 59.32 |
| b4_blk128_rtn_asym_acc1 | 0.7205 | 0.3796 | FAIL | 771.1 | 1411.7 | 59.7 | 65.12 |

Source: `tools/maya_export/bench_backbone.py`, one run per variant, values from the saved log `sw_bench.log`
(scratchpad). All speed rows were taken while another agent ran a ~575% CPU job outside the model lock, so ms/step
values are noisy upper bounds. Re-runs made outside the log (not cited here) showed wo8 and b8 acc1 varying
substantially, so treat ratios as rough. Speed/RSS rows for failing variants are informational only.

## Findings

1. Bug in the earlier int4 builds: `MatMulNBitsQuantizer(..., algo_config=cfg)` takes `bits` and `accuracy_level`
   from `cfg`, not from the constructor. The old builds therefore had `accuracy_level` unset (fp32 compute) and the
   first sweep run of "bits 8" silently produced bits=4. The tool now sets both on the config; the built graphs were
   checked (`bits` attribute) after the fix.
2. The "acc4 causes the top-1 loss" hypothesis is refuted for int4: block 128 asym gives identical numbers
   (0.7205 / 0.3796) with and without an explicit accuracy level. int4 RTN fails because of weight error.
3. The current wo8 baseline itself fails the gate (top-1 0.9677). fp32-compute 8-bit (acc1) is better (0.981,
   KL 0.0017) but still below 0.99 on top-1, and has no fast kernel in ORT (about 17-23x slower than wo8 in the same logged run: 882 vs 38 ms, 768 vs 45 ms).
4. HQQ: `HQQWeightOnlyQuantConfig` ignores `is_symmetric` and `accuracy_level`; the tool now rejects `--asym` and
   `--accuracy-level` with `--algo hqq` and tags such builds `b{bits}_blk{block}_hqq_default`. HQQ is untested with the
   corrected config. The old `int4_*` builds (including the int4 HQQ and RTN block 32 runs) had `accuracy_level` unset.
5. No variant passes. No smallest-RSS passing variant exists, so there is no <= 1.3x wo8 recommendation.

## Recommendation

No quantized variant meets the gate. Best quality is `b8_blk32_rtn_sym_acc1` (top-1 0.981, KL 0.0017) but it is
unusable for speed (~770-880 ms/step). Keep wo8 (also below the 0.99 gate; must be re-baselined) or the fp32 backbone.
Further ideas not tried here: 8-bit with block 16 or asymmetric, excluding sensitive layers (first/last, v/o proj),
GPTQ-style weight search.

Note on row 3: the existing `int4_b128_rtn_asym` build was evaluated and gave the same result as the rebuilt
`b4_blk128_rtn_asym_acc1`; the rebuilt graph is the authoritative one.
