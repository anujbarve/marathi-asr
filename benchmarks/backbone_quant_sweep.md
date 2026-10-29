# Backbone quantization sweep (teacher-forced gate vs fp32)

Gate: top-1 >= 0.99 and mean KL <= 0.01 (75 files, 4272 frames). Tool: `tools/maya_export/quant_sweep_backbone.py`.
Speed/RSS: runtime `maya_fast.backbone.Backbone`, ORT 1.26 system python, T=1 steps, 100 steps after a prefill of
`past_len` tokens, median ms/step. RSS = `ps` resident size right after load (before any step) and after the 100+100 steps.

| tag | top-1 | KL | gate | RSS after load (MB) | RSS end (MB) | ms/step past32 | ms/step past256 |
|---|---|---|---|---|---|---|---|
| wo8 baseline (b8 blk32 sym, acc4, assets/v1) | 0.9677 | 0.00336 | FAIL | 1958 | 2673 | 22.8 | 28.9 |
| b8_blk32_rtn_sym_acc1 | 0.9810 | 0.00170 | FAIL | 97 (mmap, lazy) | 2012 | 799 | 866 |
| b4_blk128_rtn_asym (existing build, see note) | 0.7205 | 0.3796 | FAIL | not measured | | | |
| b4_blk32_rtn_asym_acc1 | 0.7821 | 0.2211 | FAIL | 770 | 1632 | 52.6 | 59.3 |
| b4_blk128_rtn_asym_acc1 | 0.7205 | 0.3796 | FAIL | 771 | 1412 | 59.7 | 65.1 |

Speed rows for b4 were measured while another agent had a ~575% CPU job running, so they are upper bounds.
b8 and wo8 were re-measured twice; b8 acc1 was 882/768 ms and then 799/866 ms; wo8 was 38/45 ms under load
and 22.8/28.9 ms in a quieter run. Speed/RSS rows for failing variants are informational only.

## Findings

1. Bug in the earlier int4 builds: `MatMulNBitsQuantizer(..., algo_config=cfg)` takes `bits` and `accuracy_level`
   from `cfg`, not from the constructor. The old builds therefore had `accuracy_level` unset (fp32 compute) and the
   first sweep run of "bits 8" silently produced bits=4. The tool now sets both on the config; the built graphs were
   checked (`bits` attribute) after the fix.
2. The "acc4 causes the top-1 loss" hypothesis is refuted for int4: block 128 asym gives identical numbers
   (0.7205 / 0.3796) with and without an explicit accuracy level. int4 RTN fails because of weight error.
3. The current wo8 baseline itself fails the gate (top-1 0.9677). fp32-compute 8-bit (acc1) is better (0.981,
   KL 0.0017) but still below 0.99 on top-1, and has no fast kernel in ORT (about 35x slower than wo8).
4. No variant passes. No smallest-RSS passing variant exists, so there is no <= 1.3x wo8 recommendation.

## Recommendation

No quantized variant meets the gate. Best quality is `b8_blk32_rtn_sym_acc1` (top-1 0.981, KL 0.0017) but it is
unusable for speed (~800 ms/step). Keep wo8 (also below the 0.99 gate; must be re-baselined) or the fp32 backbone.
Further ideas not tried here: 8-bit with block 16 or asymmetric, excluding sensitive layers (first/last, v/o proj),
GPTQ-style weight search.

Note on row 3: the existing `int4_b128_rtn_asym` build was evaluated and gave the same result as the rebuilt
`b4_blk128_rtn_asym_acc1`; the rebuilt graph is the authoritative one.
