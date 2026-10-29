# Depth graph RAM reduction (2026-10-29)

Target (plan Task 2): depth steady RSS <= 1000 MB, load spike <= 1500 MB, <= 80 ms/frame, teacher-forced gate
(per level top-1 >= 0.98 and KL <= 0.02) on all 15 levels.

**Result: no candidate meets all four targets** (the fp16 baseline included: 66-99 ms/frame on the M5, 3 of 4 runs over 80 ms). Keep `depth_frame_fp16.onnx` for now: the only build passing the gate at near-target speed. Details below.

Machine: Apple M5 (arm64), 16 GB. ORT 1.26.0 (runtime python) for RSS / speed / codes; ORT 1.31.1 (`.venv-tts`)
for the teacher-forced gate. Every run under `tools/memguard.py`; swap 372 -> 364 MB over the session (no growth).

Tools: `tools/maya_export/quant_depth_frame.py` (streamed MatMulNBits conversion of the fp16 depth graphs, one
tensor in memory at a time, peak 211-220 MB), `tools/maya_export/bench_depth_ram.py` (RSS / speed / 40-frame codes).

## Candidates

All are built from the fp16-storage graphs (`depth_build/fp16/`): fp16 -> fp32 upcast is exact, so they are
quantized from the fp32 weights. MatMulNBits bits=8, symmetric, unless noted. "keep X" = those weights stay
fp16-stored behind their Cast (ORT folds them to fp32 at load).

| tag | bits / block / acc | kept fp16 | weight data (depth_all build) |
|---|---|---|---|
| fp16 | baseline | all | 873 MB |
| q8a1 | 8 / 32 / 1 | - | 498 MB |
| q8a1_kh | 8 / 32 / 1 | heads | 526 MB |
| q8a1_kg8 | 8 / 32 / 1 | group 8 (levels 9..15) | 535 MB |
| q8a1_khc | 8 / 32 / 1 | heads, cross | 540 MB |
| q8a1_khg8 | 8 / 32 / 1 | heads, group 8 | 562 MB |
| q8a4 | 8 / 32 / 4 | - | 498 MB |
| q8a4_kg8 | 8 / 32 / 4 | group 8 | 535 MB |
| q8a4b16_kg8 | 8 / 16 / 4 | group 8 | 583 MB |
| q4a1 | 4 / 32 / 1 | - | 284 MB |

## Quality gate (teacher-forced, 4272 frames, 75 files)

Top-1 per level 1..15 (gate 0.98). KL never failed (max shown).

| tag | top-1 levels 1..15 | min | max KL | pass |
|---|---|---|---|---|
| fp16 | .9965 .9967 .9986 .9979 .9979 .9995 .9977 .9991 .9986 .9993 .9988 .9993 .9984 .9991 .9984 | .9965 | 0.00001 | yes |
| q8a1 | .9927 .9867 .9899 .9869 .9860 .9862 .9822 .9806 .9712 .9668 .9724 .9696 .9686 .9691 .9583 | .9583 | 0.0023 | no (L9-15) |
| q8a1_kh | .9946 .9911 .9916 .9883 .9869 .9881 .9848 .9808 .9717 .9684 .9750 .9693 .9675 .9698 .9581 | .9581 | 0.0023 | no (L9-15) |
| q8a1_khc | .9946 .9911 .9916 .9885 .9869 .9881 .9848 .9808 .9717 .9684 .9747 .9696 .9675 .9700 .9581 | .9581 | 0.0023 | no (L9-15) |
| **q8a1_kg8** | .9927 .9867 .9899 .9869 .9860 .9862 .9822 .9806 .9906 .9920 .9925 .9925 .9925 .9899 .9862 | .9806 | 0.0005 | **yes** |
| q8a1_khg8 | .9946 .9911 .9916 .9883 .9869 .9881 .9848 .9808 .9916 .9934 .9932 .9932 .9927 .9934 .9888 | .9808 | 0.0005 | yes |
| q8a4 | .9864 .9782 .9764 .9670 .9745 .9696 .9665 .9632 .9412 .9431 .9417 .9387 .9441 .9300 .9277 | .9277 | 0.0080 | no |
| q8a4_kg8 | .9864 .9782 .9764 .9670 .9745 .9696 .9665 .9632 .9822 .9850 .9820 .9806 .9799 .9789 .9728 | .9632 | 0.0024 | no |
| q8a4b16_kg8 | .9885 .9846 .9787 .9759 .9761 .9771 .9745 .9703 .9848 .9853 .9874 .9836 .9838 .9841 .9771 | .9703 | 0.0015 | no |
| q4a1 | .8825 .8427 .8301 .8052 .7767 .7753 .7566 .7432 .5826 .6037 .5581 .5602 .5325 .5328 .4981 | .4981 | 0.574 | no |

Findings: (1) activation quantization (acc 4) is what breaks levels 1..8 (acc 1: >= 0.9806; acc 4: down to
0.9632, block 16: 0.9703). (2) At acc 1 the only failing weights are group 8 (shared by levels 9..15): keeping
group 8 fp16 lifts levels 9..15 from 0.958-0.975 to 0.986-0.9925; keeping heads / cross does nothing. Level 8
(0.9806-0.9808) is the tightest margin of any passing build.

## RSS / speed (sampled frame graph, ORT 1.26, runtime session options)

load_peak = ru_maxrss after `InferenceSession()`; first5 = sampler peak during frames 1..5; steady = RSS after 73
frames; ms = median of 30 frames; codes = fraction of the 40x16 bench codes equal to `loader_bench/base.codes.npy`.

| tag | thr | load RSS | load peak | first-5 peak | steady | memguard peak | ms/frame | codes |
|---|---|---|---|---|---|---|---|---|
| fp16 (run 2) | 8 | 2152 | 3353 | 2161 | 2165 | 3350 | 99.0 | identical |
| fp16 (run 2) | 4 | 2211 | 3351 | 2218 | 2222 | 3120 | 88.7 | identical |
| fp16 (run 1) | 8 | 2224 | n/a* | 2229 | 2233 | 3115 | 83.8 | identical |
| fp16 (run 1) | 4 | 2317 | 3387 | 2321 | 2208 | 3147 | 66.0 | identical |
| q8a1 | 8 | 96 | 96* | 613 | 618 | 618 | 345.9 | 0.908 |
| q8a1 | 4 | 96 | 96* | 608 | 613 | 613 | 437.6 | 0.908 |
| q8a1_kg8 | 8 | 478 | 478 | 897 | 902 | 902 | 244.7 | 0.928 |
| q8a1_kg8 | 4 | 473 | 473 | 891 | 896 | 896 | 249.6 | 0.928 |
| q8a4_kg8 | 8 | 1117 | 1119 | 1123 | 1127 | 1127 | 73.0 | 0.830 |
| q8a4_kg8 | 4 | 1073 | 1080 | 1082 | 1088 | 1088 | 65.9 | 0.830 |
| q8a4b16_kg8 | 8 | 1291 | 1292 | 1297 | 1303 | 1303 | 80.1 | 0.841 |
| q8a4b16_kg8 | 4 | 1247 | 1253 | 1253 | 1261 | 1261 | 72.6 | 0.841 |

\* first two runs used an in-process sampler for the load peak, which is starved while `InferenceSession()` holds the
GIL (fp16 showed 2224 vs memguard 3115); fixed to ru_maxrss before the other runs. memguard's external peak is
the cross-check. Not benched: q8a1_kh/khc (fail gate), q8a1_khg8 (passes, but strictly more RAM than kg8 on the
same slow path), q8a4, q4a1 (fail gate).

fp16 ms/frame varied 66-99 between identical runs on this machine (another agent's jobs share it); compare
candidates within the same row pair only.

## Why int8 acc 1 is slow (M5-measured; i5 behaviour extrapolated from ORT source, unmeasured on x86)

ORT MLAS has no 8-bit fp32-compute kernel on any CPU (`GetQNBitGemmVariant`: 8-bit -> CompInt8 or CompFp16 only).
MatMulNBits 8-bit acc 1 runs `ComputeBUnpacked`: dequantize the whole B into an fp32 scratch buffer on every call,
then SGEMM. Group 8 (7 uses/frame) and the cross blocks (15 uses/frame) are dequantized on every use, which is why
keeping group 8 fp32 also cuts 346 -> 245 ms. This path is generic C++, so the slowdown is expected on x86 too (extrapolated from ORT source, unmeasured on
x86: all speed numbers here are from the M5); on AVX2 the native kernels
are 8-bit CompInt8 (acc 4) and 4-bit CompFp32 (checked in onnxruntime main: `sqnbitgemm_kernel_avx2.cpp`).

## fp16 compute (plan Step 3): dropped

The CPU EP registers MLFloat16 MatMul / Gemm only under `ORT_CPU_FP16_GEMM_MATMUL_REGISTRATION_ENABLED`, defined
only when `MLAS_HALF_GEMM_ACCELERATION_POSSIBLE` (ARM64 with fp16 intrinsics, KleidiAI ARM64, or RISC-V). No fp16
MatMul kernel on x86, so an fp16-compute graph cannot run on the i5. Not built.

## Recommendation

- Keep `depth_frame_fp16.onnx` in the runtime: the only build that passes the gate at near-target speed (66-99 ms/frame
  on the M5, noisy; 3 of 4 runs over the 80 ms target; the 48.5 ms in the constraints file did not reproduce),
  2.2 GB steady, 3.35 GB load peak.
- Best RAM-and-quality build: `depth_build/q8a1_kg8/depth_frame_q8a1_kg8.onnx` (902 MB steady, no load spike, gate
  pass, min top-1 0.9806) at 245 ms/frame: meets RAM and gate, fails speed by 3x (M5-measured; on the i5 extrapolated from ORT source, unmeasured on x86). Use only if depth speed stops
  mattering (e.g. a low-RAM mode).
- Fast int8 (acc 4) builds run at 66-80 ms/frame on the M5 (ISA-specific kernels, unmeasured on x86) but fail the gate at levels 1..8 and do not reach 1000 MB (1088-1303 MB).
- Not tried, would need work outside ORT's stock kernels: an 8-bit fp32-compute kernel (custom op), or a
  per-tensor sensitivity search for acc 4 (which tensors of the private groups can stay int8).
