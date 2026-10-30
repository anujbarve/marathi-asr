"""Backbone weight-only quantization sweep (MatMulNBits) with the teacher-forced gate.

Quantizes the fp32 per-layer graphs (backbone_build/fp32) with MatMulNBitsQuantizer, merges them with the same
export_backbone.merge() used for wo8, and runs the existing backbone gate. Writes only under
work_dir()/backbone_build/<tag>/.

Tag: b{bits}_blk{block}_{algo}_{sym|asym}_acc{level}

Usage (from the worktree root, under tools/memguard.py):
  python -m tools.maya_export.quant_sweep_backbone build --bits 4 --block 128 --asym --accuracy-level 1
  python -m tools.maya_export.quant_sweep_backbone eval  --bits 4 --block 128 --asym --accuracy-level 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnx

from tools.maya_export import eval_teacher_forced as E
from tools.maya_export import export_backbone as EB
from tools.maya_export.common import work_dir

N_LAYERS = EB.N_LAYERS


def tag(bits: int, block: int, algo: str, asym: bool, acc: int) -> str:
    if algo == "hqq":  # HQQWeightOnlyQuantConfig has no symmetric/accuracy_level knobs: do not claim them in the tag
        return f"b{bits}_blk{block}_hqq_default"
    return f"b{bits}_blk{block}_{algo}_{'asym' if asym else 'sym'}_acc{acc}"


def out_dir(name: str) -> Path:
    d = work_dir() / "backbone_build" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_quantizer(model, bits: int, block: int, algo: str, asym: bool, acc: int):
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig, HQQWeightOnlyQuantConfig, MatMulNBitsQuantizer)
    if algo == "rtn":
        # algo_config overrides the quantizer's own bits/accuracy_level, so they must be set on the config
        cfg = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=not asym, accuracy_level=acc, bits=bits)
    else:
        # HQQ ignores is_symmetric and accuracy_level (rejected on the CLI); defaults are untested here
        cfg = HQQWeightOnlyQuantConfig(bits=bits, block_size=block)
    return MatMulNBitsQuantizer(model, bits=bits, block_size=block, is_symmetric=not asym, accuracy_level=acc,
                                algo_config=cfg)


def quantize_layer(src: Path, dst: Path, **kw) -> None:
    model = onnx.load(str(src))  # loads external data of the fp32 layer
    q = make_quantizer(model, **kw)
    q.process()
    EB._save_external(q.model.model, dst)


def build(bits: int, block: int, algo: str, asym: bool, acc: int) -> Path:
    name = tag(bits, block, algo, asym, acc)
    od = out_dir(name)
    fp32 = work_dir() / "backbone_build" / "fp32"
    kw = dict(bits=bits, block=block, algo=algo, asym=asym, acc=acc)
    layer_paths = []
    for i in range(N_LAYERS):
        dst = od / f"layer_{i:02d}.onnx"
        t0 = time.time()
        quantize_layer(fp32 / f"layer_{i:02d}.onnx", dst, **kw)
        layer_paths.append(dst)
        print(f"layer {i:02d} {name} ok {time.time() - t0:.1f}s", flush=True)
    out = od / f"backbone_{name}.onnx"
    EB.merge(layer_paths, fp32 / "head.onnx", out, name)
    print(f"wrote {out} and {out.with_suffix('.data').name}", flush=True)
    return out


def evaluate(bits: int, block: int, algo: str, asym: bool, acc: int, onnx_path: Path | None = None) -> dict:
    name = tag(bits, block, algo, asym, acc)
    out = onnx_path or out_dir(name) / f"backbone_{name}.onnx"
    orig = E._run_candidate
    E._run_candidate = lambda _unused, items, c, s: orig(out, items, c, s)
    try:
        res = E.backbone(name)
    finally:
        E._run_candidate = orig
    res.update({"tag": name, "bits": bits, "block": block, "algo": algo, "asym": asym, "accuracy_level": acc,
                "onnx": str(out)})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "eval"))
    ap.add_argument("--bits", type=int, choices=(4, 8), required=True)
    ap.add_argument("--block", type=int, choices=(32, 64, 128), required=True)
    ap.add_argument("--algo", choices=("rtn", "hqq"), default="rtn")
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--accuracy-level", type=int, choices=(0, 1, 2, 3, 4), default=None,
                    help="rtn only (default 4); not supported for hqq")
    ap.add_argument("--onnx", type=Path, default=None, help="eval an existing merged graph instead of the tag's")
    a = ap.parse_args()
    if a.algo == "hqq" and (a.asym or a.accuracy_level is not None):
        ap.error("--algo hqq ignores --asym and --accuracy-level (HQQWeightOnlyQuantConfig has no such options); omit them")
    kw = dict(bits=a.bits, block=a.block, algo=a.algo, asym=a.asym,
              acc=4 if a.accuracy_level is None else a.accuracy_level)
    if a.mode == "build":
        build(**kw)
    else:
        print(json.dumps(evaluate(**kw, onnx_path=a.onnx), indent=2), flush=True)
