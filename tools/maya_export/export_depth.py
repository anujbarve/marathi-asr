"""Export the 15 per-level depth graphs (fp32) to work_dir()/depth_build/fp32/level_XX.onnx.

Each level's weights are read from the checkpoint and loaded into a fresh DepthLevel; one level at a time.

    .venv/bin/python tools/memguard.py --rss-limit-mb 4000 --min-free-mb 4000 -- \
        .venv-tts/bin/python -m tools.maya_export.export_depth
"""
from __future__ import annotations

import torch

from tts.model.depth_v2 import GROUP_OF_LEVEL
from tts.tokens import N_Q

from tools.maya_export.common import ckpt_paths, work_dir
from tools.maya_export.depth_frame import DepthLevel, Q_SERVE
from tools.maya_export.export_backbone import D_MODEL, _export, _quantize_wo8, _save_external
from tools.maya_export.streamed_backbone import _Weights, shard_files

D_DEPTH = 1024
N_LEVELS = 15  # serving q_serve = 16: levels 1..15 (the checkpoint has up to 31)


def ckpt_name(name: str, level: int) -> str:
    """DepthLevel parameter name -> checkpoint key (without the model. prefix)."""
    if name.startswith("trunk."):
        _, li, rest = name.split(".", 2)
        return f"depth.groups.{GROUP_OF_LEVEL[level]}.{li}.{rest}"
    if name.startswith("head."):
        return f"depth.heads.{level - 1}.weight"
    return f"depth.{name}"


def _head_nodes(model) -> list[str]:
    """Names of the node(s) producing the `logits` output (the level head's MatMul): kept in fp32 for wo8_fph."""
    names = []
    for node in model.graph.node:
        if "logits" in node.output:
            names.append(node.name)
    return names


def export_level(level: int, weights: _Weights, build, precision: str) -> None:
    mod = DepthLevel(level).eval()
    state = {name: weights.get(ckpt_name(name, level)).float() for name, _ in mod.named_parameters()}
    mod.load_state_dict(state, strict=True)
    del state
    h = torch.randn(1, D_MODEL) * 0.02
    codes = torch.zeros(1, Q_SERVE, dtype=torch.long)
    mem = torch.zeros(1, 1, D_DEPTH)
    tmp = build / f"level_{level:02d}_raw.onnx"
    _export(mod, (h, codes, mem), tmp, input_names=["h", "codes", "mem"], output_names=["logits"],
            dynamic_axes={"h": {0: "B"}, "codes": {0: "B"}, "mem": {0: "B"}, "logits": {0: "B"}})
    import onnx

    model = onnx.load(str(tmp))
    if precision == "wo8":
        model = _quantize_wo8(model)
    elif precision == "wo8_fph":  # int8 trunks, fp32 head
        model = _quantize_wo8(model, exclude=_head_nodes(model))
    _save_external(model, build / f"level_{level:02d}.onnx")
    tmp.unlink()
    tmp.with_suffix(".onnx.data").unlink(missing_ok=True)
    del mod


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=("fp32", "wo8", "wo8_fph"), default="fp32")
    precision = ap.parse_args().precision
    build = work_dir() / "depth_build" / precision
    build.mkdir(parents=True, exist_ok=True)
    paths = ckpt_paths()
    weights = _Weights(shard_files(paths["ckpt"]))
    for level in range(1, N_LEVELS + 1):
        export_level(level, weights, build, precision)
        print(f"level {level:02d} fp32 ok", flush=True)


if __name__ == "__main__":
    main()
