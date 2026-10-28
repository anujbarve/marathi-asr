"""Export the single fp16-storage depth graph (levels 1..15) to work_dir()/depth_build/fp16/depth_all.onnx.

    .venv/bin/python tools/memguard.py --rss-limit-mb 4000 --min-free-mb 4000 -- \
        .venv-tts/bin/python -m tools.maya_export.export_depth_all
"""
from __future__ import annotations

import onnx
import torch

from tools.maya_export.common import ckpt_paths, work_dir
from tools.maya_export.depth_all import LEVELS, DepthAll, load_fp16
from tools.maya_export.export_backbone import D_MODEL, _save_external
from tools.maya_export.streamed_backbone import _Weights, shard_files

Q_SERVE = 16


def ckpt_name(name: str) -> str:
    """DepthAll parameter name -> checkpoint key (without the model. prefix)."""
    if name.startswith("groups."):
        _, g, li, rest = name.split(".", 3)
        return f"depth.groups.{g}.{li}.{rest}"
    return f"depth.{name}"


def main() -> None:
    out_dir = work_dir() / "depth_build" / "fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = _Weights(shard_files(ckpt_paths()["ckpt"]))
    with torch.device("meta"):
        model = DepthAll()
    load_fp16(model, lambda n: weights.get(ckpt_name(n)))
    model.eval()
    h = torch.randn(1, D_MODEL) * 0.02
    codes = torch.zeros(1, Q_SERVE, dtype=torch.long)
    mem = torch.zeros(1, 1, 1024)
    tmp = out_dir / "depth_all_raw.onnx"
    # no constant folding: it would fold each Cast(fp16 weight) into an fp32 constant and double the stored weights
    with torch.inference_mode():
        torch.onnx.export(model, (h, codes, mem), str(tmp), input_names=["h", "codes", "mem"],
                          output_names=["logits"], opset_version=17, dynamo=False, do_constant_folding=False,
                          dynamic_axes={"h": {0: "B"}, "codes": {0: "B"}, "mem": {0: "B"}, "logits": {0: "B"}})
    _save_external(onnx.load(str(tmp)), out_dir / "depth_all.onnx")
    tmp.unlink()
    tmp.with_suffix(".onnx.data").unlink(missing_ok=True)
    print(f"wrote {out_dir / 'depth_all.onnx'} ({LEVELS} levels)", flush=True)


if __name__ == "__main__":
    main()
