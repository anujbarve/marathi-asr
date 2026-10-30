"""Rewrite work_dir()/assets/v1/manifest.json with every runtime file and the runtime shapes.

Shapes used by the runtime: rope_theta (Qwen3 backbone), n_layers / n_heads (KV heads) / head_dim, d_mem (depth
speaker memory width). Run after the backbone, depth and Mimi exports.
"""
from __future__ import annotations

import json

from tools.maya_export.common import work_dir
from tools.maya_export.manifest import write_manifest

RUNTIME_FILES = [
    "tokenizer.json", "embed_bf16.bin", "audio_in_bf16.bin", "prompt.npz", "voices.json", "controls.json",
    "backbone_wo8.onnx", "backbone_wo8.data",
    "depth_frame_fp16.onnx", "depth_frame_fp16.onnx.data",
    "mimi_decoder.onnx",
]


def main() -> None:
    out = work_dir() / "assets" / "v1"
    prev = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    shapes = dict(prev.get("shapes", {}))
    shapes.update({"rope_theta": 1000000.0, "n_layers": 28, "n_heads": 8, "head_dim": 128, "d_mem": 1024})
    write_manifest(out, RUNTIME_FILES, shapes=shapes)
    print(f"wrote {out / 'manifest.json'} ({len(RUNTIME_FILES)} files)", flush=True)


if __name__ == "__main__":
    main()
