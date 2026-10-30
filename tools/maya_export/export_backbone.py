"""Fused backbone ONNX: 28 Qwen3 decoder layers + final RMSNorm + cb0 head, with KV cache in/out.

Build (one layer resident at a time, so it fits the 4 GB limit):
  1. each layer: transformers Qwen3DecoderLayer loaded from the checkpoint (tools/maya_export/streamed_backbone.py),
     exported to ONNX (opset 17, dynamo=False), saved with external data;
     for wo8 quantized with MatMulNBitsQuantizer(bits=8, block_size=32, is_symmetric=True, accuracy_level=4).
  2. head: final RMSNorm + cb0_head (fp32).
  3. merge: graph structure only (weights stay in the per-part data files), tensors renamed, then the weight bytes
     are copied into one data file next to the merged model.

Output: work_dir()/assets/v1/backbone_{precision}.onnx + backbone_{precision}.data
Per-part files: work_dir()/backbone_build/{precision}/ (kept for the layer-by-layer reference in the tests).

    .venv/bin/python tools/memguard.py --rss-limit-mb 4000 --min-free-mb 5000 -- \
        .venv-tts/bin/python -m tools.maya_export.export_backbone --precision wo8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper

from tools.maya_export.common import ckpt_paths, work_dir
from tools.maya_export.streamed_backbone import StreamedBackbone, shard_files

N_LAYERS = 28
N_KV_HEADS = 8
HEAD_DIM = 128
D_MODEL = 2048
N_CB0 = 2049  # 2048 codes + <aeos>
OPSET = 17
PREFILL = 8   # trace length for the export; dynamic axes cover any T
COPY_CHUNK = 1 << 24


class LayerKV(nn.Module):
    """One decoder layer, KV cache as flat tensors, additive attention mask passed in."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, h, cos, sin, mask, past_k, past_v):
        from transformers import DynamicCache

        cache = DynamicCache()
        cache.update(past_k, past_v, 0)
        out = self.layer(h, attention_mask=mask, position_ids=None, past_key_values=cache,
                         use_cache=True, cache_position=None, position_embeddings=(cos, sin))
        if isinstance(out, tuple):
            out = out[0]
        return out, cache.layers[0].keys, cache.layers[0].values


class Head(nn.Module):
    def __init__(self, norm: nn.Module, cb0_head: nn.Linear) -> None:
        super().__init__()
        self.norm = norm
        self.cb0_head = cb0_head

    def forward(self, hid):
        x = self.norm(hid)
        return x, self.cb0_head(x)


def masks(T: int, past: int) -> torch.Tensor:
    """Additive causal mask (1, 1, T, past+T): query i sees keys j <= past + i."""
    i = torch.arange(T)[:, None]
    j = torch.arange(past + T)[None, :]
    return torch.zeros(T, past + T).masked_fill(j > past + i, float("-inf"))[None, None]


def rope(rot, T: int, past: int) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(past, past + T)[None]
    return rot(torch.zeros(1, T, 1), pos)


def _export(module: nn.Module, args: tuple, path: Path, **kw) -> None:
    with torch.inference_mode():
        torch.onnx.export(module, args, str(path), opset_version=OPSET, dynamo=False,
                          do_constant_folding=True, **kw)


def _save_external(model: onnx.ModelProto, path: Path) -> None:
    # onnx appends to an existing data file; remove it so a rebuild does not leave stale bytes behind
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=path.name + ".data", size_threshold=1024)  # small tensors inline: ORT shape inference needs them


def _quantize_wo8(model: onnx.ModelProto, exclude: list[str] | None = None) -> onnx.ModelProto:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    q = MatMulNBitsQuantizer(model, bits=8, block_size=32, is_symmetric=True, accuracy_level=4,
                             nodes_to_exclude=exclude or [])
    q.process()
    return q.model.model


def export_layer(sb: StreamedBackbone, i: int, precision: str, build: Path) -> Path:
    sb._load_layer(i)
    sb.layer.self_attn.layer_idx = 0  # LayerKV owns a one-slot cache
    wrap = LayerKV(sb.layer).eval()
    h0 = torch.randn(1, PREFILL, D_MODEL) * 0.02
    cos, sin = rope(sb.rotary, PREFILL, 0)
    pk = torch.zeros(1, N_KV_HEADS, 0, HEAD_DIM)
    pv = torch.zeros(1, N_KV_HEADS, 0, HEAD_DIM)
    tmp = build / f"layer_{i:02d}_raw.onnx"
    _export(wrap, (h0, cos, sin, masks(PREFILL, 0), pk, pv), tmp,
            input_names=["h", "cos", "sin", "mask", "past_k", "past_v"],
            output_names=["out", "new_k", "new_v"],
            dynamic_axes={"h": {1: "t"}, "cos": {1: "t"}, "sin": {1: "t"}, "mask": {2: "t", 3: "total"},
                          "past_k": {2: "past"}, "past_v": {2: "past"},
                          "out": {1: "t"}, "new_k": {2: "total"}, "new_v": {2: "total"}})
    model = onnx.load(str(tmp))
    if precision == "wo8":
        model = _quantize_wo8(model)
    out = build / f"layer_{i:02d}.onnx"
    _save_external(model, out)
    tmp.unlink()
    tmp.with_suffix(".onnx.data").unlink(missing_ok=True)
    return out


def export_head(sb: StreamedBackbone, build: Path) -> Path:
    cb0 = nn.Linear(D_MODEL, N_CB0, bias=False)
    cb0.weight.data.copy_(sb.weights.get("backbone.cb0_head.weight").float())
    head = Head(sb.norm, cb0).eval()
    tmp = build / "head_raw.onnx"
    _export(head, (torch.randn(1, PREFILL, D_MODEL),), tmp, input_names=["hid"],
            output_names=["hidden", "cb0_logits"],
            dynamic_axes={"hid": {1: "t"}, "hidden": {1: "t"}, "cb0_logits": {1: "t"}})
    out = build / "head.onnx"
    _save_external(onnx.load(str(tmp)), out)
    tmp.unlink()
    tmp.with_suffix(".onnx.data").unlink(missing_ok=True)
    return out


def _rename(model: onnx.ModelProto, prefix: str, mapping: dict[str, str], keep: set[str]):
    """Rename every tensor and node of a part. Names in `mapping` are replaced; names in `keep` are shared
    across parts (cos/sin/mask); everything else gets `prefix`."""
    def new(name: str) -> str:
        if name == "":
            return name
        if name in mapping:
            return mapping[name]
        if name in keep:
            return name
        return prefix + name

    g = model.graph
    nodes = []
    for node in g.node:
        n = onnx.NodeProto()
        n.CopyFrom(node)
        del n.input[:]
        n.input.extend(new(x) for x in node.input)
        del n.output[:]
        n.output.extend(new(x) for x in node.output)
        n.name = prefix + node.name if node.name else ""
        nodes.append(n)
    inits = []
    for init in g.initializer:
        t = TensorProto()
        t.CopyFrom(init)
        t.name = new(init.name)
        inits.append(t)
    return nodes, inits


def merge(layer_paths: list[Path], head_path: Path, out_path: Path, precision: str) -> None:
    all_nodes, all_inits, opsets = [], [], {}
    for i, path in enumerate(layer_paths):
        m = onnx.load(str(path), load_external_data=False)
        src_in = "embeds" if i == 0 else f"hid_{i - 1}"
        mapping = {"h": src_in, "past_k": f"past_k_{i}", "past_v": f"past_v_{i}",
                   "out": f"hid_{i}", "new_k": f"present_k_{i}", "new_v": f"present_v_{i}"}
        nodes, inits = _rename(m, f"L{i}_", mapping, keep={"cos", "sin", "mask"})
        all_nodes += nodes
        all_inits += inits
        for op in m.opset_import:
            opsets[op.domain] = max(opsets.get(op.domain, 0), op.version)
    hm = onnx.load(str(head_path), load_external_data=False)
    nodes, inits = _rename(hm, "H_", {"hid": f"hid_{len(layer_paths) - 1}", "hidden": "hidden",
                                      "cb0_logits": "cb0_logits"}, keep=set())
    all_nodes += nodes
    all_inits += inits
    for op in hm.opset_import:
        opsets[op.domain] = max(opsets.get(op.domain, 0), op.version)

    data_name = out_path.with_suffix(".data").name
    _consolidate(all_inits, layer_paths + [head_path], out_path.parent, data_name)

    inputs = [helper.make_tensor_value_info("embeds", TensorProto.FLOAT, [1, "T", D_MODEL]),
              helper.make_tensor_value_info("cos", TensorProto.FLOAT, [1, "T", HEAD_DIM]),
              helper.make_tensor_value_info("sin", TensorProto.FLOAT, [1, "T", HEAD_DIM]),
              helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, "T", "total"])]
    outputs = [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, "T", D_MODEL]),
               helper.make_tensor_value_info("cb0_logits", TensorProto.FLOAT, [1, "T", N_CB0])]
    n = len(layer_paths)
    for i in range(n):
        inputs.append(helper.make_tensor_value_info(f"past_k_{i}", TensorProto.FLOAT, [1, N_KV_HEADS, "P", HEAD_DIM]))
        inputs.append(helper.make_tensor_value_info(f"past_v_{i}", TensorProto.FLOAT, [1, N_KV_HEADS, "P", HEAD_DIM]))
        outputs.append(helper.make_tensor_value_info(f"present_k_{i}", TensorProto.FLOAT, [1, N_KV_HEADS, "total", HEAD_DIM]))
        outputs.append(helper.make_tensor_value_info(f"present_v_{i}", TensorProto.FLOAT, [1, N_KV_HEADS, "total", HEAD_DIM]))

    graph = helper.make_graph(all_nodes, f"maya2_backbone_{precision}", inputs, outputs, all_inits)
    opset_imports = [helper.make_opsetid(d, v) for d, v in sorted(opsets.items())]
    ir_version = onnx.load(str(layer_paths[0]), load_external_data=False).ir_version
    model = helper.make_model(graph, opset_imports=opset_imports, ir_version=ir_version)
    onnx.save_model(model, str(out_path))


def _consolidate(inits: list[TensorProto], parts: list[Path], out_dir: Path, data_name: str) -> None:
    """Copy every external tensor's bytes into one data file, updating offset and location in place."""
    offset = 0
    with open(out_dir / data_name, "wb") as dst:
        for t in inits:
            if t.data_location != TensorProto.EXTERNAL:
                continue
            entries = {e.key: e.value for e in t.external_data}
            loc = entries["location"]
            src = _source_file(loc, parts)
            length = int(entries["length"])
            start = int(entries.get("offset", 0))
            with open(src, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise IOError(f"short read for {t.name} in {src}")
                    dst.write(chunk)
                    remaining -= len(chunk)
            del t.external_data[:]
            for k, v in (("location", data_name), ("offset", str(offset)), ("length", str(length))):
                e = t.external_data.add()
                e.key, e.value = k, v
            offset += length


def _source_file(location: str, parts: list[Path]) -> Path:
    for p in parts:
        if (p.parent / location).exists():
            return p.parent / location
    raise FileNotFoundError(location)


def build(precision: str) -> Path:
    if precision not in ("fp32", "wo8"):
        raise ValueError(precision)
    build_dir = work_dir() / "backbone_build" / precision
    build_dir.mkdir(parents=True, exist_ok=True)
    paths = ckpt_paths()
    sb = StreamedBackbone(paths["backbone_dir"], shard_files(paths["ckpt"]), n_layers=N_LAYERS)
    layer_paths = []
    for i in range(N_LAYERS):
        layer_paths.append(export_layer(sb, i, precision, build_dir))
        print(f"layer {i:02d} {precision} ok", flush=True)
    head = export_head(sb, build_dir)
    out_dir = work_dir() / "assets" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"backbone_{precision}.onnx"
    merge(layer_paths, head, out, precision)
    print(f"wrote {out} and {out.with_suffix('.data').name}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=("fp32", "wo8"), required=True)
    build(ap.parse_args().precision)
