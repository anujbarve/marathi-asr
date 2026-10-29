"""Weight-only block quantization (MatMulNBits) of the fp16-storage depth graphs, streamed one tensor at a time.

Input: an fp16-storage depth graph from export_depth_all (depth_frame_fp16.onnx, the sampled runtime graph, or
depth_all.onnx, the teacher-forced graph used by the quality gate). In both, each Linear weight W (N, K) is an fp16
initializer used as Cast(W -> fp32) followed by either Transpose(perm 1,0) -> MatMul(s) or Gemm(transB=1).
Each such use becomes MatMulNBits(A, Wq, scales) with Wq quantized from the fp32 upcast of W (lossless upcast of
the fp16 source). Shared weights (group 8 for levels 9..15, the cross blocks for all levels) are quantized once
and referenced by every MatMulNBits node that used them.

Weights matching --keep (regex on the initializer name) stay fp16-stored behind their Cast. Embeddings (Gather)
are left as they are. Graph I/O is unchanged.

Never holds more than one weight (plus its quantized copy) in memory: initializers are read from the source
external-data file by offset and written sequentially to the destination data file.

    PYTHONPATH=. .venv-tts/bin/python -m tools.maya_export.quant_depth_frame \
        --src ~/.cache/maya_fast/depth_build/fp16/depth_frame_fp16.onnx \
        --dst ~/.cache/maya_fast/depth_build/q8a1/depth_frame_q8a1.onnx --bits 8 --block 32 --accuracy-level 1

Note (ORT MLAS, checked on main): 8-bit MatMulNBits has native kernels only for accuracy_level 4 (int8 compute);
accuracy_level 1 runs the generic path on every CPU: dequantize B to an fp32 scratch buffer on each call, then
SGEMM. That path is the same C++ code on x86 and ARM.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

NP = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16, TensorProto.INT64: np.int64,
      TensorProto.INT32: np.int32, TensorProto.INT8: np.int8, TensorProto.UINT8: np.uint8,
      TensorProto.BOOL: np.bool_, TensorProto.DOUBLE: np.float64}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def _read(t: TensorProto, src_dir: str) -> np.ndarray:
    if t.data_location == TensorProto.EXTERNAL:
        info = {e.key: e.value for e in t.external_data}
        off = int(info.get("offset", 0))
        n = int(np.prod(list(t.dims), dtype=np.int64)) * np.dtype(NP[t.data_type]).itemsize
        length = int(info.get("length", n))
        with open(os.path.join(src_dir, info["location"]), "rb") as fh:
            fh.seek(off)
            raw = fh.read(length)
        return np.frombuffer(raw, dtype=NP[t.data_type]).reshape(tuple(t.dims))
    return onnx.numpy_helper.to_array(t)


def quant_block(w_kn: np.ndarray, bits: int, block: int, symmetric: bool):
    """w_kn (K, N) fp32 (MatMul B layout). Returns packed (N, k_blocks, blob) uint8, scales (N, k_blocks) fp32,
    zero points (N, ceil(k_blocks / kpack)) uint8. Same C++ kernels as onnxruntime's MatMulNBitsQuantizer."""
    from onnxruntime.capi._pybind_state import quantize_matmul_4bits, quantize_matmul_8bits

    K, N = w_kn.shape
    kpack = 8 // bits
    k_blocks = (K + block - 1) // block
    blob = (block + kpack - 1) // kpack
    pad = k_blocks * block - K
    if pad:
        w_kn = np.pad(w_kn, ((0, pad), (0, 0)))
    w_kn = np.ascontiguousarray(w_kn, dtype=np.float32)
    packed = np.zeros((N, k_blocks, blob), dtype=np.uint8)
    zp = np.zeros((N, (k_blocks + kpack - 1) // kpack), dtype=np.uint8)
    scales = np.zeros((N, k_blocks), dtype=np.float32)
    fn = quantize_matmul_8bits if bits == 8 else quantize_matmul_4bits
    fn(packed, w_kn, scales, zp, block, N, K, symmetric)
    return packed, scales, zp


def plan(model: onnx.ModelProto, keep: re.Pattern | None):
    """Find the quantizable weights. Returns {init_name: [(kind, node, extra)]} and the set of node names to drop."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    out = {}
    unsupported = set()  # weights with at least one Cast use that cannot be converted
    drop = set()
    for n in g.node:
        if n.op_type != "Cast" or n.input[0] not in inits:
            continue
        t = inits[n.input[0]]
        if t.data_type != TensorProto.FLOAT16 or len(t.dims) != 2 or _attr(n, "to") != TensorProto.FLOAT:
            continue
        if keep is not None and keep.search(t.name):
            continue
        uses, ok, local_drop = [], True, {n.name}
        for c in cons[n.output[0]]:
            if c.op_type == "Transpose" and list(_attr(c, "perm", [])) == [1, 0]:
                local_drop.add(c.name)
                for mm in cons[c.output[0]]:
                    if mm.op_type == "MatMul" and mm.input[1] == c.output[0]:
                        uses.append(("matmul", mm))
                    else:
                        ok = False
            elif (c.op_type == "Gemm" and c.input[1] == n.output[0] and _attr(c, "transB", 0) == 1
                  and _attr(c, "transA", 0) == 0 and _attr(c, "alpha", 1.0) == 1.0 and _attr(c, "beta", 1.0) == 1.0):
                uses.append(("gemm", c))
            else:
                ok = False
        if ok and uses:
            out.setdefault(t.name, []).extend(uses)  # one weight may feed several Casts
            drop |= local_drop
        else:
            unsupported.add(t.name)
    mixed = sorted(set(out) & unsupported)
    if mixed:
        raise ValueError(f"weights with both convertible and unconvertible Cast uses: {mixed}")
    return out, drop


def convert(src: Path, dst: Path, bits: int, block: int, accuracy_level: int, symmetric: bool,
            keep: str | None) -> dict:
    model = onnx.load(str(src), load_external_data=False)
    g = model.graph
    keep_re = re.compile(keep) if keep else None
    targets, drop_nodes = plan(model, keep_re)
    replaced = {}  # node name -> list of new nodes
    dst.parent.mkdir(parents=True, exist_ok=True)
    loc = dst.name + ".data"
    data_path = dst.with_name(loc)
    data_path.unlink(missing_ok=True)
    src_dir = str(src.parent)
    new_inits, pos = [], 0
    stats = collections.Counter()
    used = set()
    for n in g.node:
        if n.name not in drop_nodes:
            used.update(n.input)

    def add_tensor(fh, name, arr, dtype):
        nonlocal pos
        raw = np.ascontiguousarray(arr).tobytes()
        t = TensorProto()
        t.name, t.data_type = name, dtype
        t.dims.extend(arr.shape)
        if len(raw) < 1024:  # small tensors inline (ORT shape inference needs them)
            t.raw_data = raw
        else:
            fh.write(raw)
            for k, v in (("location", loc), ("offset", str(pos)), ("length", str(len(raw)))):
                e = t.external_data.add()
                e.key, e.value = k, v
            t.data_location = TensorProto.EXTERNAL
            pos += len(raw)
        new_inits.append(t)

    with open(data_path, "wb") as fh:
        for t in g.initializer:
            if t.name in targets:
                w = _read(t, src_dir).astype(np.float32)  # (N, K)
                N, K = w.shape
                packed, scales, zp = quant_block(w.T, bits, block, symmetric)
                del w
                qn, sn, zn = t.name + f"_Q{bits}", t.name + "_scales", t.name + "_zp"
                add_tensor(fh, qn, packed, TensorProto.UINT8)
                add_tensor(fh, sn, scales, TensorProto.FLOAT)
                ins = [qn, sn]
                if not symmetric:
                    add_tensor(fh, zn, zp, TensorProto.UINT8)
                    ins.append(zn)
                stats["quantized_weights"] += 1
                stats["quantized_params"] += N * K
                for kind, node in targets[t.name]:
                    attrs = dict(K=K, N=N, bits=bits, block_size=block)
                    if accuracy_level:
                        attrs["accuracy_level"] = accuracy_level
                    nodes = []
                    if kind == "gemm" and len(node.input) > 2 and node.input[2]:
                        mid = node.output[0] + "_nbits"
                        nodes.append(helper.make_node("MatMulNBits", [node.input[0]] + ins, [mid],
                                                      name=node.name + "_nbits", domain="com.microsoft", **attrs))
                        nodes.append(helper.make_node("Add", [mid, node.input[2]], [node.output[0]],
                                                      name=node.name + "_bias"))
                    else:
                        nodes.append(helper.make_node("MatMulNBits", [node.input[0]] + ins, [node.output[0]],
                                                      name=node.name + "_nbits", domain="com.microsoft", **attrs))
                    replaced[node.name] = nodes
                    stats["matmulnbits_nodes"] += 1
                continue
            if t.name not in used:
                stats["dropped_inits"] += 1
                continue
            arr = _read(t, src_dir)
            if t.data_type == TensorProto.FLOAT16 and arr.ndim == 2:
                stats["kept_fp16_params"] += arr.size
            add_tensor(fh, t.name, arr, t.data_type)
    nodes = []
    for n in g.node:
        if n.name in drop_nodes:
            continue
        nodes.extend(replaced.get(n.name, [n]))
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    names = {t.name for t in new_inits}
    keep_inputs = [i for i in g.input if i.name not in names or i.name in {"h", "cb0", "mem", "noise", "codes"}]
    del g.input[:]
    g.input.extend(keep_inputs)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save_model(model, str(dst))
    stats["data_bytes"] = pos
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--accuracy-level", type=int, default=1)
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--keep", default=None, help="regex: initializers kept fp16-stored (e.g. 'heads\\.|cross\\.')")
    a = ap.parse_args()
    res = convert(a.src.expanduser(), a.dst.expanduser(), a.bits, a.block, a.accuracy_level, not a.asym, a.keep)
    print(res, flush=True)


if __name__ == "__main__":
    main()
