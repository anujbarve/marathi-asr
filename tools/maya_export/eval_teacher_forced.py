"""Teacher-forced quality gate for the fused backbone.

Reference: fp32 per-layer graphs, one layer resident at a time, run over each reference `seq_emb`
(work_dir()/golden/ref/*.npz, from golden_reference.py), then the fp32 head.
Candidate: the fused graph (backbone_{candidate}.onnx), run over the same sequences.
Compared at the generating positions only: the frame-i prediction is made at position n_p - 1 + i.

Metrics: top-1 agreement of cb0 argmax, and mean KL(ref || cand) over the distribution softmax(logits / 0.7)
(the sampling temperature), in nats.

    .venv/bin/python tools/memguard.py --rss-limit-mb 4000 -- \
        .venv-tts/bin/python -m tools.maya_export.eval_teacher_forced backbone --candidate wo8
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from tools.maya_export import export_backbone as EB
from tools.maya_export.common import ckpt_paths, work_dir
from tools.maya_export.export_depth import N_LEVELS as DEPTH_LEVELS
from tools.maya_export.streamed_backbone import _Weights, shard_files

N_LAYERS = EB.N_LAYERS
TEMP = 0.7
TOP1_GATE = 0.99
KL_GATE = 0.01
DEPTH_TOP1_GATE = 0.98
DEPTH_KL_GATE = 0.02


def _sess(path: Path, threads: int = 8) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _rope_table(max_len: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = AutoConfig.from_pretrained(ckpt_paths()["backbone_dir"])
    cfg._attn_implementation = "sdpa"
    rot = Qwen3RotaryEmbedding(config=cfg).eval()
    cos, sin = EB.rope(rot, max_len, 0)
    return cos.numpy(), sin.numpy()


def _load_refs(ref_dir: Path, n_prompt: dict[str, int]):
    items = []
    for path in sorted(glob.glob(str(ref_dir / "*_s*.npz"))):
        name = Path(path).stem
        with np.load(path) as z:
            seq = z["seq_emb"].astype(np.float32)
            F = int(z["codes"].shape[1])
        s_id = name.rsplit("_s", 1)[0]
        n_p = n_prompt[s_id]
        assert seq.shape[0] == n_p + F, (name, seq.shape, n_p, F)
        items.append({"name": name, "seq": seq, "n_p": n_p, "F": F})
    return items


def _gen_positions(n_p: int, F: int) -> np.ndarray:
    return n_p - 1 + np.arange(F)


def _run_candidate(backbone_path: Path, items, cos_tab, sin_tab):
    sess = _sess(backbone_path)
    out = []
    for it in items:
        T = it["seq"].shape[0]
        zeros = np.zeros((1, EB.N_KV_HEADS, 0, EB.HEAD_DIM), np.float32)
        feed = {"embeds": it["seq"][None], "cos": cos_tab[:, :T], "sin": sin_tab[:, :T],
                "mask": EB.masks(T, 0).numpy()}
        for i in range(N_LAYERS):
            feed[f"past_k_{i}"] = zeros
            feed[f"past_v_{i}"] = zeros
        res = sess.run(None, feed)
        cb0 = res[1][0]  # (T, 2049)
        out.append(cb0[_gen_positions(it["n_p"], it["F"])])
    del sess
    return out


def _run_reference(build_dir: Path, items, cos_tab, sin_tab):
    """One fp32 layer at a time over every sequence; the hidden state of every item is kept between layers."""
    hidden = [it["seq"][None].copy() for it in items]
    for i in range(N_LAYERS):
        sess = _sess(build_dir / f"layer_{i:02d}.onnx", threads=4)
        for k, it in enumerate(items):
            T = it["seq"].shape[0]
            zeros = np.zeros((1, EB.N_KV_HEADS, 0, EB.HEAD_DIM), np.float32)
            out, _, _ = sess.run(None, {"h": hidden[k], "cos": cos_tab[:, :T], "sin": sin_tab[:, :T],
                                        "mask": EB.masks(T, 0).numpy(), "past_k": zeros, "past_v": zeros})
            hidden[k] = out
        del sess
    head = _sess(build_dir / "head.onnx", threads=4)
    out = []
    for k, it in enumerate(items):
        _, cb0 = head.run(None, {"hid": hidden[k]})
        out.append(cb0[0][_gen_positions(it["n_p"], it["F"])])
    return out


def _softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = x.astype(np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def backbone(candidate: str) -> dict:
    ref_dir = work_dir() / "golden" / "ref"
    light = json.loads((work_dir() / "golden" / "light.json").read_text(encoding="utf-8"))
    voice = "Charvi"
    golden = np.load(work_dir() / "golden" / "light.npz")
    n_prompt = {s_id: int(golden[f"prompt/{s_id}/{voice}/n_prompt"]) for s_id in light["sentences"]}

    items = _load_refs(ref_dir, n_prompt)
    max_len = max(it["seq"].shape[0] for it in items)
    cos_tab, sin_tab = _rope_table(max_len)

    cand = _run_candidate(work_dir() / "assets" / "v1" / f"backbone_{candidate}.onnx", items, cos_tab, sin_tab)
    ref = _run_reference(work_dir() / "backbone_build" / "fp32", items, cos_tab, sin_tab)

    top1_hits, total, kls = 0, 0, []
    for c, r in zip(cand, ref):
        top1_hits += int((c.argmax(-1) == r.argmax(-1)).sum())
        total += r.shape[0]
        p = _softmax(r, TEMP)
        q = _softmax(c, TEMP)
        kls.append(float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)), axis=-1).sum()))
    result = {"top1": top1_hits / total, "kl_mean": sum(kls) / total, "frames": total,
              "files": len(items), "gate": {"top1>=": TOP1_GATE, "kl_mean<=": KL_GATE}}
    result["pass"] = bool(result["top1"] >= TOP1_GATE and result["kl_mean"] <= KL_GATE)
    return result


def depth(candidate: str) -> dict:
    """Per-level depth gate. Candidate: per-level graphs (depth_build/{candidate}); one level resident at a time,
    every file run through it. Reference: the stored depth_logits (fp16) of the corpus."""
    ref_dir = work_dir() / "golden" / "ref"
    items = []
    for path in sorted(glob.glob(str(ref_dir / "*_s*.npz"))):
        with np.load(path) as z:
            items.append({"codes": z["codes"].T.astype(np.int64), "h": z["h_frames"].astype(np.float32),
                          "ref": z["depth_logits"].astype(np.float32)})
    mem_w = _Weights(shard_files(ckpt_paths()["ckpt"]))
    null_mem = mem_w.get("depth.null_mem").float().numpy().reshape(1, 1, 1024)
    build = work_dir() / "depth_build" / candidate
    single = build / "depth_all.onnx"
    per_top1, per_kl = [], []
    if single.exists():
        # one graph, all levels: run each file once and read the 15 level outputs
        sess = _sess(single, threads=8)
        outs = []
        for it in items:
            F = it["h"].shape[0]
            outs.append(sess.run(None, {"h": it["h"], "codes": it["codes"],
                                        "mem": np.repeat(null_mem, F, axis=0)})[0])
        del sess
        levels = [(level, [o[:, level - 1] for o in outs]) for level in range(1, DEPTH_LEVELS + 1)]
    else:
        levels = None
    for level in range(1, DEPTH_LEVELS + 1):
        if levels is None:
            sess = _sess(build / f"level_{level:02d}.onnx", threads=8)
            got_per_item = []
            for it in items:
                F = it["h"].shape[0]
                mem = np.repeat(null_mem, F, axis=0)
                got_per_item.append(sess.run(None, {"h": it["h"], "codes": it["codes"], "mem": mem})[0])
            del sess
        else:
            got_per_item = levels[level - 1][1]
        hits, total, kl_sum = 0, 0, 0.0
        for it, got in zip(items, got_per_item):
            F = it["h"].shape[0]
            want = it["ref"][:, level - 1]
            hits += int((got.argmax(-1) == want.argmax(-1)).sum())
            total += F
            p = _softmax(want, TEMP)
            q = _softmax(got, TEMP)
            kl_sum += float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
        per_top1.append(hits / total)
        per_kl.append(kl_sum / total)
    passes = [t >= DEPTH_TOP1_GATE and k <= DEPTH_KL_GATE for t, k in zip(per_top1, per_kl)]
    return {"candidate": candidate, "frames": total, "files": len(items),
            "top1_per_level": per_top1, "kl_per_level": per_kl,
            "gate": {"top1>=": DEPTH_TOP1_GATE, "kl<=": DEPTH_KL_GATE},
            "pass_per_level": passes, "pass": all(passes)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["backbone", "depth"])
    ap.add_argument("--candidate", required=True,
                    help="build tag: depth_build/<tag>/depth_all.onnx (depth) or assets/v1/backbone_<tag>.onnx")
    a = ap.parse_args()
    res = backbone(a.candidate) if a.target == "backbone" else depth(a.candidate)
    print(json.dumps(res, indent=2))
