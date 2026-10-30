import unittest

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from tools.maya_export import export_backbone as EB
from tools.maya_export.common import ckpt_paths, work_dir

# The full fp32 fused graph is 5.6 GB and cannot be loaded under the 4 GB limit, so the merge is checked on the
# first two layers (same merge code, layer count 2). The full model is then the same graph over 28 layers.
N = 2
BUILD = work_dir() / "backbone_build" / "fp32"
TOL = 1e-4


def _sess(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class TestBackboneMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = [BUILD / f"layer_{i:02d}.onnx" for i in range(N)] + [BUILD / "head.onnx"]
        if not all(p.exists() for p in paths):
            raise unittest.SkipTest("per-layer fp32 graphs not built (run export_backbone --precision fp32)")
        cls.layers = [_sess(BUILD / f"layer_{i:02d}.onnx") for i in range(N)]
        cls.head = _sess(BUILD / "head.onnx")
        cls.fused_path = BUILD / f"fused{N}_fp32.onnx"
        if not cls.fused_path.exists():
            EB.merge([BUILD / f"layer_{i:02d}.onnx" for i in range(N)], BUILD / "head.onnx", cls.fused_path, "fp32")
        cls.fused = _sess(cls.fused_path)
        cfg = AutoConfig.from_pretrained(ckpt_paths()["backbone_dir"])
        cfg._attn_implementation = "sdpa"
        cls.rot = Qwen3RotaryEmbedding(config=cfg).eval()

    def _rope(self, T, past):
        cos, sin = EB.rope(self.rot, T, past)
        return cos.numpy(), sin.numpy()

    def test_fused_matches_layers(self):
        rng = np.random.default_rng(0)
        steps = [(8, 0), (1, 8), (1, 9), (1, 10), (1, 11)]  # 8 prefill tokens, then 4 decode steps
        kv_fused = {f"past_{kv}_{i}": np.zeros((1, EB.N_KV_HEADS, 0, EB.HEAD_DIM), np.float32)
                    for i in range(N) for kv in ("k", "v")}
        kv_chain = [(np.zeros((1, EB.N_KV_HEADS, 0, EB.HEAD_DIM), np.float32),
                     np.zeros((1, EB.N_KV_HEADS, 0, EB.HEAD_DIM), np.float32)) for _ in range(N)]
        for T, past in steps:
            x = (rng.standard_normal((1, T, EB.D_MODEL)) * 0.02).astype(np.float32)
            cos, sin = self._rope(T, past)
            mask = EB.masks(T, past).numpy()

            feed = {"embeds": x, "cos": cos, "sin": sin, "mask": mask, **kv_fused}
            hid_f, cb_f, *present = self.fused.run(None, feed)
            for i in range(N):
                kv_fused[f"past_k_{i}"] = present[2 * i]
                kv_fused[f"past_v_{i}"] = present[2 * i + 1]

            h = x
            for i in range(N):
                out, nk, nv = self.layers[i].run(None, {"h": h, "cos": cos, "sin": sin, "mask": mask,
                                                        "past_k": kv_chain[i][0], "past_v": kv_chain[i][1]})
                kv_chain[i] = (nk, nv)
                h = out
            hid_c, cb_c = self.head.run(None, {"hid": h})

            rel_h = float(np.abs(hid_f - hid_c).max() / np.abs(hid_c).max())
            rel_c = float(np.abs(cb_f - cb_c).max() / np.abs(cb_c).max())
            self.assertLessEqual(rel_h, TOL, f"hidden T={T} past={past}")
            self.assertLessEqual(rel_c, TOL, f"cb0 T={T} past={past}")


if __name__ == "__main__":
    unittest.main()
