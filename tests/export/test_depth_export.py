import unittest

import numpy as np
import onnxruntime as ort
import torch

from tools.maya_export.common import ckpt_paths, work_dir
from tools.maya_export.streamed_backbone import _Weights, shard_files

BUILD = work_dir() / "depth_build" / "fp32"
REF = work_dir() / "golden" / "ref" / "conv_01_s0.npz"
FRAMES = 20
N_LEVELS = 15


def _sess(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class TestDepthLevelParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all((BUILD / f"level_{i:02d}.onnx").exists() for i in range(1, N_LEVELS + 1)):
            raise unittest.SkipTest("per-level depth graphs not built")
        if not REF.exists():
            raise unittest.SkipTest("reference corpus not found")
        cls.weights = _Weights(shard_files(ckpt_paths()["ckpt"]))

    def test_chain_matches_reference_depth_logits(self):
        """The reference is stored in fp16. Logits here reach about 23, where one fp16 ulp is 0.0156, so the
        check is: fp16-rounded outputs equal the stored values except for one-ulp rounding ties (<= 1%), and the
        absolute error is at most half an ulp. The fp32 port is checked against upstream PyTorch fp32 separately
        (tools/maya_export/depth_check.py)."""
        with np.load(REF) as z:
            codes = z["codes"][:, :FRAMES].T.astype(np.int64)  # (F, 16)
            h = z["h_frames"][:FRAMES].astype(np.float32)  # (F, 2048)
            ref = z["depth_logits"][:FRAMES].astype(np.float32)  # (F, 15, 2048), fp16 storage
        mem = self.weights.get("depth.null_mem").float().numpy().reshape(1, 1, 1024)
        mem = np.repeat(mem, FRAMES, axis=0)
        mismatch, total, worst = 0, 0, 0.0
        for level in range(1, N_LEVELS + 1):
            sess = _sess(BUILD / f"level_{level:02d}.onnx")
            got = sess.run(None, {"h": h, "codes": codes, "mem": mem})[0]
            del sess
            want = ref[:, level - 1]
            g16 = got.astype(np.float16).astype(np.float32)
            mismatch += int((g16 != want).sum())
            total += want.size
            worst = max(worst, float(np.abs(g16 - want).max()))
        print(f"fp16-rounded mismatches {mismatch}/{total}, max abs {worst:.7f}", flush=True)
        self.assertLessEqual(mismatch / total, 0.01)
        self.assertLessEqual(worst, 0.0078125)

if __name__ == "__main__":
    unittest.main()
