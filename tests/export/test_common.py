import unittest
import tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file

from tools.maya_export.common import iter_ckpt_tensors


class TestCommon(unittest.TestCase):
    def test_iter_ckpt_tensors_skips_prefix(self):
        # tiny safetensors file in a temp dir with keys backbone.lm.model.layers.0.w and depth.x;
        # skip backbone.lm.model.layers. -> only depth.x yielded, once
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.safetensors"
            tensors = {
                "backbone.lm.model.layers.0.w": torch.tensor([1.0, 2.0]),
                "depth.x": torch.tensor([3.0, 4.0, 5.0]),
                "model.depth.y": torch.tensor([6.0]),
            }
            save_file(tensors, str(p))

            res = list(iter_ckpt_tensors(skip_prefixes=("backbone.lm.model.layers.",), files=[str(p)]))
            names = [name for name, _ in res]
            self.assertIn("depth.x", names)
            self.assertIn("depth.y", names)  # model. prefix stripped
            self.assertNotIn("backbone.lm.model.layers.0.w", names)
            self.assertEqual(len(res), 2)


if __name__ == "__main__":
    unittest.main()
