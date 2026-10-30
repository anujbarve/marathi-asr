"""plan() on tiny synthetic graphs (no real weights)."""
from __future__ import annotations

import re
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.maya_export.quant_depth_frame import plan


def _graph(consumers):
    """One fp16 (4, 3) weight 'w' with one Cast per entry of consumers ('mm' = Transpose->MatMul, 'bad' = Add)."""
    w = numpy_helper.from_array(np.ones((4, 3), np.float16), "w")
    nodes, outs = [], []
    for i, kind in enumerate(consumers):
        nodes.append(helper.make_node("Cast", ["w"], [f"wf{i}"], name=f"cast{i}", to=TensorProto.FLOAT))
        if kind == "mm":
            nodes.append(helper.make_node("Transpose", [f"wf{i}"], [f"wt{i}"], name=f"tr{i}", perm=[1, 0]))
            nodes.append(helper.make_node("MatMul", ["x", f"wt{i}"], [f"y{i}"], name=f"mm{i}"))
        else:
            nodes.append(helper.make_node("Add", [f"wf{i}", f"wf{i}"], [f"y{i}"], name=f"add{i}"))
        outs.append(helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT, None))
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    return helper.make_model(helper.make_graph(nodes, "g", [x], outs, initializer=[w]))


class TestPlan(unittest.TestCase):
    def test_single_cast(self):
        targets, drop = plan(_graph(["mm"]), None)
        self.assertEqual([k for k, _ in targets["w"]], ["matmul"])
        self.assertEqual(drop, {"cast0", "tr0"})

    def test_shared_weight_accumulates_all_uses(self):
        targets, drop = plan(_graph(["mm", "mm"]), None)
        self.assertEqual([n.name for _, n in targets["w"]], ["mm0", "mm1"])
        self.assertEqual(drop, {"cast0", "tr0", "cast1", "tr1"})

    def test_mixed_convertible_and_not_raises(self):
        with self.assertRaises(ValueError):
            plan(_graph(["mm", "bad"]), None)

    def test_keep_regex_skips(self):
        targets, _ = plan(_graph(["mm"]), re.compile("^w$"))
        self.assertEqual(targets, {})


if __name__ == "__main__":
    unittest.main()
