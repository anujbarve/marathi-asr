from __future__ import annotations

import os
import onnxruntime as ort


def session_options(threads: int | None = None) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    if threads is None:
        cpu_count = os.cpu_count() or 1
        threads = min(8, cpu_count)
    opts.intra_op_num_threads = max(1, threads)
    opts.inter_op_num_threads = 1
    # Global Constraints:
    # intra_op_num_threads = requested threads (default min(8, physical cores)),
    # session.intra_op.allow_spinning=0, enable_cpu_mem_arena=False, ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.enable_cpu_mem_arena = False
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts
