"""Backbone-alone RSS and ms/step with the runtime Backbone class (run under tools/memguard.py with the runtime python).
Usage: python bench_backbone.py <backbone.onnx>   (prefill past_len 32 / 256, then 100 T=1 steps each)."""
import sys, os, json, time, subprocess, statistics
import numpy as np
sys.path.insert(0, "/Users/anujbarve/Documents/maya-fast-loader/src")
from maya_fast.backbone import Backbone
path = sys.argv[1]
def rss(): return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).split()[0]) / 1024
t0 = time.perf_counter(); bb = Backbone(path, manifest_path="/nonexistent"); load_s = time.perf_counter() - t0
rss_load = rss()
rng = np.random.default_rng(0)
out = {"path": path, "load_s": round(load_s, 2), "rss_after_load_mb": round(rss_load, 1)}
for P in (32, 256):
    bb.reset()
    bb.step((rng.standard_normal((P, bb.d_model)) * 0.02).astype(np.float32))
    ts = []
    for _ in range(100):
        x = (rng.standard_normal((1, bb.d_model)) * 0.02).astype(np.float32)
        t = time.perf_counter(); bb.step(x); ts.append((time.perf_counter() - t) * 1000)
    out[f"ms_median_past{P}"] = round(statistics.median(ts), 2)
    out[f"ms_p90_past{P}"] = round(sorted(ts)[89], 2)
out["rss_end_mb"] = round(rss(), 1)
print(json.dumps(out))
