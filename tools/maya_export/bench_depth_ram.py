"""RAM / speed / codes-parity bench for one sampled depth graph (the runtime's Depth.frame graph).

Session options match the runtime (maya_fast.ort_opts): intra_op threads, allow_spinning 0, no CPU arena,
ORT_ENABLE_ALL. RSS sampled by a 2 ms background thread:
  load_rss_mb      RSS right after InferenceSession() returns
  load_peak_mb     peak RSS during InferenceSession() (ru_maxrss: bytes on macOS, KiB on Linux)
  first5_peak_mb   peak RSS during the first 5 frames
  steady_rss_mb    RSS after 40 parity frames + 3 warm-up + 30 timed frames
  run_peak_mb      peak RSS over all frames after load
  ms_per_frame     median of 30 timed frames (same feed)
Parity: 40 frames with feeds from seeds 1234+i (same feeds as the loader bench); codes saved to OUT.codes.npy and
compared to --ref when given.

Run one process per configuration under memguard (one model at a time):
    python tools/memguard.py --rss-limit-mb 5000 --min-free-mb 5500 -- \
        python tools/maya_export/bench_depth_ram.py MODEL OUT --threads 8 --ref base.codes.npy
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
import psutil

N_PARITY = 40


class Sampler:
    def __init__(self) -> None:
        self.p = psutil.Process()
        self.peak = 0
        self._stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self) -> None:
        while not self._stop:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.002)

    def window(self) -> float:
        """Return the peak since the last call (MiB) and reset it to the current RSS."""
        cur = self.p.memory_info().rss
        peak = max(self.peak, cur)
        self.peak = cur
        return peak / 2**20

    def stop(self) -> None:
        self._stop = True
        self.t.join()


def feed(i: int) -> dict:
    r = np.random.default_rng(1234 + i)
    return {
        "h": (r.standard_normal((1, 2048)) * 0.5).astype(np.float32),
        "cb0": np.array([r.integers(0, 2048)], dtype=np.int64),
        "mem": (r.standard_normal((1, 1, 1024)) * 0.1).astype(np.float32),
        "noise": r.gumbel(size=(1, 15, 2048)).astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()

    o = ort.SessionOptions()
    o.intra_op_num_threads = a.threads
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    o.enable_cpu_mem_arena = False
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    proc = psutil.Process()
    feeds = [feed(i) for i in range(N_PARITY)]
    rss0 = proc.memory_info().rss / 2**20
    s = Sampler()
    s.window()
    t0 = time.perf_counter()
    sess = ort.InferenceSession(a.model, o, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    # the sampler thread is starved while InferenceSession() holds the GIL: use the kernel high-water mark
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss_mb = maxrss / 2**20 if sys.platform == "darwin" else maxrss / 2**10  # macOS: bytes; Linux: KiB
    load_peak = max(s.window(), maxrss_mb)
    load_rss = proc.memory_info().rss / 2**20

    codes = []
    for i in range(N_PARITY):
        codes.append(sess.run(None, feeds[i])[0].reshape(-1).astype(np.int64))
        if i == 4:
            first5_peak = s.window()
    codes = np.stack(codes)
    f = feeds[0]
    for _ in range(3):
        sess.run(None, f)
    ts = []
    for _ in range(30):
        t = time.perf_counter()
        sess.run(None, f)
        ts.append(time.perf_counter() - t)
    run_peak = max(first5_peak, s.window())
    steady = proc.memory_info().rss / 2**20
    s.stop()

    res = dict(model=os.path.basename(a.model), ort=ort.__version__, threads=a.threads,
               rss_before_mb=round(rss0), load_rss_mb=round(load_rss), load_peak_mb=round(load_peak),
               first5_peak_mb=round(first5_peak), steady_rss_mb=round(steady), run_peak_mb=round(run_peak),
               load_s=round(load_s, 2), ms_per_frame=round(statistics.median(ts) * 1e3, 1),
               ms_p90=round(sorted(ts)[int(0.9 * len(ts))] * 1e3, 1))
    np.save(a.out + ".codes.npy", codes)
    if a.ref:
        ref = np.load(a.ref)
        res["codes_identical"] = bool(np.array_equal(ref, codes))
        res["codes_match_frac"] = round(float((ref == codes).mean()), 4)
        res["codes_match_frac_levels1_15"] = round(float((ref[:, 1:] == codes[:, 1:]).mean()), 4)
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
