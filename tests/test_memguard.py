import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil

from tools.memguard import GuardResult, run_guarded


class TestMemguard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="memguard_test_")
        self._saved_work = os.environ.get("MAYA_FAST_WORK")
        os.environ["MAYA_FAST_WORK"] = self._tmp

    def tearDown(self):
        if self._saved_work is None:
            os.environ.pop("MAYA_FAST_WORK", None)
        else:
            os.environ["MAYA_FAST_WORK"] = self._saved_work

    def test_kills_on_rss_limit(self):
        child = (
            "import time\n"
            "b = bytearray(300 * 1024 * 1024)\n"
            "for i in range(0, len(b), 4096):\n"
            "    b[i] = 1\n"
            "time.sleep(10)\n"
        )
        r = run_guarded(
            [sys.executable, "-c", child],
            rss_limit_mb=100,
            min_free_mb=0,
            poll_s=0.1,
        )
        self.assertTrue(r.killed, f"expected kill, got {r}")
        self.assertEqual(r.reason, "rss limit")
        self.assertIsInstance(r, GuardResult)

    def test_refuses_when_not_enough_free(self):
        r = run_guarded(
            [sys.executable, "-c", "pass"],
            rss_limit_mb=500,
            min_free_mb=int(psutil.virtual_memory().total / (1024 * 1024)) + 1,
        )
        self.assertEqual(r.returncode, -1)
        self.assertEqual(r.reason, "insufficient free memory")
        self.assertFalse(r.killed)

    def test_clean_run_reports_peak(self):
        child = (
            "import time\n"
            "b = bytearray(50 * 1024 * 1024)\n"
            "for i in range(0, len(b), 4096):\n"
            "    b[i] = 1\n"
            "time.sleep(1.5)\n"
        )
        r = run_guarded(
            [sys.executable, "-c", child],
            rss_limit_mb=500,
            min_free_mb=0,
            poll_s=0.1,
        )
        self.assertEqual(r.returncode, 0)
        self.assertIsNone(r.reason)
        self.assertFalse(r.killed)
        self.assertGreaterEqual(r.peak_rss_mb, 40)

    def test_lock_serializes_runs(self):
        marker = Path(self._tmp) / "markers.txt"
        child = (
            "import time\n"
            f"with open({str(marker)!r}, 'a') as f:\n"
            "    f.write('S\\n')\n"
            "    f.flush()\n"
            "time.sleep(0.7)\n"
            f"with open({str(marker)!r}, 'a') as f:\n"
            "    f.write('E\\n')\n"
            "    f.flush()\n"
        )
        barrier = threading.Barrier(2)
        results = []
        errors = []

        def run():
            try:
                barrier.wait(timeout=10)
                results.append(
                    run_guarded(
                        [sys.executable, "-c", child],
                        rss_limit_mb=500,
                        min_free_mb=0,
                        poll_s=0.1,
                    )
                )
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2, "both guarded runs must finish")
        for r in results:
            self.assertEqual(r.returncode, 0, f"run failed: {r}")
            self.assertIsNone(r.reason)

        events = marker.read_text().split()
        self.assertEqual(events.count("S"), 2)
        self.assertEqual(events.count("E"), 2)
        depth = 0
        for ev in events:
            depth += 1 if ev == "S" else -1
            self.assertGreaterEqual(depth, 0, f"overlapping runs: {events}")
            self.assertLessEqual(depth, 1, f"overlapping runs: {events}")
        self.assertEqual(depth, 0, f"unbalanced run markers: {events}")


if __name__ == "__main__":
    unittest.main()
