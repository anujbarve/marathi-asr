import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import maya_fast_eval as E
from benchmarks import maya_fast_listen as L


def row(system, cat, cer, wer=0.0, frames=10, rtf=1.0, bb=0.1, dp=0.2, mm=0.05, i=0):
    return {"system": system, "id": f"{cat}_{i}", "category": cat, "seed": 0,
            "cer": cer, "wer": wer, "cer_lenient": cer / 2, "wer_lenient": wer / 2,
            "frames": frames, "rtf": rtf, "audio_s": frames * 0.08,
            "backbone_s": bb, "depth_s": dp, "mimi_s": mm}


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.rows = [row("candidate", "a", 0.1, i=0), row("candidate", "a", 0.3, i=1),
                     row("candidate", "b", 0.5, frames=30, rtf=2.0, bb=0.3, dp=0.6, mm=0.15, i=2),
                     {**row("reference", "a", 0.2), "frames": 0}]
        self.s = E.aggregate(self.rows, {"candidate": 1234.0})

    def test_category_and_overall_means(self):
        c = self.s["candidate"]
        self.assertAlmostEqual(c["categories"]["a"]["cer"], 0.2)
        self.assertAlmostEqual(c["categories"]["b"]["cer"], 0.5)
        self.assertAlmostEqual(c["overall"]["cer"], 0.3)
        self.assertEqual(c["overall"]["n"], 3)
        self.assertAlmostEqual(c["overall"]["cer_lenient"], 0.15)

    def test_timing(self):
        c = self.s["candidate"]
        self.assertAlmostEqual(c["rtf_mean"], 4 / 3)
        self.assertEqual(c["rtf_max"], 2.0)
        # frame-weighted: (0.1+0.1+0.3) s / 50 frames = 10 ms
        self.assertAlmostEqual(c["ms_per_frame_backbone"], 10.0)
        self.assertAlmostEqual(c["ms_per_frame_depth"], 20.0)
        self.assertAlmostEqual(c["ms_per_frame_mimi"], 5.0)
        self.assertEqual(c["peak_rss_mb"], 1234.0)

    def test_reference_has_no_timing(self):
        self.assertNotIn("rtf_mean", self.s["reference"])

    def test_delta(self):
        self.assertAlmostEqual(E.delta_vs(self.s, "candidate", "reference")["cer"], 0.1)

    def test_table(self):
        t = E.format_table(self.s, "x")
        self.assertIn("| candidate CER |", t)
        self.assertIn("| reference WER |", t)
        self.assertIn("1234", t)
        self.assertIn("0.300", t)

    def test_limit_and_testset(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.json")
            json.dump([{"id": str(i), "category": "c", "text": "x"} for i in range(5)], open(p, "w"))
            self.assertEqual(len(E.load_testset(p, 3)), 3)
            self.assertEqual(len(E.load_testset(p, None)), 5)


class TestListen(unittest.TestCase):
    def test_pick_covers_categories(self):
        items = [{"id": f"{c}{i}", "category": c, "text": "t"} for c in "abcde" for i in range(5)]
        got = L.pick_sentences(items)
        self.assertEqual(len(got), 10)
        self.assertEqual({g["category"] for g in got}, set("abcde"))
        self.assertEqual(len({g["id"] for g in got}), 10)

    def test_sides_deterministic(self):
        self.assertEqual(L.assign_sides(10, random.Random(1)), L.assign_sides(10, random.Random(1)))
        self.assertTrue(set(L.assign_sides(50, random.Random(1))) <= {"A", "B"})

    def test_score(self):
        key = {"01": {"candidate": "A"}, "02": {"candidate": "B"}, "03": {"candidate": "A"}, "04": {"candidate": "B"}}
        r = L.score_answers(key, {"01": "A", "02": "A", "03": "tie", "04": "A"})
        self.assertEqual((r["candidate_preferred"], r["tied"], r["reference_preferred"]), (1, 1, 2))
        self.assertAlmostEqual(r["rate"], 0.5)
        with self.assertRaises(ValueError):
            L.score_answers(key, {"01": "C"})


if __name__ == "__main__":
    unittest.main()
