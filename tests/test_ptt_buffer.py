import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ptt_buffer import ClipAssembler


def chunk(value, n=160):
    return np.full(n, value, dtype=np.float32)


class TestClipAssembler(unittest.TestCase):
    def test_clip_includes_pre_and_post_roll(self):
        a = ClipAssembler()
        for _ in range(100):
            self.assertIsNone(a.feed(chunk(1)))
        self.assertIsNone(a.start())
        for _ in range(50):
            self.assertIsNone(a.feed(chunk(2)))
        a.stop()
        outs = [a.feed(chunk(3)) for _ in range(25)]
        self.assertTrue(all(o is None for o in outs[:-1]))
        clip = outs[-1]
        self.assertEqual(len(clip), 4800 + 8000 + 4000)
        self.assertTrue(np.all(clip[:4800] == 1) and np.all(clip[4800:12800] == 2) and np.all(clip[12800:] == 3))

    def test_pre_roll_limited_to_available_history(self):
        a = ClipAssembler()
        for _ in range(10):
            a.feed(chunk(1))
        self.assertIsNone(a.start())
        a.feed(chunk(2))
        a.stop()
        clip = [a.feed(chunk(3)) for _ in range(25)][-1]
        self.assertEqual(len(clip), 1600 + 160 + 4000)
        self.assertTrue(np.all(clip[:1600] == 1))
        self.assertTrue(np.all(clip[1600:1760] == 2))

    def test_post_roll_trimmed_exactly_with_odd_chunk_size(self):
        a = ClipAssembler()
        for _ in range(20):
            a.feed(chunk(1, 512))
        a.start()
        for _ in range(10):
            a.feed(chunk(2, 512))
        a.stop()
        outs = [a.feed(chunk(3, 512)) for _ in range(8)]
        self.assertTrue(all(o is None for o in outs[:-1]))
        clip = outs[-1]
        held = 10 * 512
        # pre-roll is capped at 4800 samples even though 10240 were available
        self.assertEqual(len(clip), 4800 + held + 4000)
        self.assertTrue(np.all(clip[-4000:] == 3))

    def test_restart_during_post_roll_flushes_without_duplication(self):
        a = ClipAssembler()
        a.start()
        for _ in range(10):
            a.feed(chunk(2))
        a.stop()
        for _ in range(5):
            a.feed(chunk(3))
        first = a.start()
        self.assertEqual(len(first), 1600 + 800)
        for _ in range(10):
            a.feed(chunk(4))
        a.stop()
        second = [a.feed(chunk(5)) for _ in range(25)][-1]
        self.assertTrue(np.all(second[:1600] == 4))
        self.assertEqual(len(second), 1600 + 4000)

    def test_is_recording_tracks_hold_only(self):
        a = ClipAssembler()
        self.assertFalse(a.is_recording)
        a.start()
        self.assertTrue(a.is_recording)
        a.stop()
        self.assertFalse(a.is_recording)
        for _ in range(5):
            a.feed(chunk(3))
        self.assertFalse(a.is_recording)


if __name__ == "__main__":
    unittest.main()
