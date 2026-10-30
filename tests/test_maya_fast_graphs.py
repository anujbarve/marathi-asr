"""Graph file names come from the manifest ("graphs"), with v1 defaults."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from tests.fixtures.tiny_assets import make_tiny_assets
from src.maya_fast.assets import DEFAULT_GRAPHS, AssetError, graph_names, load_manifest, write_manifest
from src.maya_fast.engine import MayaFast
from tools.make_asset_variant import derive, external_data_files


def _rewrite_manifest(assets_dir: Path, graphs: dict | None, rename: dict[str, str] | None = None) -> None:
    m = json.loads((assets_dir / "manifest.json").read_text())
    names = list(m["files"])
    for old, new in (rename or {}).items():
        (assets_dir / old).rename(assets_dir / new)
        names[names.index(old)] = new
    write_manifest(assets_dir, names, shapes=m["shapes"], graphs=graphs)


class TestGraphNames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.assets_dir = make_tiny_assets(Path(self.tmp.name) / "v1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_without_graphs_key(self):
        m = load_manifest(self.assets_dir)
        self.assertNotIn("graphs", m)
        self.assertEqual(graph_names(m), DEFAULT_GRAPHS)
        self.assertEqual(DEFAULT_GRAPHS["backbone"], "backbone_wo8.onnx")
        self.assertEqual(DEFAULT_GRAPHS["depth"], "depth_frame_fp16.onnx")

    def test_partial_override_keeps_other_defaults(self):
        g = graph_names({"graphs": {"depth": "depth_other.onnx"}})
        self.assertEqual(g["depth"], "depth_other.onnx")
        self.assertEqual(g["backbone"], DEFAULT_GRAPHS["backbone"])
        self.assertEqual(g["mimi"], DEFAULT_GRAPHS["mimi"])

    def test_unknown_role_raises(self):
        with self.assertRaises(AssetError):
            graph_names({"graphs": {"vocoder": "x.onnx"}})

    def test_graph_not_in_files_map_raises(self):
        with self.assertRaises(AssetError):
            graph_names({"files": {"a.onnx": "x"}, "graphs": {"depth": "b.onnx"}})
        m = json.loads((self.assets_dir / "manifest.json").read_text())
        m["graphs"] = {"depth": "not_listed.onnx"}
        (self.assets_dir / "manifest.json").write_text(json.dumps(m))
        for verify in (True, False):
            with self.assertRaises(AssetError):
                load_manifest(self.assets_dir, verify=verify)

    def test_engine_loads_renamed_graphs(self):
        ref, _ = MayaFast(assets_dir=self.assets_dir, threads=1).synthesize("t_1 t_2", seed=3)
        _rewrite_manifest(
            self.assets_dir,
            graphs={"depth": "depth_alt.onnx", "backbone": "backbone_alt.onnx"},
            rename={"depth_frame_fp16.onnx": "depth_alt.onnx", "backbone_wo8.onnx": "backbone_alt.onnx"},
        )
        self.assertFalse((self.assets_dir / "depth_frame_fp16.onnx").exists())
        eng = MayaFast(assets_dir=self.assets_dir, threads=1)
        self.assertEqual(eng.graphs["depth"], "depth_alt.onnx")
        self.assertEqual(eng.graphs["backbone"], "backbone_alt.onnx")
        wav, _ = eng.synthesize("t_1 t_2", seed=3)
        np.testing.assert_array_equal(wav, ref)

    def test_engine_without_graphs_key_needs_default_names(self):
        # Renamed file, no "graphs" key: the engine looks for the v1 name and fails.
        m = json.loads((self.assets_dir / "manifest.json").read_text())
        (self.assets_dir / "depth_frame_fp16.onnx").rename(self.assets_dir / "depth_alt.onnx")
        names = [n for n in m["files"] if n != "depth_frame_fp16.onnx"] + ["depth_alt.onnx"]
        write_manifest(self.assets_dir, names, shapes=m["shapes"])
        load_manifest(self.assets_dir, verify=True)
        with self.assertRaises(Exception) as ctx:
            MayaFast(assets_dir=self.assets_dir, threads=1)
        self.assertIn("depth_frame_fp16.onnx", str(ctx.exception))


class TestDeriveVariant(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        self.src = make_tiny_assets(root / "v1")
        # A replacement depth graph with external data, as the real builds have.
        build = root / "build"
        build.mkdir()
        m = onnx.load(str(self.src / "depth_frame_fp16.onnx"))
        # A 15 x 64 float weight (noise * 1) large enough to go to the external file.
        m.graph.initializer.append(onnx.numpy_helper.from_array(np.ones((15, 64), np.float32), "noise_scale"))
        argmax = next(n for n in m.graph.node if n.op_type == "ArgMax")
        argmax.input[0] = "noise_scaled"
        m.graph.node.insert(0, onnx.helper.make_node("Mul", ["noise", "noise_scale"], ["noise_scaled"]))
        self.new_depth = build / "depth_frame_new.onnx"
        onnx.save(m, str(self.new_depth), save_as_external_data=True,
                  location="depth_frame_new.onnx.data", size_threshold=1024)
        (self.src / "backbone_fp32.extra").write_bytes(b"unhashed extra")
        self.before = {p.name: p.read_bytes() for p in self.src.iterdir()}

    def tearDown(self):
        self.tmp.cleanup()

    def test_derive_swaps_depth_and_rehashes(self):
        dst = Path(self.tmp.name) / "v2"
        m = derive(self.src, dst, {"depth": self.new_depth})
        self.assertEqual(m["graphs"], {"depth": "depth_frame_new.onnx"})
        self.assertIn("depth_frame_new.onnx", m["files"])
        self.assertIn("depth_frame_new.onnx.data", m["files"])
        self.assertNotIn("depth_frame_fp16.onnx", m["files"])
        self.assertFalse((dst / "depth_frame_fp16.onnx").exists())
        self.assertEqual(external_data_files(dst / "depth_frame_new.onnx"), ["depth_frame_new.onnx.data"])
        # unhashed extras in the source are not linked
        self.assertFalse((dst / "backbone_fp32.extra").exists())
        self.assertEqual(set(p.name for p in dst.iterdir()), set(m["files"]) | {"manifest.json"})
        # hard links, not copies
        self.assertEqual(os.stat(dst / "prompt.npz").st_ino, os.stat(self.src / "prompt.npz").st_ino)
        self.assertEqual(os.stat(dst / "depth_frame_new.onnx").st_ino, os.stat(self.new_depth).st_ino)
        load_manifest(dst, verify=True)
        # source untouched
        self.assertEqual({p.name: p.read_bytes() for p in self.src.iterdir()}, self.before)
        eng = MayaFast(assets_dir=dst, threads=1)
        self.assertEqual(eng.graphs["depth"], "depth_frame_new.onnx")
        ref, _ = MayaFast(assets_dir=self.src, threads=1).synthesize("t_1 t_2", seed=5)
        wav, _ = eng.synthesize("t_1 t_2", seed=5)
        np.testing.assert_array_equal(wav, ref)

    def test_derive_refuses_existing_dst(self):
        dst = Path(self.tmp.name) / "v2"
        dst.mkdir()
        with self.assertRaises(FileExistsError):
            derive(self.src, dst, {"depth": self.new_depth})

    def test_derive_refuses_missing_external_data(self):
        (self.new_depth.parent / "depth_frame_new.onnx.data").unlink()
        with self.assertRaises(FileNotFoundError):
            derive(self.src, Path(self.tmp.name) / "v2", {"depth": self.new_depth})


if __name__ == "__main__":
    unittest.main()
