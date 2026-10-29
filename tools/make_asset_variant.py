"""Derive an asset set from an existing one: hard-link the manifest files, swap some graphs, rehash.

  python3 tools/make_asset_variant.py --src ~/.cache/maya_fast/assets/v1 \
      --dst ~/.cache/maya_fast/assets/v2-lowram \
      --graph depth=~/.cache/maya_fast/depth_build/q8a1_kg8/depth_frame_q8a1_kg8.onnx

Each ``--graph role=path`` hard-links ``path`` (and its external-data files, read
from the ONNX ``location`` fields, which must sit beside it) into the new set
under the same file names, drops the old graph of that role (and its data files)
and records the new name in the manifest ``graphs`` key. The source set is never
modified. Only files listed in the source manifest ``files`` map (plus the
replacement graph files) are linked; unhashed extras in the source directory
(e.g. ``backbone_fp32.*``) are not carried over.

Hard links share inodes with the source: rewriting a source file in place (for
example ``onnx.save_model`` over an existing ``.onnx``) changes the variant
too. The manifest sha256 check then fails loudly at load time. Write new
files instead of overwriting old ones. No weights are loaded (only the ONNX protobuf header, without external
data).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.maya_fast.assets import DEFAULT_GRAPHS, graph_names, write_manifest  # noqa: E402


def external_data_files(onnx_path: Path) -> list[str]:
    """Sorted external-data file names referenced by an ONNX graph."""
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    locs = set()
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key == "location":
                locs.add(e.value)
    for loc in locs:
        if Path(loc).name != loc:
            raise ValueError(f"{onnx_path}: external data location {loc!r} is not a plain file name")
    return sorted(locs)


def derive(src: Path, dst: Path, replace: dict[str, Path]) -> dict:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; refusing to overwrite")
    for role in replace:
        if role not in DEFAULT_GRAPHS:
            raise ValueError(f"unknown graph role {role!r}; one of {sorted(DEFAULT_GRAPHS)}")
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    graphs = graph_names(manifest)

    drop: set[str] = set()
    add: dict[str, Path] = {}  # name in dst -> source path
    for role, path in replace.items():
        old = graphs[role]
        drop |= {old, *external_data_files(src / old)}
        path = Path(path)
        add[path.name] = path
        for loc in external_data_files(path):
            if not (path.parent / loc).exists():
                raise FileNotFoundError(f"{path}: external data {loc} not beside it")
            add[loc] = path.parent / loc
        graphs[role] = path.name

    dst.mkdir(parents=True)
    names = [n for n in manifest["files"] if n not in drop]
    for n in names:
        if n not in add:
            os.link(src / n, dst / n)
    for name, path in add.items():
        os.link(path, dst / name)
        names.append(name)
    changed = {r: n for r, n in graphs.items() if n != DEFAULT_GRAPHS[r]}
    return write_manifest(dst, names, shapes=manifest.get("shapes", {}), graphs=changed,
                          version=manifest.get("version", 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--graph", action="append", default=[], metavar="ROLE=PATH")
    args = ap.parse_args(argv)
    replace = {}
    for g in args.graph:
        role, _, path = g.partition("=")
        replace[role] = Path(os.path.expanduser(path))
    m = derive(Path(os.path.expanduser(args.src)), Path(os.path.expanduser(args.dst)), replace)
    print(json.dumps({"graphs": m.get("graphs", {}), "files": m["files"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
