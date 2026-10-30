from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(assets_dir: Path, names: list[str], shapes: dict, version: int = 1) -> dict:
    """Hash every named file in assets_dir and write manifest.json next to them.
    The runtime refuses any asset whose sha256 differs (src/maya_fast/assets.py)."""
    manifest = {
        "version": version,
        "files": {name: sha256_file(assets_dir / name) for name in names},
        "shapes": shapes,
    }
    with open(assets_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest
