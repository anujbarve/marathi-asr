from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np


class AssetError(Exception):
    pass


def default_work_dir() -> Path:
    env = os.environ.get("MAYA_FAST_WORK")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "maya_fast"


def default_assets_dir() -> Path:
    return default_work_dir() / "assets" / "v1"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(assets_dir: str | Path, verify: bool = True) -> dict[str, Any]:
    assets_dir = Path(assets_dir)
    manifest_file = assets_dir / "manifest.json"
    if not manifest_file.exists():
        raise AssetError(f"manifest.json missing in {assets_dir}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        try:
            manifest = json.load(f)
        except Exception as e:
            raise AssetError(f"Corrupt manifest.json in {assets_dir}: {e}")

    if verify:
        files = manifest.get("files", {})
        for name, expected_sha in files.items():
            fpath = assets_dir / name
            if not fpath.exists():
                raise AssetError(f"Required asset file missing: {name}")
            actual_sha = compute_file_sha256(fpath)
            if actual_sha != expected_sha:
                raise AssetError(f"Asset sha256 mismatch for file: {name}")

    return manifest


class PromptAssets:
    def __init__(self, assets_dir: str | Path, verify: bool = True) -> None:
        self.assets_dir = Path(assets_dir)
        self.manifest = load_manifest(self.assets_dir, verify=verify)
        self.shapes = self.manifest.get("shapes", {})

        from .embed import EmbeddingTable
        from .prompt import AudioIn

        d_model = self.shapes.get("d_model", 2048)
        vocab = self.shapes.get("vocab", 151936)
        n_q = self.shapes.get("n_q", 16)
        n_codes = self.shapes.get("n_codes", 2048)
        self.embed_table = EmbeddingTable(self.assets_dir / "embed_bf16.bin", rows=vocab, dim=d_model)
        audio_table = EmbeddingTable(self.assets_dir / "audio_in_bf16.bin", rows=n_q * n_codes, dim=d_model)

        # Controls (text_frames_prior, guard, sampling, text)
        with open(self.assets_dir / "controls.json", "r", encoding="utf-8") as f:
            self.controls = json.load(f)

        # Speaker anchor weights and audio input norm/aeos (fp32; the audio table is bf16 on disk)
        prompt_npz = np.load(self.assets_dir / "prompt.npz")
        self.prompt_params = {k: prompt_npz[k] for k in prompt_npz.files}
        self.audio_in = AudioIn(audio_table, self.prompt_params["audio_in/aeos_in"],
                                self.prompt_params["audio_in/norm"], n_q=n_q)

        # Voices: name -> {mode, use_icl, use_mem}
        with open(self.assets_dir / "voices.json", "r", encoding="utf-8") as f:
            self.voices_info = json.load(f)

        # Tokenizer (extended upstream tokenizer, fast format)
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.assets_dir / "tokenizer.json"))
