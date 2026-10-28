"""Export the prompt-side assets (tokenizer, bf16 embedding tables, speaker anchor, audio input, voices,
sentence controls) to MAYA_FAST_WORK/assets/v1 for the numpy runtime. Run guarded:

    .venv/bin/python tools/memguard.py --rss-limit-mb 4000 --min-free-mb 4000 -- \
        .venv-tts/bin/python -m tools.maya_export.export_prompt_assets
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from maya3 import guards as G
from maya3.engine import ROOT
from maya3.prompt import MODES
from scripts import s2_serve_controls as S

from tools.maya_export.common import build_light_model, ckpt_paths, work_dir
from tools.maya_export.manifest import write_manifest

N_Q = 16
N_CODES = 2048
ECAPA_DIM = 192
HARD_CAP = 1500          # frame cap ceiling used by the goldens (golden_light.py: min(1500, ...))
CHUNK_S = 16.0           # upstream default in s2_serve_controls.py (golden_light.py: chunk_s)


def _bf16_to_file(t: torch.Tensor, path: Path) -> int:
    """Write a bf16 tensor as raw little-endian uint16 (row-major). Returns the row count."""
    raw = t.detach().contiguous().view(torch.int16).numpy().astype("<u2")
    raw.tofile(path)
    return raw.shape[0]


def _controls(prior_path: str) -> dict:
    coef_model = G.DurationPrior()
    g = G.GuardConfig()
    prior = S.load_prior(prior_path)
    return {
        "text_frames_prior": prior,
        "prior_default": {"a": S.PRIOR_DEFAULT["a"], "b": S.PRIOR_DEFAULT["b"],
                          "resid_p99": S.PRIOR_DEFAULT["resid_p99"]},
        "text": {"terminals": S.TERMINALS, "fps": S.FPS, "frame_s": G.FRAME_S, "chunk_s": CHUNK_S},
        "guard": {
            "enabled": bool(g.enabled),
            "dur_k": float(g.dur_k),
            "dur_margin_s": float(g.dur_margin_s),
            "hard_cap": HARD_CAP,
            "dur_prior_coefs": {lang: [float(x) for x in c] for lang, c in coef_model.coef.items()},
            "dur_prior_default": [float(x) for x in coef_model.default],
        },
        "sampling": {"q_serve": 16, "temp": 0.7, "top_k": 64, "tau": 0.35, "hazard": 0.6,
                     "cfg_scale": 0.0, "aeos_class": N_CODES},
    }


def main() -> None:
    out = work_dir() / "assets" / "v1"
    out.mkdir(parents=True, exist_ok=True)

    model = build_light_model()
    with torch.inference_mode():
        # tokenizer: the extended tokenizer the model was trained with (backend, fast format)
        model.backbone.tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))

        embed_rows = _bf16_to_file(model.backbone.lm.model.embed_tokens.weight, out / "embed_bf16.bin")
        audio_rows = _bf16_to_file(model.audio_in.table.weight, out / "audio_in_bf16.bin")

        anchor = model.speaker_anchor
        prompt = {
            "anchor_fc1_w": anchor.fc1.weight.detach().float().numpy(),
            "anchor_fc1_b": anchor.fc1.bias.detach().float().numpy(),
            "anchor_fc2_w": anchor.fc2.weight.detach().float().numpy(),
            "anchor_fc2_b": anchor.fc2.bias.detach().float().numpy(),
            "anchor_null": anchor.null_vec.detach().float().numpy(),
            "audio_in/aeos_in": model.audio_in.aeos_in.detach().float().numpy(),
            "audio_in/norm": model.audio_in.norm.weight.detach().float().numpy(),
        }
    np.savez(out / "prompt.npz", **prompt)

    # voices: every trained named voice, with the conditioning flags the upstream builder uses
    paths = ckpt_paths()
    with open(paths["voices_json"], "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["voices"] if isinstance(raw, dict) and "voices" in raw else raw
    voices = {}
    for it in items:
        name = it.get("name")
        if not name:
            continue
        mode = it.get("mode", "named")
        use_icl, use_mem = MODES[mode]
        voices[name] = {"mode": mode, "use_icl": bool(use_icl), "use_mem": bool(use_mem)}
    (out / "voices.json").write_text(json.dumps(voices, indent=2, ensure_ascii=False), encoding="utf-8")

    controls = _controls(str(Path(ROOT) / "configs" / "text_frames_prior.json"))
    (out / "controls.json").write_text(json.dumps(controls, indent=2, ensure_ascii=False), encoding="utf-8")

    names = ["tokenizer.json", "embed_bf16.bin", "audio_in_bf16.bin", "prompt.npz",
             "voices.json", "controls.json"]
    write_manifest(out, names, shapes={
        "d_model": int(model.audio_in.aeos_in.shape[0]),
        "vocab": embed_rows,
        "audio_rows": audio_rows,
        "n_q": N_Q,
        "n_codes": N_CODES,
        "ecapa_dim": ECAPA_DIM,
    })
    print(f"wrote prompt assets to {out}")


if __name__ == "__main__":
    main()
