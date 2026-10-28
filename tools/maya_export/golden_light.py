from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from maya3 import guards as G
from maya3.engine import ROOT
from maya3.prompt import DEFAULT_BUILDER, MODES, VoiceRegistry, load_named_voices
from scripts import s2_serve_controls as S
from tts.data.packing_v2 import PackerV2

from tools.maya_export.common import build_light_model, ckpt_paths, work_dir


def main() -> None:
    out_dir = work_dir() / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    testset_path = Path("benchmarks/tts_testset_mr.json")
    if not testset_path.exists():
        testset_path = Path("/Users/anujbarve/Documents/marathi-helper/benchmarks/tts_testset_mr.json")
    with open(testset_path, "r", encoding="utf-8") as f:
        test_sentences = json.load(f)

    paths = ckpt_paths()
    prior_path = Path(ROOT) / "configs" / "text_frames_prior.json"
    prior = S.load_prior(str(prior_path))

    registry = VoiceRegistry()
    load_named_voices(registry, paths["voices_json"])
    # Also load the voices.json file to get the list of voice names
    with open(paths["voices_json"], "r", encoding="utf-8") as f:
        raw_v = json.load(f)
    v_items = raw_v["voices"] if isinstance(raw_v, dict) and "voices" in raw_v else raw_v
    voice_names = [v["name"] for v in v_items if isinstance(v, dict) and "name" in v]

    model = build_light_model()
    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)

    g = G.GuardConfig()
    dur = G.DurationPrior()
    chunk_s = 16.0  # upstream default in s2_serve_controls.py

    npz_data = {}
    json_data = {
        "voices": {},
        "sentences": {},
    }

    # 1. Per voice: mem, ecapa
    print(f"Generating voice goldens for {len(voice_names)} voices...")
    for vname in voice_names:
        voice = registry.get(vname)
        use_icl, use_mem = MODES[voice.mode]
        builder = voice.builder or DEFAULT_BUILDER
        voice.prefix = builder.prefix(packer, voice)
        voice.spk_refs = (
            np.asarray(voice.ecapa, np.float32).reshape(1, 192)
            if voice.ecapa is not None
            else np.zeros((1, 192), np.float32)
        )
        refs_t = torch.as_tensor(voice.spk_refs, device="cpu")[None]
        with torch.inference_mode():
            mem = model.depth.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([not use_mem], device="cpu"))

        npz_data[f"voice/{vname}/mem"] = mem.cpu().float().numpy()
        npz_data[f"voice/{vname}/ecapa"] = voice.spk_refs

        json_data["voices"][vname] = {
            "mode": voice.mode,
            "use_mem": bool(use_mem),
            "use_icl": bool(use_icl),
        }

    # 2. Per sentence × every voice
    print(f"Generating prompt goldens for {len(test_sentences)} sentences × {len(voice_names)} voices...")
    lang = "mr"
    for s_idx, item in enumerate(test_sentences):
        s_id = item["id"]
        text = item["text"]

        # Sentence-level controls
        ef, er = S.expected_frames(prior, lang, len(text))
        loose = min(1500, int(2.0 * (ef + er) + 40))
        cap = loose
        if g.enabled:
            cap = min(cap, max(int(ef * 0.5) + 8, dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
        minimum = max(4, int(0.5 * ef))
        split_chunks = S.split_text(text, lang, prior, chunk_s)

        json_data["sentences"][s_id] = {
            "ef": float(ef),
            "er": float(er),
            "cap": int(cap),
            "minimum": int(minimum),
            "split_text": split_chunks,
        }

        for vname in voice_names:
            voice = registry.get(vname)
            builder = voice.builder or DEFAULT_BUILDER
            with torch.inference_mode():
                ex = builder.example(packer, voice, text, lang)
                prompt_emb, n_p = S.prompt_from_example(ex, model, "cpu", torch.float32, 16)

            prefix_key = f"prompt/{s_id}/{vname}"
            npz_data[f"{prefix_key}/token_ids"] = np.asarray(ex["tokens"], dtype=np.int64)
            npz_data[f"{prefix_key}/kinds"] = np.asarray(ex["kind"], dtype=np.int64)
            npz_data[f"{prefix_key}/prompt_emb"] = prompt_emb[0].cpu().float().numpy()
            npz_data[f"{prefix_key}/n_prompt"] = np.int64(n_p)

    # 3. Fixed random code rows (seed 0) audio_in frames and aeos
    print("Generating audio_in and aeos goldens...")
    torch.manual_seed(0)
    test_codes = torch.randint(0, 2048, (8, 16), dtype=torch.int64)
    with torch.inference_mode():
        audio_in_frames = model.audio_in.frames(test_codes, 16)
        aeos = model.audio_in.aeos(1)[0]

    npz_data["audio_in_test_codes"] = test_codes.numpy()
    npz_data["audio_in_frames"] = audio_in_frames.cpu().float().numpy()
    npz_data["aeos"] = aeos.cpu().float().numpy()

    npz_path = out_dir / "light.npz"
    json_path = out_dir / "light.json"

    np.savez(npz_path, **npz_data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully wrote {npz_path} and {json_path}")


if __name__ == "__main__":
    main()
