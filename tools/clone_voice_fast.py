#!/usr/bin/env python3
"""True Voice Cloning with Maya-Fast ONNX Engine.

Extracts:
  1. ECAPA speaker embedding (192-dim) from reference recording.
  2. Mimi discrete acoustic codes (32, F) from reference recording.
  3. Uses Maya2's V5Builder & PackerV2 to construct the in-context conditioning prefix.
  4. Feeds the in-context prompt and custom speaker memory directly into Maya-Fast's
     optimized ONNX Backbone + Depth + Mimi pipeline!
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from maya3.engine import ROOT as MAYA3_ROOT
sys.path.insert(0, str(MAYA3_ROOT))

import scripts.s2_serve_controls as S
from tts.data.packing_v2 import PackerV2
from maya3.prompt import Voice, V5Builder
from tools.maya_export.common import build_light_model
from src.maya_fast.engine import MayaFast, _softmax_topk
from src.maya_fast.control import StopPolicy, sample_cb0
from speechbrain.inference.speaker import EncoderClassifier
from transformers import MimiModel
import torchaudio.functional as AF


def clone_voice(
    ref_audio_path: str | Path,
    ref_text: str,
    target_text: str,
    out_path: str | Path = "my_cloned_voice.wav",
    lang: str = "mr",
    threads: int = 8,
    seed: int = 0,
    temp: float = 0.7,
    top_k: int = 64,
    min_frames: int = 10,
    cap_frames: int = 600,
) -> Dict[str, Any]:
    """Clones a voice from reference audio + text and synthesizes target_text."""
    ref_audio_path = Path(ref_audio_path)
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    print("=" * 60)
    print("🧬 CLONING VOICE FROM REFERENCE AUDIO")
    print("=" * 60)
    print(f"Reference Audio: {ref_audio_path}")
    print(f"Reference Text : {ref_text}")
    print(f"Target Text    : {target_text}\n")

    t_start = time.perf_counter()

    # 1. Load audio and compute ECAPA embedding
    print("Step 1/4: Extracting ECAPA speaker embedding...")
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )
    wav, sr = sf.read(ref_audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    w16 = AF.resample(torch.from_numpy(wav), sr, 16000)
    with torch.no_grad():
        ecapa_vec = clf.encode_batch(w16[None]).squeeze().numpy()
    ecapa_vec = ecapa_vec / (np.linalg.norm(ecapa_vec) + 1e-12)
    print(f"         ECAPA embedding ready (dim={ecapa_vec.shape[0]})")

    # 2. Extract discrete acoustic codes with Mimi
    print("Step 2/4: Extracting acoustic tokens with Mimi encoder...")
    mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
    w24 = AF.resample(torch.from_numpy(wav), sr, 24000)
    with torch.no_grad():
        codes = mimi.encode(w24[None, None]).audio_codes[0].numpy()  # (32, F)
    print(f"         Acoustic tokens ready (shape={codes.shape})")

    # 3. Build in-context prompt and depth speaker memory
    print("Step 3/4: Building in-context conditioning prefix and speaker memory...")
    model = build_light_model()
    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
    builder = V5Builder()

    voice = Voice(
        "custom_clone",
        mode="icl_spk",
        clips=[{"text": ref_text, "codes": codes, "lang": lang}],
        ecapa=ecapa_vec
    )
    voice.prefix = builder.prefix(packer, voice)
    voice.spk_refs = np.asarray(voice.ecapa, np.float32).reshape(1, 192)

    with torch.inference_mode():
        ex = builder.example(packer, voice, target_text, lang)
        prompt_emb_t, n_prompt = S.prompt_from_example(ex, model, "cpu", torch.float32, 16)
        prompt_emb = prompt_emb_t[0].cpu().numpy().astype(np.float32)

        refs_t = torch.as_tensor(voice.spk_refs, device="cpu")[None]
        spk_mem_t = model.depth.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([False], device="cpu"))
        spk_mem = spk_mem_t[0].cpu().numpy().astype(np.float32)  # (1, 1024)

    print(f"         In-context prompt ready ({prompt_emb.shape[0]} tokens, mem shape {spk_mem.shape})")

    # 4. Synthesize with fast ONNX runtime engine
    print("\nStep 4/4: Synthesizing speech with Maya-Fast ONNX engine...")
    engine = MayaFast(threads=threads)

    engine.backbone.reset()
    rng = np.random.default_rng(seed)

    # Prefill prompt
    hidden, cb0_logits = engine.backbone.step(prompt_emb)

    gen_codes = []
    stop_policy = StopPolicy(min_frames=min_frames)

    t_synth = time.perf_counter()
    for frame_idx in range(cap_frames):
        p = _softmax_topk(cb0_logits, temp=temp, top_k=top_k)
        stop_reason = stop_policy.check(frame_idx, p)
        if stop_reason is not None and frame_idx >= min_frames:
            break

        noise_cb0 = rng.gumbel(size=2049).astype(np.float32)
        cb0 = sample_cb0(p, noise_cb0)

        noise_depth = rng.gumbel(size=(15, 2048)).astype(np.float32)
        frame_codes = engine.depth.frame(hidden, cb0, spk_mem, noise_depth)
        gen_codes.append(frame_codes)

        # Feed frame back to backbone
        embed_in = engine.assets.audio_in.frames(frame_codes[None, :])
        hidden, cb0_logits = engine.backbone.step(embed_in)

    if not gen_codes:
        raise RuntimeError("No audio codes generated during synthesis.")

    all_codes = np.stack(gen_codes, axis=1)  # (16, F)
    out_wav = engine.mimi.decode(all_codes)
    synth_time = time.perf_counter() - t_synth
    total_time = time.perf_counter() - t_start

    out_file = Path(out_path)
    sf.write(out_file, out_wav, 24000)

    audio_dur = len(out_wav) / 24000.0
    rtf = synth_time / audio_dur if audio_dur > 0 else 0

    print("\n" + "=" * 60)
    print("🎉 VOICE CLONING COMPLETE!")
    print(f"Generated File : {out_file.resolve()}")
    print(f"Audio Duration : {audio_dur:.2f}s")
    print(f"Synthesis Time : {synth_time:.2f}s (RTF: {rtf:.2f})")
    print("=" * 60)

    return {
        "out_file": str(out_file.resolve()),
        "audio_dur_s": audio_dur,
        "synth_time_s": synth_time,
        "total_time_s": total_time,
        "rtf": rtf,
        "num_frames": len(gen_codes),
    }


def main():
    parser = argparse.ArgumentParser(description="True Voice Cloning with Maya-Fast")
    parser.add_argument("--ref-audio", default="my_sample_recording.wav", help="Path to reference recording")
    parser.add_argument("--ref-text", default="हॅलो मी काहीतरी मराठीमध्ये बोलत आहे तुम्हाला कळतय का",
                        help="Exact transcript of the reference recording")
    parser.add_argument("--text", default="नमस्कार! हा माझा स्वतःचा आवाज आहे आणि माया-फास्ट मॉडेल आता माझ्या आवाजात बोलत आहे.",
                        help="Text to synthesize with the cloned voice")
    parser.add_argument("--out", default="my_actual_cloned_voice.wav", help="Output file path")
    parser.add_argument("--lang", default="mr", help="Language code (default: mr)")
    parser.add_argument("--threads", type=int, default=8, help="Number of ORT threads")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    clone_voice(
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        target_text=args.text,
        out_path=args.out,
        lang=args.lang,
        threads=args.threads,
        seed=args.seed,
        temp=args.temp,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
