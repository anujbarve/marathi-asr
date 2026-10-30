"""Heavy reference corpus: upstream serving path (controlled stop, temp 0.7, top-k 64, depth sampler) with the
backbone streamed one layer at a time (tools/maya_export/streamed_backbone.py), so it fits 4 GB.

Per sentence the three seeds run as one batch (same prompt, no padding). Each row has its own torch.Generator for
the cb0 draw; the depth draws use the global RNG, seeded per step from the sentence index, so a rerun reproduces.

Per (sentence, seed) writes work_dir()/golden/ref/{id}_s{seed}.npz with:
  codes        (16, F)   int64  generated codes (cb0 first)
  wav          (F*1920,) fp32   Mimi decode of codes, 24 kHz
  reason       str               stop reason ("cap", "aeos_argmax", "aeos_tau", "aeos_hazard")
  h_frames     (F, 2048) fp32   backbone hidden fed to depth for each frame
  cb0_logits   (F, 2049) fp16   raw cb0 head logits per frame
  depth_logits (F, 15, 2048) fp16  teacher-forced depth head logits for levels 1..15, from h_frames and codes
  seq_emb      (n_p + F, 2048) fp32 backbone input sequence: prompt embeddings, then the fed-back frame embeddings

Run guarded (one model process at a time):
    .venv/bin/python tools/memguard.py --rss-limit-mb 4000 --min-free-mb 4000 -- \
        .venv-tts/bin/python -m tools.maya_export.golden_reference
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from maya3.engine import ROOT
from maya3.prompt import DEFAULT_BUILDER, MODES, VoiceRegistry, load_named_voices
from scripts import s2_serve_controls as S
from tts.data.packing_v2 import PackerV2
from tts.model.depth_v2 import GROUP_OF_LEVEL
from tts.tokens import AEOS_CLASS

from tools.maya_export.common import build_light_model, ckpt_paths, work_dir
from tools.maya_export.streamed_backbone import StreamedBackbone, shard_files

VOICE = "Charvi"
SEEDS = (0, 1, 2)
TEMP = 0.7
TOP_K = 64
TAU = 0.35
HAZARD = 0.6
Q_SERVE = 16
N_LEVELS = Q_SERVE - 1


def _cb0_probs(logits: torch.Tensor) -> torch.Tensor:
    """Upstream ServeSession._cb0_probs for a batch: temperature, top-k mask, softmax."""
    lg = logits.float() / TEMP
    kth = lg.topk(TOP_K, -1).values[:, -1:]
    lg = lg.masked_fill(lg < kth, -float("inf"))
    return lg.softmax(-1)


def _voice_mem(model, voice) -> torch.Tensor:
    """Depth speaker memory for the voice (named voices: null memory, as the golden prompts)."""
    _, use_mem = MODES[voice.mode]
    refs = np.asarray(voice.ecapa, np.float32).reshape(1, 192) if voice.ecapa is not None else np.zeros((1, 192), np.float32)
    refs_t = torch.as_tensor(refs, device="cpu")[None]
    return model.depth.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([not use_mem], device="cpu"))


def _teacher_forced_depth(model, h_frames: torch.Tensor, codes: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
    """Logits of every depth head for levels 1..15 on the generated codes. h_frames (F, D), codes (16, F)."""
    depth = model.depth
    F = h_frames.shape[0]
    mem_f = mem.expand(F, -1, -1)
    ref_mask = torch.ones(F, 1, dtype=torch.bool)
    out = []
    codes_t = codes.T.contiguous()  # (F, 16)
    for level in range(1, Q_SERVE):
        x = depth._prefix(h_frames, codes_t[:, :level], level)
        h = depth._run(x, GROUP_OF_LEVEL[level], mem_f, ref_mask)
        out.append(depth.heads[level - 1](h).float())
    return torch.stack(out, 1)  # (F, 15, 2048)


def main() -> None:
    out_dir = work_dir() / "golden" / "ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    golden = json.loads((work_dir() / "golden" / "light.json").read_text(encoding="utf-8"))
    sentences = golden["sentences"]
    limit = int(os.environ.get("MAYA_REF_LIMIT", "0"))  # smoke runs only; 0 = whole corpus
    if limit:
        sentences = dict(list(sentences.items())[:limit])

    model = build_light_model()
    paths = ckpt_paths()
    registry = VoiceRegistry()
    load_named_voices(registry, paths["voices_json"])
    voice = registry.get(VOICE)
    builder = voice.builder or DEFAULT_BUILDER
    packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
    voice.prefix = builder.prefix(packer, voice)
    voice.spk_refs = np.zeros((1, 192), np.float32)
    mem = _voice_mem(model, voice)
    audio_in = model.audio_in
    cb0_head = model.backbone.cb0_logits

    stream = StreamedBackbone(paths["backbone_dir"], shard_files(paths["ckpt"]))
    B = len(SEEDS)

    for s_idx, (s_id, item) in enumerate(sentences.items()):
        text, cap, min_f = item["text"], int(item["cap"]), int(item["minimum"])
        ex = builder.example(packer, voice, text, "mr")
        with torch.inference_mode():
            prompt_emb, n_p = S.prompt_from_example(ex, model, "cpu", torch.float32, Q_SERVE)
        prompt_emb = prompt_emb.float()  # (1, n_p, D)

        gens = [torch.Generator().manual_seed(seed) for seed in SEEDS]
        stream.reset()
        with torch.inference_mode():
            h_all = stream.forward(prompt_emb.expand(B, -1, -1))
            h = h_all[:, -1]
            H = np.zeros(B)
            reason = [None] * B
            frames = [[] for _ in range(B)]
            h_rec = [[] for _ in range(B)]
            logit_rec = [[] for _ in range(B)]
            fed = [[] for _ in range(B)]

            for t in range(cap):
                logits = cb0_head(h).float()
                p = _cb0_probs(logits)
                p_aeos = p[:, AEOS_CLASS].numpy()
                active = [i for i in range(B) if reason[i] is None]
                if not active:
                    break
                for i in active:
                    if t >= min_f:
                        H[i] = 1.0 - (1.0 - H[i]) * (1.0 - p_aeos[i])
                        if int(p[i].argmax()) == AEOS_CLASS:
                            reason[i] = "aeos_argmax"
                        elif p_aeos[i] >= TAU:
                            reason[i] = "aeos_tau"
                        elif H[i] >= HAZARD:
                            reason[i] = "aeos_hazard"
                cb0 = torch.zeros(B, dtype=torch.long)
                for i in range(B):
                    if reason[i] is not None:
                        continue
                    q = p[i].clone()
                    q[AEOS_CLASS] = 0.0
                    q = q / q.sum()
                    cb0[i] = int(torch.multinomial(q, 1, generator=gens[i]))
                for i in active:
                    if reason[i] is None:
                        logit_rec[i].append(logits[i].half())
                        h_rec[i].append(h[i].float())

                torch.manual_seed(1000 * s_idx + t)
                mem_b = mem.expand(B, -1, -1)
                frame = S.depth_generate(model.depth, h.float(), cb0, mem_b, Q_SERVE, TEMP, TOP_K, 0.0,
                                         torch.ones(B, 1, dtype=torch.bool))
                fb = audio_in.frames(frame, Q_SERVE)  # (B, D)
                for i in active:
                    if reason[i] is None:
                        frames[i].append(frame[i])
                        fed[i].append(fb[i].clone())
                h = stream.forward(fb[:, None, :])[:, -1]

            for i in range(B):
                if reason[i] is None:
                    reason[i] = "cap"

        for i, seed in enumerate(SEEDS):
            F = len(frames[i])
            if F == 0:
                raise RuntimeError(f"{s_id} seed {seed}: no frames generated (reason {reason[i]})")
            codes = torch.stack(frames[i], -1)  # (16, F)
            h_frames = torch.stack(h_rec[i], 0)  # (F, D)
            depth_logits = _teacher_forced_depth(model, h_frames, codes, mem).half()
            seq = torch.cat([prompt_emb[0], torch.stack(fed[i], 0)], 0)
            np.savez(
                out_dir / f"{s_id}_s{seed}.npz",
                codes=codes.numpy().astype(np.int64),
                reason=np.array(reason[i]),
                h_frames=h_frames.numpy().astype(np.float32),
                cb0_logits=torch.stack(logit_rec[i], 0).numpy().astype(np.float16),
                depth_logits=depth_logits.numpy(),
                seq_emb=seq.numpy().astype(np.float32),
            )
        print(f"{s_id}: frames {[len(f) for f in frames]} reasons {reason} cap {cap} min {min_f}", flush=True)

    _decode_wav(out_dir, sentences.keys())


def _decode_wav(out_dir: Path, s_ids) -> None:
    """Mimi decode of every saved codes array, written into the same npz as `wav`."""
    from transformers import MimiModel

    mimi = MimiModel.from_pretrained(ckpt_paths()["mimi_dir"], dtype=torch.float32).eval()
    for s_id in s_ids:
        for seed in SEEDS:
            path = out_dir / f"{s_id}_s{seed}.npz"
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files}
            with torch.inference_mode():
                wav = mimi.decode(torch.from_numpy(arrays["codes"])[None])[0][0, 0].float().numpy()
            arrays["wav"] = wav
            np.savez(path, **arrays)
    print(f"wrote wav for {len(s_ids) * len(SEEDS)} files", flush=True)


if __name__ == "__main__":
    main()
