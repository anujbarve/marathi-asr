from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
import numpy as np
import psutil

from .assets import PromptAssets, default_assets_dir
from .backbone import Backbone
from .depth import Depth
from .codec import MimiDecoder
from .prompt import AudioIn, build_prompt
from .sampling import gumbel
from .control import (
    expected_frames,
    frame_cap,
    min_frames,
    split_text,
    StopPolicy,
    sample_cb0,
)

# Speakable text check: must contain at least one alphanumeric character
ALPHANUM_PATTERN = re.compile(r"[\w\d]", re.UNICODE)


def _softmax_topk(logits: np.ndarray, temp: float = 0.7, top_k: int = 64) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    k = min(top_k, logits.shape[-1])
    topk_idx = np.argpartition(logits, -k)[-k:]
    masked = np.full_like(logits, -np.inf)
    masked[topk_idx] = logits[topk_idx] / temp
    max_val = np.max(masked[topk_idx])
    exp_vals = np.exp(masked - max_val)
    exp_vals[masked == -np.inf] = 0.0
    sum_exp = np.sum(exp_vals)
    if sum_exp > 0:
        return exp_vals / sum_exp
    probs = np.zeros_like(logits)
    probs[topk_idx] = 1.0 / len(topk_idx)
    return probs


class MayaFast:
    def __init__(
        self,
        assets_dir: str | Path | None = None,
        threads: int | None = None,
    ) -> None:
        if assets_dir is None:
            self.assets_dir = default_assets_dir()
        else:
            self.assets_dir = Path(assets_dir)

        self.threads = threads
        self.assets = PromptAssets(self.assets_dir, verify=True)
        self.shapes = self.assets.shapes
        self.controls = self.assets.controls
        self.sampling_cfg = self.controls.get("sampling", {})

        # Load models
        backbone_path = self.assets_dir / "backbone_wo8.onnx"
        self.backbone = Backbone(
            backbone_path,
            threads=self.threads,
            config=self.shapes,
        )

        depth_path = self.assets_dir / "depth_frame_wo8.onnx"
        self.depth = Depth(depth_path, threads=self.threads)

        mimi_path = self.assets_dir / "mimi_decoder.onnx"
        self.mimi = MimiDecoder(mimi_path, threads=self.threads)

        self.audio_in = AudioIn(self.assets.prompt_params)

    @property
    def voices(self) -> list[str]:
        return sorted(list(self.assets.voices_info.keys()))

    def synthesize(
        self,
        text: str,
        voice: str = "Charvi",
        seed: int = 0,
        chunk_s: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Synthesize text into 24 kHz float32 audio.
        Returns:
            (wav, metrics)
        """
        # Review focus #2: Empty or punctuation-only text: expected ValueError("no speakable text")
        if not text or not ALPHANUM_PATTERN.search(text):
            raise ValueError("no speakable text")

        if voice not in self.assets.voices_info:
            raise KeyError(f"Unknown voice {voice!r}. Available: {self.voices}")

        t_wall_start = time.perf_counter()
        prior = self.controls.get("text_frames_prior", {})
        sampling_cfg = self.sampling_cfg
        q_serve = int(sampling_cfg.get("q_serve", 16))
        temp = float(sampling_cfg.get("temp", 0.7))
        top_k = int(sampling_cfg.get("top_k", 64))
        tau = float(sampling_cfg.get("tau", 0.35))
        hazard = float(sampling_cfg.get("hazard", 0.6))
        aeos_class = int(self.shapes.get("vocab", 2048))

        # Split text into chunks
        c_s = 12.0 if chunk_s is None else chunk_s
        chunks = split_text(text, "mr", prior, chunk_s=c_s)

        rng = np.random.default_rng(seed)

        # Voice speaker memory for Depth
        mem_key = f"{voice}/mem"
        mem = self.assets.voices_arrays.get(mem_key)
        if mem is None:
            # If plain mode or no memory
            d_mem = int(self.shapes.get("d_mem", 1024))
            mem = np.zeros((1, 1, d_mem), dtype=np.float32)

        gen_codes: list[np.ndarray] = []
        reasons: list[str] = []
        caps: list[int] = []
        max_past_len = 0
        max_chunk_bound = 0

        backbone_time = 0.0
        depth_time = 0.0
        mimi_time = 0.0

        for chunk_text in chunks:
            # Review focus #1: Each piece a fresh KV session
            self.backbone.reset()

            prompt = build_prompt(chunk_text, voice, self.assets, lang="mr")

            ef, er = expected_frames(prior, "mr", len(chunk_text))
            cap = frame_cap(chunk_text, "mr", ef, er, self.controls)
            min_f = min_frames(ef)
            caps.append(cap)
            chunk_bound = prompt.n_prompt + cap
            if chunk_bound > max_chunk_bound:
                max_chunk_bound = chunk_bound

            # Initial backbone step on prompt
            t0 = time.perf_counter()
            h, cb0_logits = self.backbone.step(prompt.embeds)
            backbone_time += time.perf_counter() - t0

            if self.backbone.past_len > max_past_len:
                max_past_len = self.backbone.past_len

            stop_policy = StopPolicy(min_frames=min_f, tau=tau, hazard=hazard)
            chunk_frames: list[np.ndarray] = []
            reason = "cap"

            for t in range(cap):
                # 1. Softmax with top_k mask over cb0_logits
                p = _softmax_topk(cb0_logits, temp=temp, top_k=top_k)

                # 2. StopPolicy check
                stop_reason = stop_policy.check(t, p, aeos_class=aeos_class)
                if stop_reason is not None:
                    reason = stop_reason
                    break

                # 3. Sample cb0 (aeos zeroed, Gumbel-max over the rest)
                cb0_noise = gumbel(rng, (cb0_logits.shape[-1],))
                cb0 = sample_cb0(p, cb0_noise, aeos_class=aeos_class)

                # 4. Depth frame
                depth_noise = gumbel(rng, (15, int(self.shapes.get("vocab", 2048))))
                t_d = time.perf_counter()
                frame_codes = self.depth.frame(h, cb0, mem, depth_noise)
                depth_time += time.perf_counter() - t_d

                chunk_frames.append(frame_codes)

                # 5. AudioIn feedback
                fb = self.audio_in.frames(frame_codes[None, :], q_use=q_serve)

                # 6. Next backbone step
                t_b = time.perf_counter()
                h, cb0_logits = self.backbone.step(fb)
                backbone_time += time.perf_counter() - t_b

                if self.backbone.past_len > max_past_len:
                    max_past_len = self.backbone.past_len

            reasons.append(reason)
            # Note on upstream aeos step:
            # Upstream does one self._step(audio_in.aeos(1)) after the loop if reason.startswith("aeos").
            # However, because each chunk resets the KV session and no further steps read from this KV session,
            # running the final aeos step changes nothing returned and wastes compute. We skip it here.

            if chunk_frames:
                # chunk_frames: list of (16,)
                chunk_codes_arr = np.stack(chunk_frames, axis=1)  # (16, F_chunk)
                gen_codes.append(chunk_codes_arr)

        if gen_codes:
            all_codes = np.concatenate(gen_codes, axis=1)  # (16, Total_F)
            total_frames = all_codes.shape[1]
            t_m = time.perf_counter()
            wav = self.mimi.decode(all_codes)
            mimi_time += time.perf_counter() - t_m
        else:
            total_frames = 0
            wav = np.zeros(0, dtype=np.float32)

        wall_s = time.perf_counter() - t_wall_start
        audio_s = len(wav) / 24000.0
        rtf = wall_s / max(1e-6, audio_s)

        # Peak RSS in MB
        process = psutil.Process()
        peak_rss_mb = process.memory_info().rss / (1024 * 1024)

        metrics = {
            "audio_s": audio_s,
            "wall_s": wall_s,
            "rtf": rtf,
            "frames": total_frames,
            "reasons": reasons,
            "caps": caps,
            "backbone_s": backbone_time,
            "depth_s": depth_time,
            "mimi_s": mimi_time,
            "peak_rss_mb": peak_rss_mb,
            "max_past_len": max_past_len,
            "max_chunk_bound": max_chunk_bound,
        }

        return wav, metrics
