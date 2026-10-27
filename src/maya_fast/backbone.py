from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

from .ort_opts import session_options


class Backbone:
    def __init__(
        self,
        path: str | Path,
        threads: int | None = None,
        manifest_path: str | Path | None = None,
        config: dict | None = None,
    ) -> None:
        self.path = Path(path)
        self.opts = session_options(threads)
        self.session = ort.InferenceSession(str(self.path), self.opts, providers=["CPUExecutionProvider"])

        # Determine shapes and hyperparameters
        # Default real Qwen3 / Maya2 backbone parameters:
        # n_layers=28, n_heads=8 (kv heads), head_dim=128, rope_theta=10000.0, d_model=2048
        self.n_layers = 28
        self.n_heads = 8
        self.head_dim = 128
        self.rope_theta = 10000.0
        self.d_model = 2048

        # If manifest or config provided, override:
        if config is not None:
            self._apply_config(config)
        elif manifest_path is not None and Path(manifest_path).exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                shapes = data.get("shapes", {})
                self._apply_config(shapes)
        else:
            # Check if manifest.json exists next to model
            maybe_manifest = self.path.parent / "manifest.json"
            if maybe_manifest.exists():
                with open(maybe_manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    shapes = data.get("shapes", {})
                    self._apply_config(shapes)
            else:
                # Discover n_layers from session inputs
                past_k_inputs = [inp.name for inp in self.session.get_inputs() if inp.name.startswith("past_k_")]
                if past_k_inputs:
                    self.n_layers = len(past_k_inputs)

        # Inspect session input shapes if available
        for inp in self.session.get_inputs():
            if inp.name == "past_k_0":
                shape = inp.shape
                # shape: [1, n_heads, P, head_dim]
                if len(shape) == 4:
                    if isinstance(shape[1], int):
                        self.n_heads = shape[1]
                    if isinstance(shape[3], int):
                        self.head_dim = shape[3]
            elif inp.name == "embeds":
                if len(inp.shape) == 3 and isinstance(inp.shape[2], int):
                    self.d_model = inp.shape[2]

        self.reset()

    def _apply_config(self, cfg: dict) -> None:
        if "n_layers" in cfg:
            self.n_layers = int(cfg["n_layers"])
        if "n_heads" in cfg:
            self.n_heads = int(cfg["n_heads"])
        if "head_dim" in cfg:
            self.head_dim = int(cfg["head_dim"])
        if "rope_theta" in cfg:
            self.rope_theta = float(cfg["rope_theta"])
        if "d_model" in cfg:
            self.d_model = int(cfg["d_model"])

    def reset(self) -> None:
        self.past_len = 0
        self.past_k = [np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32) for _ in range(self.n_layers)]
        self.past_v = [np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32) for _ in range(self.n_layers)]

    def _compute_rope(self, T: int, past: int) -> tuple[np.ndarray, np.ndarray]:
        # RoPE for positions [past, past + T)
        # dim = head_dim
        pos = np.arange(past, past + T, dtype=np.float32)[:, None]  # (T, 1)
        dim_indices = np.arange(0, self.head_dim, 2, dtype=np.float32)  # (head_dim/2,)
        inv_freq = 1.0 / (self.rope_theta ** (dim_indices / self.head_dim))  # (head_dim/2,)
        sinusoid = pos * inv_freq[None, :]  # (T, head_dim/2)
        # duplicate across second half to match typical rotary implementation: [sinusoid, sinusoid]
        sinusoid = np.concatenate([sinusoid, sinusoid], axis=-1)  # (T, head_dim)
        cos = np.cos(sinusoid)[None, :, :]  # (1, T, head_dim)
        sin = np.sin(sinusoid)[None, :, :]  # (1, T, head_dim)
        return cos.astype(np.float32), sin.astype(np.float32)

    def _compute_mask(self, T: int, past: int) -> np.ndarray:
        # Additive causal mask (1, 1, T, past+T): query i sees keys j <= past + i
        total_len = past + T
        i = np.arange(T)[:, None]
        j = np.arange(total_len)[None, :]
        mask = np.zeros((T, total_len), dtype=np.float32)
        mask[j > (past + i)] = -np.inf
        return mask[None, None, :, :]

    def step(self, embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        embeds: (T, d_model) or (1, T, d_model)
        Returns:
            hidden: (d_model,) last position
            cb0_logits: (vocab + 1,) last position
        """
        if embeds.ndim == 2:
            embeds = embeds[None, :, :]
        elif embeds.ndim != 3:
            raise ValueError(f"Expected embeds to have 2 or 3 dims, got shape {embeds.shape}")

        embeds = embeds.astype(np.float32)
        T = embeds.shape[1]
        P = self.past_len

        cos, sin = self._compute_rope(T, P)
        mask = self._compute_mask(T, P)

        feed = {
            "embeds": embeds,
            "cos": cos,
            "sin": sin,
            "mask": mask,
        }
        for i in range(self.n_layers):
            feed[f"past_k_{i}"] = self.past_k[i]
            feed[f"past_v_{i}"] = self.past_v[i]

        outputs = self.session.run(None, feed)

        # Output ordering from session:
        # output[0] = hidden (1, T, d_model)
        # output[1] = cb0_logits (1, T, vocab + 1)
        # output[2..] = present_k_i, present_v_i
        hidden_out = outputs[0]
        cb0_logits_out = outputs[1]

        idx = 2
        for i in range(self.n_layers):
            self.past_k[i] = outputs[idx]
            self.past_v[i] = outputs[idx + 1]
            idx += 2

        self.past_len = P + T

        last_hidden = hidden_out[0, -1, :]
        last_cb0_logits = cb0_logits_out[0, -1, :]
        return last_hidden, last_cb0_logits
