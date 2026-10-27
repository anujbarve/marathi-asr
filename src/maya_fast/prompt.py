from __future__ import annotations

import math
import dataclasses
from typing import Any
import numpy as np

from .assets import PromptAssets


def _rms_norm(x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    # x: (..., D)
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * (1.0 / np.sqrt(variance + eps))
    if weight is not None:
        normed = normed * weight
    return normed.astype(np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:
    # PyTorch default GELU
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


class AudioIn:
    def __init__(self, params: dict[str, np.ndarray]) -> None:
        # params:
        # "audio_in/table" or "table"
        # "audio_in/aeos_in" or "aeos_in"
        # "audio_in/norm" or "norm"
        self.table = params.get("audio_in/table", params.get("table"))
        self.aeos_in = params.get("audio_in/aeos_in", params.get("aeos_in"))
        self.norm_w = params.get("audio_in/norm", params.get("norm"))
        if self.table is None or self.aeos_in is None:
            raise ValueError("Missing audio_in parameters in dict")

        self.d_backbone = self.table.shape[1]
        self.n_q = 16
        self.n_codes = self.table.shape[0] // self.n_q

    def frames(self, codes: np.ndarray, q_use: int = 16) -> np.ndarray:
        """
        codes: (N, 16) int64 in [0, n_codes)
        Returns: (N, 2048) float32
        """
        codes = np.asarray(codes, dtype=np.int64)
        if np.any((codes[:, :q_use] < 0) | (codes[:, :q_use] >= self.n_codes)):
            raise ValueError("audio code outside [0, n_codes) reached the input table")

        lvl = np.arange(q_use, dtype=np.int64)[None, :]  # (1, q_use)
        flat_indices = codes[:, :q_use] + lvl * self.n_codes  # (N, q_use)
        e = self.table[flat_indices]  # (N, q_use, d_backbone)
        sum_e = np.sum(e, axis=1) / math.sqrt(q_use)  # (N, d_backbone)
        return _rms_norm(sum_e, self.norm_w)

    def aeos(self) -> np.ndarray:
        """
        Returns: (2048,) float32
        """
        return _rms_norm(self.aeos_in, self.norm_w)


@dataclasses.dataclass
class Prompt:
    token_ids: np.ndarray
    embeds: np.ndarray
    n_prompt: int


def _speaker_anchor_forward(
    refs: np.ndarray,
    null_mask: bool,
    fc1_w: np.ndarray,
    fc1_b: np.ndarray,
    fc2_w: np.ndarray,
    fc2_b: np.ndarray,
    null_vec: np.ndarray,
) -> np.ndarray:
    """
    refs: (R, ecapa_dim) or (ecapa_dim,)
    returns: (d_model,) float32
    """
    if null_mask or refs is None or len(refs) == 0:
        x = null_vec
    else:
        refs = np.asarray(refs, dtype=np.float32)
        if refs.ndim == 1:
            x = refs
        else:
            x = np.mean(refs, axis=0)

    # fc1: x @ fc1_w.T + fc1_b (or x @ fc1_w depending on weight layout)
    # in PyTorch Linear(in_f, out_f): weight is (out_f, in_f)
    if fc1_w.shape[1] == len(x):
        h1 = x @ fc1_w.T + fc1_b
    else:
        h1 = x @ fc1_w + fc1_b

    act = _gelu(h1)

    if fc2_w.shape[1] == len(act):
        out = act @ fc2_w.T + fc2_b
    else:
        out = act @ fc2_w + fc2_b

    return out.astype(np.float32)


def build_prompt(
    text: str,
    voice: str,
    assets: PromptAssets,
    lang: str = "mr",
) -> Prompt:
    """
    Port of the inference path:
    text -> tokens
    prefix tokens: <|lang:mr|> ... <|audio|>
    embed lookup + audio_in + speaker_anchor added at <|audio|> sentinel.
    """
    if voice not in assets.voices_info:
        raise KeyError(f"Unknown voice {voice!r}. Available: {list(assets.voices_info.keys())}")

    vinfo = assets.voices_info[voice]
    mode = vinfo.get("mode", "spk")

    # Encode text
    # In maya3 / tokenizer:
    # Text tokens: target text encoded
    encoded = assets.tokenizer.encode(text)
    text_token_ids = list(encoded.ids)

    # Find sentinel token IDs
    lang_tag = f"<|lang:{lang}|>"
    lang_id = assets.tokenizer.token_to_id(lang_tag)
    if lang_id is None:
        lang_id = assets.tokenizer.token_to_id("<|lang:mr|>")

    audio_id = assets.tokenizer.token_to_id("<|audio|>")

    tokens: list[int] = []
    # If there is prefix in voice:
    prefix = vinfo.get("prefix")
    if prefix is not None and "tokens" in prefix:
        tokens.extend(list(prefix["tokens"]))
    else:
        if lang_id is not None:
            tokens.append(lang_id)
        tokens.extend(text_token_ids)
        if audio_id is not None:
            tokens.append(audio_id)

    token_arr = np.array(tokens, dtype=np.int64)
    embeds = assets.embed_table.lookup(token_arr)  # (T, d_model)

    # Speaker anchor at <|audio|> sentinel
    ecapa_key = f"{voice}/ecapa"
    ecapa = assets.voices_arrays.get(ecapa_key)
    null_mask = mode in ("plain", "joint_null", "icl_only") or ecapa is None

    fc1_w = assets.prompt_params.get("anchor_fc1_w")
    fc1_b = assets.prompt_params.get("anchor_fc1_b")
    fc2_w = assets.prompt_params.get("anchor_fc2_w")
    fc2_b = assets.prompt_params.get("anchor_fc2_b")
    null_vec = assets.prompt_params.get("anchor_null")

    if fc1_w is not None and fc2_w is not None and null_vec is not None:
        anchor_vec = _speaker_anchor_forward(
            ecapa,
            null_mask=null_mask,
            fc1_w=fc1_w,
            fc1_b=fc1_b,
            fc2_w=fc2_w,
            fc2_b=fc2_b,
            null_vec=null_vec,
        )
        # Find index of sentinel <|audio|>
        if audio_id is not None:
            sentinel_indices = np.where(token_arr == audio_id)[0]
            for s_idx in sentinel_indices:
                embeds[s_idx] += anchor_vec

    return Prompt(token_ids=token_arr, embeds=embeds, n_prompt=len(token_arr))
