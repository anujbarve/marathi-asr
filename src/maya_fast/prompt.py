from __future__ import annotations

import dataclasses
import math

import numpy as np

from .assets import PromptAssets
from .embed import EmbeddingTable

# Port of the inference path for named voices (maya3.prompt.NamedSpeakerBuilder with V5Builder.example, the
# packer's _span and collate_v2 truncation, TTSModelV2.compose_inputs and SpeakerAnchor in eval mode).
# Checked token for token and value for value against tools/maya_export/golden_light.py by the golden tests.

RMS_EPS = float(np.finfo(np.float32).eps)  # torch.nn.RMSNorm default eps for float32
_erf = np.frompyfunc(math.erf, 1, 1)


def _rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + RMS_EPS)) * weight).astype(np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:
    # exact (erf) GELU, torch's default
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float32))).astype(np.float32)


class AudioIn:
    """Audio code frames -> backbone input (port of tts/model/audio_io.py AudioInputEmbed, eval mode)."""

    def __init__(self, table: EmbeddingTable, aeos_in: np.ndarray, norm_w: np.ndarray, n_q: int = 16) -> None:
        self.table = table
        self.aeos_in = np.asarray(aeos_in, dtype=np.float32)
        self.norm_w = np.asarray(norm_w, dtype=np.float32)
        self.n_q = int(n_q)
        if table.rows % self.n_q:
            raise ValueError("audio table rows must be a multiple of n_q")
        self.n_codes = table.rows // self.n_q

    def frames(self, codes: np.ndarray, q_use: int = 16) -> np.ndarray:
        """
        codes: (N, 16) int64 in [0, n_codes)
        Returns: (N, d_model) float32
        """
        codes = np.asarray(codes, dtype=np.int64)
        if np.any((codes[:, :q_use] < 0) | (codes[:, :q_use] >= self.n_codes)):
            raise ValueError("audio code outside [0, n_codes) reached the input table")
        lvl = np.arange(q_use, dtype=np.int64)[None, :]
        e = self.table.lookup(codes[:, :q_use] + lvl * self.n_codes)  # (N, q_use, d)
        return _rms_norm(e.sum(axis=1) / math.sqrt(q_use), self.norm_w)

    def aeos(self) -> np.ndarray:
        """Returns: (d_model,) float32"""
        return _rms_norm(self.aeos_in[None, :], self.norm_w)[0]


def speaker_anchor_null(p: dict[str, np.ndarray]) -> np.ndarray:
    """SpeakerAnchor for a null speaker row: fc2(gelu(fc1(null_vec))). Returns (d_model,) float32."""
    h = _gelu(p["anchor_null"] @ p["anchor_fc1_w"].T + p["anchor_fc1_b"])
    return (h @ p["anchor_fc2_w"].T + p["anchor_fc2_b"]).astype(np.float32)


@dataclasses.dataclass
class Prompt:
    token_ids: np.ndarray  # (n_prompt,) int64, ends with the <|audio|> sentinel
    embeds: np.ndarray     # (n_prompt, d_model) float32
    n_prompt: int


def build_prompt(text: str, voice: str, assets: PromptAssets, lang: str = "mr") -> Prompt:
    """
    Named voice prompt:  <|lang:xx|> "{Name}: " text <|audio|>
    The speaker anchor (null for named voices) is added at the <|audio|> sentinel, the last prompt position.
    """
    name = _resolve_voice(voice, assets)
    info = assets.voices_info[name]
    if info.get("mode") != "named":
        raise NotImplementedError(f"voice {name!r} uses mode {info.get('mode')!r}; only named voices are supported")

    rendered = "{" + name + "}: " + text.strip()
    tok = assets.tokenizer
    lang_id = tok.token_to_id(f"<|lang:{lang}|>")
    audio_id = tok.token_to_id("<|audio|>")
    if lang_id is None or audio_id is None:
        raise KeyError(f"tokenizer lacks <|lang:{lang}|> or <|audio|>")

    ids = [lang_id, *tok.encode(rendered, add_special_tokens=False).ids, audio_id]
    token_ids = np.asarray(ids, dtype=np.int64)
    embeds = assets.embed_table.lookup(token_ids)
    embeds[-1] += speaker_anchor_null(assets.prompt_params)
    return Prompt(token_ids=token_ids, embeds=embeds, n_prompt=len(token_ids))


def _resolve_voice(voice: str, assets: PromptAssets) -> str:
    if voice in assets.voices_info:
        return voice
    low = str(voice).lower()  # voices are case-insensitive
    hits = [k for k in assets.voices_info if k.lower() == low]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"Unknown voice {voice!r}. Available: {sorted(assets.voices_info)}")
