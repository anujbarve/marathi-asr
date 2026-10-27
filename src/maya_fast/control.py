from __future__ import annotations

import re
import numpy as np

PRIOR_DEFAULT = {"a": 4.0, "b": 0.44, "resid_p99": 18.0}
TERMINALS = "।॥.!?\n"
FPS = 12.5
PAUSE = re.compile(r"[,;:.!?।॥…\-]")


def expected_frames(prior: dict, lang: str, nchar: int) -> tuple[float, float]:
    p = prior.get(lang, PRIOR_DEFAULT)
    return float(p["a"] + p["b"] * nchar), float(p["resid_p99"])


def min_frames(ef: float, min_frac: float = 0.5) -> int:
    return max(4, int(min_frac * ef))


def _text_feats(t: str) -> np.ndarray:
    letters = sum(1 for ch in t if ch.isalpha() or (0x900 <= ord(ch) <= 0xDFF))
    latin = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    return np.array([1.0, letters - latin, latin, len(PAUSE.findall(t)), sum(ch.isdigit() for ch in t)], dtype=np.float32)


def frame_cap(text: str, lang: str, ef: float, er: float, controls: dict) -> int:
    guard_cfg = controls.get("guard", {})
    hard_cap = guard_cfg.get("hard_cap", 1500)
    if not guard_cfg.get("enabled", True):
        return hard_cap

    # From upstream tts_quant_spike:
    # loose = min(1500, int(2.0 * (ef + er) + 40))
    # cap = loose
    # if g.enabled:
    #     cap = min(cap, max(int(ef * 0.5) + 8, dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
    dur_k = float(guard_cfg.get("dur_k", 2.2))
    dur_margin_s = float(guard_cfg.get("dur_margin_s", 1.2))

    # Duration prior estimation
    dur_prior_coefs = guard_cfg.get("dur_prior_coefs")
    if dur_prior_coefs and lang in dur_prior_coefs:
        coef = np.asarray(dur_prior_coefs[lang], dtype=np.float32)
        exp_s = float(max(0.35, float(_text_feats(text) @ coef)))
    else:
        # Fallback to ef * 0.08 s
        exp_s = float(max(0.35, ef * 0.08))

    dur_cap = int((dur_k * exp_s + dur_margin_s) / 0.08)
    loose = min(hard_cap, int(2.0 * (ef + er) + 40))
    cap = min(loose, max(int(ef * 0.5) + 8, dur_cap))
    return int(cap)


def split_text(text: str, lang: str, prior: dict, chunk_s: float = 12.0) -> list[str]:
    """Split at sentence terminals into pieces of about chunk_s seconds (prior-estimated).
    Falls back to commas, then whitespace, for run-on text."""
    if chunk_s <= 0:
        return [text]
    target = chunk_s * FPS
    pieces = [p.strip() for p in re.split(rf"(?<=[{re.escape(TERMINALS)}])\s+", text) if p.strip()]
    out: list[str] = []
    cur = ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if cur and expected_frames(prior, lang, len(cand))[0] > target:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)

    final: list[str] = []
    for piece in out:
        if expected_frames(prior, lang, len(piece))[0] <= target * 1.3:
            final.append(piece)
            continue
        words = re.split(r"(?<=[,،])\s+|\s+", piece)
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip() if cur else w
            if cur and expected_frames(prior, lang, len(cand))[0] > target:
                final.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            final.append(cur)
    return final if final else [text]


class StopPolicy:
    def __init__(self, min_frames: int, tau: float = 0.35, hazard: float = 0.6) -> None:
        self.min_frames = int(min_frames)
        self.tau = float(tau)
        self.hazard = float(hazard)
        self.H = 0.0

    def check(self, t: int, p: np.ndarray, aeos_class: int = 2048) -> str | None:
        """
        p: (vocab + 1,) probability distribution over cb0 classes.
        Returns: 'aeos_argmax', 'aeos_tau', 'aeos_hazard', or None
        """
        if t < self.min_frames:
            return None

        p_aeos = float(p[aeos_class])
        self.H = 1.0 - (1.0 - self.H) * (1.0 - p_aeos)

        if int(np.argmax(p)) == aeos_class:
            return "aeos_argmax"
        if p_aeos >= self.tau:
            return "aeos_tau"
        if self.H >= self.hazard:
            return "aeos_hazard"
        return None


def sample_cb0(p: np.ndarray, noise: np.ndarray, aeos_class: int = 2048) -> int:
    """
    aeos zeroed, Gumbel-max over the rest.
    """
    q = np.array(p, dtype=np.float32, copy=True)
    q[aeos_class] = 0.0
    s = np.sum(q)
    if s > 0:
        q = q / s
    else:
        # Uniform over classes except aeos
        q = np.ones_like(q)
        q[aeos_class] = 0.0
        q = q / np.sum(q)

    # Gumbel-max on probabilities q: argmax(log(q) + noise)
    # where q == 0, log(q) = -inf
    with np.errstate(divide="ignore"):
        log_q = np.log(q)
    perturbed = log_q + noise
    perturbed[aeos_class] = -np.inf
    return int(np.argmax(perturbed))
