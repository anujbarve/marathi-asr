from __future__ import annotations

import re

import numpy as np

# Port of maya3.guards (DurationPrior, text_feats) and scripts/s2_serve_controls.py (expected_frames, split_text).
# The numbers come from controls.json, written by tools/maya_export/export_prompt_assets.py from upstream.
PAUSE = re.compile(r"[,;:.!?।॥…\-]")
PRIOR_DEFAULT = {"a": 2.0, "b": 0.72, "resid_p99": 90.0}  # scripts/s2_serve_controls.py
FRAME_S = 0.08  # maya3.guards.FRAME_S (one frame = 80 ms of audio)


def text_feats(t: str) -> np.ndarray:
    letters = sum(1 for ch in t if ch.isalpha() or (0x900 <= ord(ch) <= 0xDFF))
    latin = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    return np.array([1.0, letters - latin, latin, len(PAUSE.findall(t)), sum(ch.isdigit() for ch in t)])


def expected_frames(prior: dict, lang: str, nchar: int) -> tuple[float, float]:
    p = prior.get(lang, PRIOR_DEFAULT)
    return p["a"] + p["b"] * nchar, p["resid_p99"]


def min_frames(ef: float) -> int:
    return max(4, int(0.5 * ef))


def expected_seconds(text: str, lang: str, guard: dict) -> float:
    coefs = guard["dur_prior_coefs"]
    coef = np.asarray(coefs.get(lang, guard["dur_prior_default"]), dtype=np.float64)
    return float(max(0.35, text_feats(text) @ coef))


def frame_cap(text: str, lang: str, ef: float, er: float, controls: dict) -> int:
    """Duration cap as in tools/maya_export/golden_light.py: min(loose, max(0.5*ef + 8, duration-prior cap))."""
    guard = controls["guard"]
    loose = min(int(guard["hard_cap"]), int(2.0 * (ef + er) + 40))
    if not guard["enabled"]:
        return loose
    dur_cap = int((float(guard["dur_k"]) * expected_seconds(text, lang, guard)
                   + float(guard["dur_margin_s"])) / FRAME_S)
    return int(min(loose, max(int(ef * 0.5) + 8, dur_cap)))


def split_text(text: str, lang: str, prior: dict, chunk_s: float = 12.0, *, fps: float = 12.5,
               terminals: str = "।.!?;।॥") -> list[str]:
    """Split at sentence terminals into pieces of about chunk_s seconds (prior-estimated).
    Falls back to commas, then whitespace, for run-on text."""
    if chunk_s <= 0:
        return [text]
    target = chunk_s * fps
    pieces = [p.strip() for p in re.split(rf"(?<=[{re.escape(terminals)}])\s+", text) if p.strip()]
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
    return final


class StopPolicy:
    """The "controlled" stop rule of ServeSession.speak: argmax, tau and hazard tests, none before min_frames."""

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
    aeos zeroed and renormalised, then Gumbel-max over the rest (same distribution as upstream multinomial).
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
