"""Blind A/B listening pairs: upstream reference vs maya-fast candidate.

  generate: python3 tools/memguard.py --rss-limit-mb 5000 --min-free-mb 5500 -- \
                python3 benchmarks/maya_fast_listen.py --assets DIR
            writes pair_XX_A.wav / pair_XX_B.wav and key.json to <work>/listen/
  score:    python3 benchmarks/maya_fast_listen.py --score answers.json
            answers.json: {"01": "A" | "B" | "tie", ...}
            prints the candidate-preferred-or-tied rate (pass >= 45%).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

TESTSET = ROOT / "benchmarks" / "tts_testset_mr.json"
N_PAIRS = 10


def work_dir() -> Path:
    env = os.environ.get("MAYA_FAST_WORK")
    return Path(env) if env else Path.home() / ".cache" / "maya_fast"


def pick_sentences(items: list[dict], n: int = N_PAIRS) -> list[dict]:
    """Round-robin over categories (file order) so every category is covered."""
    cats: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        cats[it["category"]].append(it)
    out, i = [], 0
    while len(out) < min(n, len(items)):
        for lst in cats.values():
            if i < len(lst) and len(out) < n:
                out.append(lst[i])
        i += 1
    return out


def assign_sides(n: int, rng: random.Random) -> list[str]:
    """Which side holds the candidate, per pair: 'A' or 'B'."""
    return [rng.choice("AB") for _ in range(n)]


TARGET_RMS = 0.05
PEAK_LIMIT = 0.98


def normalize_rms(wav: np.ndarray, target: float = TARGET_RMS, peak_limit: float = PEAK_LIMIT) -> np.ndarray:
    """Scale to a common RMS so loudness cannot unblind a pair; scale is capped so the peak stays <= peak_limit."""
    wav = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if rms < 1e-9:
        return wav
    gain = min(target / rms, peak_limit / peak) if peak > 0 else 1.0
    return (wav * gain).astype(np.float32)


def renormalize_dir(d: Path) -> int:
    """Re-normalize every pair_*_{A,B}.wav in d in place (no re-synthesis). Returns files rewritten."""
    import soundfile as sf

    n = 0
    for f in sorted(Path(d).glob("pair_*_[AB].wav")):
        x, sr = sf.read(f, dtype="float32")
        sf.write(f, normalize_rms(x), sr)
        n += 1
    return n


def score_answers(key: dict, answers: dict) -> dict:
    pref = tie = ref = 0
    for pid, ans in answers.items():
        cand_side = key[pid]["candidate"]
        if ans == "tie":
            tie += 1
        elif ans == cand_side:
            pref += 1
        elif ans in ("A", "B"):
            ref += 1
        else:
            raise ValueError(f"bad answer {ans!r} for pair {pid}")
    n = pref + tie + ref
    return {"n": n, "candidate_preferred": pref, "tied": tie, "reference_preferred": ref,
            "rate": (pref + tie) / n if n else float("nan")}


def generate(args) -> None:
    import soundfile as sf
    from src.maya_fast.engine import MayaFast

    items = pick_sentences(json.loads(Path(args.testset).read_text(encoding="utf-8")))
    out = Path(args.out_dir) if args.out_dir else work_dir() / "listen"
    out.mkdir(parents=True, exist_ok=True)
    sides = assign_sides(len(items), random.Random(args.shuffle_seed))
    eng = MayaFast(assets_dir=args.assets, threads=args.threads)
    key = {}
    for i, (it, cand_side) in enumerate(zip(items, sides), start=1):
        pid = f"{i:02d}"
        ref = np.load(Path(args.ref_dir) / f"{it['id']}_s0.npz")["wav"].astype(np.float32)
        cand, _ = eng.synthesize(it["text"], voice=args.voice, seed=0)
        cand, ref = normalize_rms(cand), normalize_rms(ref)
        a, b = (cand, ref) if cand_side == "A" else (ref, cand)
        sf.write(out / f"pair_{pid}_A.wav", a, 24000)
        sf.write(out / f"pair_{pid}_B.wav", b, 24000)
        key[pid] = {"id": it["id"], "text": it["text"], "candidate": cand_side}
        print(f"[listen] pair {pid} {it['id']}", flush=True)
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
    print(f"wrote {len(key)} pairs to {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets")
    ap.add_argument("--ref-dir", default=str(work_dir() / "golden" / "ref"))
    ap.add_argument("--testset", default=str(TESTSET))
    ap.add_argument("--voice", default="Charvi")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--shuffle-seed", type=int, default=12345)
    ap.add_argument("--out-dir", help="where pairs and key.json go (default <work>/listen)")
    ap.add_argument("--renormalize", metavar="DIR", help="re-normalize existing pair wavs in DIR in place and exit")
    ap.add_argument("--score", help="answers.json to score against <out-dir>/key.json")
    args = ap.parse_args(argv)
    if args.renormalize:
        print(f"renormalized {renormalize_dir(Path(args.renormalize))} files")
        return 0
    if args.score:
        out = Path(args.out_dir) if args.out_dir else work_dir() / "listen"
        key = json.loads((out / "key.json").read_text())
        r = score_answers(key, json.loads(Path(args.score).read_text()))
        print(json.dumps(r, indent=1))
        print("PASS" if r["rate"] >= 0.45 else "FAIL", f"(rate {r['rate']:.2f}, gate 0.45)")
        return 0
    if not args.assets:
        ap.error("--assets is required to generate pairs")
    generate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
