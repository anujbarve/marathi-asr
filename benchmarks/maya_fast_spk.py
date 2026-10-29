"""Speaker similarity: ECAPA cosine of each eval WAV to the voice's reference embedding.

Needs speechbrain + torch, so it runs in an env that has them (not the
torch-free runtime venv). Embeddings to compare against come from
<work>/golden/light.npz key ``voice/<voice>/ecapa``; the ECAPA model is the
cached ``speechbrain/spkrec-ecapa-voxceleb``.

  python3 tools/memguard.py --rss-limit-mb 4000 --min-free-mb 5000 -- \
      <venv-with-speechbrain>/bin/python benchmarks/maya_fast_spk.py --run-dir <work>/eval/<run>
Scores <run-dir>/<system>/*.wav for every system dir, plus the reference npz
corpus (--ref-dir) for the same ids/seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def work_dir() -> Path:
    env = os.environ.get("MAYA_FAST_WORK")
    return Path(env) if env else Path.home() / ".cache" / "maya_fast"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--voice", default="Charvi")
    ap.add_argument("--golden", default=str(work_dir() / "golden" / "light.npz"))
    ap.add_argument("--ref-dir", default=str(work_dir() / "golden" / "ref"))
    args = ap.parse_args(argv)

    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    import torchaudio.functional as AF

    target = np.load(args.golden, allow_pickle=True)[f"voice/{args.voice}/ecapa"]
    clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         run_opts={"device": "cpu"})

    def embed(wav24: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            w = AF.resample(torch.from_numpy(wav24.astype(np.float32)), 24000, 16000)
            return clf.encode_batch(w[None]).squeeze().numpy()

    target_kind = "golden ecapa"
    if not np.any(target):
        # Named voices store a zero ECAPA vector: use the centroid of the
        # full reference corpus (all ids and seeds) as the voice target.
        embs = [embed(np.load(p)["wav"]) for p in sorted(Path(args.ref_dir).glob("*.npz"))]
        target = np.mean(embs, axis=0)
        target_kind = f"reference-corpus centroid (n={len(embs)}; biases the reference score up slightly)"
    print("target:", target_kind)
    run = Path(args.run_dir)
    scores: dict[str, list[float]] = {}
    ids = set()
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.wav")):
            wav, _ = sf.read(f, dtype="float32")
            scores.setdefault(d.name, []).append(cosine(embed(wav), target))
            ids.add(f.stem)
    refd = Path(args.ref_dir)
    for stem in sorted(ids):
        p = refd / f"{stem}.npz"
        if p.exists():
            scores.setdefault("reference", []).append(cosine(embed(np.load(p)["wav"]), target))
    summary = {k: {"mean_cosine": float(np.mean(v)), "n": len(v)} for k, v in scores.items()}
    (run / "spk.json").write_text(json.dumps({"target": target_kind, **summary}, indent=1))
    for k, v in summary.items():
        print(f"{k}: mean ECAPA cosine {v['mean_cosine']:.4f} (n={v['n']})")
    if "reference" in summary and "candidate" in summary:
        d = summary["candidate"]["mean_cosine"] - summary["reference"]["mean_cosine"]
        print(f"candidate - reference = {d:+.4f} (gate >= -0.02)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
