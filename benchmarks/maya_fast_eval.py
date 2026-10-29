"""End-to-end eval for maya-fast: synthesize, transcribe, score.

Systems compared (all scored by the same Marathi ASR, text = test-set text):
  candidate  MayaFast(--assets)
  baseline   MayaFast(--baseline-assets), optional (use when no reference corpus)
  reference  WAVs stored in --ref-dir/{id}_s{seed}.npz (upstream PyTorch), if present

The TTS model and the ASR model never share a process: the driver runs
``--phase tts`` then ``--phase asr`` as separate child processes, so the
parent (and memguard) sees one model resident at a time.

Run guarded:
  python3 tools/memguard.py --rss-limit-mb 5000 --min-free-mb 5500 -- \
      python3 benchmarks/maya_fast_eval.py --assets DIR --limit 3 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

TESTSET = ROOT / "benchmarks" / "tts_testset_mr.json"
METRIC_KEYS = ("cer", "wer", "cer_lenient", "wer_lenient")
PHASES = ("backbone_s", "depth_s", "mimi_s")


def work_dir() -> Path:
    env = os.environ.get("MAYA_FAST_WORK")
    return Path(env) if env else Path.home() / ".cache" / "maya_fast"


def max_rss_mb() -> float:
    """Peak RSS of this process in MiB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def load_testset(path: Path, limit: int | None) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(rows: list[dict], peak_rss: dict[str, float] | None = None) -> dict:
    """rows: one dict per (system, id, seed) with cer/wer/..., category and,
    for synthesized systems, rtf/frames/audio_s/backbone_s/depth_s/mimi_s.
    Returns {system: {"overall": {...}, "categories": {cat: {...}}, ...}}."""
    peak_rss = peak_rss or {}
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    out = {}
    for system, rs in by_sys.items():
        def block(sub: list[dict]) -> dict:
            b = {"n": len(sub)}
            for k in METRIC_KEYS:
                b[k] = _mean([r[k] for r in sub])
            return b
        cats: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            cats[r["category"]].append(r)
        entry = {
            "overall": block(rs),
            "categories": {c: block(v) for c, v in sorted(cats.items())},
        }
        timed = [r for r in rs if r.get("frames")]
        if timed:
            frames = sum(r["frames"] for r in timed)
            entry["rtf_mean"] = _mean([r["rtf"] for r in timed])
            entry["rtf_max"] = max(r["rtf"] for r in timed)
            entry["frames"] = frames
            for p in PHASES:
                entry["ms_per_frame_" + p[:-2]] = 1000.0 * sum(r[p] for r in timed) / frames
        if system in peak_rss:
            entry["peak_rss_mb"] = peak_rss[system]
        out[system] = entry
    return out


def format_table(summary: dict, title: str = "") -> str:
    lines = []
    if title:
        lines += [f"### {title}", ""]
    cats = sorted({c for s in summary.values() for c in s["categories"]})
    lines.append("| system | " + " | ".join(cats) + " | overall | n |")
    lines.append("|---|" + "---|" * (len(cats) + 2))
    for metric in ("cer", "wer", "cer_lenient", "wer_lenient"):
        for system, s in summary.items():
            cells = [f"{s['categories'][c][metric]:.3f}" if c in s["categories"] else "-" for c in cats]
            lines.append(f"| {system} {metric.upper()} | " + " | ".join(cells)
                         + f" | {s['overall'][metric]:.3f} | {s['overall']['n']} |")
    lines += ["", "| system | RTF mean | RTF max | peak RSS MB | backbone ms/frame | depth ms/frame | mimi ms/frame |",
              "|---|---|---|---|---|---|---|"]
    for system, s in summary.items():
        if "rtf_mean" not in s:
            continue
        lines.append(
            f"| {system} | {s['rtf_mean']:.3f} | {s['rtf_max']:.3f} | "
            f"{s.get('peak_rss_mb', float('nan')):.0f} | {s['ms_per_frame_backbone']:.1f} | "
            f"{s['ms_per_frame_depth']:.1f} | {s['ms_per_frame_mimi']:.1f} |")
    return "\n".join(lines) + "\n"


def delta_vs(summary: dict, system: str, other: str) -> dict:
    a, b = summary[system]["overall"], summary[other]["overall"]
    return {k: a[k] - b[k] for k in METRIC_KEYS}


# ---------------------------------------------------------------- phases

def phase_tts(args) -> None:
    from src.maya_fast.engine import MayaFast
    import soundfile as sf

    run = Path(args.run_dir)
    items = load_testset(args.testset, args.limit)
    systems = {"candidate": args.assets}
    if args.baseline_assets:
        systems["baseline"] = args.baseline_assets
    meta = {"items": [], "peak_rss_mb": {}}
    for system, assets in systems.items():
        (run / system).mkdir(parents=True, exist_ok=True)
        eng = MayaFast(assets_dir=assets, threads=args.threads)
        for it in items:
            for seed in args.seeds:
                wav, m = eng.synthesize(it["text"], voice=args.voice, seed=seed)
                rel = f"{system}/{it['id']}_s{seed}.wav"
                sf.write(run / rel, wav, 24000)
                meta["items"].append({
                    "system": system, "id": it["id"], "category": it["category"],
                    "seed": seed, "wav": rel,
                    **{k: m[k] for k in ("audio_s", "wall_s", "rtf", "frames") + PHASES}})
                print(f"[tts] {system} {it['id']} s{seed} rtf={m['rtf']:.2f} "
                      f"frames={m['frames']} rss_max={max_rss_mb():.0f}MB", flush=True)
        # ru_maxrss is process-wide and monotonic; with several systems in
        # one process it is a running max, so the last system's value covers all.
        meta["peak_rss_mb"][system] = max_rss_mb()
        del eng
        import gc; gc.collect()
    (run / "tts_meta.json").write_text(json.dumps(meta, indent=1))


def phase_asr(args) -> None:
    from src.audio_utils import resample
    from src.metrics import score_transcript
    from src.models import IndicConformerONNXASR
    import soundfile as sf

    run = Path(args.run_dir)
    items = load_testset(args.testset, args.limit)
    meta = json.loads((run / "tts_meta.json").read_text())
    jobs = []  # (row-base, wav 24k)
    for r in meta["items"]:
        wav, sr = sf.read(run / r["wav"], dtype="float32")
        jobs.append((dict(r), wav, sr))
    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    if ref_dir and ref_dir.is_dir():
        for it in items:
            for seed in args.seeds:
                p = ref_dir / f"{it['id']}_s{seed}.npz"
                if p.exists():
                    wav = np.load(p)["wav"].astype(np.float32)
                    jobs.append(({"system": "reference", "id": it["id"],
                                  "category": it["category"], "seed": seed}, wav, 24000))
    text = {it["id"]: it["text"] for it in items}
    asr = IndicConformerONNXASR(num_threads=args.threads or 4)
    asr.load()
    rows = []
    for base, wav, sr in jobs:
        hyp = asr.transcribe(resample(wav, sr, 16000), 16000)
        row = {**base, **score_transcript(text[base["id"]], hyp), "hyp": hyp}
        rows.append(row)
        print(f"[asr] {row['system']} {row['id']} s{row['seed']} cer={row['cer']:.3f}", flush=True)
    summary = aggregate(rows, meta["peak_rss_mb"])
    (run / "results.json").write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1))
    table = format_table(summary, title=f"maya-fast eval {run.name}")
    (run / "results.md").write_text(table)
    print(table)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="candidate assets dir")
    ap.add_argument("--baseline-assets", help="second assets dir, synthesized and scored the same way")
    ap.add_argument("--ref-dir", default=str(work_dir() / "golden" / "ref"),
                    help="upstream reference npz corpus (wav key); skipped if missing")
    ap.add_argument("--testset", default=str(TESTSET))
    ap.add_argument("--threads", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--voice", default="Charvi")
    ap.add_argument("--run-name")
    ap.add_argument("--run-dir")
    ap.add_argument("--phase", choices=("all", "tts", "asr"), default="all")
    args = ap.parse_args(argv)
    if not args.run_dir:
        name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
        args.run_dir = str(work_dir() / "eval" / name)
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    if args.phase == "tts":
        phase_tts(args); return 0
    if args.phase == "asr":
        phase_asr(args); return 0
    base = [sys.executable, os.path.abspath(__file__)] + [a for a in (argv if argv is not None else sys.argv[1:])]
    for ph in ("tts", "asr"):
        cmd = base + ["--phase", ph, "--run-dir", args.run_dir]
        rc = subprocess.call(cmd)
        if rc:
            print(f"phase {ph} failed rc={rc}", file=sys.stderr)
            return rc
    print(f"results: {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
