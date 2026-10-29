from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path
import numpy as np

from .engine import MayaFast


def main() -> None:
    parser = argparse.ArgumentParser(description="Maya-Fast TTS CLI")
    parser.add_argument("text", type=str, help="Text to speak")
    parser.add_argument("-o", "--out", type=str, default="out.wav", help="Output WAV path")
    parser.add_argument("--voice", type=str, default="Charvi", help="Voice name (default: Charvi)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--lang", type=str, default="mr", help="Language tag: mr, hi, ... (default: mr)")
    parser.add_argument("--threads", type=int, default=None, help="ORT threads (default: min(8, cores))")
    parser.add_argument("--assets", type=str, default=None, help="Assets directory")

    args = parser.parse_args()

    engine = MayaFast(assets_dir=args.assets, threads=args.threads)
    wav, metrics = engine.synthesize(args.text, voice=args.voice, seed=args.seed, lang=args.lang)

    # Convert float32 [-1.0, 1.0] to 16-bit PCM WAV using stdlib wave
    wav_clipped = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav_clipped * 32767.0).astype(np.int16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm16.tobytes())

    print(
        f"audio_s={metrics['audio_s']:.2f} "
        f"wall_s={metrics['wall_s']:.2f} "
        f"rtf={metrics['rtf']:.2f} "
        f"peak_rss_mb={metrics['peak_rss_mb']:.1f}"
    )


if __name__ == "__main__":
    main()
