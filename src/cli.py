#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import psutil

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_utils import load_and_resample
from src.models import get_model
from src.normalizer import normalize_marathi_text

def main():
    parser = argparse.ArgumentParser(
        description="Lightweight Marathi Automatic Speech Recognition (ASR) CLI"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the input audio file (.wav, .flac, etc.)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="indicconformer-mr-int8",
        choices=["indicconformer-mr-int8", "whisper-tiny", "whisper-base", "whisper-small"],
        help="ASR model engine to use (default: indicconformer-mr-int8)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format with metadata"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply standard Devanagari text normalization"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found at '{args.audio_file}'", file=sys.stderr)
        sys.exit(1)

    # 1. Load and resample audio
    try:
        t0 = time.perf_counter()
        audio_data, duration_s = load_and_resample(args.audio_file, target_sr=16000)
    except Exception as e:
        print(f"Error reading audio file: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load model
    try:
        asr_engine = get_model(args.model)
        load_t0 = time.perf_counter()
        asr_engine.load()
        load_time_s = time.perf_counter() - load_t0
    except Exception as e:
        print(f"Error initializing model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Transcribe
    infer_t0 = time.perf_counter()
    raw_transcript = asr_engine.transcribe(audio_data, sample_rate=16000)
    infer_time_s = time.perf_counter() - infer_t0

    final_transcript = normalize_marathi_text(raw_transcript) if args.normalize else raw_transcript
    rtf = infer_time_s / max(duration_s, 0.001)

    # Memory usage check
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)

    if args.json:
        result = {
            "model": asr_engine.model_id,
            "transcript": final_transcript,
            "raw_transcript": raw_transcript,
            "audio_duration_s": round(duration_s, 2),
            "inference_latency_s": round(infer_time_s, 3),
            "model_load_latency_s": round(load_time_s, 3),
            "real_time_factor": round(rtf, 4),
            "peak_ram_mb": round(ram_mb, 2),
            "language": "mr"
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(final_transcript)

if __name__ == "__main__":
    main()
