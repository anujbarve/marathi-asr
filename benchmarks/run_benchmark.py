import argparse
import csv
import json
import os
import platform
import sys
import time
import psutil
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_utils import load_and_resample
from src.models import get_model
from src.normalizer import normalize_marathi_text
from src.metrics import score_transcript

DEFAULT_MODELS = ["indicconformer-mr-int8"]
# Whisper is opt-in (--models): it is too slow for the i5 target to run by default.

def run_benchmark(models_to_test):
    manifest_path = "benchmarks/dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Hardware: {platform.processor() or platform.machine()} | CPU count: {os.cpu_count()} | {platform.system()} {platform.release()}")

    results = []
    output_csv = "benchmarks/benchmark_results.csv"

    for model_name in models_to_test:
        print(f"\n==========================================")
        print(f"Benchmarking Model: {model_name}")
        print(f"==========================================")

        engine = get_model(model_name)

        # Measure cold start load time
        t_load_start = time.perf_counter()
        engine.load()
        cold_load_s = time.perf_counter() - t_load_start
        print(f"Loaded in {cold_load_s:.3f}s | Disk Size: {engine.disk_size_mb} MB")

        # Warm up with dummy audio
        warmup_audio = np.zeros(16000, dtype=np.float32)
        engine.transcribe(warmup_audio)

        for s_idx, sample in enumerate(samples):
            audio_path = sample["file"]
            audio_data, duration_s = load_and_resample(audio_path, target_sr=16000)
            reference = sample["reference"]
            norm_ref = normalize_marathi_text(reference)

            # Measure inference latency & peak RAM
            proc = psutil.Process(os.getpid())
            ram_before = proc.memory_info().rss / (1024 * 1024)

            t_infer_start = time.perf_counter()
            raw_pred = engine.transcribe(audio_data, sample_rate=16000)
            infer_s = time.perf_counter() - t_infer_start

            ram_after = proc.memory_info().rss / (1024 * 1024)
            peak_ram = max(ram_before, ram_after)

            norm_pred = normalize_marathi_text(raw_pred)

            scores = score_transcript(reference, raw_pred)

            rtf = infer_s / max(duration_s, 0.001)

            res = {
                "model": engine.model_id,
                "sample_id": sample["id"],
                "category": sample["category"],
                "duration_s": round(duration_s, 2),
                "latency_s": round(infer_s, 3),
                "rtf": round(rtf, 4),
                "wer": round(scores["wer"], 4),
                "cer": round(scores["cer"], 4),
                "wer_lenient": round(scores["wer_lenient"], 4),
                "cer_lenient": round(scores["cer_lenient"], 4),
                "peak_ram_mb": round(peak_ram, 1),
                "disk_size_mb": engine.disk_size_mb,
                "cold_load_s": round(cold_load_s, 3),
                "reference": norm_ref,
                "prediction": norm_pred
            }
            results.append(res)
            print(f"[{sample['id']} - {sample['category']}] "
                  f"Dur: {duration_s:.1f}s | Lat: {infer_s:.3f}s | RTF: {rtf:.3f} | "
                  f"WER: {scores['wer']:.2%} | CER: {scores['cer']:.2%} | "
                  f"WER(lenient): {scores['wer_lenient']:.2%}")

    # Write CSV
    fieldnames = [
        "model", "sample_id", "category", "duration_s", "latency_s", "rtf",
        "wer", "cer", "wer_lenient", "cer_lenient", "peak_ram_mb", "disk_size_mb",
        "cold_load_s", "reference", "prediction"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll benchmark results successfully written to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Marathi ASR models on benchmarks/dataset_manifest.json")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model ids to run (default: {' '.join(DEFAULT_MODELS)})")
    args = parser.parse_args()
    run_benchmark(args.models)

if __name__ == "__main__":
    main()
