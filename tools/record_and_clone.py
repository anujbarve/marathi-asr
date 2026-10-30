#!/usr/bin/env python3
"""Interactive microphone voice recording and true voice cloning with Maya-Fast.

1. Record a short speech sample from any selected microphone (or use an existing WAV).
2. Automatically transcribes the reference audio using the built-in IndicConformer ASR.
3. Clones the voice via Maya 2's ECAPA speaker memory & in-context conditioning.
4. Generates speech with the Maya-Fast ONNX engine in your exact voice.
5. Optionally plays the generated audio on your Mac speakers.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.clone_voice_fast import clone_voice
from src.audio_utils import load_and_resample
from src.models import get_model


def list_microphones():
    devs = sd.query_devices()
    inputs = []
    print("\n" + "=" * 55)
    print("🎤 AVAILABLE MICROPHONES")
    print("=" * 55)
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            inputs.append(i)
            print(f"  [{i}] {d['name']} (Channels: {d['max_input_channels']}, Default SR: {int(d['default_samplerate'])} Hz)")
    print("=" * 55)
    return inputs


def record_audio(device_id: int, duration_s: float = 6.0, samplerate: int = 24000) -> np.ndarray:
    print(f"\nRecording {duration_s} seconds from device [{device_id}]...")
    for count in range(3, 0, -1):
        print(f"Starting in {count}...", end="\r", flush=True)
        time.sleep(1.0)
    print("🔴 RECORDING NOW! Please speak a natural Marathi or Hindi sentence...\n")
    
    rec = sd.rec(int(duration_s * samplerate), samplerate=samplerate, channels=1, dtype='float32', device=device_id)
    sd.wait()
    print("🟢 Recording complete!")
    return rec.squeeze()


def normalize_audio(wav: np.ndarray, target_rms: float = 0.05, peak_limit: float = 0.95) -> np.ndarray:
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if rms < 1e-9 or peak < 1e-9:
        return wav
    gain = min(target_rms / rms, peak_limit / peak)
    return (wav * gain).astype(np.float32)


def auto_transcribe(audio_path: Path) -> str:
    print("\n🤖 Automatically transcribing reference speech with IndicConformer ASR...")
    model = get_model("indicconformer-mr-int8")
    wav = load_and_resample(str(audio_path), target_sr=16000)
    text = model.transcribe(wav).strip()
    print(f"📝 Transcribed text: \"{text}\"")
    return text


def main():
    parser = argparse.ArgumentParser(description="Microphone voice recording and true voice cloning")
    parser.add_argument("--sample", type=str, default=None, 
                        help="Path to existing reference recording (.wav). If provided, skips microphone recording.")
    parser.add_argument("--ref-text", type=str, default=None,
                        help="Transcript of the reference recording. If omitted, auto-transcribes with IndicConformer.")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index")
    parser.add_argument("--duration", type=float, default=6.0, help="Recording duration in seconds (default 6.0)")
    parser.add_argument("--text", type=str, 
                        default="नमस्कार! हा माझा स्वतःचा आवाज आहे आणि माया-फास्ट मॉडेल आता माझ्या आवाजात बोलत आहे.", 
                        help="Target text for the model to speak in your cloned voice")
    parser.add_argument("--out", type=str, default="my_cloned_voice.wav", help="Output audio file path")
    parser.add_argument("--lang", type=str, default="mr", help="Language code (default: mr)")
    parser.add_argument("--play", action="store_true", help="Automatically play output using macOS afplay")
    args = parser.parse_args()

    sample_path = None
    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.exists():
            print(f"❌ Error: Sample file '{sample_path}' does not exist.")
            return 1
        print(f"📁 Using existing voice sample: {sample_path.resolve()}")
    else:
        # Check if default sample already exists and ask user
        default_sample = Path("my_sample_recording.wav")
        record_new = True
        if default_sample.exists() and sys.stdin.isatty():
            choice = input(f"Found existing recording '{default_sample}'. Use it? [Y/n]: ").strip().lower()
            if choice in ("", "y", "yes"):
                sample_path = default_sample
                record_new = False

        if record_new:
            inputs = list_microphones()
            if not inputs:
                print("❌ No microphone devices detected on this system.")
                return 1

            device_id = args.device
            if device_id is None:
                default_dev = sd.default.device[0]
                if sys.stdin.isatty():
                    prompt_txt = f"Select microphone number (Press Enter for default [{default_dev}]): "
                    choice = input(prompt_txt).strip()
                    if not choice:
                        device_id = default_dev
                    else:
                        try:
                            device_id = int(choice)
                        except ValueError:
                            print(f"Invalid choice '{choice}', using default {default_dev}")
                            device_id = default_dev
                else:
                    device_id = default_dev

            if device_id not in inputs:
                print(f"Device {device_id} is not a valid input device. Using {inputs[0]}")
                device_id = inputs[0]

            sample_wav = record_audio(device_id, duration_s=args.duration, samplerate=24000)
            sample_wav = normalize_audio(sample_wav)
            sample_path = Path("my_sample_recording.wav")
            sf.write(sample_path, sample_wav, 24000)
            print(f"💾 Saved sample recording to: {sample_path.resolve()}")

    # Determine reference transcript
    ref_text = args.ref_text
    if not ref_text:
        ref_text = auto_transcribe(sample_path)
        if not ref_text:
            print("⚠️ ASR returned empty transcription. Please provide --ref-text manually.")
            return 1

    # Perform True Voice Cloning
    print("\nStarting synthesis with Maya-Fast true voice cloning...")
    metrics = clone_voice(
        ref_audio_path=sample_path,
        ref_text=ref_text,
        target_text=args.text,
        out_path=args.out,
        lang=args.lang,
    )

    out_file = Path(metrics["out_file"])
    print(f"\n✅ Cloned voice audio successfully saved to: {out_file.resolve()}")

    if args.play or sys.platform == "darwin":
        if args.play:
            print(f"🔊 Playing audio via afplay...")
            subprocess.run(["afplay", str(out_file)])
        else:
            print(f"Tip: Play the generated audio anytime by running:\n  afplay {out_file.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
