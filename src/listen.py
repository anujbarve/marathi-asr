#!/usr/bin/env python3
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_utils import resample
from src.models import get_model
from src.ptt_buffer import ClipAssembler

SAMPLE_RATE = 16000
MIN_CLIP_S = 0.3


class PushToTalkRecorder:
    """
    Keeps a mic stream open and buffers audio only while the hotkey is held,
    with a short pre-roll before the press and post-roll after the release.
    Finished clips are pushed onto a queue for the main thread to transcribe.
    """
    def __init__(self, hotkey: keyboard.Key):
        self.hotkey = hotkey
        self.clips: "queue.Queue[np.ndarray]" = queue.Queue()
        self.device_sr = SAMPLE_RATE
        self._assembler = ClipAssembler(sample_rate=SAMPLE_RATE)
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        with self._lock:
            clip = self._assembler.feed(indata[:, 0].copy())
        if clip is not None:
            self.clips.put(clip)

    def _on_press(self, key):
        # Held keys auto-repeat, so only the first press starts a clip
        if key != self.hotkey:
            return
        with self._lock:
            if self._assembler.is_recording:
                return
            flushed = self._assembler.start()
        if flushed is not None:
            self.clips.put(flushed)
        print("● recording...", flush=True)

    def _on_release(self, key):
        if key == self.hotkey:
            with self._lock:
                self._assembler.stop()

    def _make_stream(self, sample_rate):
        return sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback
        )

    def start(self):
        try:
            self._stream = self._make_stream(SAMPLE_RATE)
            self.device_sr = SAMPLE_RATE
        except sd.PortAudioError:
            # Some devices (e.g. Windows WASAPI) refuse 16 kHz; use their default rate
            self.device_sr = int(sd.query_devices(kind="input")["default_samplerate"])
            self._stream = self._make_stream(self.device_sr)
        self._assembler = ClipAssembler(sample_rate=self.device_sr)
        self._stream.start()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self):
        self._listener.stop()
        self._stream.stop()
        self._stream.close()


def main():
    parser = argparse.ArgumentParser(
        description="Push-to-talk Marathi transcription: hold a key, speak, release"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="indicconformer-mr-int8",
        choices=["indicconformer-mr-int8", "whisper-tiny", "whisper-base", "whisper-small"],
        help="ASR model engine to use (default: indicconformer-mr-int8)"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="alt_r",
        help="pynput Key name to hold while speaking, e.g. alt_r, cmd_r, ctrl_r, f13 (default: alt_r = Right Option)"
    )
    args = parser.parse_args()

    hotkey = getattr(keyboard.Key, args.key, None)
    if hotkey is None:
        print(f"Error: Unknown key '{args.key}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model '{args.model}'...", flush=True)
    try:
        asr_engine = get_model(args.model)
        load_t0 = time.perf_counter()
        asr_engine.load()
        load_time_s = time.perf_counter() - load_t0
    except Exception as e:
        print(f"Error initializing model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    recorder = PushToTalkRecorder(hotkey)
    try:
        recorder.start()
    except Exception as e:
        print(f"Error opening microphone: {e}", file=sys.stderr)
        sys.exit(1)

    if recorder.device_sr != SAMPLE_RATE:
        print(f"Microphone does not support {SAMPLE_RATE} Hz; capturing at {recorder.device_sr} Hz and resampling.", flush=True)
    print(f"Model loaded in {load_time_s:.2f}s. Ready, hold '{args.key}' to talk (Ctrl+C to quit).", flush=True)

    try:
        while True:
            try:
                audio_data = recorder.clips.get(timeout=0.5)
            except queue.Empty:
                continue

            audio_data = resample(audio_data, recorder.device_sr, SAMPLE_RATE)
            duration_s = len(audio_data) / SAMPLE_RATE
            if duration_s < MIN_CLIP_S:
                print(f"(ignored {duration_s:.2f}s clip)", flush=True)
                continue

            infer_t0 = time.perf_counter()
            transcript = asr_engine.transcribe(audio_data, sample_rate=SAMPLE_RATE)
            infer_time_s = time.perf_counter() - infer_t0

            print(f"→ {transcript or '(no speech detected)'}")
            print(f"  [{duration_s:.2f}s audio, {infer_time_s:.3f}s inference]", flush=True)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
