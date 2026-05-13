#!/usr/bin/env python3
"""Benchmark RTF (Real-Time Factor) for ASR models.

Measures:
  - Model loading time
  - Transcription time for audio files
  - Real-Time Factor (RTF) = processing_time / audio_duration

Usage:
    # Default: Use settings from .env
    uv run python scripts/benchmark_asr_rtf.py

    # Specific model and files
    uv run python scripts/benchmark_asr_rtf.py --model sense_voice --files scripts/audios/*.wav

    # Test all files in a directory
    uv run python scripts/benchmark_asr_rtf.py --dir scripts/audios/
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src to path so we can import audio_server
sys.path.append(str(Path(__file__).parent.parent / "src"))

from audio_server.asr_engine import ASREngine
from audio_server.config import settings

# Disable verbose logging from the engine during benchmark
logging.getLogger("audio_server").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RTF for ASR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model type to test (e.g., sense_voice, funasr_nano, qwen3_asr). "
             "If not set, uses SHERPA_ONNX_MODEL_TYPE from .env",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific WAV files to benchmark",
    )
    parser.add_argument(
        "--dir",
        default="scripts/audios",
        help="Directory containing WAV files (default: %(default)s)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs per file (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before measurement (default: %(default)s)",
    )

    return parser.parse_args()


def load_audio(path: str) -> np.ndarray:
    """Load audio and resample to 16kHz mono float32."""
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        # Simple decimation/interpolation for benchmark purposes
        # (Real app uses a better resampler, but this is fast for tests)
        indices = np.round(np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))).astype(int)
        data = data[indices]
    return data.astype(np.float32)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


async def run_benchmark():
    args = parse_args()

    # Override model type if provided
    if args.model:
        os.environ["SHERPA_ONNX_MODEL_TYPE"] = args.model
        # We need to reload settings if we want it to take effect globally,
        # but ASREngine will read it when it initializes the backend.

    # ── Resolve files ──────────────────────────────────────────────
    files = []
    if args.files:
        for f in args.files:
            p = Path(f)
            if p.is_file():
                files.append(p)
    else:
        d = Path(args.dir)
        if d.is_dir():
            files = sorted(list(d.glob("*.wav")))

    if not files:
        print(f"Error: No WAV files found in {args.dir}")
        sys.exit(1)

    # ── Load model ─────────────────────────────────────────────────
    print("=" * 80)
    print("ASR RTF Benchmark")
    print("=" * 80)
    print(f"  Model Type: {args.model or settings.sherpa_onnx_model_type}")
    print(f"  Provider:   {settings.sherpa_onnx_provider}")
    print(f"  Threads:    {settings.sherpa_onnx_num_threads}")
    print(f"  Files:      {len(files)}")
    print(f"  Runs:       {args.runs} (warmup={args.warmup})")
    print()

    engine = ASREngine()
    print("Loading model...", end=" ", flush=True)
    t0 = time.monotonic()
    engine.load()
    load_time = time.monotonic() - t0
    print(f"done ({format_duration(load_time)})")
    print()

    # ── Benchmark ──────────────────────────────────────────────────
    results = [] # (name, duration, inference_ms, rtf, text)

    for wav_path in files:
        name = wav_path.name
        audio = load_audio(str(wav_path))
        audio_dur = len(audio) / 16000

        print(f"  [{name}] ({audio_dur:.2f}s)")

        # Warmup
        for _ in range(args.warmup):
            await engine.transcribe(audio)

        # Measurement
        latencies = []
        text = ""
        for i in range(args.runs):
            t0 = time.monotonic()
            text, _ = await engine.transcribe(audio)
            elapsed = (time.monotonic() - t0) * 1000
            latencies.append(elapsed)
            print(f"    Run {i+1}/{args.runs}: {elapsed:.1f}ms (RTF: {elapsed/(audio_dur*1000):.3f})")

        avg_latency = float(np.mean(latencies))
        rtf = avg_latency / (audio_dur * 1000)
        results.append((name, audio_dur, avg_latency, rtf, text))

        print(f"    ──> Avg: {avg_latency:.1f}ms | RTF: {rtf:.3f} | Text: {text[:50]}...")
        print()

    # ── Summary ────────────────────────────────────────────────────
    print("-" * 80)
    print(f"{'File':<25} {'Audio':<10} {'Inference':<12} {'RTF':<10}")
    print("-" * 80)
    for name, dur, inf, rtf, _ in results:
        print(f"{name[:24]:<25} {dur:<10.2f}s {inf:<10.1f}ms {rtf:<10.3f}")

    avg_rtf = float(np.mean([r[3] for r in results]))
    print("-" * 80)
    print(f"{'AVERAGE':<36} {avg_rtf:<10.3f}")
    print("-" * 80)

    # ── Rating ─────────────────────────────────────────────────────
    if avg_rtf < 0.1:
        grade = "🟢 Excellent (RTF < 0.1)"
    elif avg_rtf < 0.3:
        grade = "🟢 Good (RTF < 0.3)"
    elif avg_rtf < 0.5:
        grade = "🟡 Acceptable (RTF < 0.5)"
    else:
        grade = "🔴 Slow (RTF >= 0.5)"

    print(f"  ASR RTF Rating: {grade}")
    print(f"  Processing Speed: {1/avg_rtf:.1f}x real-time")
    print()

    await engine.unload()


if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        pass
