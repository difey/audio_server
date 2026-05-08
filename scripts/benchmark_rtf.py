#!/usr/bin/env python3
"""Benchmark RTF (Real-Time Factor) for Qwen3-TTS models.

Measures:
  - Model loading time
  - Synthesis time for sentences of varying lengths
  - Real-Time Factor (RTF) = processing_time / audio_duration

Usage:
    # Default: 0.6B model on CUDA
    uv run python scripts/benchmark_rtf.py

    # 1.7B model, CPU, float32
    uv run python scripts/benchmark_rtf.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --device cpu --dtype float32

    # Local model directory
    uv run python scripts/benchmark_rtf.py \
        --model /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
        --device cuda:0 --dtype bfloat16

    # Custom text list
    uv run python scripts/benchmark_rtf.py \
        --sentences "你好。" "今天天气真不错。" "这是一段较长的中文测试文本。"

    # Custom language/speaker
    uv run python scripts/benchmark_rtf.py \
        --language English --speaker Ryan

Requirements:
    pip install qwen-tts torch numpy
"""

import argparse
import sys
import time

import numpy as np


# ── Default test sentences ──────────────────────────────────────────
DEFAULT_SENTENCES: list[tuple[str, str, str]] = [
    # (label, text, language)
    ("short_zh",   "你好，世界。",                                      "Chinese"),
    ("medium_zh",  "其实我真的有发现，我是一个特别善于观察别人情绪的人。",   "Chinese"),
    ("long_zh",    "人工智能是计算机科学的一个分支，它企图了解智能的实质，"
                   "并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
                   "该领域的研究包括机器人、语言识别、图像识别、"
                   "自然语言处理和专家系统等。",                         "Chinese"),
    ("short_en",   "Hello, world.",                                    "English"),
    ("medium_en",  "She said she would be here by noon, "
                   "but I haven't seen her yet.",                       "English"),
    ("long_en",    "Artificial intelligence is a branch of computer "
                   "science that aims to understand the essence of "
                   "intelligence and produce new intelligent machines "
                   "that respond in ways similar to human intelligence. "
                   "Research in this field includes robotics, language "
                   "recognition, image recognition, natural language "
                   "processing, and expert systems.",                    "English"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RTF for Qwen3-TTS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="HuggingFace model ID or local path  (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Device (e.g. "cuda:0", "cpu")  (default: %(default)s)',
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type  (default: %(default)s)",
    )

    # Benchmark config
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs per sentence  (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before measurement  (default: %(default)s)",
    )

    # Speaker / language override
    parser.add_argument(
        "--speaker",
        default=None,
        help='Default speaker name (e.g. "Vivian", "Ryan")',
    )
    parser.add_argument(
        "--language",
        default=None,
        help='Default language (e.g. "Chinese", "English")',
    )

    # Custom sentences (overrides defaults)
    parser.add_argument(
        "--sentences",
        nargs="+",
        default=None,
        help="Custom list of sentences to benchmark",
    )

    return parser.parse_args()


def resolve_dtype(dtype_str: str):
    import torch
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def resolve_speaker(language: str, fallback: str | None) -> str:
    """Pick a sensible speaker for the given language."""
    if fallback:
        return fallback
    lang_to_speaker = {
        "Chinese": "Vivian",
        "English": "Ryan",
        "Japanese": "Ono_Anna",
        "Korean": "Sohee",
    }
    return lang_to_speaker.get(language, "Vivian")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.1f}m"


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    print("=" * 80)
    print("Qwen3-TTS RTF Benchmark")
    print("=" * 80)
    print(f"  Model:   {args.model}")
    print(f"  Device:  {args.device}")
    print(f"  Dtype:   {args.dtype}")
    print(f"  Runs:    {args.runs}  (warmup={args.warmup})")
    print()

    # ── Build sentence list ────────────────────────────────────────
    if args.sentences:
        sentences: list[tuple[str, str, str]] = [
            (f"custom_{i}", s, args.language or "Chinese")
            for i, s in enumerate(args.sentences)
        ]
    else:
        sentences = DEFAULT_SENTENCES

    # ── Load model ─────────────────────────────────────────────────
    from qwen_tts import Qwen3TTSModel

    print("Loading model...", end=" ", flush=True)
    t0 = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
    )
    load_time = time.monotonic() - t0
    sr = getattr(model, "sample_rate", 24000)
    print(f"done  ({format_duration(load_time)})")
    print(f"  Sample rate: {sr} Hz")
    print()

    # ── Benchmark ──────────────────────────────────────────────────
    results: list[tuple[str, int, float, float, float]] = []  # label, chars, time, audio_dur, rtf

    supports_instruct = "1.7B" in args.model and "CustomVoice" in args.model
    if supports_instruct:
        print("  * Model supports `instruct` (1.7B) — not used in benchmark")

    for label, text, lang in sentences:
        speaker = resolve_speaker(lang, args.speaker)
        label_display = f"{label} ({lang}, {speaker})"

        print(f"  [{label_display}]  ({len(text)} chars, {len(text.split())} words)")
        print(f"    Text: {text[:80]}{'…' if len(text) > 80 else ''}")

        kwargs: dict = {
            "text": text,
            "language": lang,
            "speaker": speaker,
        }

        # Warmup
        for _ in range(args.warmup):
            model.generate_custom_voice(**kwargs)

        # Measurement
        times: list[float] = []
        audio_durs: list[float] = []
        for run in range(1, args.runs + 1):
            t0 = time.monotonic()
            wavs, sr_out = model.generate_custom_voice(**kwargs)
            elapsed = time.monotonic() - t0
            times.append(elapsed)

            audio_dur = len(wavs[0]) / sr_out if sr_out > 0 else 0
            audio_durs.append(audio_dur)
            rtf_run = elapsed / audio_dur if audio_dur > 0 else 0
            print(f"    Run {run}/{args.runs}:  synth={format_duration(elapsed)}, "
                  f"audio={audio_dur:.1f}s, RTF={rtf_run:.3f}")

        avg_time = float(np.mean(times))
        avg_audio_dur = float(np.mean(audio_durs))
        rtf = avg_time / avg_audio_dur if avg_audio_dur > 0 else 0
        results.append((label, len(text), avg_time, avg_audio_dur, rtf))

        print(f"    ──> Avg: synth={format_duration(avg_time)}, "
              f"audio={avg_audio_dur:.1f}s, RTF={rtf:.3f}")
        print()

    # ── Summary table ──────────────────────────────────────────────
    print()
    print("─" * 80)
    print("SUMMARY")
    print("─" * 80)
    print(f"  Model:     {args.model}")
    print(f"  Device:    {args.device}")
    print(f"  Dtype:     {args.dtype}")
    print(f"  Load time: {format_duration(load_time)}")
    print()
    print(f"  {'Sentence':<20} {'Chars':<8} {'Synth':<14} {'Audio':<14} {'RTF':<10}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 14} {'─' * 14} {'─' * 10}")
    for label, chars, t, dur, rtf in results:
        # Find the original sentence for language display
        orig_lang = ""
        for lbl, _, lang in sentences:
            if lbl == label:
                orig_lang = lang
                break
        print(f"  {label:<20} {chars:<8} {format_duration(t):<14} {dur:<10.1f}s  {rtf:<10.3f}")

    avg_rtf = float(np.mean([r for _, _, _, _, r in results]))
    max_rtf = float(max(r for _, _, _, _, r in results))
    min_rtf = float(min(r for _, _, _, _, r in results))
    total_chars = sum(c for _, c, _, _, _ in results)

    print(f"  {'─' * 20} {'─' * 8} {'─' * 14} {'─' * 14} {'─' * 10}")
    print(f"  {'Total chars':<20} {total_chars:<8}")
    print(f"  {'Min RTF':<20} {min_rtf:<.3f}")
    print(f"  {'Max RTF':<20} {max_rtf:<.3f}")
    print(f"  {'Average RTF':<20} {avg_rtf:<.3f}")
    print()

    # ── Rating ─────────────────────────────────────────────────────
    if avg_rtf < 0.3:
        grade = "🟢 Excellent (RTF < 0.3) — real-time feasible"
    elif avg_rtf < 0.5:
        grade = "🟢 Good (RTF < 0.5)"
    elif avg_rtf < 1.0:
        grade = "🟡 Acceptable (RTF < 1.0)"
    elif avg_rtf < 2.0:
        grade = "🟠 Slow (RTF < 2.0)"
    else:
        grade = "🔴 Very slow (RTF >= 2.0)"

    print(f"  RTF Rating: {grade}")
    print()

    # ── Rendering speed ────────────────────────────────────────────
    # Equivalent to how many seconds of audio can be generated per second
    print(f"  Rendering speed: {1/avg_rtf:.1f}x real-time  (1s audio in {avg_rtf*1000:.0f}ms)")
    print()
    print("─" * 80)


if __name__ == "__main__":
    main()
