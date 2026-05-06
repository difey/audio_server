#!/usr/bin/env python3
"""Python test client for Bondi-Whisper WebSocket ASR."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:
    print("Install websockets: uv add websockets")
    sys.exit(1)


async def send_file(ws_url: str, wav_path: str, chunk_ms: int = 30) -> None:
    """Read a WAV file and stream it as PCM chunks via WebSocket.

    Args:
        ws_url: WebSocket URL (e.g. ws://localhost:8000/ws/transcribe)
        wav_path: Path to WAV file (will be resampled to 16kHz mono if needed)
        chunk_ms: Chunk duration in milliseconds (default: 30ms = 480 samples)
    """
    print(f"Reading {wav_path}...")
    data, sr = sf.read(wav_path)

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16kHz
    if sr != 16000:
        from scipy import signal
        target_len = int(len(data) * 16000 / sr)
        data = signal.resample(data, target_len)
        print(f"Resampled {sr}Hz -> 16000Hz")

    # Convert to Int16
    data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
    pcm_bytes = data_int16.tobytes()

    chunk_size = int(16000 * chunk_ms / 1000) * 2  # bytes for chunk_ms
    total_chunks = (len(pcm_bytes) + chunk_size - 1) // chunk_size

    print(f"Connecting to {ws_url}...")
    print(f"Audio: {len(data_int16)} samples ({len(data_int16)/16000:.2f}s)")
    print(f"Chunk size: {chunk_ms}ms ({chunk_size} bytes), {total_chunks} chunks")
    print("-" * 50)

    async with websockets.connect(ws_url) as ws:
        # Start streaming
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            await ws.send(chunk)

            # Simulate real-time by sleeping for chunk duration
            await asyncio.sleep(chunk_ms / 1000)

            # Check for responses (non-blocking)
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                    if isinstance(msg, bytes):
                        continue
                    data = json.loads(msg)
                    _print_result(data)
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

        # Wait for final transcription
        print("-" * 50)
        print("Waiting for remaining results...")
        await asyncio.sleep(2.0)

        # Collect any final messages
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                if isinstance(msg, bytes):
                    continue
                data = json.loads(msg)
                _print_result(data)
        except asyncio.TimeoutError:
            pass

    print("-" * 50)
    print("Done.")


async def send_mic(ws_url: str, chunk_ms: int = 30) -> None:
    """Record from microphone and stream in real-time."""
    try:
        import pyaudio
    except ImportError:
        print("Install pyaudio for mic input: uv add pyaudio")
        print("On macOS: brew install portaudio && uv add pyaudio")
        sys.exit(1)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = int(RATE * chunk_ms / 1000)

    p = pyaudio.PyAudio()

    # Find input device
    info = p.get_host_api_info_by_index(0)
    device_count = info.get("deviceCount", 0)
    input_device = None
    for i in range(device_count):
        dev_info = p.get_device_info_by_host_api_device_index(0, i)
        if dev_info.get("maxInputChannels", 0) > 0:
            input_device = i
            break

    if input_device is None:
        print("No input device found")
        sys.exit(1)

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CHUNK,
    )

    print(f"Connecting to {ws_url}...")
    print("Recording... Press Ctrl+C to stop.")
    print("-" * 50)

    async with websockets.connect(ws_url) as ws:
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                await ws.send(data)

                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                        if isinstance(msg, bytes):
                            continue
                        result = json.loads(msg)
                        _print_result(result)
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    pass

                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def _print_result(data: dict) -> None:
    """Print a transcription result."""
    t = data.get("type", "")
    text = data.get("text", "")
    if t == "speech_start":
        print(f"[{t}] 🎤 Speech detected")
    elif t == "interim":
        print(f"[{t}] ✍️ {text}", end="\r")
    elif t == "final":
        print(f"\n[{t}] ✅ {text}")
    elif t == "error":
        print(f"\n[{t}] ❌ {data.get('message', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="Test client for Bondi-Whisper streaming ASR"
    )
    parser.add_argument(
        "-u", "--url",
        default="ws://localhost:8000/ws/transcribe",
        help="WebSocket URL (default: ws://localhost:8000/ws/transcribe)",
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to WAV file to transcribe",
    )
    parser.add_argument(
        "-m", "--mic",
        action="store_true",
        help="Use microphone input",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=30,
        help="Audio chunk size in ms (default: 30)",
    )

    args = parser.parse_args()

    if args.file:
        asyncio.run(send_file(args.url, args.file, args.chunk_ms))
    elif args.mic:
        try:
            asyncio.run(send_mic(args.url, args.chunk_ms))
        except KeyboardInterrupt:
            pass
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
