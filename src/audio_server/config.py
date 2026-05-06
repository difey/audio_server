"""Configuration management for audio-server."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    vad_frame_ms: int = field(
        default_factory=lambda: int(os.getenv("VAD_FRAME_MS", "30"))
    )
    # webrtcvad mode override (0-3, higher = more aggressive filtering).
    # Uses 0 by default (least aggressive, best for low-volume environments).
    # If set via env, it takes precedence over vad_threshold.
    vad_mode_override: int = field(
        default_factory=lambda: int(os.getenv("VAD_MODE", "0"))
    )
    vad_threshold: float = field(
        default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.0"))
    )
    silence_duration_ms: int = field(
        default_factory=lambda: int(os.getenv("SILENCE_DURATION_MS", "600"))
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.getenv("SAMPLE_RATE", "16000"))
    )
    host: str = field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", "8000"))
    )
    max_connections: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONNECTIONS", "4"))
    )

    # ── ASR (Automatic Speech Recognition) ──
    asr_enabled: bool = field(
        default_factory=lambda: os.getenv("ASR_ENABLED", "true").lower() == "true"
    )

    # ── sherpa-onnx backend ──
    # Model types: sense_voice | funasr_nano | funasr_mlt_nano | qwen3_asr | moonshine_v2
    # Auto-detected from model name, or override via SHERPA_ONNX_MODEL_TYPE.
    sherpa_onnx_model: str = field(
        default_factory=lambda: os.getenv(
            "SHERPA_ONNX_MODEL",
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        )
    )
    sherpa_onnx_model_dir: str = field(
        default_factory=lambda: os.getenv("SHERPA_ONNX_MODEL_DIR", "")
    )
    sherpa_onnx_num_threads: int = field(
        default_factory=lambda: int(os.getenv("SHERPA_ONNX_NUM_THREADS", "2"))
    )
    sherpa_onnx_language: str = field(
        default_factory=lambda: os.getenv("SHERPA_ONNX_LANGUAGE", "")
    )
    sherpa_onnx_itn: bool = field(
        default_factory=lambda: os.getenv("SHERPA_ONNX_ITN", "true").lower() == "true"
    )

    @property
    def vad_mode(self) -> int:
        """WebRTC VAD mode (0-3). 0=least aggressive, 3=most aggressive."""
        if self.vad_mode_override != 0:
            return self.vad_mode_override
        # Map threshold 0.0-1.0 to mode 0-3
        return min(3, int(self.vad_threshold * 4))

    @property
    def vad_frame_samples(self) -> int:
        """Number of samples per VAD frame."""
        return int(self.sample_rate * self.vad_frame_ms / 1000)

    @property
    def silence_frame_count(self) -> int:
        """Number of consecutive silence frames to end a speech segment."""
        return max(1, self.silence_duration_ms // self.vad_frame_ms)

    # ── TTS (Text-to-Speech) ──
    tts_enabled: bool = field(
        default_factory=lambda: os.getenv("TTS_ENABLED", "false").lower() == "true"
    )
    tts_model: str = field(
        default_factory=lambda: os.getenv("TTS_MODEL", "matcha-icefall-zh-en")
    )
    tts_model_dir: str = field(
        default_factory=lambda: os.getenv("TTS_MODEL_DIR", "")
    )
    tts_vocoder: str = field(
        default_factory=lambda: os.getenv("TTS_VOCODER", "vocos-16khz-univ")
    )
    tts_num_threads: int = field(
        default_factory=lambda: int(os.getenv("TTS_NUM_THREADS", "4"))
    )
    tts_provider: str = field(
        default_factory=lambda: os.getenv("TTS_PROVIDER", "cpu")
    )
    tts_sid: int = field(
        default_factory=lambda: int(os.getenv("TTS_SID", "0"))
    )
    tts_speed: float = field(
        default_factory=lambda: float(os.getenv("TTS_SPEED", "1.0"))
    )
    tts_max_num_sentences: int = field(
        default_factory=lambda: int(os.getenv("TTS_MAX_NUM_SENTENCES", "1"))
    )

    @property
    def model_cache_dir(self) -> Path:
        """Directory for caching downloaded models."""
        raw = os.getenv("MODEL_CACHE_DIR", "")
        return Path(raw) if raw else Path.home() / ".cache" / "audio-server"


settings = Settings()
