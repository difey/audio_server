"""TTS engine — sherpa-onnx OfflineTts with Matcha-TTS (zh-en)."""

import io
import logging
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from .config import settings

logger = logging.getLogger(__name__)

_TTS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
)
_VOCODER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models"
)

# Known TTS models
_TTS_MODELS = {
    "matcha-icefall-zh-en": {
        "archive": "matcha-icefall-zh-en.tar.bz2",
        "acoustic_model": "model-steps-3.onnx",
        "vocoder": "vocos-16khz-univ.onnx",
        "lexicon": "lexicon.txt",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "rule_fsts": ["phone-zh.fst", "date-zh.fst", "number-zh.fst"],
    },
}


class _SherpaOnnxTtsBackend:
    """Wrapper around sherpa-onnx OfflineTts (Matcha-TTS)."""

    def __init__(self):
        import sherpa_onnx

        model_name = settings.tts_model
        if model_name not in _TTS_MODELS:
            raise ValueError(
                f"Unknown TTS model '{model_name}'. "
                f"Supported: {list(_TTS_MODELS.keys())}"
            )

        model_info = _TTS_MODELS[model_name]
        model_dir = self._ensure_model(model_name, model_info)
        vocoder_path = self._ensure_vocoder(model_info["vocoder"])

        acoustic_model = str(model_dir / model_info["acoustic_model"])
        lexicon = str(model_dir / model_info["lexicon"])
        tokens = str(model_dir / model_info["tokens"])
        data_dir = str(model_dir / model_info["data_dir"])

        # Build rule_fsts string
        rule_fst_paths = [
            str(model_dir / fst) for fst in model_info["rule_fsts"]
        ]
        rule_fsts = ",".join(rule_fst_paths)

        provider = settings.tts_provider
        num_threads = settings.tts_num_threads
        max_num_sentences = settings.tts_max_num_sentences

        logger.info(
            "Loading TTS model '%s' (provider=%s, threads=%d)...",
            model_name,
            provider,
            num_threads,
        )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=acoustic_model,
                    vocoder=str(vocoder_path),
                    lexicon=lexicon,
                    tokens=tokens,
                    data_dir=data_dir,
                ),
                provider=provider,
                num_threads=num_threads,
                debug=False,
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=max_num_sentences,
        )

        if not tts_config.validate():
            raise ValueError("TTS config validation failed — check model files")

        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sample_rate = self._tts.sample_rate
        logger.info(
            "TTS model loaded (sample_rate=%d)", self._sample_rate
        )

    def synthesize(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Run TTS synthesis.

        Args:
            text: Input text to synthesize.
            sid: Speaker ID (default from config).
            speed: Speed factor (default from config).

        Returns:
            Tuple of (audio_samples_float32, sample_rate).
        """
        if not text.strip():
            raise ValueError("Input text is empty")

        sid = sid if sid is not None else settings.tts_sid
        speed = speed if speed is not None else settings.tts_speed

        t0 = time.monotonic()
        audio = self._tts.generate(text, sid=sid, speed=speed)
        elapsed = time.monotonic() - t0

        if len(audio.samples) == 0:
            raise RuntimeError("TTS generated empty audio")

        audio_duration = len(audio.samples) / audio.sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        logger.info(
            "TTS: %.1fs audio generated in %.2fs (RTF=%.2f, text='%s')",
            audio_duration, elapsed, rtf, text[:50],
        )

        return audio.samples, audio.sample_rate

    def synthesize_to_wav_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Synthesize text and return WAV bytes."""
        samples, sample_rate = self.synthesize(text, sid=sid, speed=speed)
        buf = io.BytesIO()
        sf.write(buf, samples, samplerate=sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()

    def synthesize_to_pcm_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, int]:
        """Synthesize text and return raw Int16 PCM bytes + sample rate.

        Returns:
            Tuple of (raw_pcm_bytes, sample_rate).
        """
        samples, sample_rate = self.synthesize(text, sid=sid, speed=speed)
        int16_data = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        return int16_data.tobytes(), sample_rate

    def synthesize_to_mp3_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Synthesize text and return MP3 bytes via ffmpeg."""
        samples, sample_rate = self.synthesize(text, sid=sid, speed=speed)
        # Write WAV to temp buffer
        wav_buf = io.BytesIO()
        sf.write(wav_buf, samples, samplerate=sample_rate, format="WAV")
        wav_buf.seek(0)

        # Pipe through ffmpeg to encode MP3
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f", "wav",
                "-i", "pipe:0",
                "-f", "mp3",
                "-b:a", "128k",
                "-bitexact",
                "pipe:1",
            ],
            input=wav_buf.read(),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg MP3 encoding failed: {proc.stderr.decode(errors='replace')}"
            )
        return proc.stdout

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # ── Model download ──────────────────────────────────────────────

    def _ensure_model(self, model_name: str, model_info: dict) -> Path:
        """Resolve model directory, downloading if needed."""
        # 1. User-specified model dir
        model_dir = settings.tts_model_dir
        if model_dir:
            p = Path(model_dir)
            if p.is_dir():
                logger.info("Using TTS model from TTS_MODEL_DIR=%s", p)
                return p
            logger.warning("TTS_MODEL_DIR=%s not found, falling back", model_dir)

        # 2. Cached model directory
        cache = settings.model_cache_dir / "sherpa-onnx"
        cached_dir = cache / model_name
        if cached_dir.is_dir():
            return cached_dir

        # 3. Auto-download
        archive_name = model_info["archive"]
        url = f"{_TTS_MODEL_URL}/{archive_name}"
        archive = cache / archive_name
        cache.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading TTS model from %s ...", url)
        t0 = time.time()
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        logger.info(
                            "  Download: %.0f%% (%d MB)",
                            pct, downloaded // 1024 // 1024,
                        )
        logger.info("Downloaded in %.1fs, extracting...", time.time() - t0)

        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=cache)
        archive.unlink()

        logger.info("TTS model extracted to %s", cached_dir)
        return cached_dir

    def _ensure_vocoder(self, vocoder_name: str) -> Path:
        """Download vocoder model if not already cached."""
        cache = settings.model_cache_dir / "sherpa-onnx" / "vocoders"
        vocoder_path = cache / vocoder_name
        if vocoder_path.is_file():
            return vocoder_path

        url = f"{_VOCODER_URL}/{vocoder_name}"
        cache.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading vocoder from %s ...", url)
        t0 = time.time()
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(vocoder_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        logger.info("Vocoder downloaded in %.1fs", time.time() - t0)

        return vocoder_path


class TTSEngine:
    """Singleton wrapper around the sherpa-onnx TTS backend."""

    _instance: Optional["TTSEngine"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._backend: Optional[_SherpaOnnxTtsBackend] = None
        self._initialized = False

    def load(self) -> None:
        """Load the TTS model. Called once at startup if TTS is enabled."""
        if self._backend is not None:
            return
        self._backend = _SherpaOnnxTtsBackend()
        self._initialized = True

    def synthesize(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Input text.
            sid: Speaker ID (default: from config).
            speed: Speed factor (default: from config).

        Returns:
            Tuple of (float32 samples, sample_rate).
        """
        if self._backend is None:
            raise RuntimeError("TTS engine not loaded. Call load() first.")
        return self._backend.synthesize(text, sid=sid, speed=speed)

    def synthesize_to_wav_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Synthesize text and return WAV bytes."""
        if self._backend is None:
            raise RuntimeError("TTS engine not loaded. Call load() first.")
        return self._backend.synthesize_to_wav_bytes(text, sid=sid, speed=speed)

    def synthesize_to_pcm_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, int]:
        """Synthesize text and return raw Int16 PCM bytes + sample rate."""
        if self._backend is None:
            raise RuntimeError("TTS engine not loaded. Call load() first.")
        return self._backend.synthesize_to_pcm_bytes(text, sid=sid, speed=speed)

    def synthesize_to_mp3_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Synthesize text and return MP3 bytes."""
        if self._backend is None:
            raise RuntimeError("TTS engine not loaded. Call load() first.")
        return self._backend.synthesize_to_mp3_bytes(text, sid=sid, speed=speed)

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def sample_rate(self) -> int:
        if self._backend is None:
            return 0
        return self._backend.sample_rate
