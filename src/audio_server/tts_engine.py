"""TTS engine — sherpa-onnx OfflineTts (Matcha-TTS, VITS/Piper) / Qwen3-TTS."""

import asyncio
import io
import logging
import os
import subprocess
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from .config import settings

logger = logging.getLogger(__name__)


def _is_qwen3_tts_model(model_name: str) -> bool:
    """Check if model name refers to a Qwen3-TTS HuggingFace model."""
    return model_name.startswith("Qwen/Qwen3-TTS")

_TTS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
)
_VOCODER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models"
)

# Known TTS models
_TTS_MODELS = {
    "matcha-icefall-zh-en": {
        "type": "matcha",
        "archive": "matcha-icefall-zh-en.tar.bz2",
        "acoustic_model": "model-steps-3.onnx",
        "vocoder": "vocos-16khz-univ.onnx",
        "lexicon": "lexicon.txt",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "rule_fsts": ["phone-zh.fst", "date-zh.fst", "number-zh.fst"],
    },
    "vits-piper-zh_CN-chaowen-medium": {
        "type": "vits",
        "archive": "vits-piper-zh_CN-chaowen-medium.tar.bz2",
        "model_file": "zh_CN-chaowen-medium.onnx",
        "tokens": "tokens.txt",
        "lexicon": "lexicon.txt",
        "rule_fsts": ["phone.fst", "date.fst", "number.fst"],
    },
    "vits-piper-en_GB-jenny_dioco-medium": {
        "type": "vits",
        "archive": "vits-piper-en_GB-jenny_dioco-medium.tar.bz2",
        "model_file": "en_GB-jenny_dioco-medium.onnx",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
    },
}


class _SherpaOnnxTtsBackend:
    """Wrapper around sherpa-onnx OfflineTts (Matcha-TTS / VITS-Piper)."""

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

        provider = settings.tts_provider
        num_threads = settings.tts_num_threads
        max_num_sentences = settings.tts_max_num_sentences

        model_type = model_info.get("type", "matcha")

        if model_type == "vits":
            # VITS / Piper models: single ONNX with built-in vocoder
            model_path = str(model_dir / model_info["model_file"])
            tokens = str(model_dir / model_info["tokens"])

            # Some VITS models use espeak-ng-data, others use lexicon + rule_fsts
            vits_config_kwargs = {
                "model": model_path,
                "tokens": tokens,
            }
            if "data_dir" in model_info:
                vits_config_kwargs["data_dir"] = str(
                    model_dir / model_info["data_dir"]
                )
            if "lexicon" in model_info:
                vits_config_kwargs["lexicon"] = str(
                    model_dir / model_info["lexicon"]
                )

            logger.info(
                "Loading VITS TTS model '%s' (provider=%s, threads=%d)...",
                model_name, provider, num_threads,
            )

            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    **vits_config_kwargs,
                ),
                provider=provider,
                num_threads=num_threads,
                debug=False,
            )

        else:
            # Matcha-TTS models: acoustic model + separate vocoder
            vocoder_path = self._ensure_vocoder(model_info["vocoder"])

            acoustic_model = str(model_dir / model_info["acoustic_model"])
            lexicon = str(model_dir / model_info["lexicon"])
            tokens = str(model_dir / model_info["tokens"])
            data_dir = str(model_dir / model_info["data_dir"])

            rule_fst_paths = [
                str(model_dir / fst) for fst in model_info["rule_fsts"]
            ]
            rule_fsts = ",".join(rule_fst_paths)

            logger.info(
                "Loading Matcha TTS model '%s' (provider=%s, threads=%d)...",
                model_name, provider, num_threads,
            )

            model_config = sherpa_onnx.OfflineTtsModelConfig(
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
            )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
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

        # Convert to float32 numpy array (sherpa-onnx may return a list)
        samples = np.array(audio.samples, dtype=np.float32)

        audio_duration = len(samples) / audio.sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        logger.info(
            "TTS: %.1fs audio generated in %.2fs (RTF=%.2f, text='%s')",
            audio_duration, elapsed, rtf, text[:50],
        )

        return samples, audio.sample_rate

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


# ── Qwen3-TTS Backend (qwen-tts package) ──────────────────────────────


class _Qwen3TtsBackend:
    """Wrapper around Qwen3-TTS from the ``qwen-tts`` package.

    Supports both 0.6B and 1.7B CustomVoice models:
      - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
      - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    """

    def __init__(self):
        model_id = settings.tts_qwen3_model
        device = settings.tts_qwen3_device
        dtype_str = settings.tts_qwen3_dtype

        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        logger.info(
            "Loading Qwen3-TTS model '%s' (device=%s, dtype=%s)...",
            model_id, device, dtype_str,
        )
        t0 = time.monotonic()

        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
        )

        # Detect whether this model supports ``instruct``
        # (1.7B CustomVoice supports it, 0.6B does not).
        self._supports_instruct = "1.7B" in model_id and "CustomVoice" in model_id

        self._sample_rate = getattr(self._model, "sample_rate", 24000)
        elapsed = time.monotonic() - t0
        logger.info(
            "Qwen3-TTS model loaded in %.1fs (sample_rate=%d, instruct=%s)",
            elapsed, self._sample_rate, self._supports_instruct,
        )

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text using Qwen3-TTS CustomVoice.

        Args:
            text: Input text to synthesize.
            language: Language (e.g. ``"Chinese"``, ``"English"``, ``"Auto"``).
                      Defaults to ``TTS_QWEN3_LANGUAGE``.
            speaker: Speaker name (e.g. ``"Vivian"``, ``"Ryan"``).
                     Defaults to ``TTS_QWEN3_SPEAKER``.
            instruct: Natural-language instruction (1.7B only; ignored for 0.6B).

        Returns:
            Tuple of (float32 samples, sample_rate).
        """
        if not text.strip():
            raise ValueError("Input text is empty")

        language = language or settings.tts_qwen3_language
        speaker = speaker or settings.tts_qwen3_speaker

        kwargs: dict = {
            "text": text,
            "language": language,
            "speaker": speaker,
        }
        # Only pass ``instruct`` if the model supports it
        if self._supports_instruct and instruct:
            kwargs["instruct"] = instruct

        t0 = time.monotonic()
        wavs, sr = self._model.generate_custom_voice(**kwargs)
        elapsed = time.monotonic() - t0

        if not wavs or len(wavs[0]) == 0:
            raise RuntimeError("Qwen3-TTS generated empty audio")

        samples = np.asarray(wavs[0], dtype=np.float32)
        audio_duration = len(samples) / sr if sr > 0 else 0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        logger.info(
            "Qwen3-TTS: %.1fs audio generated in %.2fs (RTF=%.2f, text='%s')",
            audio_duration, elapsed, rtf, text[:50],
        )

        return samples, sr

    # ── Output format helpers (reuse generic logic) ────────────────────

    def _to_wav_bytes(self, samples: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, samples, samplerate=sr, format="WAV")
        buf.seek(0)
        return buf.read()

    def _to_pcm_bytes(self, samples: np.ndarray, sr: int) -> tuple[bytes, int]:
        int16_data = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        return int16_data.tobytes(), sr

    def _to_mp3_bytes(self, samples: np.ndarray, sr: int) -> bytes:
        wav_buf = io.BytesIO()
        sf.write(wav_buf, samples, samplerate=sr, format="WAV")
        wav_buf.seek(0)
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


# ── Format helpers (sync, for use in thread pool) ───────────────────


def _encode_mp3_sync(samples: np.ndarray, sr: int) -> bytes:
    """Run ffmpeg MP3 encoding synchronously.

    This is intended to be called via ``run_in_executor`` so the
    subprocess does not block the async event loop.
    """
    wav_buf = io.BytesIO()
    sf.write(wav_buf, samples, samplerate=sr, format="WAV")
    wav_buf.seek(0)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "wav", "-i", "pipe:0",
         "-f", "mp3", "-b:a", "128k", "-bitexact", "pipe:1"],
        input=wav_buf.read(),
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg MP3 encoding failed: {proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


# ── Dynamic Batch Scheduler (TTS) ────────────────────────────────────


@dataclass
class _TTSRequest:
    """A single TTS request waiting in the scheduler queue."""

    text: str
    sid: Optional[int]
    speed: Optional[float]
    future: "asyncio.Future[tuple[np.ndarray, int]]"


class TTSScheduler:
    """Queues TTS requests and processes them on a thread pool.

    Each :class:`_SherpaOnnxTtsBackend` instance runs in its own thread,
    allowing multiple GPU inferences to proceed in parallel without
    blocking the async event loop.
    """

    def __init__(
        self,
        backends: list[_SherpaOnnxTtsBackend],
        max_queue_size: int = 30,
    ):
        self._backends = backends
        self._queue: asyncio.Queue[_TTSRequest] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._pool = ThreadPoolExecutor(max_workers=len(backends))
        self._tasks: list[asyncio.Task] = []

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Launch one background worker per backend instance."""
        for i, backend in enumerate(self._backends):
            task = asyncio.create_task(self._worker(i, backend))
            self._tasks.append(task)
        logger.info(
            "TTSScheduler started (%d instances, queue=%d)",
            len(self._backends), self._queue.maxsize,
        )

    async def stop(self) -> None:
        """Cancel all workers and drain the queue."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._pool.shutdown(wait=False)

        # Fail any remaining requests so callers don't hang
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                if not req.future.done():
                    req.future.set_exception(
                        RuntimeError("TTS server shutting down")
                    )
            except asyncio.QueueEmpty:
                break
        logger.info("TTSScheduler stopped")

    # ── Public API ─────────────────────────────────────────────────

    async def submit(
        self, text: str, sid: Optional[int], speed: Optional[float],
    ) -> tuple[np.ndarray, int]:
        """Submit a TTS request.

        Returns a Future that resolves when a worker finishes processing.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        req = _TTSRequest(text=text, sid=sid, speed=speed, future=future)
        await self._queue.put(req)
        return await future

    # ── Background worker ──────────────────────────────────────────

    async def _worker(
        self, worker_id: int, backend: _SherpaOnnxTtsBackend,
    ) -> None:
        """Process one TTS request at a time in this worker's thread."""
        while True:
            try:
                req = await self._queue.get()
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._pool, backend.synthesize,
                        req.text, req.sid, req.speed,
                    )
                    if not req.future.done():
                        req.future.set_result(result)
                except Exception as e:
                    if not req.future.done():
                        req.future.set_exception(e)
            except asyncio.CancelledError:
                break


# ── Unified Singleton Wrapper ─────────────────────────────────────────


class TTSEngine:
    """Singleton wrapper with async interface backed by :class:`TTSScheduler`.

    For sherpa-onnx models (the primary path), multiple backend instances
    are created and load-balanced via ``TTSScheduler``.

    For Qwen3-TTS models, a single backend is used directly (synchronous).
    """

    _instance: Optional["TTSEngine"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._backend: Optional[_SherpaOnnxTtsBackend] = None
        self._qwen3_backend: Optional[_Qwen3TtsBackend] = None
        self._scheduler: Optional[TTSScheduler] = None

        # Qwen3-TTS uses an asyncio lock + single-thread executor
        # to serialise GPU access without blocking the event loop.
        self._qwen3_lock: Optional[asyncio.Lock] = None
        self._qwen3_executor: Optional[ThreadPoolExecutor] = None
        self._initialized = False

    def load(self) -> None:
        """Load TTS model(s) and start the scheduler / executor.

        Called once at application startup.
        - Sherpa-onnx: N backends via ``TTSScheduler``.
        - Qwen3-TTS: single backend + ``asyncio.Lock`` + 1-thread executor.
        """
        if self._backend is not None or self._qwen3_backend is not None:
            return

        if _is_qwen3_tts_model(settings.tts_model):
            self._qwen3_backend = _Qwen3TtsBackend()
            self._qwen3_lock = asyncio.Lock()
            self._qwen3_executor = ThreadPoolExecutor(max_workers=1)
            logger.info(
                "Qwen3-TTS ready (lock + 1-thread executor)"
            )
        else:
            num = settings.tts_num_instances
            backends = [_SherpaOnnxTtsBackend() for _ in range(num)]
            self._backend = backends[0]  # reference for properties
            self._scheduler = TTSScheduler(
                backends=backends,
                max_queue_size=settings.tts_max_queue_size,
            )
            self._scheduler.start()

        self._initialized = True

    async def unload(self) -> None:
        """Stop the scheduler and release resources."""
        if self._scheduler:
            await self._scheduler.stop()
            self._scheduler = None
        if self._qwen3_executor:
            self._qwen3_executor.shutdown(wait=False)
            self._qwen3_executor = None
        self._qwen3_lock = None
        self._backend = None
        self._qwen3_backend = None
        self._initialized = False

    # ── Sherpa-onnx interface (async via scheduler) ──────────────────

    async def synthesize(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text to audio (sherpa-onnx backend, async).

        Args:
            text: Input text.
            sid: Speaker ID (default: from config).
            speed: Speed factor (default: from config).

        Returns:
            Tuple of (float32 samples, sample_rate).
        """
        if self._scheduler is None:
            raise RuntimeError("Sherpa-onnx TTS not loaded.")
        return await self._scheduler.submit(text, sid, speed)

    async def synthesize_to_wav_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Synthesize text and return WAV bytes (async)."""
        samples, sr = await self.synthesize(text, sid=sid, speed=speed)
        buf = io.BytesIO()
        sf.write(buf, samples, samplerate=sr, format="WAV")
        buf.seek(0)
        return buf.read()

    async def synthesize_to_pcm_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, int]:
        """Synthesize text and return raw Int16 PCM bytes (async)."""
        samples, sr = await self.synthesize(text, sid=sid, speed=speed)
        int16_data = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        return int16_data.tobytes(), sr

    async def synthesize_to_mp3_bytes(
        self,
        text: str,
        sid: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Synthesize text and return MP3 bytes via ffmpeg (async).

        Both the GPU inference and the ffmpeg subprocess run in the
        thread pool so neither blocks the event loop.
        """
        samples, sr = await self.synthesize(text, sid=sid, speed=speed)
        # ffmpeg encoding is CPU-heavy, run in executor too
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # default thread pool
            _encode_mp3_sync, samples, sr,
        )

    # ── Qwen3-TTS interface (async via lock + executor) ──────────────

    async def synthesize_qwen3(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text using Qwen3-TTS (async).

        GPU access is serialised via ``asyncio.Lock`` and the blocking
        ``generate_custom_voice`` call runs in a dedicated thread so
        the event loop stays responsive.
        """
        if self._qwen3_backend is None or self._qwen3_lock is None:
            raise RuntimeError("Qwen3-TTS not loaded.")

        loop = asyncio.get_event_loop()
        async with self._qwen3_lock:
            # Run the blocking backend.synthesize() in the executor
            return await loop.run_in_executor(
                self._qwen3_executor,
                self._qwen3_backend.synthesize,
                text,
                language,
                speaker,
                instruct,
            )

    async def synthesize_qwen3_to_wav_bytes(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> bytes:
        samples, sr = await self.synthesize_qwen3(
            text, language=language, speaker=speaker, instruct=instruct,
        )
        return self._qwen3_backend._to_wav_bytes(samples, sr)

    async def synthesize_qwen3_to_pcm_bytes(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> tuple[bytes, int]:
        samples, sr = await self.synthesize_qwen3(
            text, language=language, speaker=speaker, instruct=instruct,
        )
        return self._qwen3_backend._to_pcm_bytes(samples, sr)

    async def synthesize_qwen3_to_mp3_bytes(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> bytes:
        samples, sr = await self.synthesize_qwen3(
            text, language=language, speaker=speaker, instruct=instruct,
        )
        # ffmpeg is CPU-heavy, run in default executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _encode_mp3_sync, samples, sr,
        )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None or self._qwen3_backend is not None

    @property
    def is_qwen3(self) -> bool:
        return self._qwen3_backend is not None

    @property
    def sample_rate(self) -> int:
        if self._qwen3_backend is not None:
            return self._qwen3_backend._sample_rate
        if self._backend is not None:
            return self._backend.sample_rate
        return 0
