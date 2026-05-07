"""ASR engine — sherpa-onnx backend for ONNX-based ASR models."""

import logging
import os
import time
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .config import settings

logger = logging.getLogger(__name__)


def _model_base_url() -> str:
    """Return the base URL for downloading sherpa-onnx models.

    Defaults to the official sherpa-onnx releases, but can be overridden
    via SHERPA_ONNX_MODEL_BASE_URL to point to a custom GitHub Release
    (e.g., your own repo's releases for custom models).
    """
    return settings.sherpa_onnx_model_base_url.rstrip("/")


# ── sherpa-onnx backend ────────────────────────────────────────────

class _SherpaOnnxBackend:
    """Wrapper around sherpa-onnx (SenseVoice / FunASR Nano / Qwen3-ASR / Moonshine V2).

    Auto-downloads the model from GitHub releases on first load.
    Model type is auto-detected from model name, or override via
    SHERPA_ONNX_MODEL_TYPE env var.

    SenseVoice:
        sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09
        → model.int8.onnx, tokens.txt

    FunASR Nano:
        sherpa-onnx-funasr-nano-int8-2025-12-30
        → encoder_adaptor.int8.onnx, llm_int8/llm.int8.onnx,
          embedding.int8.onnx, Qwen3-0.6B/ (tokenizer dir)

    Qwen3-ASR:
        sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
        → conv_frontend.onnx, encoder.int8.onnx, decoder.int8.onnx,
          tokenizer/ (tokenizer dir)

    Moonshine V2:
        sherpa-onnx-moonshine-base-{lang}-quantized-2026-02-27
        → encoder_model.ort, decoder_model_merged.ort, tokens.txt
    """

    _SENSE_VOICE = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
    _FUNASR_NANO = "sherpa-onnx-funasr-nano-int8-2025-12-30"
    _QWEN3_ASR = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    _MOONSHINE_V2_ZH = "sherpa-onnx-moonshine-base-zh-quantized-2026-02-27"
    _MOONSHINE_V2_EN = "sherpa-onnx-moonshine-base-en-quantized-2026-02-27"
    _MOONSHINE_V2_ES = "sherpa-onnx-moonshine-base-es-quantized-2026-02-27"

    # funasr_mlt_nano shares same file structure as funasr_nano.
    # Its model name is dynamic (date-stamped by CI), so we read it
    # from settings.sherpa_onnx_model rather than a hardcoded constant.
    _MODEL_NAMES = {
        "sense_voice": _SENSE_VOICE,
        "funasr_nano": _FUNASR_NANO,
        "funasr_mlt_nano": None,  # uses SHERPA_ONNX_MODEL
        "qwen3_asr": _QWEN3_ASR,
        "moonshine_v2": _MOONSHINE_V2_ZH,
    }

    # Known moonshine v2 variants for auto-detection
    _MOONSHINE_V2_MODELS = {
        _MOONSHINE_V2_ZH,
        _MOONSHINE_V2_EN,
        _MOONSHINE_V2_ES,
    }

    # Parts manifest for multi-file model downloads (to stay under GitHub's
    # 2GB per-file limit). Each entry:
    #   (part_filename_in_release, is_archive, extract_subdir)
    #   - part_filename: basename as uploaded to GitHub Release
    #   - is_archive: True = .tar.bz2 to extract after download
    #   - extract_subdir: subdirectory to extract into (None = flat)
    # The full download URL is: {base_url}/{model_name}_{part_filename}
    _PARTS_MANIFEST: dict[str, list[tuple[str, bool, str | None]]] = {
        "funasr_mlt_nano": [
            ("encoder_adaptor.int8.onnx", False, None),
            ("embedding.int8.onnx", False, None),
            ("llm_int8.tar.bz2", True, None),
            ("Qwen3-0.6B.tar.bz2", True, None),
        ],
    }

    def __init__(self):
        import sherpa_onnx

        self._model_type = self._detect_model_type()
        model_dir = self._ensure_model()

        lang = settings.sherpa_onnx_language or ""
        t0 = time.time()

        if self._model_type in ("funasr_nano", "funasr_mlt_nano"):
            self._init_funasr_nano(sherpa_onnx, model_dir, lang)
        elif self._model_type == "qwen3_asr":
            self._init_qwen3_asr(sherpa_onnx, model_dir)
        elif self._model_type == "moonshine_v2":
            self._init_moonshine_v2(sherpa_onnx, model_dir)
        else:
            self._init_sense_voice(sherpa_onnx, model_dir, lang)

        logger.info("Model loaded in %.2fs", time.time() - t0)

    # ── Init helpers ────────────────────────────────────────────────

    def _init_sense_voice(self, sherpa_onnx, model_dir: Path, lang: str) -> None:
        logger.info(
            "Loading sherpa-onnx SenseVoice (provider=%s, threads=%d, language=%s)...",
            settings.sherpa_onnx_provider,
            settings.sherpa_onnx_num_threads,
            lang or "auto",
        )
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=settings.sherpa_onnx_num_threads,
            language=lang,
            use_itn=True,
            debug=False,
            provider=settings.sherpa_onnx_provider,
        )

    def _init_funasr_nano(self, sherpa_onnx, model_dir: Path, lang: str) -> None:
        model_label = "FunASR" if self._model_type == "funasr_nano" else "FunASR-MLT"
        logger.info(
            "Loading sherpa-onnx %s (provider=%s, threads=%d, language=%s, itn=%s)...",
            model_label,
            settings.sherpa_onnx_provider,
            settings.sherpa_onnx_num_threads,
            lang or "auto",
            settings.sherpa_onnx_itn,
        )
        llm_path = model_dir / "llm_int8" / "llm.int8.onnx"
        if not llm_path.is_file():
            llm_path = model_dir / "llm.int8.onnx"

        tok_dir = model_dir / "Qwen3-0.6B"
        if not tok_dir.is_dir():
            tok_dir = model_dir

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor=str(model_dir / "encoder_adaptor.int8.onnx"),
            llm=str(llm_path),
            embedding=str(model_dir / "embedding.int8.onnx"),
            tokenizer=str(tok_dir),
            num_threads=settings.sherpa_onnx_num_threads,
            language=lang,
            itn=settings.sherpa_onnx_itn,
            debug=False,
            provider=settings.sherpa_onnx_provider,
        )

    def _init_qwen3_asr(self, sherpa_onnx, model_dir: Path) -> None:
        logger.info(
            "Loading sherpa-onnx Qwen3-ASR 0.6B (provider=%s, threads=%d)...",
            settings.sherpa_onnx_provider,
            settings.sherpa_onnx_num_threads,
        )
        # Qwen3-ASR uses feature_dim=128 (vs usual 80)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(model_dir / "conv_frontend.onnx"),
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            tokenizer=str(model_dir / "tokenizer"),
            num_threads=settings.sherpa_onnx_num_threads,
            feature_dim=128,
            max_new_tokens=128,
            temperature=1e-6,
            top_p=0.8,
            debug=False,
            provider=settings.sherpa_onnx_provider,
        )

    def _init_moonshine_v2(self, sherpa_onnx, model_dir: Path) -> None:
        logger.info(
            "Loading sherpa-onnx Moonshine V2 (provider=%s, threads=%d)...",
            settings.sherpa_onnx_provider,
            settings.sherpa_onnx_num_threads,
        )
        # Moonshine V2 models contain encoder_model.ort, decoder_model_merged.ort, tokens.txt
        encoder = model_dir / "encoder_model.ort"
        decoder = model_dir / "decoder_model_merged.ort"
        tokens = model_dir / "tokens.txt"

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            num_threads=settings.sherpa_onnx_num_threads,
            debug=False,
            provider=settings.sherpa_onnx_provider,
        )

    # ── Transcribe ──────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.monotonic()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(settings.sample_rate, audio)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        inference_ms = (time.monotonic() - t0) * 1000
        return text, inference_ms

    # ── Model download ──────────────────────────────────────────────

    def _detect_model_type(self) -> str:
        """Auto-detect model type from SHERPA_ONNX_MODEL name, or
        use the configured SHERPA_ONNX_MODEL_TYPE as override."""
        override = os.getenv("SHERPA_ONNX_MODEL_TYPE", "")
        if override in self._MODEL_NAMES or override == "moonshine_v2":
            return override

        model_name = settings.sherpa_onnx_model

        # Check known moonshine v2 models first (direct name match)
        if model_name in self._MOONSHINE_V2_MODELS:
            return "moonshine_v2"

        for mtype, mname in self._MODEL_NAMES.items():
            if mname == model_name:
                return mtype

        # Pattern match fallback
        if "moonshine" in model_name.lower():
            return "moonshine_v2"
        if "qwen3" in model_name.lower() or "qwen3_asr" in model_name.lower():
            return "qwen3_asr"
        if "mlt" in model_name.lower() and "funasr" in model_name.lower():
            return "funasr_mlt_nano"
        if "funasr" in model_name.lower():
            return "funasr_nano"
        return "sense_voice"

    def _ensure_model(self) -> Path:
        """Resolve model directory.

        Priority:
          1. SHERPA_ONNX_MODEL_DIR (if set and exists)
          2. Cached model directory
          3. Auto-download from GitHub
        """
        # 1. User-specified model dir
        model_dir = settings.sherpa_onnx_model_dir
        if model_dir:
            p = Path(model_dir)
            if p.is_dir():
                logger.info("Using model from SHERPA_ONNX_MODEL_DIR=%s", p)
                return p
            logger.warning("SHERPA_ONNX_MODEL_DIR=%s not found, falling back", model_dir)

        # 2. Resolve model name
        mapped = self._MODEL_NAMES.get(self._model_type)
        if mapped is None or self._model_type == "moonshine_v2":
            model_name = settings.sherpa_onnx_model
        else:
            model_name = mapped

        cache = settings.model_cache_dir / "sherpa-onnx"
        cached_dir = cache / model_name

        # Check cache
        if cached_dir.is_dir():
            required = self._minimal_dir_content()
            if all((cached_dir / r).exists() for r in required):
                return cached_dir
            logger.warning("Cache incomplete for %s, re-downloading", model_name)

        # 3. Auto-download
        cache.mkdir(parents=True, exist_ok=True)
        parts = self._PARTS_MANIFEST.get(self._model_type)

        if parts:
            self._download_parts(model_name, cache, cached_dir, parts)
        else:
            self._download_single_archive(model_name, cache, cached_dir)

        return cached_dir

    def _minimal_dir_content(self) -> list[str]:
        """Return a list of files/dirs that must exist for a valid model dir."""
        if self._model_type in ("funasr_nano", "funasr_mlt_nano"):
            return ["encoder_adaptor.int8.onnx", "embedding.int8.onnx", "Qwen3-0.6B"]
        if self._model_type == "qwen3_asr":
            return ["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx"]
        if self._model_type == "moonshine_v2":
            return ["encoder_model.ort"]
        # sense_voice
        return ["model.int8.onnx"]

    def _download_single_archive(
        self, model_name: str, cache: Path, cached_dir: Path
    ) -> None:
        """Download a single .tar.bz2 archive and extract it."""
        url = f"{_model_base_url()}/{model_name}.tar.bz2"
        archive = cache / f"{model_name}.tar.bz2"

        self._do_download(url, archive)

        logger.info("Extracting %s...", archive.name)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=cache)
        archive.unlink()
        logger.info("Model extracted to %s", cached_dir)

    def _download_parts(
        self, model_name: str, cache: Path, cached_dir: Path, parts: list[tuple[str, bool, str | None]]
    ) -> None:
        """Download model as multiple parts (files + archives).

        Each part is downloaded from ``{base_url}/{model_name}_{part_name}``.
        Archives are extracted in-place within `cached_dir`.
        """
        cached_dir.mkdir(parents=True, exist_ok=True)
        base_url = _model_base_url()

        for part_name, is_archive, extract_subdir in parts:
            full_name = f"{model_name}_{part_name}"
            url = f"{base_url}/{full_name}"
            dest = cached_dir / part_name

            logger.info("Downloading part %s ...", full_name)
            self._do_download(url, dest)

            if is_archive:
                logger.info("Extracting %s...", dest.name)
                target = cached_dir / extract_subdir if extract_subdir else cached_dir
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(dest, "r:bz2") as tar:
                    tar.extractall(path=target)
                dest.unlink()

    @staticmethod
    def _do_download(url: str, dest: Path) -> None:
        """Download a single file from *url* to *dest* with progress logging."""
        logger.info("Downloading from %s ...", url)
        t0 = time.time()
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        logger.info(
                            "  Download: %.0f%% (%d MB)",
                            pct,
                            downloaded // 1024 // 1024,
                        )
        logger.info("Downloaded in %.1fs (%s)", time.time() - t0, dest.name)


# ── Engine ───────────────────────────────────────────────────────────

class ASREngine:
    """Singleton wrapper around the sherpa-onnx backend."""

    _instance: Optional["ASREngine"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._backend: Optional[_SherpaOnnxBackend] = None
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="sherpa",
        )
        self._initialized = False

    def load(self) -> None:
        """Load the model. Called once at startup."""
        if self._backend is not None:
            return
        self._backend = _SherpaOnnxBackend()
        self._initialized = True

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Run transcription.

        Args:
            audio: 16kHz mono float32, values in [-1, 1].

        Returns:
            Tuple of (text, inference_time_ms).
        """
        if self._backend is None:
            raise RuntimeError("ASR engine not loaded. Call load() first.")

        if len(audio) == 0:
            return "", 0.0

        text, inference_ms = self._backend.transcribe(audio)
        logger.debug(
            "Transcribed %.2fs audio -> '%s' (%.1fms)",
            len(audio) / settings.sample_rate,
            text,
            inference_ms,
        )
        return text, inference_ms

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor
