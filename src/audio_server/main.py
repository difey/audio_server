"""Audio server FastAPI application entry point."""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response

from .asr_engine import ASREngine
from .config import settings
from .session import Session, session_manager
from .tts_engine import TTSEngine

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("audio_server")

# ── ASR Engine (global singleton) ────────────────────────────────────
_asr_engine = ASREngine()

# ── TTS Engine (global singleton, optional) ──────────────────────────
_tts_engine = TTSEngine()




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Starting Audio server...")

    if settings.asr_enabled:
        _asr_engine.load()
    else:
        logger.info("ASR is disabled (set ASR_ENABLED=true to enable)")

    if settings.tts_enabled:
        logger.info("TTS is enabled, loading TTS engine...")
        _tts_engine.load()
    else:
        logger.info("TTS is disabled (set TTS_ENABLED=true to enable)")

    logger.info("Ready. Listening on %s:%d", settings.host, settings.port)
    yield
    logger.info("Shutting down.")
    if _asr_engine.is_loaded:
        await _asr_engine.unload()
    if _tts_engine.is_loaded:
        await _tts_engine.unload()


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Audio Server", version="0.1.0", lifespan=lifespan)

# ── TTS request model ────────────────────────────────────────────────


class TTSRequest(BaseModel):
    """Request body for POST /v1/audio/speech."""

    input: str = Field(
        ...,
        description="Text to synthesize",
        min_length=1,
        max_length=2000,
    )
    voice: str | None = Field(
        default=None,
        description="Speaker ID (e.g. '0', '1') for sherpa-onnx, or speaker name (e.g. 'Vivian') for Qwen3-TTS",
    )
    response_format: str = Field(
        default="wav",
        description="Audio format: wav, pcm (raw Int16), or mp3",
        pattern=r"^(wav|pcm|mp3)$",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speed factor (0.5-2.0, sherpa-onnx only)",
    )
    # ── Qwen3-TTS specific ──
    language: str | None = Field(
        default=None,
        description="Language for Qwen3-TTS (e.g. 'Chinese', 'English', 'Auto')",
    )
    instruct: str | None = Field(
        default=None,
        description="Natural-language instruction for voice control (1.7B model only)",
    )


# ── TTS endpoint ─────────────────────────────────────────────────────


@app.post(
    "/v1/audio/speech",
    responses={
        200: {
            "content": {
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/L16": {"schema": {"type": "string", "format": "binary"}},
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
            },
            "description": "Audio file (WAV, PCM, or MP3)",
        },
    },
)
async def text_to_speech(req: TTSRequest):
    """Synthesize text to speech (OpenAI-compatible endpoint).

    Supports both sherpa-onnx models and Qwen3-TTS models.

    Sherpa-onnx models:
    - `matcha-icefall-zh-en` — Matcha-TTS (zh/en)
    - `vits-piper-zh_CN-chaowen-medium` — VITS Piper Chinese (chaowen)
    - `vits-piper-en_GB-jenny_dioco-medium` — VITS Piper English UK (jenny_dioco)

    Qwen3-TTS models (set ``TTS_MODEL`` to model ID like
    ``Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice``):
    - ``voice`` → speaker name (e.g. ``"Vivian"``, ``"Ryan"``)
    - ``language`` → language (e.g. ``"Chinese"``, ``"English"``)
    - ``instruct`` → optional instruction (1.7B only)

    Supported response formats:
    - `wav` — WAV file (audio/wav)
    - `pcm` — raw Int16 PCM (audio/L16)
    - `mp3` — MP3 via ffmpeg (audio/mpeg)

    Request (sherpa-onnx):
        ```json
        {
          "model": "matcha-icefall-zh-en",
          "input": "你好，世界！",
          "voice": "0",
          "response_format": "wav",
          "speed": 1.0
        }
        ```

    Request (Qwen3-TTS):
        ```json
        {
          "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
          "input": "你好，世界！",
          "voice": "Vivian",
          "language": "Chinese",
          "response_format": "wav"
        }
        ```

    Returns:
        Audio bytes (WAV / PCM / MP3) with the appropriate Content-Type.
    """
    if not _tts_engine.is_loaded:
        return Response(
            content='{"error": "TTS engine not loaded. Enable TTS_ENABLED=true"}',
            status_code=503,
            media_type="application/json",
        )

    try:
        fmt = req.response_format

        if _tts_engine.is_qwen3:
            # ── Qwen3-TTS path (async via lock + executor) ──────────────
            if fmt == "wav":
                audio_bytes = await _tts_engine.synthesize_qwen3_to_wav_bytes(
                    text=req.input,
                    language=req.language,
                    speaker=req.voice,
                    instruct=req.instruct,
                )
                media_type = "audio/wav"
                filename = "speech.wav"
            elif fmt == "pcm":
                audio_bytes, sample_rate = await _tts_engine.synthesize_qwen3_to_pcm_bytes(
                    text=req.input,
                    language=req.language,
                    speaker=req.voice,
                    instruct=req.instruct,
                )
                media_type = f"audio/L16; rate={sample_rate}; channels=1"
                filename = "speech.pcm"
            elif fmt == "mp3":
                audio_bytes = await _tts_engine.synthesize_qwen3_to_mp3_bytes(
                    text=req.input,
                    language=req.language,
                    speaker=req.voice,
                    instruct=req.instruct,
                )
                media_type = "audio/mpeg"
                filename = "speech.mp3"
            else:
                return Response(
                    content=f'{{"error": "Unsupported response_format: {fmt}"}}',
                    status_code=400,
                    media_type="application/json",
                )
        else:
            # ── Sherpa-onnx path (async via TTSScheduler) ──────────────
            sid = int(req.voice) if req.voice is not None else None

            if fmt == "wav":
                audio_bytes = await _tts_engine.synthesize_to_wav_bytes(
                    text=req.input, sid=sid, speed=req.speed,
                )
                media_type = "audio/wav"
                filename = "speech.wav"
            elif fmt == "pcm":
                audio_bytes, sample_rate = await _tts_engine.synthesize_to_pcm_bytes(
                    text=req.input, sid=sid, speed=req.speed,
                )
                media_type = f"audio/L16; rate={sample_rate}; channels=1"
                filename = "speech.pcm"
            elif fmt == "mp3":
                audio_bytes = await _tts_engine.synthesize_to_mp3_bytes(
                    text=req.input, sid=sid, speed=req.speed,
                )
                media_type = "audio/mpeg"
                filename = "speech.mp3"
            else:
                return Response(
                    content=f'{{"error": "Unsupported response_format: {fmt}"}}',
                    status_code=400,
                    media_type="application/json",
                )

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
            },
        )
    except asyncio.QueueFull:
        logger.warning("TTS queue full, rejecting request")
        return Response(
            content='{"error": "Server busy, please try again later"}',
            status_code=503,
            media_type="application/json",
        )
    except Exception as e:
        logger.error("TTS synthesis error: %s", e, exc_info=True)
        return Response(
            content=f'{{"error": "TTS synthesis failed: {e}"}}',
            status_code=500,
            media_type="application/json",
        )


# ── ASR file transcription endpoint ───────────────────────────────────


@app.post("/v1/audio/transcriptions")
async def transcribe_file(
    file: UploadFile = File(...),
    response_format: str = Form("text"),
):
    """Transcribe an audio file (WAV, MP3, FLAC, OGG, etc.).

    OpenAI-compatible endpoint:

        curl -X POST http://localhost:8000/v1/audio/transcriptions \\
          -F "file=@speech.wav" \\
          -F "response_format=json"

    Supported `response_format`:
    - ``text`` — plain text (default)
    - ``json`` — ``{"text": "...", "duration_sec": 1.2, "inference_ms": 85}``
    """
    if not _asr_engine.is_loaded:
        raise HTTPException(status_code=503, detail="ASR engine not loaded")

    ext = Path(file.filename or "audio.wav").suffix.lower()
    supported = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma"}

    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {', '.join(sorted(supported))}",
        )

    # Save uploaded file to a temp location
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load audio → 16kHz mono float32
        audio = _load_audio(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    text, inference_ms = _asr_engine.transcribe(audio)

    duration_sec = len(audio) / 16000
    rtf = inference_ms / (duration_sec * 1000) if duration_sec > 0 else 0

    logger.info(
        "ASR file (%s) done: text=%.80r duration=%.2fs inference=%.0fms rtf=%.2f",
        Path(file.filename or "audio.wav").name,
        text,
        duration_sec,
        inference_ms,
        rtf,
    )

    if response_format == "json":
        return {
            "text": text,
            "duration_sec": round(duration_sec, 2),
            "inference_ms": round(inference_ms, 1),
        }
    return Response(content=text, media_type="text/plain")


def _load_audio(path: str) -> "np.ndarray":
    """Load any audio file to 16kHz mono float32 [-1, 1]."""
    import numpy as np
    import soundfile as sf
    import io

    try:
        data, sr = sf.read(path)
        # soundfile handles WAV/FLAC/OGG natively
    except Exception:
        # For MP3/other formats, use pydub (requires ffmpeg)
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError(
                "pydub is required for MP3/other audio formats. "
                "Install: pip install pydub  (and system ffmpeg)"
            )
        audio_seg = AudioSegment.from_file(path)
        buf = io.BytesIO()
        audio_seg.export(buf, format="wav")
        buf.seek(0)
        data, sr = sf.read(buf)

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        data = _resample(data, sr, 16000)

    # Ensure float32 in [-1, 1]
    if data.dtype != np.float32:
        if data.dtype in (np.int16, np.int32):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)

    return data


def _resample(data: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
    """Resample audio using simple linear interpolation (lightweight)."""
    import numpy as np
    import math

    duration = len(data) / orig_sr
    target_len = int(math.ceil(duration * target_sr))
    indices = np.linspace(0, len(data) - 1, target_len)
    # Linear interpolation via floor (nearest-neighbor is fast enough for ASR)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, len(data) - 1)
    return data[indices]


# ── Static files (Vue SPA) ───────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR_EXISTS = STATIC_DIR.is_dir()


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def serve_frontend(full_path: str = ""):
    """Serve the Vue SPA or static assets.

    This catch-all route only handles HTTP GET/HEAD, not WebSocket.
    WebSocket upgrade requests at /ws/transcribe are handled by
    the dedicated route before reaching here.
    """
    # Skip for WebSocket paths (shouldn't reach here, but just in case)
    if full_path.startswith("ws/"):
        return HTMLResponse(status_code=404)

    if not STATIC_DIR_EXISTS:
        return {
            "service": "Audio Server",
            "model": settings.sherpa_onnx_model,
            "status": "running",
            "ws_endpoint": "/ws/transcribe",
        }

    # Serve specific file if it exists
    if full_path:
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))

    # SPA fallback: always serve index.html
    index_content = (STATIC_DIR / "index.html").read_bytes()
    return Response(content=index_content, media_type="text/html")


# ── WebSocket ────────────────────────────────────────────────────────
@app.websocket("/ws/transcribe")
async def websocket_endpoint(ws: WebSocket):
    if not _asr_engine.is_loaded:
        await ws.accept()
        await ws.send_json({
            "type": "error",
            "message": "ASR is disabled (set ASR_ENABLED=true to enable)",
        })
        await ws.close(code=1011, reason="ASR disabled")
        return

    await ws.accept()
    session = await session_manager.create_session(ws)
    if session is None:
        return  # rejected — already closed by create_session

    logger.info(
        "Client connected (%d active)",
        session_manager.active_count,
    )

    try:
        while True:
            data = await ws.receive_bytes()
            await session.handle_audio(data)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        await session_manager.remove_session(ws)
        logger.info(
            "Session cleaned up (%d active)",
            session_manager.active_count,
        )


# ── Entry point ──────────────────────────────────────────────────────
def main():
    uvicorn.run(
        "audio_server.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_max_size=1024 * 1024,  # 1MB max WS message
    )


if __name__ == "__main__":
    main()
