"""WebSocket session manager for streaming ASR."""

import asyncio
import base64
import json
import logging
import time
from enum import Enum
from typing import Optional

import numpy as np
from fastapi import WebSocket

from .asr_engine import ASREngine
from .audio_buffer import AudioBuffer
from .config import settings
from .vad import VAD

logger = logging.getLogger(__name__)

# Send interim (partial) results every N ms during speech
INTERIM_INTERVAL_MS = 400


class SessionState(Enum):
    IDLE = "idle"
    SPEECH = "speech"
    TRANSCRIBING = "transcribing"


class Session:
    """Manages a single WebSocket streaming session.

    State machine:
        IDLE → (VAD speech) → SPEECH
        SPEECH → (VAD silence timeout / interim timer) → TRANSCRIBING
        TRANSCRIBING → (transcription done) → IDLE

    During SPEECH, interim transcriptions fire every INTERIM_INTERVAL_MS
    so the client sees partial results as the user speaks.
    """

    def __init__(self, websocket: WebSocket):
        self._ws = websocket
        self._vad = VAD()
        self._buffer = AudioBuffer()
        self._asr = ASREngine()
        self._state = SessionState.IDLE
        self._silence_counter = 0
        self._speech_start_time: Optional[float] = None
        self._pending_pcm = AudioBuffer(max_duration_sec=2.0)
        self._task: Optional[asyncio.Task] = None

        # ── Interim / metrics state ──
        self._last_interim_time: float = 0.0
        self._interim_running: bool = False  # guard: only one interim at a time
        self._last_interim_text: str = ""

    async def handle_audio(self, data: bytes) -> None:
        """Process an incoming PCM chunk.

        Args:
            data: Raw Int16 PCM bytes.
        """
        self._pending_pcm.push(data)

        frame_bytes = self._vad.frame_bytes
        total_bytes = b"".join(self._pending_pcm._buffer)
        self._pending_pcm.clear()

        offset = 0
        while offset + frame_bytes <= len(total_bytes):
            frame = total_bytes[offset : offset + frame_bytes]
            offset += frame_bytes
            self._process_vad_frame(frame)

        if offset < len(total_bytes):
            self._pending_pcm.push(total_bytes[offset:])

    def _process_vad_frame(self, frame: bytes) -> None:
        """Process a single VAD frame and check interim timer."""
        is_speech = self._vad.is_speech(frame)

        if self._state == SessionState.IDLE:
            if is_speech:
                logger.info("Speech started")
                self._state = SessionState.SPEECH
                self._speech_start_time = time.time()
                self._last_interim_time = time.time()
                self._buffer.push(frame)
                self._schedule_speech_start()

        elif self._state == SessionState.SPEECH:
            self._buffer.push(frame)
            if is_speech:
                self._silence_counter = 0
                # Check if it's time to send an interim result
                if (not self._interim_running
                        and (time.time() - self._last_interim_time) * 1000 >= INTERIM_INTERVAL_MS):
                    self._fire_interim()
            else:
                self._silence_counter += 1
                if self._silence_counter >= settings.silence_frame_count:
                    self._transition_to_final()

        elif self._state == SessionState.TRANSCRIBING:
            # Buffer speech frames so they aren't lost while ASR runs.
            # They'll be picked up by the next handle_audio() call once
            # state returns to IDLE.
            if is_speech:
                self._pending_pcm.push(frame)

    # ── helpers ────────────────────────────────────────────────────────

    def _build_metrics(self, audio: np.ndarray, inference_ms: float) -> dict:
        """Build a metrics dict for the current segment."""
        now = time.time()
        e2e = (now - (self._speech_start_time or now)) * 1000
        return {
            "audio_duration_ms": round(len(audio) / settings.sample_rate * 1000, 1),
            "inference_ms": round(inference_ms, 1),
            "e2e_ms": round(e2e, 1),
        }

    def _schedule_speech_start(self) -> None:
        """Send speech_start notification to client."""
        asyncio.ensure_future(self._send_json({
            "type": "speech_start",
            "timestamp": time.time(),
        }))

    # ── Interim transcription ─────────────────────────────────────────

    def _fire_interim(self) -> None:
        """Launch an interim transcription task (non-blocking)."""
        self._interim_running = True
        audio = self._buffer.as_float32()  # snapshot, don't clear
        # Only run if there's enough new audio since last interim
        if len(audio) >= settings.sample_rate * 0.15:  # ≥ 150ms
            asyncio.ensure_future(self._do_interim(audio))
        else:
            self._interim_running = False

    async def _do_interim(self, audio: np.ndarray) -> None:
        """Transcribe a snapshot and send interim result."""
        try:
            text, inference_ms = await self._asr.transcribe(audio)
            if text and text != self._last_interim_text:
                self._last_interim_text = text
                self._last_interim_time = time.time()
                await self._send_json({
                    "type": "interim",
                    "text": text,
                    "metrics": self._build_metrics(audio, inference_ms),
                })
        except Exception as e:
            logger.info("Interim transcription error: %s", e)
        finally:
            self._interim_running = False

    # ── Final transcription ───────────────────────────────────────────

    def _transition_to_final(self) -> None:
        """Transition from SPEECH to TRANSCRIBING and launch final inference."""
        self._state = SessionState.TRANSCRIBING
        audio = self._buffer.extract_and_clear()
        self._silence_counter = 0
        self._task = asyncio.ensure_future(self._do_final(audio, vad_triggered=True))

    async def _do_final(self, audio: np.ndarray, vad_triggered: bool = True) -> None:
        """Run ASR inference and send final result with metrics.

        Args:
            audio: Audio array to transcribe.
            vad_triggered: Whether this final was triggered naturally by VAD
                           silence detection (True) or forced (flush on
                           disconnect, False).
        """
        if len(audio) < settings.sample_rate * 0.1:
            logger.info("Audio too short (%.2fs), skipping", len(audio) / settings.sample_rate)
            self._state = SessionState.IDLE
            return

        try:
            logger.info("Final transcription of %.2fs audio...", len(audio) / settings.sample_rate)
            text, inference_ms = await self._asr.transcribe(audio)
            logger.info("Final result: '%s' (%.1fms)", text, inference_ms)
            duration = len(audio) / settings.sample_rate
            # Encode audio as base64 Int16 PCM for client-side playback
            int16_data = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            pcm_bytes = int16_data.tobytes()
            audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
            await self._send_json({
                "type": "final",
                "text": text,
                "audio": audio_b64,
                "vad": vad_triggered,
                "metrics": self._build_metrics(audio, inference_ms),
                "start": self._speech_start_time or 0.0,
                "end": (self._speech_start_time or 0.0) + duration,
            })
        except Exception as e:
            logger.error("Transcription error: %s", e, exc_info=True)
            await self._send_json({
                "type": "error",
                "message": f"Transcription failed: {e}",
            })
        finally:
            self._state = SessionState.IDLE
            self._speech_start_time = None
            self._last_interim_text = ""

    # ── WebSocket send ────────────────────────────────────────────────

    async def _send_json(self, data: dict) -> None:
        """Send a JSON message over WebSocket."""
        try:
            await self._ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.warning("Failed to send message: %s", e)

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def flush(self) -> None:
        """Force-transcribe any buffered audio on disconnect.

        This ensures the last speech segment is transcribed even if
        the VAD silence threshold hasn't been reached yet.
        """
        if self._state == SessionState.SPEECH:
            audio = self._buffer.extract_and_clear()
            if len(audio) >= settings.sample_rate * 0.1:
                logger.info("Flushing %d samples (%.2fs) from SPEECH state",
                             len(audio), len(audio) / settings.sample_rate)
                self._state = SessionState.TRANSCRIBING
                await self._do_final(audio, vad_triggered=False)
            else:
                self._buffer.clear()
        self._state = SessionState.IDLE
        self._speech_start_time = None

    async def cleanup(self) -> None:
        """Clean up session resources."""
        await self.flush()
        self._buffer.clear()
        self._pending_pcm.clear()
        if self._task and not self._task.done():
            self._task.cancel()


# ── SessionManager ──────────────────────────────────────────────────

class SessionManager:
    """Manages all active WebSocket sessions."""

    def __init__(self):
        self._sessions: dict[int, Session] = {}
        self._semaphore = asyncio.Semaphore(settings.max_connections)

    async def create_session(self, websocket: WebSocket) -> Session | None:
        """Create a new session with connection limit.

        Returns None if the server is busy (max connections reached).
        The caller should close the WebSocket in that case.
        """
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Rejected connection: max sessions (%d) reached",
                           settings.max_connections)
            await websocket.send_json({
                "type": "error",
                "message": f"Server busy (max {settings.max_connections} connections)",
            })
            await websocket.close(code=1013, reason="server busy")
            return None

        session = Session(websocket)
        self._sessions[id(websocket)] = session
        return session

    async def remove_session(self, websocket: WebSocket) -> None:
        """Remove and clean up a session."""
        session = self._sessions.pop(id(websocket), None)
        if session:
            await session.cleanup()
            self._semaphore.release()

    @property
    def active_count(self) -> int:
        return len(self._sessions)


session_manager = SessionManager()
