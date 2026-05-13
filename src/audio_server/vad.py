"""Voice Activity Detection using webrtcvad."""

import logging
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


class VAD:
    """Wrapper around webrtcvad with configurable sensitivity."""

    def __init__(self):
        try:
            import webrtcvad
        except ImportError:
            raise ImportError("webrtcvad is not installed. Install with: pip install .[asr]")

        self._vad = webrtcvad.Vad()
        self._vad.set_mode(settings.vad_mode)
        self._frame_ms = settings.vad_frame_ms
        self._sample_rate = settings.sample_rate

        logger.debug(
            "VAD initialized: mode=%d, frame=%dms, rate=%d",
            settings.vad_mode, self._frame_ms, self._sample_rate,
        )

    def is_speech(self, pcm_frame: bytes) -> bool:
        """Check if a PCM frame contains speech.

        Args:
            pcm_frame: Raw Int16 PCM bytes, must be exactly
                       frame_ms * sample_rate / 1000 samples.

        Returns:
            True if speech detected.
        """
        return self._vad.is_speech(pcm_frame, self._sample_rate)

    @property
    def frame_bytes(self) -> int:
        """Size of one VAD frame in bytes (Int16)."""
        return int(self._sample_rate * self._frame_ms / 1000) * 2  # 2 bytes per sample

    @property
    def frame_samples(self) -> int:
        """Number of samples per VAD frame."""
        return int(self._sample_rate * self._frame_ms / 1000)
