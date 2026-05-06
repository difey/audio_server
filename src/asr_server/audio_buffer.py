"""Audio ring buffer for streaming PCM data."""

import struct
from collections import deque
from typing import Optional

import numpy as np


class AudioBuffer:
    """Buffer for accumulating Int16 PCM audio chunks.

    Accumulates raw Int16 bytes and provides methods to
    convert to float32 numpy arrays for whisper inference.
    """

    def __init__(self, max_duration_sec: float = 30.0, sample_rate: int = 16000):
        max_samples = int(max_duration_sec * sample_rate)
        self._buffer = deque[bytes]()
        self._total_samples = 0
        self._max_samples = max_samples
        self._sample_rate = sample_rate

    def push(self, pcm_bytes: bytes) -> None:
        """Append raw Int16 PCM bytes to the buffer.

        Args:
            pcm_bytes: Raw Int16 PCM data (little-endian).
        """
        self._buffer.append(pcm_bytes)
        self._total_samples += len(pcm_bytes) // 2

        # Trim oldest data if buffer exceeds max duration
        while self._total_samples > self._max_samples and self._buffer:
            oldest = self._buffer.popleft()
            self._total_samples -= len(oldest) // 2

    def clear(self) -> None:
        """Clear all buffered audio."""
        self._buffer.clear()
        self._total_samples = 0

    def as_float32(self) -> np.ndarray:
        """Convert buffered audio to float32 numpy array.

        Returns:
            Float32 array in [-1, 1], 16kHz mono.
        """
        if not self._buffer:
            return np.array([], dtype=np.float32)

        all_bytes = b"".join(self._buffer)
        int16_data = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32)
        return int16_data / 32767.0

    def extract_and_clear(self) -> np.ndarray:
        """Get audio as float32 and clear the buffer.

        Returns:
            Float32 array of the accumulated audio.
        """
        audio = self.as_float32()
        self.clear()
        return audio

    @property
    def duration_sec(self) -> float:
        """Current buffered duration in seconds."""
        return self._total_samples / self._sample_rate

    @property
    def is_empty(self) -> bool:
        return self._total_samples == 0
