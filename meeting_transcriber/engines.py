from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .config import TranscriptionConfig
from .parakeet_engine import ParakeetEngine
from .whisperx_engine import WhisperXEngine


class TranscriptionEngine(Protocol):
    def transcribe(self, source: Path) -> dict[str, Any]: ...


def create_engine(config: TranscriptionConfig) -> TranscriptionEngine:
    if config.backend == "whisperx":
        return WhisperXEngine(config)
    if config.backend == "parakeet":
        return ParakeetEngine(config)
    raise ValueError(f"unsupported transcription backend: {config.backend}")
