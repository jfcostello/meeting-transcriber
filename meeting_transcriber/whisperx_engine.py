from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from .config import TranscriptionConfig
from .render import utterances


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class WhisperXEngine:
    """Runs ASR, alignment and mandatory speaker diarization sequentially."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config

    def transcribe(self, source: Path) -> dict[str, Any]:
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        token = os.getenv(self.config.hf_token_env, "")
        if not token:
            raise RuntimeError(
                f"{self.config.hf_token_env} is required for diarization"
            )

        audio = whisperx.load_audio(str(source))
        asr_model = whisperx.load_model(
            self.config.model,
            self.config.device,
            compute_type=self.config.compute_type,
            language=self.config.language,
        )
        try:
            result = asr_model.transcribe(audio, batch_size=self.config.batch_size)
        finally:
            del asr_model
            _release_cuda()

        language = str(result.get("language") or self.config.language or "").strip()
        if not language:
            raise RuntimeError("WhisperX did not return a language")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, device=self.config.device
        )
        try:
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                self.config.device,
                return_char_alignments=False,
            )
        finally:
            del align_model
            _release_cuda()

        diarizer = DiarizationPipeline(
            model_name=self.config.diarization_model,
            token=token,
            device=self.config.device,
        )
        try:
            diarization = diarizer(
                str(source),
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )
            result = whisperx.assign_word_speakers(
                diarization, result, fill_nearest=True
            )
        finally:
            del diarizer
            _release_cuda()

        segments = _plain(result.get("segments", []))
        if not segments:
            raise RuntimeError("WhisperX returned no transcript segments")
        missing = [
            index
            for index, segment in enumerate(segments)
            if not segment.get("speaker")
        ]
        if missing:
            raise RuntimeError(
                f"mandatory diarization left {len(missing)} segments without speakers"
            )
        return {
            "engine": "whisperx",
            "model": self.config.model,
            "language": language,
            "diarization_model": self.config.diarization_model,
            "segments": segments,
            "utterances": utterances(segments),
            "speakers": sorted({str(segment["speaker"]) for segment in segments}),
        }
