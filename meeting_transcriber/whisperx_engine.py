from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import TranscriptionConfig
from .render import utterances

LOGGER = logging.getLogger(__name__)


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


def _is_cuda_out_of_memory(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        signal in message
        for signal in (
            "cuda out of memory",
            "cuda_error_out_of_memory",
            "out of memory",
            "failed to allocate",
        )
    )


def _elapsed(started: float) -> float:
    return round(perf_counter() - started, 3)


def _disable_telemetry() -> None:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"


class WhisperXEngine:
    """Runs ASR, alignment and mandatory speaker diarization sequentially."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config

    def transcribe(self, source: Path) -> dict[str, Any]:
        _disable_telemetry()
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        token = os.getenv(self.config.hf_token_env, "")
        if not token:
            raise RuntimeError(
                f"{self.config.hf_token_env} is required for diarization"
            )

        total_started = perf_counter()
        timings: dict[str, float] = {}

        started = perf_counter()
        audio = whisperx.load_audio(str(source))
        timings["audio_decode"] = _elapsed(started)

        started = perf_counter()
        asr_model = whisperx.load_model(
            self.config.model,
            self.config.device,
            compute_type=self.config.compute_type,
            language=self.config.language,
        )
        timings["asr_model_load"] = _elapsed(started)
        effective_batch_size = self.config.batch_size
        try:
            started = perf_counter()
            while True:
                try:
                    result = asr_model.transcribe(
                        audio, batch_size=effective_batch_size
                    )
                    break
                except RuntimeError as error:
                    if effective_batch_size == 1 or not _is_cuda_out_of_memory(error):
                        raise
                    reduced = max(1, effective_batch_size // 2)
                    LOGGER.warning(
                        "CUDA memory exhausted at batch size %d; retrying at %d",
                        effective_batch_size,
                        reduced,
                    )
                    effective_batch_size = reduced
                    _release_cuda()
            timings["asr"] = _elapsed(started)
        finally:
            del asr_model
            _release_cuda()

        language = str(result.get("language") or self.config.language or "").strip()
        if not language:
            raise RuntimeError("WhisperX did not return a language")
        started = perf_counter()
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, device=self.config.device
        )
        timings["alignment_model_load"] = _elapsed(started)
        try:
            started = perf_counter()
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                self.config.device,
                return_char_alignments=False,
            )
            timings["alignment"] = _elapsed(started)
        finally:
            del align_model
            _release_cuda()

        started = perf_counter()
        diarizer = DiarizationPipeline(
            model_name=self.config.diarization_model,
            token=token,
            device=self.config.device,
        )
        timings["diarization_model_load"] = _elapsed(started)
        try:
            started = perf_counter()
            diarization = diarizer(
                audio,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )
            result = whisperx.assign_word_speakers(
                diarization, result, fill_nearest=True
            )
            timings["diarization_and_assignment"] = _elapsed(started)
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
        timings["total"] = _elapsed(total_started)
        return {
            "engine": "whisperx",
            "model": self.config.model,
            "language": language,
            "diarization_model": self.config.diarization_model,
            "effective_batch_size": effective_batch_size,
            "timings_seconds": timings,
            "segments": segments,
            "utterances": utterances(segments),
            "speakers": sorted({str(segment["speaker"]) for segment in segments}),
        }
