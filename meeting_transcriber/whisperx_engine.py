from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import TranscriptionConfig
from .diarization import apply_mandatory_diarization
from .render import utterances
from .runtime import elapsed, is_cuda_out_of_memory, prepare_runtime, release_cuda

LOGGER = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class WhisperXEngine:
    """Runs ASR, alignment and mandatory speaker diarization sequentially."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.effective_batch_size = config.batch_size

    def transcribe(self, source: Path) -> dict[str, Any]:
        prepare_runtime(self.config.allow_tf32, self.config.cpu_threads)
        import whisperx

        total_started = perf_counter()
        timings: dict[str, float] = {}

        started = perf_counter()
        audio = whisperx.load_audio(str(source))
        timings["audio_decode"] = elapsed(started)

        started = perf_counter()
        asr_model = whisperx.load_model(
            self.config.model,
            self.config.device,
            device_index=self.config.device_index,
            compute_type=self.config.compute_type,
            language=self.config.language,
            vad_method=self.config.vad_method,
            threads=self.config.cpu_threads,
            asr_options={
                "beam_size": self.config.beam_size,
                "best_of": self.config.beam_size,
                "temperatures": [0.0],
            },
        )
        timings["asr_model_load"] = elapsed(started)
        effective_batch_size = self.effective_batch_size
        try:
            started = perf_counter()
            while True:
                try:
                    result = asr_model.transcribe(
                        audio,
                        batch_size=effective_batch_size,
                        chunk_size=self.config.chunk_size_seconds,
                    )
                    break
                except RuntimeError as error:
                    if effective_batch_size == 1 or not is_cuda_out_of_memory(error):
                        raise
                    reduced = max(1, effective_batch_size // 2)
                    LOGGER.warning(
                        "CUDA memory exhausted at batch size %d; retrying at %d",
                        effective_batch_size,
                        reduced,
                    )
                    effective_batch_size = reduced
                    self.effective_batch_size = reduced
                    release_cuda()
            timings["asr"] = elapsed(started)
        finally:
            del asr_model
            release_cuda()

        language = str(result.get("language") or self.config.language or "").strip()
        if not language:
            raise RuntimeError("WhisperX did not return a language")
        started = perf_counter()
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, device=self.config.device
        )
        timings["alignment_model_load"] = elapsed(started)
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
            timings["alignment"] = elapsed(started)
        finally:
            del align_model
            release_cuda()

        result, diarization_timings = apply_mandatory_diarization(
            audio, result, self.config
        )
        timings.update(diarization_timings)

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
        timings["total"] = elapsed(total_started)
        return {
            "engine": "whisperx",
            "backend": "whisperx",
            "model": self.config.model,
            "language": language,
            "diarization_model": self.config.diarization_model,
            "effective_batch_size": effective_batch_size,
            "timings_seconds": timings,
            "segments": segments,
            "utterances": utterances(segments),
            "speakers": sorted({str(segment["speaker"]) for segment in segments}),
        }
