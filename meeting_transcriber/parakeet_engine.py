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
SAMPLE_RATE = 16000


def _timestamp_words(timestamp_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current and current["word"].strip():
            current["word"] = current["word"].strip()
            words.append(current)
        current = None

    for item in timestamp_tokens:
        token = str(item.get("token", ""))
        if not token:
            continue
        if current is not None and token[:1].isspace():
            flush()
        clean = token.lstrip()
        if clean:
            if current is None:
                current = {
                    "word": clean,
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                }
            else:
                current["word"] += clean
                current["end"] = float(item["end"])
        if token[-1:].isspace():
            flush()
    flush()
    return words


def _segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if current and float(word["start"]) - float(current[-1]["end"]) > 1.5:
            _append_segment(segments, current)
            current = []
        current.append(word)
        duration = float(current[-1]["end"]) - float(current[0]["start"])
        if str(word["word"]).endswith((".", "?", "!")) or duration >= 15:
            _append_segment(segments, current)
            current = []
    _append_segment(segments, current)
    return segments


def _append_segment(
    segments: list[dict[str, Any]], words: list[dict[str, Any]]
) -> None:
    if not words:
        return
    segments.append(
        {
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": " ".join(str(word["word"]) for word in words),
            "words": words.copy(),
        }
    )


def _transcribe_chunks(
    model: Any,
    processor: Any,
    audio: Any,
    config: TranscriptionConfig,
    dtype: Any,
    device: str,
    chunk_seconds: int,
) -> tuple[list[dict[str, Any]], float, float, str]:
    import torch

    chunk_samples = chunk_seconds * SAMPLE_RATE
    overlap_samples = config.parakeet_overlap_seconds * SAMPLE_RATE
    total_samples = len(audio)
    words: list[dict[str, Any]] = []
    decoded_parts: list[str] = []
    feature_seconds = 0.0
    asr_seconds = 0.0

    for core_start in range(0, total_samples, chunk_samples):
        core_end = min(total_samples, core_start + chunk_samples)
        actual_start = max(0, core_start - overlap_samples)
        actual_end = min(total_samples, core_end + overlap_samples)

        started = perf_counter()
        inputs = processor(
            audio[actual_start:actual_end],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).to(device=device, dtype=dtype)
        feature_seconds += perf_counter() - started

        started = perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, return_dict_in_generate=True)
        asr_seconds += perf_counter() - started
        sequences = generated.sequences.detach().cpu()
        durations = generated.durations.detach().cpu()
        decoded, timestamp_batches = processor.decode(
            sequences,
            durations=durations,
            skip_special_tokens=True,
        )
        decoded_parts.append(decoded[0] if isinstance(decoded, list) else str(decoded))

        relative_words = _timestamp_words(
            timestamp_batches[0] if timestamp_batches else []
        )
        actual_offset = actual_start / SAMPLE_RATE
        core_start_seconds = core_start / SAMPLE_RATE
        core_end_seconds = core_end / SAMPLE_RATE
        for word in relative_words:
            absolute = {
                **word,
                "start": float(word["start"]) + actual_offset,
                "end": float(word["end"]) + actual_offset,
            }
            midpoint = (absolute["start"] + absolute["end"]) / 2
            if midpoint < core_start_seconds:
                continue
            if core_end < total_samples and midpoint >= core_end_seconds:
                continue
            words.append(absolute)
        del generated, inputs

    return (
        words,
        round(feature_seconds, 3),
        round(asr_seconds, 3),
        " ".join(decoded_parts),
    )


class ParakeetEngine:
    """Runs Parakeet TDT timestamps with mandatory local WhisperX diarization."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.effective_chunk_seconds = config.parakeet_chunk_seconds

    def transcribe(self, source: Path) -> dict[str, Any]:
        prepare_runtime(self.config.allow_tf32, self.config.cpu_threads)
        import torch
        import whisperx
        from transformers import AutoModelForTDT, AutoProcessor

        total_started = perf_counter()
        timings: dict[str, float] = {}

        started = perf_counter()
        audio = whisperx.load_audio(str(source))
        timings["audio_decode"] = elapsed(started)

        started = perf_counter()
        processor = AutoProcessor.from_pretrained(self.config.model)
        dtype = getattr(torch, self.config.parakeet_dtype)
        device = f"{self.config.device}:{self.config.device_index}"
        model = AutoModelForTDT.from_pretrained(
            self.config.model,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()
        timings["asr_model_load"] = elapsed(started)

        try:
            effective_chunk_seconds = self.effective_chunk_seconds
            while True:
                try:
                    words, feature_time, asr_time, decoded_text = _transcribe_chunks(
                        model,
                        processor,
                        audio,
                        self.config,
                        dtype,
                        device,
                        effective_chunk_seconds,
                    )
                    break
                except RuntimeError as error:
                    if effective_chunk_seconds <= 60 or not is_cuda_out_of_memory(
                        error
                    ):
                        raise
                    reduced = max(60, effective_chunk_seconds // 2)
                    LOGGER.warning(
                        "CUDA memory exhausted at %d-second Parakeet chunks; retrying at %d seconds",
                        effective_chunk_seconds,
                        reduced,
                    )
                    effective_chunk_seconds = reduced
                    self.effective_chunk_seconds = reduced
                    release_cuda()
            timings["feature_extraction"] = feature_time
            timings["asr"] = asr_time
        finally:
            del model
            release_cuda()

        segments = _segments_from_words(words)
        if not segments:
            raise RuntimeError(
                f"Parakeet returned no timestamped segments for: {decoded_text[:120]}"
            )

        aligned: dict[str, Any] = {"segments": segments}
        aligned, diarization_timings = apply_mandatory_diarization(
            audio, aligned, self.config
        )
        timings.update(diarization_timings)

        segments = aligned["segments"]
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
            "engine": "parakeet+whisperx-diarization",
            "backend": "parakeet",
            "model": self.config.model,
            "language": self.config.language or "auto",
            "diarization_model": self.config.diarization_model,
            "effective_batch_size": 1,
            "effective_chunk_seconds": effective_chunk_seconds,
            "timings_seconds": timings,
            "segments": segments,
            "utterances": utterances(segments),
            "speakers": sorted({str(segment["speaker"]) for segment in segments}),
        }
