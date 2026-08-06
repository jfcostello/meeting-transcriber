from __future__ import annotations

import os
from time import perf_counter
from typing import Any

from .config import TranscriptionConfig
from .runtime import elapsed, release_cuda


def apply_mandatory_diarization(
    audio: Any,
    result: dict[str, Any],
    config: TranscriptionConfig,
) -> tuple[dict[str, Any], dict[str, float]]:
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    token = os.getenv(config.hf_token_env, "")
    if not token:
        raise RuntimeError(f"{config.hf_token_env} is required for diarization")

    timings: dict[str, float] = {}
    started = perf_counter()
    diarizer = DiarizationPipeline(
        model_name=config.diarization_model,
        token=token,
        device=config.device,
    )
    timings["diarization_model_load"] = elapsed(started)
    try:
        started = perf_counter()
        diarization = diarizer(
            audio,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
        )
        result = whisperx.assign_word_speakers(diarization, result, fill_nearest=True)
        timings["diarization_and_assignment"] = elapsed(started)
    finally:
        del diarizer
        release_cuda()
    return result, timings
