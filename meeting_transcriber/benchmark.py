from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, TranscriptionConfig
from .media import probe_media, sha256_file
from .render import transcript_markdown
from .whisperx_engine import WhisperXEngine

EngineFactory = Callable[[TranscriptionConfig], WhisperXEngine]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _model_directory(model: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-.") or "model"
    fingerprint = hashlib.sha256(model.encode()).hexdigest()[:8]
    return f"{readable[:72]}-{fingerprint}"


def benchmark_models(
    source: Path,
    config: AppConfig,
    models: list[str],
    output_root: Path | None = None,
    engine_factory: EngineFactory = WhisperXEngine,
) -> tuple[Path, dict[str, Any]]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    unique_models = list(
        dict.fromkeys(model.strip() for model in models if model.strip())
    )
    if not unique_models:
        raise ValueError("at least one benchmark model is required")

    metadata = probe_media(source, config.timezone)
    source_hash = sha256_file(source)
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    root = (output_root or config.paths.state / "benchmarks").resolve()
    final_dir = root / f"{run_id}-{source_hash[:12]}"
    temporary = root / f".{run_id}-{source_hash[:12]}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    for model in unique_models:
        transcription_config = replace(config.transcription, model=model)
        replace(config, transcription=transcription_config).validate_runtime()
        transcription = engine_factory(transcription_config).transcribe(source)
        model_dir = temporary / _model_directory(model)
        model_dir.mkdir()
        transcript = {
            "schema_version": 1,
            "model": model,
            "language": transcription["language"],
            "speakers": transcription["speakers"],
            "utterances": transcription["utterances"],
            "segments": transcription["segments"],
        }
        _write_json(model_dir / "transcript.json", transcript)
        (model_dir / "transcript.md").write_text(
            transcript_markdown(transcript), encoding="utf-8"
        )
        timings = transcription.get("timings_seconds", {})
        total_seconds = float(timings.get("total", 0))
        realtime_multiple = (
            round(metadata.duration_seconds / total_seconds, 2)
            if metadata.duration_seconds and total_seconds > 0
            else None
        )
        results.append(
            {
                "model": model,
                "directory": model_dir.name,
                "language": transcription["language"],
                "speakers": len(transcription["speakers"]),
                "effective_batch_size": transcription.get("effective_batch_size"),
                "timings_seconds": timings,
                "realtime_multiple": realtime_multiple,
            }
        )

    report = {
        "schema_version": 1,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "source": {
            "path": str(source),
            "sha256": source_hash,
            **metadata.to_dict(),
        },
        "diarization_model": config.transcription.diarization_model,
        "results": results,
    }
    _write_json(temporary / "benchmark.json", report)
    root.mkdir(parents=True, exist_ok=True)
    temporary.replace(final_dir)
    return final_dir, report
