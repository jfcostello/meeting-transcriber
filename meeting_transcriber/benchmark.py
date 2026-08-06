from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, TranscriptionConfig
from .engines import TranscriptionEngine, create_engine
from .media import probe_media, sha256_file
from .render import transcript_markdown

EngineFactory = Callable[[TranscriptionConfig], TranscriptionEngine]


class GpuMemorySampler:
    """Samples total GPU memory so CTranslate2 and PyTorch are both visible."""

    def __init__(self, device_index: int, interval_seconds: float = 0.25):
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self) -> int | None:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            return int(value.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if (used := self._read()) is not None:
                self.peak_mib = max(self.peak_mib or used, used)

    def start(self) -> None:
        self.baseline_mib = self._read()
        self.peak_mib = self.baseline_mib
        if self.baseline_mib is not None:
            self._thread = threading.Thread(
                target=self._sample_until_stopped,
                name="gpu-memory-sampler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> dict[str, int | None]:
        if (used := self._read()) is not None:
            self.peak_mib = max(self.peak_mib or used, used)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        delta = (
            max(0, self.peak_mib - self.baseline_mib)
            if self.peak_mib is not None and self.baseline_mib is not None
            else None
        )
        return {
            "baseline_gpu_memory_mib": self.baseline_mib,
            "peak_gpu_memory_mib": self.peak_mib,
            "peak_job_gpu_memory_mib": delta,
        }


@dataclass(frozen=True)
class BenchmarkCandidate:
    backend: str
    model: str
    compute_type: str | None = None
    parakeet_dtype: str | None = None
    batch_size: int | None = None
    beam_size: int | None = None

    @property
    def label(self) -> str:
        precision = (
            self.parakeet_dtype if self.backend == "parakeet" else self.compute_type
        )
        return "-".join(
            str(value)
            for value in (self.backend, self.model, precision, self.beam_size)
            if value is not None
        )

    def apply(self, config: TranscriptionConfig) -> TranscriptionConfig:
        values: dict[str, Any] = {"backend": self.backend, "model": self.model}
        for name in ("compute_type", "parakeet_dtype", "batch_size", "beam_size"):
            if (value := getattr(self, name)) is not None:
                values[name] = value
        return replace(config, **values)


RTX_3070_CANDIDATES = (
    BenchmarkCandidate(
        "parakeet", "nvidia/parakeet-tdt-0.6b-v3", parakeet_dtype="float16"
    ),
    BenchmarkCandidate(
        "whisperx",
        "distil-whisper/distil-large-v3.5-ct2",
        compute_type="float16",
        batch_size=8,
        beam_size=1,
    ),
    BenchmarkCandidate(
        "whisperx",
        "large-v3-turbo",
        compute_type="float16",
        batch_size=8,
        beam_size=1,
    ),
    BenchmarkCandidate(
        "whisperx",
        "large-v3-turbo",
        compute_type="int8_float16",
        batch_size=8,
        beam_size=1,
    ),
    BenchmarkCandidate(
        "whisperx",
        "large-v3",
        compute_type="int8_float16",
        batch_size=4,
        beam_size=1,
    ),
)


def parse_candidate(
    value: str, default_backend: str = "whisperx"
) -> BenchmarkCandidate:
    backend, separator, model = value.strip().partition("=")
    if not separator:
        backend, model = default_backend, backend
    backend = backend.strip().lower()
    model = model.strip()
    if backend not in {"whisperx", "parakeet"} or not model:
        raise ValueError("candidate must be BACKEND=MODEL")
    return BenchmarkCandidate(backend, model)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _model_directory(label: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9._-]+", "-", label).strip("-.") or "model"
    fingerprint = hashlib.sha256(label.encode()).hexdigest()[:8]
    return f"{readable[:72]}-{fingerprint}"


def benchmark_models(
    source: Path,
    config: AppConfig,
    candidates: list[BenchmarkCandidate | str],
    output_root: Path | None = None,
    engine_factory: EngineFactory = create_engine,
) -> tuple[Path, dict[str, Any]]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    parsed = [
        parse_candidate(item, config.transcription.backend)
        if isinstance(item, str)
        else item
        for item in candidates
    ]
    unique_candidates = list(dict.fromkeys(parsed))
    if not unique_candidates:
        raise ValueError("at least one benchmark candidate is required")

    metadata = probe_media(source, config.timezone)
    source_hash = sha256_file(source)
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    root = (output_root or config.paths.state / "benchmarks").resolve()
    final_dir = root / f"{run_id}-{source_hash[:12]}"
    temporary = root / f".{run_id}-{source_hash[:12]}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    for candidate in unique_candidates:
        transcription_config = candidate.apply(config.transcription)
        replace(config, transcription=transcription_config).validate_runtime()
        memory = GpuMemorySampler(transcription_config.device_index)
        memory.start()
        try:
            transcription = engine_factory(transcription_config).transcribe(source)
        finally:
            memory_result = memory.stop()
        model_dir = temporary / _model_directory(candidate.label)
        model_dir.mkdir()
        transcript = {
            "schema_version": 1,
            "backend": candidate.backend,
            "model": candidate.model,
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
                "backend": candidate.backend,
                "model": candidate.model,
                "directory": model_dir.name,
                "compute_type": transcription_config.compute_type,
                "parakeet_dtype": transcription_config.parakeet_dtype,
                "beam_size": transcription_config.beam_size,
                "language": transcription["language"],
                "speakers": len(transcription["speakers"]),
                "effective_batch_size": transcription.get("effective_batch_size"),
                "effective_chunk_seconds": transcription.get("effective_chunk_seconds"),
                **memory_result,
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
