from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml


class ConfigError(ValueError):
    pass


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


@dataclass(frozen=True)
class PathsConfig:
    input: Path
    output: Path
    state: Path


@dataclass(frozen=True)
class WorkerConfig:
    poll_seconds: float
    stable_seconds: float
    retry_delays_seconds: tuple[int, ...]


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str
    device: str
    compute_type: str
    batch_size: int
    language: str | None
    min_speakers: int | None
    max_speakers: int | None
    diarization_model: str
    hf_token_env: str


@dataclass(frozen=True)
class SummaryConfig:
    enabled: bool
    provider: str
    base_url: str | None
    model: str | None


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    timezone: str
    worker: WorkerConfig
    transcription: TranscriptionConfig
    summary: SummaryConfig

    def validate_runtime(self) -> None:
        if not self.transcription.model:
            raise ConfigError("WHISPER_MODEL is not configured")
        token = os.getenv(self.transcription.hf_token_env, "")
        if not token:
            raise ConfigError(
                f"{self.transcription.hf_token_env} is required for mandatory diarization"
            )
        if self.transcription.device != "cuda":
            raise ConfigError("the production worker requires device=cuda")
        if self.summary.enabled and (
            not self.summary.base_url or not self.summary.model
        ):
            raise ConfigError("an enabled summary provider requires base_url and model")


def _path(base: Path, value: Any, name: str) -> Path:
    if not value:
        raise ConfigError(f"paths.{name} is required")
    result = Path(str(value)).expanduser()
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def _optional_int(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ConfigError(f"{name} must be positive")
    return parsed


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as error:
        raise ConfigError(f"config not found: {config_path}") from error
    raw = _expand(raw)
    paths_raw = raw.get("paths", {})
    worker_raw = raw.get("worker", {})
    transcription_raw = raw.get("transcription", {})
    summary_raw = raw.get("summary", {})
    base = config_path.parent

    paths = PathsConfig(
        input=_path(base, paths_raw.get("input"), "input"),
        output=_path(base, paths_raw.get("output"), "output"),
        state=_path(base, paths_raw.get("state"), "state"),
    )
    if len({paths.input, paths.output, paths.state}) != 3:
        raise ConfigError("input, output and state paths must be different")
    if paths.output.is_relative_to(paths.input) or paths.state.is_relative_to(
        paths.input
    ):
        raise ConfigError(
            "output and state paths cannot be inside the watched input path"
        )

    retry_values = worker_raw.get("retry_delays_seconds", [60, 300, 1800])
    if not isinstance(retry_values, list) or not retry_values:
        raise ConfigError("worker.retry_delays_seconds must be a non-empty list")
    worker = WorkerConfig(
        poll_seconds=max(1.0, float(worker_raw.get("poll_seconds", 5))),
        stable_seconds=max(1.0, float(worker_raw.get("stable_seconds", 30))),
        retry_delays_seconds=tuple(max(1, int(value)) for value in retry_values),
    )

    min_speakers = _optional_int(transcription_raw.get("min_speakers"), "min_speakers")
    max_speakers = _optional_int(transcription_raw.get("max_speakers"), "max_speakers")
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise ConfigError("min_speakers cannot exceed max_speakers")
    transcription = TranscriptionConfig(
        model=str(transcription_raw.get("model", "")).strip(),
        device=str(transcription_raw.get("device", "cuda")).strip(),
        compute_type=str(transcription_raw.get("compute_type", "int8_float16")).strip(),
        batch_size=max(1, int(transcription_raw.get("batch_size", 8))),
        language=(
            str(transcription_raw["language"]).strip()
            if transcription_raw.get("language")
            else None
        ),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarization_model=str(
            transcription_raw.get(
                "diarization_model", "pyannote/speaker-diarization-community-1"
            )
        ).strip(),
        hf_token_env=str(transcription_raw.get("hf_token_env", "HF_TOKEN")).strip(),
    )
    if not transcription.diarization_model:
        raise ConfigError("a diarization model is required")

    summary = SummaryConfig(
        enabled=bool(summary_raw.get("enabled", False)),
        provider=str(summary_raw.get("provider", "openai_compatible")).strip(),
        base_url=(
            str(summary_raw["base_url"]).rstrip("/")
            if summary_raw.get("base_url")
            else None
        ),
        model=(str(summary_raw["model"]).strip() if summary_raw.get("model") else None),
    )
    timezone_name = str(raw.get("timezone", "America/Toronto")).strip()
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ConfigError(f"unknown timezone: {timezone_name}") from error
    return AppConfig(
        paths=paths,
        timezone=timezone_name,
        worker=worker,
        transcription=transcription,
        summary=summary,
    )
