from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

_FILENAME_TIMESTAMPS = (
    re.compile(
        r"Audio-(\d{4}-\d{2}-\d{2})[ _](\d{2})[:.-](\d{2})[:.-](\d{2})", re.IGNORECASE
    ),
    re.compile(r"(\d{4}-\d{2}-\d{2})[ _-](\d{2})[-.](\d{2})[-.](\d{2})"),
)


@dataclass(frozen=True)
class MediaMetadata:
    duration_seconds: float | None
    recorded_at: str
    timestamp_source: str
    format_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_supported(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _filename_timestamp(path: Path, timezone_name: str) -> str | None:
    for pattern in _FILENAME_TIMESTAMPS:
        if match := pattern.search(path.stem):
            value = (
                f"{match.group(1)}T{match.group(2)}:{match.group(3)}:{match.group(4)}"
            )
            parsed = datetime.fromisoformat(value).replace(
                tzinfo=ZoneInfo(timezone_name)
            )
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return None


def probe_media(path: Path, timezone_name: str = "America/Toronto") -> MediaMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name:format_tags=creation_time:stream_tags=creation_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        raw = json.loads(subprocess.check_output(command, text=True, timeout=60))
    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(f"ffprobe failed for {path}: {error}") from error

    format_data = raw.get("format", {})
    duration_raw = format_data.get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None

    candidates: list[tuple[str, str | None]] = [
        (
            "embedded.format.creation_time",
            _iso((format_data.get("tags") or {}).get("creation_time", "")),
        ),
    ]
    for stream in raw.get("streams", []):
        candidates.append(
            (
                "embedded.stream.creation_time",
                _iso((stream.get("tags") or {}).get("creation_time", "")),
            )
        )
    candidates.append(("filename", _filename_timestamp(path, timezone_name)))

    stat = path.stat()
    birthtime = getattr(stat, "st_birthtime", None)
    if birthtime:
        candidates.append(
            (
                "filesystem.birthtime",
                datetime.fromtimestamp(birthtime, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            )
        )
    candidates.append(
        (
            "filesystem.mtime",
            datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
    )
    source, recorded_at = next((source, value) for source, value in candidates if value)
    return MediaMetadata(
        duration_seconds=duration,
        recorded_at=recorded_at,
        timestamp_source=source,
        format_name=format_data.get("format_name"),
    )
