from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .render import transcript_markdown
from .state import Job
from .summarizer import LocalOpenAISummarizer
from .whisperx_engine import WhisperXEngine


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def process_job(
    job: Job, config: AppConfig, engine: WhisperXEngine | None = None
) -> Path:
    source = Path(job.source_path)
    if not source.is_file():
        raise RuntimeError(f"source file disappeared: {source}")
    final_dir = config.paths.output / job.job_id
    manifest_path = final_dir / "job.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("source", {}).get("sha256") == job.job_id
        ):
            return final_dir

    temporary = config.paths.output / f".{job.job_id}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    audio_name = f"audio{source.suffix.lower()}"
    shutil.copy2(source, temporary / audio_name)

    transcription = (engine or WhisperXEngine(config.transcription)).transcribe(source)
    transcript_data = {
        "schema_version": 1,
        "job_id": job.job_id,
        "language": transcription["language"],
        "speakers": transcription["speakers"],
        "utterances": transcription["utterances"],
        "segments": transcription["segments"],
    }
    _write_json(temporary / "transcript.json", transcript_data)
    (temporary / "transcript.md").write_text(
        transcript_markdown(transcript_data), encoding="utf-8"
    )

    summary_manifest: dict[str, Any]
    if config.summary.enabled:
        summarizer = LocalOpenAISummarizer(
            config.summary.base_url or "", config.summary.model or ""
        )
        summary = summarizer.summarize(
            (temporary / "transcript.md").read_text(encoding="utf-8")
        )
        (temporary / "summary.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
        summary_manifest = {
            "status": "complete",
            "provider": config.summary.provider,
            "model": config.summary.model,
            "path": "summary.md",
        }
    else:
        summary_manifest = {
            "status": "pending_configuration",
            "provider": config.summary.provider,
            "model": config.summary.model,
        }

    completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "job_id": job.job_id,
        "completed_at": completed_at,
        "source": {
            "original_path": str(source),
            "sha256": job.job_id,
            "size_bytes": source.stat().st_size,
            **job.metadata,
        },
        "audio": {"path": audio_name},
        "transcription": {
            "engine": transcription["engine"],
            "model": transcription["model"],
            "language": transcription["language"],
            "diarization_model": transcription["diarization_model"],
            "speakers": transcription["speakers"],
            "effective_batch_size": transcription.get("effective_batch_size"),
            "timings_seconds": transcription.get("timings_seconds", {}),
            "transcript_json": "transcript.json",
            "transcript_markdown": "transcript.md",
        },
        "summary": summary_manifest,
        "logseq": {
            "status": "pending",
            "matched_graph": None,
            "matched_appointment_uuid": None,
        },
    }
    _write_json(temporary / "job.json", manifest)
    config.paths.output.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temporary.replace(final_dir)
    return final_dir
