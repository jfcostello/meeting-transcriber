import json
from pathlib import Path

from meeting_transcriber.config import load_config
from meeting_transcriber.pipeline import process_job
from meeting_transcriber.state import Job


class FakeEngine:
    def transcribe(self, source: Path):
        return {
            "engine": "whisperx",
            "model": "test",
            "language": "en",
            "diarization_model": "mandatory",
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Hello"},
                {"start": 2.1, "end": 3.0, "speaker": "SPEAKER_01", "text": "Hi"},
            ],
            "utterances": [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Hello"},
                {"start": 2.1, "end": 3.0, "speaker": "SPEAKER_01", "text": "Hi"},
            ],
        }


def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
paths:
  input: {tmp_path / "input"}
  output: {tmp_path / "output"}
  state: {tmp_path / "state"}
transcription:
  model: test
  device: cuda
  diarization_model: mandatory
summary:
  enabled: false
""",
        encoding="utf-8",
    )
    return path


def test_bundle_is_structured_and_idempotent(tmp_path):
    source = tmp_path / "source.webm"
    source.write_bytes(b"audio")
    config = load_config(config_file(tmp_path))
    job = Job(
        "abc123",
        str(source),
        {"recorded_at": "2026-08-06T12:00:00Z", "duration_seconds": 3},
        1,
    )
    output = process_job(job, config, FakeEngine())
    assert (output / "audio.webm").read_bytes() == b"audio"
    manifest = json.loads((output / "job.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["summary"]["status"] == "pending_configuration"
    assert manifest["logseq"]["status"] == "pending"
    assert "SPEAKER_00" in (output / "transcript.md").read_text()
    assert process_job(job, config, FakeEngine()) == output
