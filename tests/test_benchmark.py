import json
from pathlib import Path
from unittest.mock import patch

from meeting_transcriber.benchmark import (
    RTX_3070_CANDIDATES,
    benchmark_models,
    parse_candidate,
)
from meeting_transcriber.config import load_config


class FakeEngine:
    def __init__(self, config):
        self.config = config

    def transcribe(self, source: Path):
        return {
            "language": "en",
            "speakers": ["SPEAKER_00"],
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "speaker": "SPEAKER_00",
                    "text": self.config.model,
                }
            ],
            "utterances": [
                {
                    "start": 0,
                    "end": 1,
                    "speaker": "SPEAKER_00",
                    "text": self.config.model,
                }
            ],
            "effective_batch_size": 4,
            "timings_seconds": {"asr": 5.0, "total": 10.0},
        }


def test_benchmark_writes_comparable_private_results(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  input: {tmp_path / "input"}
  output: {tmp_path / "output"}
  state: {tmp_path / "state"}
transcription:
  model:
  device: cuda
  diarization_model: mandatory
""",
        encoding="utf-8",
    )
    source = tmp_path / "meeting.webm"
    source.write_bytes(b"audio")
    ffprobe = json.dumps({"format": {"duration": "60"}, "streams": []})
    with patch("subprocess.check_output", return_value=ffprobe):
        output, report = benchmark_models(
            source,
            load_config(config_path),
            ["new-model", "baseline", "new-model"],
            engine_factory=FakeEngine,
        )

    assert len(report["results"]) == 2
    assert report["results"][0]["realtime_multiple"] == 6.0
    assert (output / "benchmark.json").is_file()
    for result in report["results"]:
        transcript = output / result["directory"] / "transcript.md"
        assert result["model"] in transcript.read_text(encoding="utf-8")


def test_mixed_backend_candidates_and_3070_preset_are_explicit():
    parakeet = parse_candidate("parakeet=nvidia/parakeet-tdt-0.6b-v3")
    whisper = parse_candidate("large-v3-turbo")
    assert parakeet.backend == "parakeet"
    assert whisper.backend == "whisperx"
    assert {candidate.backend for candidate in RTX_3070_CANDIDATES} == {
        "parakeet",
        "whisperx",
    }
    assert all(candidate.beam_size in {None, 1} for candidate in RTX_3070_CANDIDATES)
