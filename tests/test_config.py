from pathlib import Path

import pytest

from meeting_transcriber.config import ConfigError, load_config


def write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
paths:
  input: input
  output: output
  state: state
worker:
  stable_seconds: 2
transcription:
  model: ${{TRANSCRIPTION_MODEL}}
  device: cuda
  diarization_model: pyannote/speaker-diarization-community-1
  hf_token_env: HF_TOKEN
summary:
  enabled: false
{extra}
""",
        encoding="utf-8",
    )
    return path


def test_runtime_requires_model_and_diarization_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSCRIPTION_MODEL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="TRANSCRIPTION_MODEL"):
        config.validate_runtime()
    monkeypatch.setenv("TRANSCRIPTION_MODEL", "selected-later")
    config = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="HF_TOKEN"):
        config.validate_runtime()


def test_runtime_accepts_explicit_gpu_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_MODEL", "selected-later")
    monkeypatch.setenv("HF_TOKEN", "secret")
    config = load_config(write_config(tmp_path))
    config.validate_runtime()
    assert (
        config.transcription.diarization_model
        == "pyannote/speaker-diarization-community-1"
    )


def test_paths_cannot_overlap(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "paths:\n  input: data\n  output: data/output\n  state: state\ntranscription:\n  diarization_model: required\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="inside"):
        load_config(path)


def test_boolean_environment_values_are_parsed_not_coerced(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_TF32", "false")
    config = load_config(
        write_config(
            tmp_path,
            "transcription:\n  allow_tf32: ${ALLOW_TF32}\n  diarization_model: required\n",
        )
    )
    assert config.transcription.allow_tf32 is False


def test_invalid_backend_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="backend"):
        load_config(
            write_config(
                tmp_path,
                "transcription:\n  backend: cloud\n  diarization_model: required\n",
            )
        )
