import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from meeting_transcriber.config import TranscriptionConfig
from meeting_transcriber.whisperx_engine import WhisperXEngine


def _config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model="test",
        device="cuda",
        compute_type="int8_float16",
        batch_size=8,
        language="en",
        min_speakers=None,
        max_speakers=None,
        diarization_model="required",
        hf_token_env="HF_TOKEN",
    )


def _fake_modules(asr):
    align_model = MagicMock()
    fake = types.ModuleType("whisperx")
    fake.load_audio = MagicMock(return_value=[0])
    fake.load_model = MagicMock(return_value=asr)
    fake.load_align_model = MagicMock(return_value=(align_model, {}))
    fake.align = MagicMock(
        return_value={"segments": [{"start": 0, "end": 1, "text": "hello"}]}
    )
    fake.assign_word_speakers = MagicMock(
        return_value={
            "segments": [
                {"start": 0, "end": 1, "text": "hello", "speaker": "SPEAKER_00"}
            ]
        }
    )
    diarize_module = types.ModuleType("whisperx.diarize")
    diarizer = MagicMock(return_value="speaker timeline")
    diarize_module.DiarizationPipeline = MagicMock(return_value=diarizer)
    return fake, diarize_module, diarizer


def test_diarization_is_always_run_without_decoding_audio_twice(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    asr = MagicMock()
    asr.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }
    fake, diarize_module, diarizer = _fake_modules(asr)

    with patch.dict(
        sys.modules, {"whisperx": fake, "whisperx.diarize": diarize_module}
    ):
        result = WhisperXEngine(_config()).transcribe(Path("audio.webm"))

    diarize_module.DiarizationPipeline.assert_called_once()
    diarizer.assert_called_once_with([0], min_speakers=None, max_speakers=None)
    fake.assign_word_speakers.assert_called_once_with(
        "speaker timeline", fake.align.return_value, fill_nearest=True
    )
    assert result["speakers"] == ["SPEAKER_00"]
    assert result["effective_batch_size"] == 8
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "0"
    assert set(result["timings_seconds"]) == {
        "audio_decode",
        "asr_model_load",
        "asr",
        "alignment_model_load",
        "alignment",
        "diarization_model_load",
        "diarization_and_assignment",
        "total",
    }


def test_cuda_memory_failure_halves_batch_size_and_retries(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    asr = MagicMock()
    successful = {
        "language": "en",
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }
    asr.transcribe.side_effect = [RuntimeError("CUDA out of memory"), successful]
    fake, diarize_module, _ = _fake_modules(asr)

    with patch.dict(
        sys.modules, {"whisperx": fake, "whisperx.diarize": diarize_module}
    ):
        result = WhisperXEngine(_config()).transcribe(Path("audio.webm"))

    assert asr.transcribe.call_args_list == [
        call([0], batch_size=8),
        call([0], batch_size=4),
    ]
    assert result["effective_batch_size"] == 4
