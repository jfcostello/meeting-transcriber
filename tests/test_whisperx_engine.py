import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from meeting_transcriber.config import TranscriptionConfig
from meeting_transcriber.whisperx_engine import WhisperXEngine


def test_diarization_is_always_run(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    asr = MagicMock()
    asr.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }
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
    config = TranscriptionConfig(
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
    with patch.dict(
        sys.modules, {"whisperx": fake, "whisperx.diarize": diarize_module}
    ):
        result = WhisperXEngine(config).transcribe(Path("audio.webm"))
    diarize_module.DiarizationPipeline.assert_called_once()
    diarizer.assert_called_once()
    fake.assign_word_speakers.assert_called_once_with(
        "speaker timeline", fake.align.return_value, fill_nearest=True
    )
    assert result["speakers"] == ["SPEAKER_00"]
