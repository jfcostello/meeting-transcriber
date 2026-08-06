import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from meeting_transcriber.config import TranscriptionConfig
from meeting_transcriber.whisperx_engine import WhisperXEngine


def _config() -> TranscriptionConfig:
    return TranscriptionConfig(
        backend="whisperx",
        model="test",
        device="cuda",
        device_index=0,
        compute_type="int8_float16",
        parakeet_dtype="float16",
        parakeet_chunk_seconds=600,
        parakeet_overlap_seconds=2,
        batch_size=8,
        beam_size=1,
        chunk_size_seconds=30,
        vad_method="silero",
        cpu_threads=4,
        allow_tf32=True,
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
    torch_module = types.ModuleType("torch")
    torch_module.set_num_threads = MagicMock()
    torch_module.set_float32_matmul_precision = MagicMock()
    torch_module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(allow_tf32=False)
    )
    torch_module.cuda = MagicMock()
    torch_module.cuda.is_available.return_value = False
    return fake, diarize_module, diarizer, torch_module


def test_diarization_is_always_run_without_decoding_audio_twice(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    asr = MagicMock()
    asr.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }
    fake, diarize_module, diarizer, torch_module = _fake_modules(asr)

    with patch.dict(
        sys.modules,
        {
            "torch": torch_module,
            "whisperx": fake,
            "whisperx.diarize": diarize_module,
        },
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
    asr.transcribe.side_effect = [
        RuntimeError("CUDA out of memory"),
        successful,
        successful,
    ]
    fake, diarize_module, _, torch_module = _fake_modules(asr)

    with patch.dict(
        sys.modules,
        {
            "torch": torch_module,
            "whisperx": fake,
            "whisperx.diarize": diarize_module,
        },
    ):
        engine = WhisperXEngine(_config())
        result = engine.transcribe(Path("audio.webm"))
        second = engine.transcribe(Path("next.webm"))

    assert asr.transcribe.call_args_list == [
        call([0], batch_size=8, chunk_size=30),
        call([0], batch_size=4, chunk_size=30),
        call([0], batch_size=4, chunk_size=30),
    ]
    assert result["effective_batch_size"] == 4
    assert second["effective_batch_size"] == 4
