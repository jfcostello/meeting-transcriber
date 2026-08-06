import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from meeting_transcriber.config import TranscriptionConfig
from meeting_transcriber.parakeet_engine import ParakeetEngine


class FakeTensor:
    def detach(self):
        return self

    def cpu(self):
        return self


class FakeInputs(dict):
    def to(self, **kwargs):
        return self


def _config() -> TranscriptionConfig:
    return TranscriptionConfig(
        backend="parakeet",
        model="nvidia/parakeet-tdt-0.6b-v3",
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


def test_parakeet_uses_native_timestamps_then_mandatory_diarization(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    torch_module = types.ModuleType("torch")
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    torch_module.inference_mode = nullcontext
    torch_module.set_num_threads = MagicMock()
    torch_module.set_float32_matmul_precision = MagicMock()
    torch_module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(allow_tf32=False)
    )
    torch_module.cuda = MagicMock()
    torch_module.cuda.is_available.return_value = False

    processor = MagicMock(return_value=FakeInputs())
    processor.decode.return_value = (
        ["Hello world."],
        [
            [
                {"token": " Hello", "start": 0.0, "end": 0.2},
                {"token": " world", "start": 0.3, "end": 0.5},
                {"token": ".", "start": 0.5, "end": 0.5},
            ]
        ],
    )
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = processor

    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = types.SimpleNamespace(
        sequences=FakeTensor(), durations=FakeTensor()
    )
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForTDT = model_class
    transformers.AutoProcessor = processor_class

    whisperx = types.ModuleType("whisperx")
    whisperx.load_audio = MagicMock(return_value=[0.0])

    def assign_speakers(diarization, result, fill_nearest):
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
        return result

    whisperx.assign_word_speakers = MagicMock(side_effect=assign_speakers)
    diarize_module = types.ModuleType("whisperx.diarize")
    diarizer = MagicMock(return_value="timeline")
    diarize_module.DiarizationPipeline = MagicMock(return_value=diarizer)

    with patch.dict(
        sys.modules,
        {
            "torch": torch_module,
            "transformers": transformers,
            "whisperx": whisperx,
            "whisperx.diarize": diarize_module,
        },
    ):
        result = ParakeetEngine(_config()).transcribe(Path("meeting.webm"))

    assert result["backend"] == "parakeet"
    assert result["segments"][0]["text"] == "Hello world."
    assert result["speakers"] == ["SPEAKER_00"]
    model_class.from_pretrained.assert_called_once()
    diarizer.assert_called_once_with([0.0], min_speakers=None, max_speakers=None)


def test_parakeet_remembers_smaller_chunk_after_cuda_oom(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private")
    torch_module = types.ModuleType("torch")
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    torch_module.inference_mode = nullcontext
    torch_module.set_num_threads = MagicMock()
    torch_module.set_float32_matmul_precision = MagicMock()
    torch_module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(allow_tf32=False)
    )
    torch_module.cuda = MagicMock()
    torch_module.cuda.is_available.return_value = False

    processor = MagicMock(return_value=FakeInputs())
    processor.decode.return_value = (
        ["Hello."],
        [[{"token": " Hello.", "start": 0.0, "end": 0.2}]],
    )
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = processor
    generated = types.SimpleNamespace(sequences=FakeTensor(), durations=FakeTensor())
    model = MagicMock()
    model.to.return_value = model
    model.generate.side_effect = [
        RuntimeError("CUDA out of memory"),
        generated,
        generated,
    ]
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForTDT = model_class
    transformers.AutoProcessor = processor_class

    whisperx = types.ModuleType("whisperx")
    whisperx.load_audio = MagicMock(return_value=[0.0])

    def assign_speakers(diarization, result, fill_nearest):
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
        return result

    whisperx.assign_word_speakers = MagicMock(side_effect=assign_speakers)
    diarize_module = types.ModuleType("whisperx.diarize")
    diarize_module.DiarizationPipeline = MagicMock(
        return_value=MagicMock(return_value="timeline")
    )

    with patch.dict(
        sys.modules,
        {
            "torch": torch_module,
            "transformers": transformers,
            "whisperx": whisperx,
            "whisperx.diarize": diarize_module,
        },
    ):
        engine = ParakeetEngine(_config())
        first = engine.transcribe(Path("meeting.webm"))
        second = engine.transcribe(Path("next.webm"))

    assert first["effective_chunk_seconds"] == 300
    assert second["effective_chunk_seconds"] == 300
    assert processor.call_args_list == [
        call([0.0], sampling_rate=16000, return_tensors="pt"),
        call([0.0], sampling_rate=16000, return_tensors="pt"),
        call([0.0], sampling_rate=16000, return_tensors="pt"),
    ]
