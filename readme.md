# Meeting Transcriber

A private GPU worker for meeting audio. It watches a durable queue, waits for files to finish copying, deduplicates identical audio, transcribes locally, applies mandatory speaker diarization, and emits a structured bundle for later Logseq insertion.

No cloud transcription or summarization provider is included.

## Architecture

- The RTX 3070 PC owns transcription compute.
- NDO remains the Logseq writer and retains the graph encryption key.
- Every completed job contains the original audio, diarized JSON and Markdown transcripts, source timestamps, and a Logseq integration placeholder.
- Originals are never moved or deleted.
- Queue state is durable in SQLite. Interrupted jobs return to the queue, failures retry with backoff, and identical audio is processed once.

## Transcription backends

Both backends use the same mandatory local `pyannote/speaker-diarization-community-1` pass.

- `parakeet`: NVIDIA Parakeet TDT 0.6B v3 through native Transformers. It supplies punctuation and word timestamps directly, so Whisper alignment is skipped.
- `whisperx`: batched faster-whisper ASR, WhisperX word alignment, then diarization.

Parakeet is an actual production option, not a benchmark shim. It does not install the much larger NeMo framework.

## Required setup

The production target is Linux, an NVIDIA RTX 3070 8 GB, and a current NVIDIA driver capable of the CUDA 12.8 runtime in the container.

WhisperX diarization requires a Hugging Face read token and acceptance of the terms for [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1).

```bash
cp .env.example .env
# Set HF_TOKEN. Leave the model unset until the benchmark is complete.
docker compose -f compose.gpu.yaml build
docker compose -f compose.gpu.yaml run --rm transcriber doctor --runtime
docker compose -f compose.gpu.yaml up -d
```

Set `TRANSCRIPTION_BACKEND` to `parakeet` or `whisperx`, and set `TRANSCRIPTION_MODEL` to the selected model. The worker refuses to process audio while the model is blank.

## RTX 3070 benchmark

Use a representative private meeting and run the complete tuned shortlist:

```bash
docker compose -f compose.gpu.yaml run --rm transcriber benchmark \
  /data/input/example.webm --preset rtx3070
```

The preset compares:

- Parakeet TDT 0.6B v3 in FP16
- Distil-Whisper large v3.5 in FP16
- Whisper large-v3-turbo in FP16 and INT8/FP16
- Whisper large-v3 in INT8/FP16 as the accuracy baseline

Each candidate produces its own fully diarized transcript. `benchmark.json` records total and per-stage time, processing-speed multiple, effective batch or chunk size, and total peak GPU memory measured through `nvidia-smi`. Compare names, specialist terms, omissions, hallucinations, speaker boundaries, speed, and VRAM before selecting the winner.

Custom cross-backend comparisons are also supported:

```bash
meeting-transcriber --config config.yaml benchmark meeting.webm \
  --candidate parakeet=nvidia/parakeet-tdt-0.6b-v3 \
  --candidate whisperx=large-v3-turbo
```

## RTX 3070 tuning already applied

- ASR, alignment, and diarization run sequentially and release VRAM between stages.
- Audio is decoded once per job and reused by every stage.
- Parakeet uses FP16 and PyTorch SDPA. Long audio is processed in overlapping 300-second windows; an out-of-memory error halves the window down to 60 seconds. The learned safe size is reused for later jobs.
- Whisper uses batched inference, fast Silero VAD, a single greedy decoding path, and fixed English language detection. An out-of-memory error halves the batch size and the worker reuses the learned safe size.
- TF32 is enabled for remaining FP32 Ampere operations.
- Lazy CUDA module loading and fragmentation-resistant PyTorch and CTranslate2 allocators are enabled.
- CPU thread counts are bounded instead of oversubscribing the machine.
- The worker processes one GPU job at a time; concurrent jobs would reduce throughput on 8 GB VRAM.

The speed defaults deliberately trade some Whisper search accuracy for throughput: `beam_size: 1` replaces beam 5, and `language: en` skips detection. Change `language` to blank for multilingual Whisper audio. The benchmark exists to measure whether those trade-offs are acceptable on the actual recordings.

`doctor --runtime` reports the exact GPU, compute capability, VRAM, CUDA/PyTorch versions, NVIDIA driver, persistence mode, power limit, and temperature. On the GPU PC, enable NVIDIA persistence mode at the host level if it is off; this avoids repeated driver initialization without changing transcript quality.

## Paths

The compose deployment uses:

- `data/input`: watched recursively for audio/video
- `data/output/<sha256>`: immutable completed bundles
- `data/state/worker.sqlite3`: queue and retry state
- `data/cache`: downloaded Hugging Face and Torch models

Supported inputs: AAC, AVI, FLAC, M4A, MKV, MOV, MP3, MP4, OGA, OGG, OPUS, WAV and WEBM. WEBM and M4A cover Logseq recordings.

## Commands

```bash
meeting-transcriber --config config.yaml doctor
meeting-transcriber --config config.yaml doctor --runtime
meeting-transcriber --config config.yaml once
meeting-transcriber --config config.yaml worker
meeting-transcriber --config config.yaml process /path/to/recording.webm
meeting-transcriber --config config.yaml status
```

`doctor` performs static checks without downloading a model. `doctor --runtime` additionally requires CUDA, `HF_TOKEN`, and a selected model.

## Completed bundle

```text
data/output/<sha256>/
├── audio.<original-extension>
├── job.json
├── transcript.json
└── transcript.md
```

`job.json` is the automation contract. It records the source hash, embedded or filesystem timestamp source, duration, backend/model, diarization version, measured timings, summary state, and eventual Logseq match.

## Privacy

- Audio and transcript processing are local.
- No external LLM client is shipped.
- The Hugging Face token is used only to download the accepted models.
- Hugging Face and pyannote telemetry are forcibly disabled.
- After models are cached, run `docker compose -f compose.gpu.yaml -f compose.offline.yaml up -d` to remove network access from the worker completely.
- The Logseq encryption key does not belong on the GPU PC.

## Development

The lightweight tests do not load models or require a GPU:

```bash
python -m pip install PyYAML pytest ruff
ruff check .
ruff format --check .
pytest
```
