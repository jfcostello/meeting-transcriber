# Meeting Transcriber

A private, durable GPU worker for meeting audio. It watches a queue, waits for files to finish copying, deduplicates by SHA-256, runs WhisperX alignment and **mandatory speaker diarization**, and emits a structured bundle for later Logseq insertion.

No cloud transcription or summarization provider is included. A local summarizer will be selected separately.

## Architecture

- The GPU PC runs this worker and owns transcription compute.
- NDO remains the Logseq writer and retains the graph encryption key.
- Every completed job contains the original audio, a diarized JSON transcript, readable Markdown, source timestamps, and a Logseq integration placeholder.
- Originals are never moved or deleted.
- Queue state is durable in SQLite. Interrupted jobs return to the queue, failures retry with backoff, and identical audio is processed once.

## Required setup

The production target is an NVIDIA GPU with a current driver capable of CUDA 12.8. The container includes the remaining CUDA/cuDNN runtime and a pinned WhisperX revision.

WhisperX diarization requires a Hugging Face read token and acceptance of the terms for [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1).

```bash
cp .env.example .env
# Set HF_TOKEN and, later, WHISPER_MODEL.
docker compose -f compose.gpu.yaml build
docker compose -f compose.gpu.yaml run --rm transcriber doctor --runtime
docker compose -f compose.gpu.yaml up -d
```

The model is intentionally unset. The worker will refuse to process audio until `WHISPER_MODEL` is explicitly selected.

## Model selection

Do not select a model from generic benchmarks. Run the same representative private meeting through the current shortlist:

```bash
docker compose -f compose.gpu.yaml run --rm transcriber benchmark /data/input/example.webm \
  --model distil-whisper/distil-large-v3.5-ct2 \
  --model large-v3-turbo \
  --model large-v3
```

This writes a diarized transcript for each model plus stage timings and the processing-speed multiple under `data/state/benchmarks`. Compare names, specialist terms, omissions and hallucinations as well as speed.

- `distil-large-v3.5` is the newest fast English candidate.
- `large-v3-turbo` is the leading multilingual and long-form candidate.
- `large-v3` is the slower accuracy baseline.

The worker automatically halves the configured batch size and retries if the 8 GB GPU runs out of memory. The successful batch size and every processing-stage duration are recorded in `job.json`.

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

`doctor` performs static checks without downloading or loading a model. `doctor --runtime` additionally requires CUDA, `HF_TOKEN`, and `WHISPER_MODEL`.

## Completed bundle

```text
data/output/<sha256>/
├── audio.<original-extension>
├── job.json
├── transcript.json
└── transcript.md
```

`job.json` is the automation contract. It records the source hash, embedded or filesystem timestamp source, duration, transcription/diarization versions, summary state and eventual Logseq match.

## Privacy

- Audio and transcript processing are local.
- No external LLM client is shipped.
- The Hugging Face token is used only to obtain the accepted diarization model.
- Hugging Face and pyannote telemetry are forcibly disabled.
- After models are cached, run with `docker compose -f compose.gpu.yaml -f compose.offline.yaml up -d` to remove network access from the worker completely.
- The Logseq E2EE key does not belong on the GPU PC.

## Development

The lightweight tests do not load WhisperX or require a GPU:

```bash
python -m pip install PyYAML pytest
pytest
```
