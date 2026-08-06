FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYANNOTE_METRICS_ENABLED=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml readme.md license.md ./

RUN python3 -m pip install --break-system-packages \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    && python3 -m pip install --break-system-packages \
      "PyYAML>=6.0.2,<7" \
      "whisperx @ git+https://github.com/m-bain/whisperX.git@2cfd7b7c5c7bba144954364db747319b50e8232b"

COPY meeting_transcriber ./meeting_transcriber
COPY main.py config.yaml config.container.yaml ./

RUN python3 -m pip install --break-system-packages --no-deps .

RUN useradd --create-home --uid 1000 worker \
    && mkdir -p /data/input /data/output /data/state /cache \
    && chown -R worker:worker /data /cache

USER worker
ENTRYPOINT ["meeting-transcriber", "--config", "/app/config.yaml"]
CMD ["worker"]
