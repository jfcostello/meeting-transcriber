from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .benchmark import RTX_3070_CANDIDATES, benchmark_models, parse_candidate
from .config import AppConfig, ConfigError, load_config
from .media import probe_media, sha256_file
from .pipeline import process_job
from .state import Job, StateStore
from .worker import Worker


def _nvidia_smi(device_index: int) -> dict[str, str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    fields = (
        "driver_version",
        "persistence_mode",
        "memory.total",
        "power.limit",
        "temperature.gpu",
    )
    try:
        values = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    parts = [item.strip() for item in values.splitlines()[0].split(",")]
    return dict(zip(fields, parts, strict=False)) if len(parts) == len(fields) else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeting-transcriber")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("worker", help="watch and process continuously")
    subparsers.add_parser("once", help="run one discovery and processing cycle")
    process = subparsers.add_parser("process", help="process one file immediately")
    process.add_argument("path")
    benchmark = subparsers.add_parser(
        "benchmark", help="compare full diarized results from multiple ASR models"
    )
    benchmark.add_argument("path")
    benchmark_selection = benchmark.add_mutually_exclusive_group(required=True)
    benchmark_selection.add_argument(
        "--model",
        action="append",
        help="model for the configured backend; repeat to compare models",
    )
    benchmark_selection.add_argument(
        "--candidate",
        action="append",
        help="BACKEND=MODEL; repeat to compare backends",
    )
    benchmark_selection.add_argument(
        "--preset",
        choices=["rtx3070"],
        help="benchmark the full tuned RTX 3070 candidate set",
    )
    benchmark.add_argument(
        "--output", help="benchmark directory; defaults under the state directory"
    )
    doctor = subparsers.add_parser(
        "doctor", help="validate configuration and dependencies"
    )
    doctor.add_argument(
        "--runtime", action="store_true", help="also require model credentials and CUDA"
    )
    subparsers.add_parser("status", help="show durable queue counts")
    return parser


def _doctor(config: AppConfig, runtime: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "config": "ok",
        "ffprobe": shutil.which("ffprobe") or "missing",
        "input": str(config.paths.input),
        "output": str(config.paths.output),
        "state": str(config.paths.state),
        "diarization": "mandatory",
    }
    if result["ffprobe"] == "missing":
        raise RuntimeError("ffprobe is not installed")
    if runtime:
        config.validate_runtime()
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        result["cuda"] = {
            "available": True,
            "device": torch.cuda.get_device_name(config.transcription.device_index),
            "device_index": config.transcription.device_index,
            "compute_capability": ".".join(
                str(value)
                for value in torch.cuda.get_device_capability(
                    config.transcription.device_index
                )
            ),
            "vram_bytes": torch.cuda.get_device_properties(
                config.transcription.device_index
            ).total_memory,
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "nvidia_smi": _nvidia_smi(config.transcription.device_index),
        }
        result["hf_token"] = (
            "present" if os.getenv(config.transcription.hf_token_env) else "missing"
        )
        result["backend"] = config.transcription.backend
        result["model"] = config.transcription.model
    return result


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            print(json.dumps(_doctor(config, args.runtime), indent=2))
            return 0
        if args.command in {"worker", "once", "process"}:
            config.validate_runtime()
        if args.command == "worker":
            worker = Worker(config)
            try:
                worker.run_forever()
            finally:
                worker.close()
            return 0
        if args.command == "once":
            worker = Worker(config)
            try:
                print(json.dumps(worker.run_once(), indent=2))
            finally:
                worker.close()
            return 0
        if args.command == "process":
            source = Path(args.path).expanduser().resolve()
            digest = sha256_file(source)
            metadata = probe_media(source, config.timezone).to_dict()
            output = process_job(Job(digest, str(source), metadata, 1), config)
            print(json.dumps({"job_id": digest, "output": str(output)}, indent=2))
            return 0
        if args.command == "benchmark":
            if args.preset == "rtx3070":
                candidates = list(RTX_3070_CANDIDATES)
            elif args.candidate:
                candidates = [parse_candidate(value) for value in args.candidate]
            else:
                candidates = [
                    parse_candidate(value, config.transcription.backend)
                    for value in args.model
                ]
            output_dir, report = benchmark_models(
                Path(args.path),
                config,
                candidates,
                Path(args.output).expanduser().resolve() if args.output else None,
            )
            print(
                json.dumps(
                    {"output": str(output_dir), "results": report["results"]},
                    indent=2,
                )
            )
            return 0
        if args.command == "status":
            config.paths.state.mkdir(parents=True, exist_ok=True)
            state = StateStore(config.paths.state / "worker.sqlite3")
            try:
                print(json.dumps(state.stats(), indent=2))
            finally:
                state.close()
            return 0
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1


def main() -> int:
    return run()
