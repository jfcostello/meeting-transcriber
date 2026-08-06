from __future__ import annotations

import logging
import time

from .config import AppConfig
from .engines import create_engine
from .media import is_supported, probe_media, sha256_file
from .pipeline import process_job
from .state import StateStore

LOGGER = logging.getLogger(__name__)


class Worker:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.paths.input.mkdir(parents=True, exist_ok=True)
        self.config.paths.output.mkdir(parents=True, exist_ok=True)
        self.config.paths.state.mkdir(parents=True, exist_ok=True)
        self.state = StateStore(self.config.paths.state / "worker.sqlite3")
        self.engine = create_engine(self.config.transcription)

    def close(self) -> None:
        self.state.close()

    def discover(self) -> int:
        queued = 0
        for path in sorted(self.config.paths.input.rglob("*")):
            if not is_supported(path):
                continue
            try:
                if not self.state.observe(path, self.config.worker.stable_seconds):
                    continue
                digest = sha256_file(path)
                metadata = probe_media(path, self.config.timezone).to_dict()
                if self.state.enqueue(path, digest, metadata):
                    LOGGER.info("queued %s as %s", path, digest)
                    queued += 1
            except (FileNotFoundError, PermissionError, RuntimeError) as error:
                LOGGER.warning("discovery failed for %s: %s", path, error)
        return queued

    def process_ready(self) -> int:
        processed = 0
        while job := self.state.claim():
            try:
                output = process_job(job, self.config, self.engine)
                self.state.complete(job.job_id, output)
                LOGGER.info("completed %s", job.job_id)
                processed += 1
            except Exception as error:
                LOGGER.exception("job %s failed", job.job_id)
                self.state.fail(
                    job, str(error), self.config.worker.retry_delays_seconds
                )
        return processed

    def run_once(self) -> dict[str, int]:
        discovered = self.discover()
        processed = self.process_ready()
        return {"discovered": discovered, "processed": processed, **self.state.stats()}

    def run_forever(self) -> None:
        LOGGER.info("watching %s", self.config.paths.input)
        while True:
            self.run_once()
            time.sleep(self.config.worker.poll_seconds)
