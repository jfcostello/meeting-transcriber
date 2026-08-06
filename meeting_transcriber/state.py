from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Job:
    job_id: str
    source_path: str
    metadata: dict[str, Any]
    attempts: int


class StateStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations (
              path TEXT PRIMARY KEY,
              size INTEGER NOT NULL,
              mtime_ns INTEGER NOT NULL,
              stable_since REAL NOT NULL,
              last_seen REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              source_path TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('queued','processing','complete','failed')),
              attempts INTEGER NOT NULL DEFAULT 0,
              next_attempt_at REAL NOT NULL DEFAULT 0,
              error TEXT,
              output_path TEXT,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
              path TEXT PRIMARY KEY,
              job_id TEXT NOT NULL REFERENCES jobs(job_id),
              discovered_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_ready_idx ON jobs(status, next_attempt_at, created_at);
            """
        )
        self.connection.execute(
            "UPDATE jobs SET status='queued', next_attempt_at=0, error='worker interrupted' WHERE status='processing'"
        )

    def observe(
        self, path: Path, stable_seconds: float, now: float | None = None
    ) -> bool:
        now = time.time() if now is None else now
        stat = path.stat()
        row = self.connection.execute(
            "SELECT * FROM observations WHERE path=?", (str(path),)
        ).fetchone()
        if (
            row is None
            or row["size"] != stat.st_size
            or row["mtime_ns"] != stat.st_mtime_ns
        ):
            self.connection.execute(
                "INSERT INTO observations(path,size,mtime_ns,stable_since,last_seen) VALUES(?,?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET size=excluded.size,mtime_ns=excluded.mtime_ns,stable_since=excluded.stable_since,last_seen=excluded.last_seen",
                (str(path), stat.st_size, stat.st_mtime_ns, now, now),
            )
            return False
        self.connection.execute(
            "UPDATE observations SET last_seen=? WHERE path=?", (now, str(path))
        )
        return now - row["stable_since"] >= stable_seconds

    def enqueue(
        self,
        source: Path,
        job_id: str,
        metadata: dict[str, Any],
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO jobs(job_id,source_path,metadata_json,status,created_at,updated_at) VALUES(?,?,?,'queued',?,?)",
            (job_id, str(source), encoded, now, now),
        )
        self.connection.execute(
            "INSERT INTO sources(path,job_id,discovered_at) VALUES(?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET job_id=excluded.job_id,discovered_at=excluded.discovered_at",
            (str(source), job_id, now),
        )
        return cursor.rowcount == 1

    def claim(self, now: float | None = None) -> Job | None:
        now = time.time() if now is None else now
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT * FROM jobs WHERE (status='queued' OR status='failed') AND next_attempt_at<=? ORDER BY created_at LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            attempts = row["attempts"] + 1
            self.connection.execute(
                "UPDATE jobs SET status='processing',attempts=?,updated_at=?,error=NULL WHERE job_id=?",
                (attempts, now, row["job_id"]),
            )
            self.connection.execute("COMMIT")
            return Job(
                row["job_id"],
                row["source_path"],
                json.loads(row["metadata_json"]),
                attempts,
            )
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def complete(
        self, job_id: str, output_path: Path, now: float | None = None
    ) -> None:
        now = time.time() if now is None else now
        self.connection.execute(
            "UPDATE jobs SET status='complete',output_path=?,updated_at=?,error=NULL WHERE job_id=?",
            (str(output_path), now, job_id),
        )

    def fail(
        self,
        job: Job,
        error: str,
        retry_delays: tuple[int, ...],
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        delay = retry_delays[min(job.attempts - 1, len(retry_delays) - 1)]
        self.connection.execute(
            "UPDATE jobs SET status='failed',next_attempt_at=?,updated_at=?,error=? WHERE job_id=?",
            (now + delay, now, error[:4000], job.job_id),
        )

    def stats(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status,COUNT(*) count FROM jobs GROUP BY status"
        ).fetchall()
        result = {"queued": 0, "processing": 0, "complete": 0, "failed": 0}
        result.update({row["status"]: row["count"] for row in rows})
        return result
