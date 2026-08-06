from meeting_transcriber.state import StateStore


def test_file_must_remain_stable(tmp_path):
    source = tmp_path / "audio.webm"
    source.write_bytes(b"a")
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        assert not state.observe(source, stable_seconds=10, now=100)
        assert not state.observe(source, stable_seconds=10, now=109)
        assert state.observe(source, stable_seconds=10, now=110)
        source.write_bytes(b"changed")
        assert not state.observe(source, stable_seconds=10, now=111)
    finally:
        state.close()


def test_hash_deduplication_and_retry(tmp_path):
    one = tmp_path / "one.webm"
    two = tmp_path / "two.webm"
    one.write_bytes(b"same")
    two.write_bytes(b"same")
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        assert state.enqueue(one, "hash", {"duration": 1}, now=1)
        assert not state.enqueue(two, "hash", {"duration": 1}, now=2)
        job = state.claim(now=3)
        assert job is not None and job.attempts == 1
        state.fail(job, "temporary", (10,), now=4)
        assert state.claim(now=13) is None
        retry = state.claim(now=14)
        assert retry is not None and retry.attempts == 2
        state.complete(retry.job_id, tmp_path / "output", now=15)
        assert state.stats()["complete"] == 1
    finally:
        state.close()
