import json
from unittest.mock import patch

from meeting_transcriber.media import is_supported, probe_media, sha256_file


def test_hash_and_webm_support(tmp_path):
    path = tmp_path / "recording.webm"
    path.write_bytes(b"private audio")
    assert is_supported(path)
    assert (
        sha256_file(path)
        == "302c2335ee1f5463ebf564f299622e3c9481d8fbc9b6ce9f92ca883de69816b3"
    )


def test_embedded_timestamp_wins(tmp_path):
    path = tmp_path / "Audio-2026-08-06 12:30:00.m4a"
    path.write_bytes(b"audio")
    payload = json.dumps(
        {
            "format": {
                "duration": "61.5",
                "format_name": "mov,mp4,m4a",
                "tags": {"creation_time": "2026-08-06T10:00:00Z"},
            },
            "streams": [],
        }
    )
    with patch("subprocess.check_output", return_value=payload):
        metadata = probe_media(path)
    assert metadata.recorded_at == "2026-08-06T10:00:00Z"
    assert metadata.timestamp_source == "embedded.format.creation_time"
    assert metadata.duration_seconds == 61.5


def test_logseq_filename_timestamp_is_understood(tmp_path):
    path = tmp_path / "Audio-2026-08-06 12:30:00.webm"
    path.write_bytes(b"audio")
    with patch("subprocess.check_output", return_value='{"format": {}, "streams": []}'):
        metadata = probe_media(path)
    assert metadata.timestamp_source == "filename"
    assert metadata.recorded_at == "2026-08-06T16:30:00Z"
