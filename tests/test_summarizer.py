import pytest

from meeting_transcriber.summarizer import LocalOpenAISummarizer


def test_public_summary_endpoint_is_rejected():
    with pytest.raises(RuntimeError, match="private-network"):
        LocalOpenAISummarizer("https://api.example.com/v1", "model")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1",
        "http://192.168.2.81:1234/v1",
        "http://model.local:8000/v1",
    ],
)
def test_private_summary_endpoint_is_accepted(url):
    assert LocalOpenAISummarizer(url, "model").base_url == url
