from __future__ import annotations

import ipaddress
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse


class Summarizer(Protocol):
    def summarize(self, transcript: str) -> str: ...


@dataclass(frozen=True)
class PendingSummarizer:
    """Explicit placeholder until a private local model is selected."""

    reason: str = "local summary model is not configured"

    def summarize(self, transcript: str) -> str:
        raise RuntimeError(self.reason)


def _private_endpoint(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    if host in {"localhost", "host.docker.internal"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


@dataclass(frozen=True)
class LocalOpenAISummarizer:
    base_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2

    def __post_init__(self) -> None:
        if not _private_endpoint(self.base_url):
            raise RuntimeError(
                "summary endpoint must be loopback or a private-network address"
            )

    def summarize(self, transcript: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Summarize this private meeting transcript. Preserve decisions, actions, owners, dates, risks and open questions. Do not invent details.",
                },
                {"role": "user", "content": transcript},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"content-type": "application/json"}
        if token := os.getenv("LOCAL_LLM_API_KEY"):
            headers["authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read().decode("utf-8"))
        try:
            return str(result["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "local summary endpoint returned an invalid response"
            ) from error
