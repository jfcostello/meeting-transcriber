from __future__ import annotations

from typing import Any


def timestamp(seconds: float | None) -> str:
    value = max(0, int(seconds or 0))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def utterances(
    segments: list[dict[str, Any]], maximum_gap: float = 1.5
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = str(segment.get("speaker") or "UNKNOWN")
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        previous = result[-1] if result else None
        if (
            previous
            and previous["speaker"] == speaker
            and start - previous["end"] <= maximum_gap
        ):
            previous["text"] = f"{previous['text']} {text}"
            previous["end"] = end
        else:
            result.append(
                {"speaker": speaker, "start": start, "end": end, "text": text}
            )
    return result


def transcript_markdown(data: dict[str, Any]) -> str:
    lines = ["# Transcript", ""]
    for item in data["utterances"]:
        lines.extend(
            [
                f"## {item['speaker']} · {timestamp(item['start'])}–{timestamp(item['end'])}",
                "",
                item["text"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
