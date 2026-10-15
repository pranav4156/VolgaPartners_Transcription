from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type

from .models import TranscriptionResult


def _ts_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")


class Formatter(ABC):
    extension: str = ""

    @abstractmethod
    def render(self, result: TranscriptionResult) -> str:
        ...

    def write(self, result: TranscriptionResult, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.render(result), encoding="utf-8")
        return out_path


class JSONFormatter(Formatter):
    extension = "json"

    def render(self, result: TranscriptionResult) -> str:
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


class SRTFormatter(Formatter):
    extension = "srt"

    def render(self, result: TranscriptionResult) -> str:
        blocks = []
        for n, seg in enumerate(result.segments, start=1):
            blocks.append(
                f"{n}\n{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}\n{seg.text}\n"
            )
        return "\n".join(blocks)


class VTTFormatter(Formatter):
    extension = "vtt"

    def render(self, result: TranscriptionResult) -> str:
        lines = ["WEBVTT", ""]
        for seg in result.segments:
            lines.append(f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)


_FORMATTERS: Dict[str, Type[Formatter]] = {
    "json": JSONFormatter,
    "srt": SRTFormatter,
    "vtt": VTTFormatter,
}


def get_formatter(fmt: str) -> Formatter:
    try:
        return _FORMATTERS[fmt.lower()]()
    except KeyError:
        raise ValueError(f"unknown format '{fmt}'. choose: {', '.join(_FORMATTERS)}")


def available_formats() -> list[str]:
    return list(_FORMATTERS)
