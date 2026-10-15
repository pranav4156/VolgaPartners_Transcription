from __future__ import annotations

import json

import pytest

from transcriber.models import Segment, TranscriptionResult, Word
from transcriber.output import get_formatter


def _result() -> TranscriptionResult:
    return TranscriptionResult(
        text="hello world goodbye",
        language="en",
        duration=5.0,
        segments=[
            Segment(0, 0.0, 1.5, "hello world",
                    words=[Word("hello", 0.0, 0.7, 0.98), Word("world", 0.7, 1.5, 0.95)]),
            Segment(1, 2.0, 3.25, "goodbye"),
        ],
    )


def test_json_roundtrips_and_rounds_times():
    data = json.loads(get_formatter("json").render(_result()))
    assert data["language"] == "en"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["words"][0]["word"] == "hello"
    assert data["segments"][0]["start"] == 0.0


def test_srt_format():
    srt = get_formatter("srt").render(_result())
    assert "1\n00:00:00,000 --> 00:00:01,500\nhello world" in srt
    assert "2\n00:00:02,000 --> 00:00:03,250\ngoodbye" in srt


def test_vtt_format():
    vtt = get_formatter("vtt").render(_result())
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_write_creates_file(tmp_path):
    out = get_formatter("srt").write(_result(), tmp_path / "x.srt")
    assert out.exists() and out.read_text().strip()


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        get_formatter("docx")
