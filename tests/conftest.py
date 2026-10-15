from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from transcriber.config import load_settings
from transcriber.engine import Transcriber
from transcriber.models import Segment, Word


@pytest.fixture
def settings():
    return load_settings()


def _make_audio(path: Path, seconds: float, fmt_args: List[str]) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}", *fmt_args, str(path)],
        capture_output=True, check=True,
    )
    return path


@pytest.fixture
def sample_mp3(tmp_path) -> Path:
    return _make_audio(tmp_path / "sample.mp3", 3.0, ["-ac", "2", "-ar", "44100"])


@pytest.fixture
def sample_wav_long(tmp_path) -> Path:
    return _make_audio(tmp_path / "long.wav", 4.0, ["-ac", "1", "-ar", "16000"])


class StubTranscriber(Transcriber):
    def load(self) -> None:
        pass

    def transcribe_file(self, wav_path, language=None) -> Tuple[List[Segment], str]:
        seg = Segment(
            id=0, start=0.0, end=1.0, text="hello world",
            words=[Word("hello", 0.0, 0.5, 0.9), Word("world", 0.5, 1.0, 0.9)],
        )
        return [seg], "en"


@pytest.fixture
def stub_transcriber(settings):
    return StubTranscriber(settings)
