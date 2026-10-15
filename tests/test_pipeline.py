from __future__ import annotations

import pytest

from transcriber.audio import AudioChunker
from transcriber.pipeline import TranscriptionPipeline


def test_pipeline_short_file(stub_transcriber, sample_mp3, tmp_path):
    pipeline = TranscriptionPipeline(stub_transcriber)
    result = pipeline.run(sample_mp3, work_dir=tmp_path / "work")

    assert result.language == "en"
    assert result.duration == pytest.approx(3.0, abs=0.3)
    assert len(result.segments) == 1
    assert result.text == "hello world"
    assert result.provider == stub_transcriber.settings.provider


def test_pipeline_long_file_stitches_timestamps(stub_transcriber, sample_wav_long, tmp_path):
    chunker = AudioChunker(max_chunk_seconds=2.0, overlap_seconds=0.5)
    pipeline = TranscriptionPipeline(stub_transcriber, chunker=chunker)
    result = pipeline.run(sample_wav_long, work_dir=tmp_path / "work")

    assert len(result.segments) == 3
    assert [round(s.start, 2) for s in result.segments] == [0.0, 1.5, 3.0]
    assert result.segments[1].words[0].start == pytest.approx(1.5)
    assert [s.id for s in result.segments] == [0, 1, 2]
