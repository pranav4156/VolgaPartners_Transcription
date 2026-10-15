from __future__ import annotations

import pytest

from transcriber.audio import AudioChunker, AudioError, AudioProcessor


def test_probe_reads_metadata(sample_mp3):
    meta = AudioProcessor().probe(sample_mp3)
    assert meta.duration == pytest.approx(3.0, abs=0.3)
    assert meta.channels == 2
    assert meta.sample_rate == 44100


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        AudioProcessor().probe(tmp_path / "nope.wav")


def test_probe_rejects_non_audio(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_text("this is not audio")
    with pytest.raises(AudioError):
        AudioProcessor().probe(junk)


def test_normalize_converts_any_format_to_16k_mono(sample_mp3, tmp_path):
    out = AudioProcessor().normalize(sample_mp3, tmp_path / "norm.wav")
    meta = AudioProcessor().probe(out)
    assert meta.sample_rate == 16000
    assert meta.channels == 1
    assert meta.codec == "pcm_s16le"


def test_short_file_is_single_chunk(sample_wav_long):
    chunker = AudioChunker(max_chunk_seconds=600)
    chunks = chunker.split(sample_wav_long, duration=4.0, work_dir=sample_wav_long.parent)
    assert len(chunks) == 1
    assert chunks[0].offset == 0.0


def test_long_file_is_split_with_offsets(sample_wav_long, tmp_path):
    chunker = AudioChunker(max_chunk_seconds=2.0, overlap_seconds=0.5)
    chunks = chunker.split(sample_wav_long, duration=4.0, work_dir=tmp_path / "chunks")
    assert len(chunks) > 1
    assert chunks[0].offset == 0.0
    assert chunks[1].offset == pytest.approx(1.5)
    for c in chunks:
        assert c.path.exists()


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        AudioChunker(max_chunk_seconds=2.0, overlap_seconds=2.0)
