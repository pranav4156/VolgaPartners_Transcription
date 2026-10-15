from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class AudioError(Exception):
    pass


@dataclass(frozen=True)
class AudioMeta:
    duration: float
    codec: str
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class Chunk:
    path: Path
    offset: float
    index: int


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise AudioError(
            f"'{tool}' not found on PATH. Install ffmpeg "
            f"(brew install ffmpeg / apt-get install ffmpeg)."
        )


class AudioProcessor:
    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE, channels: int = TARGET_CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels

    def probe(self, path: str | Path) -> AudioMeta:
        path = Path(path)
        if not path.exists():
            raise AudioError(f"file not found: {path}")
        if path.stat().st_size == 0:
            raise AudioError(f"file is empty: {path}")
        _require("ffprobe")

        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise AudioError(f"ffprobe could not read {path.name}: {proc.stderr.strip()}")

        info = json.loads(proc.stdout or "{}")
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        if not streams:
            raise AudioError(f"no audio stream in {path.name}")

        s, fmt = streams[0], info.get("format", {})
        return AudioMeta(
            duration=float(s.get("duration") or fmt.get("duration") or 0.0),
            codec=s.get("codec_name", "unknown"),
            sample_rate=int(s.get("sample_rate", 0) or 0),
            channels=int(s.get("channels", 0) or 0),
        )

    def normalize(self, path: str | Path, out_path: str | Path) -> Path:
        path, out_path = Path(path), Path(out_path)
        _require("ffmpeg")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(path),
             "-vn", "-ac", str(self.channels), "-ar", str(self.sample_rate),
             "-c:a", "pcm_s16le", "-f", "wav", str(out_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out_path.exists():
            raise AudioError(f"ffmpeg failed on {path.name}: {proc.stderr.strip()[-400:]}")
        return out_path


class AudioChunker:
    def __init__(self, max_chunk_seconds: float = 600.0, overlap_seconds: float = 2.0):
        if overlap_seconds >= max_chunk_seconds:
            raise ValueError("overlap must be smaller than the chunk length")
        self.max_chunk_seconds = max_chunk_seconds
        self.overlap_seconds = overlap_seconds

    def needs_chunking(self, duration: float) -> bool:
        return duration > self.max_chunk_seconds

    def split(self, wav_path: str | Path, duration: float, work_dir: str | Path) -> List[Chunk]:
        wav_path, work_dir = Path(wav_path), Path(work_dir)
        if not self.needs_chunking(duration):
            return [Chunk(path=wav_path, offset=0.0, index=0)]

        _require("ffmpeg")
        work_dir.mkdir(parents=True, exist_ok=True)
        chunks: List[Chunk] = []
        start, index = 0.0, 0
        step = self.max_chunk_seconds - self.overlap_seconds

        while start < duration:
            out = work_dir / f"chunk_{index:04d}.wav"
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-y",
                 "-ss", f"{start:.3f}", "-t", f"{self.max_chunk_seconds:.3f}",
                 "-i", str(wav_path), "-c", "copy", str(out)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0 or not out.exists():
                raise AudioError(f"ffmpeg failed to cut chunk {index}: {proc.stderr.strip()[-300:]}")
            chunks.append(Chunk(path=out, offset=start, index=index))
            index += 1
            start += step

        return chunks
