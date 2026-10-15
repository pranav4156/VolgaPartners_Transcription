from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

from .audio import AudioChunker, AudioProcessor
from .engine import Transcriber
from .models import Segment, TranscriptionResult


class TranscriptionPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        audio_processor: Optional[AudioProcessor] = None,
        chunker: Optional[AudioChunker] = None,
    ):
        self.transcriber = transcriber
        self.audio = audio_processor or AudioProcessor()
        self.chunker = chunker or AudioChunker()

    def run(self, input_path: str | Path, work_dir: str | Path | None = None) -> TranscriptionResult:
        input_path = Path(input_path)
        meta = self.audio.probe(input_path)

        workdir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="transcribe_"))
        workdir.mkdir(parents=True, exist_ok=True)

        wav = self.audio.normalize(input_path, workdir / "normalized.wav")
        chunks = self.chunker.split(wav, meta.duration, workdir / "chunks")

        merged: List[Segment] = []
        language: Optional[str] = None

        for chunk in chunks:
            segments, lang = self.transcriber.transcribe_file(chunk.path, language)
            language = language or lang
            self._merge(merged, segments, offset=chunk.offset)

        for new_id, seg in enumerate(merged):
            seg.id = new_id

        full_text = " ".join(s.text for s in merged if s.text).strip()
        return TranscriptionResult(
            text=full_text,
            language=language or "unknown",
            duration=meta.duration,
            segments=merged,
            model=self.transcriber.settings.model_size,
            device=self.transcriber.settings.device,
            provider=self.transcriber.settings.provider,
        )

    @staticmethod
    def _merge(acc: List[Segment], new: List[Segment], offset: float) -> None:
        last_end = acc[-1].end if acc else 0.0
        for seg in new:
            start = seg.start + offset
            end = seg.end + offset
            if offset > 0 and end <= last_end + 1e-3:
                continue
            seg.start, seg.end = start, end
            for w in seg.words:
                w.start += offset
                w.end += offset
            acc.append(seg)
