from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Settings
from .models import Segment, Word


class Transcriber(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def transcribe_file(
        self, wav_path: str | Path, language: Optional[str] = None
    ) -> Tuple[List[Segment], str]:
        ...

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def close(self) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class WhisperXTranscriber(Transcriber):
    def load(self) -> None:
        import whisperx
        self._wx = whisperx
        self._model = whisperx.load_model(
            self.settings.model_size,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
            language=self.settings.language,
        )
        self._align_model = None
        self._align_meta = None

    def _ensure_align_model(self, language_code: str) -> None:
        if not self.settings.word_timestamps:
            return
        if self._align_model is None:
            self._align_model, self._align_meta = self._wx.load_align_model(
                language_code=language_code, device=self.settings.device
            )

    def transcribe_file(self, wav_path, language=None):
        self.ensure_loaded()
        audio = self._wx.load_audio(str(wav_path))
        result = self._model.transcribe(
            audio, batch_size=16, language=language or self.settings.language
        )
        detected = result.get("language", language or self.settings.language or "en")

        raw_segments = result.get("segments", [])
        if self.settings.word_timestamps and raw_segments:
            self._ensure_align_model(detected)
            aligned = self._wx.align(
                raw_segments, self._align_model, self._align_meta,
                audio, self.settings.device, return_char_alignments=False,
            )
            raw_segments = aligned.get("segments", raw_segments)

        return self._to_segments(raw_segments), detected

    @staticmethod
    def _to_segments(raw) -> List[Segment]:
        segments: List[Segment] = []
        for i, seg in enumerate(raw):
            words = [
                Word(
                    word=w.get("word", "").strip(),
                    start=float(w.get("start", seg.get("start", 0.0))),
                    end=float(w.get("end", seg.get("end", 0.0))),
                    probability=w.get("score"),
                )
                for w in seg.get("words", [])
                if w.get("start") is not None and w.get("end") is not None
            ]
            segments.append(Segment(
                id=i,
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=seg.get("text", "").strip(),
                words=words,
            ))
        return segments

    def close(self) -> None:
        self._model = None
        self._align_model = None


class FasterWhisperTranscriber(Transcriber):
    def load(self) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.settings.model_size,
            device=self.settings.device,
            compute_type=self.settings.compute_type,
        )

    @staticmethod
    def _load_wav_f32(wav_path):
        import wave
        import numpy as np
        with wave.open(str(wav_path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
                raise ValueError("expected 16 kHz mono 16-bit WAV from the normaliser")
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe_file(self, wav_path, language=None):
        self.ensure_loaded()
        audio = self._load_wav_f32(wav_path)
        seg_iter, info = self._model.transcribe(
            audio,
            beam_size=self.settings.beam_size,
            language=language or self.settings.language,
            word_timestamps=self.settings.word_timestamps,
            vad_filter=self.settings.vad_filter,
        )
        segments: List[Segment] = []
        for i, seg in enumerate(seg_iter):
            words = [
                Word(word=w.word.strip(), start=float(w.start),
                     end=float(w.end), probability=getattr(w, "probability", None))
                for w in (seg.words or [])
            ] if self.settings.word_timestamps else []
            segments.append(Segment(
                id=i, start=float(seg.start), end=float(seg.end),
                text=seg.text.strip(), words=words,
            ))
        return segments, info.language

    def close(self) -> None:
        self._model = None


_ENGINES = {
    "whisperx": WhisperXTranscriber,
    "faster-whisper": FasterWhisperTranscriber,
}


def build_transcriber(settings: Settings) -> Transcriber:
    try:
        cls = _ENGINES[settings.provider]
    except KeyError:
        raise ValueError(
            f"unknown provider '{settings.provider}'. choose one of: {', '.join(_ENGINES)}"
        )
    return cls(settings)
