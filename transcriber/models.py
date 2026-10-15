from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: Optional[float] = None


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: List[Segment]
    model: str = ""
    device: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for seg in d["segments"]:
            seg["start"] = round(seg["start"], 3)
            seg["end"] = round(seg["end"], 3)
            for w in seg["words"]:
                w["start"] = round(w["start"], 3)
                w["end"] = round(w["end"], 3)
                if w.get("probability") is not None:
                    w["probability"] = round(w["probability"], 4)
        d["duration"] = round(d["duration"], 3)
        return d
