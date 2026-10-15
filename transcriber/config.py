from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _detect_device() -> tuple[str, str]:
    try:
        import torch
    except Exception:
        return "cpu", "torch not installed; defaulting to CPU"
    try:
        if torch.cuda.is_available():
            return "cuda", f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "cpu", "Apple Silicon (mps) detected, but the Whisper engine is CPU-only here"
    except Exception:
        pass
    return "cpu", "no GPU detected; using CPU"


_DEFAULT_MODEL = {"cuda": "large-v3", "cpu": "small"}
_DEFAULT_COMPUTE = {"cuda": "float16", "cpu": "int8"}


@dataclass(frozen=True)
class Settings:
    provider: str
    model_size: str
    device: str
    compute_type: str
    language: Optional[str]
    word_timestamps: bool
    vad_filter: bool
    beam_size: int
    device_note: str

    def summary(self) -> str:
        return (
            f"provider={self.provider} model={self.model_size} "
            f"device={self.device} compute={self.compute_type} "
            f"word_ts={self.word_timestamps} vad={self.vad_filter} "
            f"lang={self.language or 'auto'} ({self.device_note})"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_settings() -> Settings:
    detected_device, note = _detect_device()
    device = _env("DEVICE", detected_device) or "cpu"
    if device != detected_device:
        note = f"device forced to {device} via DEVICE env (auto-detected {detected_device})"
    model_size = _env("MODEL_SIZE", _DEFAULT_MODEL.get(device, "small"))
    compute_type = _env("COMPUTE_TYPE", _DEFAULT_COMPUTE.get(device, "int8"))
    return Settings(
        provider=_env("PROVIDER", "whisperx"),
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=_env("LANGUAGE", None),
        word_timestamps=_env("WORD_TIMESTAMPS", "true").lower() == "true",
        vad_filter=_env("VAD_FILTER", "true").lower() == "true",
        beam_size=int(_env("BEAM_SIZE", "5")),
        device_note=note,
    )
