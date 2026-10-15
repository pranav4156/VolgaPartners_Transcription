#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from transcriber.audio import AudioError
from transcriber.config import load_settings
from transcriber.engine import build_transcriber
from transcriber.output import available_formats, get_formatter
from transcriber.pipeline import TranscriptionPipeline


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe an audio file to timestamped text (JSON/SRT/VTT).",
    )
    p.add_argument("input", help="path to an audio file (wav, mp3, m4a, flac, ogg, ...)")
    p.add_argument("--out", default="out", help="output directory (default: ./out)")
    p.add_argument("--formats", default="json,srt,vtt",
                   help=f"comma-separated: {', '.join(available_formats())}")
    p.add_argument("--provider", help="whisperx | faster-whisper")
    p.add_argument("--model", help="model size, e.g. small, large-v3, distil-large-v3")
    p.add_argument("--device", help="cuda | cpu (default: auto-detect)")
    p.add_argument("--language", help="force language code, e.g. en (default: auto)")
    p.add_argument("--no-word-timestamps", action="store_true",
                   help="segment-level timestamps only (faster)")
    p.add_argument("--max-chunk-seconds", type=float, default=None,
                   help="split audio longer than this into chunks (default: 600)")
    return p.parse_args(argv)


def _apply_overrides(args) -> None:
    import os
    mapping = {
        "PROVIDER": args.provider,
        "MODEL_SIZE": args.model,
        "DEVICE": args.device,
        "LANGUAGE": args.language,
    }
    for key, val in mapping.items():
        if val:
            os.environ[key] = val
    if args.no_word_timestamps:
        os.environ["WORD_TIMESTAMPS"] = "false"


def _quiet_third_party() -> None:
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    for name in ("pyannote", "whisperx", "lightning", "lightning.pytorch",
                 "pytorch_lightning", "torch", "speechbrain", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main(argv=None) -> int:
    args = parse_args(argv)
    _quiet_third_party()
    _apply_overrides(args)

    settings = load_settings()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    for f in formats:
        if f not in available_formats():
            print(f"error: unknown format '{f}'. choose: {', '.join(available_formats())}",
                  file=sys.stderr)
            return 2

    print(f"[config] {settings.summary()}", file=sys.stderr)
    print(f"[input ] {args.input}", file=sys.stderr)

    transcriber = build_transcriber(settings)
    if args.max_chunk_seconds is not None:
        from transcriber.audio import AudioChunker
        pipeline = TranscriptionPipeline(
            transcriber, chunker=AudioChunker(max_chunk_seconds=args.max_chunk_seconds)
        )
    else:
        pipeline = TranscriptionPipeline(transcriber)

    t0 = time.time()
    try:
        result = pipeline.run(args.input)
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        transcriber.close()
    elapsed = time.time() - t0

    stem = Path(args.input).stem
    out_dir = Path(args.out)
    written = []
    for f in formats:
        path = get_formatter(f).write(result, out_dir / f"{stem}.{f}")
        written.append(str(path))

    print(
        f"[done  ] {len(result.segments)} segments, "
        f"{result.duration:.1f}s audio, lang={result.language}, took {elapsed:.1f}s",
        file=sys.stderr,
    )
    for w in written:
        print(f"[write ] {w}", file=sys.stderr)
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
