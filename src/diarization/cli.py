"""Run diarization on a strict 16 kHz mono WAV and write RTTM."""

import argparse
from pathlib import Path

from .audio import read_wav
from .pipeline import Diarizer
from .rttm import format_rttm


def main(argv=None):
    parser = argparse.ArgumentParser(description="NCNN speaker diarization with overlap-aware RTTM output")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    segments = Diarizer(root, args.min_speakers, args.max_speakers).diarize(read_wav(args.input))
    args.output.write_text(format_rttm(args.input.stem, segments), encoding="utf-8")


if __name__ == "__main__":
    main()
