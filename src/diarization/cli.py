"""Run diarization on a strict 16 kHz mono WAV and write RTTM."""

import argparse
from pathlib import Path

from .audio import read_wav
from .models import close_vulkan
from .pipeline import Diarizer
from .rttm import format_rttm


def main(argv=None):
    parser = argparse.ArgumentParser(description="NCNN speaker diarization with overlap-aware RTTM output")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="directory containing the NCNN, projection, and PLDA model files")
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--gpu-index", type=int, help="NCNN Vulkan GPU index (default: CPU)")
    args = parser.parse_args(argv)
    diarizer = None
    try:
        diarizer = Diarizer(args.models_dir, args.min_speakers, args.max_speakers, gpu_index=args.gpu_index)
        segments = diarizer.diarize(read_wav(args.input))
    finally:
        del diarizer
        if args.gpu_index is not None:
            close_vulkan()
    args.output.write_text(format_rttm(args.input.stem, segments), encoding="utf-8")


if __name__ == "__main__":
    main()
