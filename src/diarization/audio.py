"""Strict WAV input and WeSpeaker-compatible fbank extraction."""

from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000


def read_wav(path: Path) -> np.ndarray:
    info = sf.info(path)
    if info.format != "WAV" or info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError("input must be a 16 kHz mono WAV file")
    if info.subtype not in {"PCM_16", "FLOAT"}:
        raise ValueError("input must use 16-bit PCM or 32-bit float samples")
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    if not np.isfinite(audio).all():
        raise ValueError("input contains non-finite samples")
    return audio


def fbank_options() -> knf.FbankOptions:
    options = knf.FbankOptions()
    options.frame_opts.dither = 0.0
    options.frame_opts.snip_edges = True
    options.frame_opts.samp_freq = SAMPLE_RATE
    options.frame_opts.frame_length_ms = 25.0
    options.frame_opts.frame_shift_ms = 10.0
    options.frame_opts.window_type = "hamming"
    options.mel_opts.num_bins = 80
    options.mel_opts.low_freq = 20.0
    options.mel_opts.high_freq = 0.0
    options.energy_floor = 0.0
    return options


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    extractor = knf.OnlineFbank(fbank_options())
    extractor.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio * np.float32(32768)))
    extractor.input_finished()
    features = np.array([extractor.get_frame(i) for i in range(extractor.num_frames_ready)], dtype=np.float32)
    if not len(features):
        return np.empty((0, 80), dtype=np.float32)
    features -= features.mean(axis=0, keepdims=True)
    return features
