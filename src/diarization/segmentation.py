"""Sliding segmentation windows and powerset speaker activity."""

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .audio import SAMPLE_RATE


CHUNK_SAMPLES = 10 * SAMPLE_RATE
STEP_SAMPLES = SAMPLE_RATE
FRAME_DURATION = 0.0619375
FRAME_STEP = 0.016875


def recording_frame_mask(starts: Sequence[int], frames: int, duration_samples: int) -> np.ndarray:
    """Mark window frames whose centers fall inside the real recording."""
    offsets = np.rint(np.asarray(starts) / SAMPLE_RATE / FRAME_STEP).astype(int)
    times = (offsets[:, None] + np.arange(frames)) * FRAME_STEP + FRAME_DURATION / 2
    return times < duration_samples / SAMPLE_RATE
POWERSET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start: int
    samples: np.ndarray


@dataclass(frozen=True)
class WindowSegmentation:
    index: int
    start: int
    activity: np.ndarray


@dataclass(frozen=True)
class SegmentationResult:
    activity: np.ndarray
    starts: tuple[int, ...]
    duration_samples: int


def _window_starts(duration_samples: int) -> tuple[int, ...]:
    if duration_samples < 0:
        raise ValueError("audio length must be non-negative")
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + CHUNK_SAMPLES > duration_samples:
            break
        start += STEP_SAMPLES
    return tuple(starts)


def make_windows(audio: np.ndarray) -> Iterator[AudioWindow]:
    """Yield numbered, padded windows without retaining the whole recording's windows."""
    for index, start in enumerate(_window_starts(len(audio))):
        samples = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        part = audio[start:start + CHUNK_SAMPLES]
        samples[:len(part)] = part
        yield AudioWindow(index, start, samples)


def decode_powerset(scores: np.ndarray) -> np.ndarray:
    if scores.shape[-1] != 7:
        raise ValueError("segmentation must have seven powerset classes")
    return POWERSET[np.argmax(scores, axis=-1)]


def segment_window(window: AudioWindow, model) -> WindowSegmentation:
    """Run one window; the caller owns the model used by this worker."""
    scores = model.run(window.samples[None, None])[0]
    return WindowSegmentation(window.index, window.start, decode_powerset(scores))


def assemble_segmentation(results: Sequence[WindowSegmentation], duration_samples: int) -> SegmentationResult:
    """Validate a complete window set and restore recording order."""
    starts = _window_starts(duration_samples)
    ordered: list[WindowSegmentation | None] = [None] * len(starts)
    for result in results:
        if not 0 <= result.index < len(starts) or result.start != starts[result.index]:
            raise ValueError("segmentation window index or start does not match the recording")
        if ordered[result.index] is not None:
            raise ValueError("duplicate segmentation window index")
        if result.activity.ndim != 2 or result.activity.shape[1] != 3:
            raise ValueError("segmentation window activity must have shape (frames, 3)")
        ordered[result.index] = result
    if any(result is None for result in ordered):
        raise ValueError("missing segmentation window result")
    activity = np.stack([result.activity for result in ordered if result is not None])
    return SegmentationResult(activity, starts, duration_samples)


def segment_audio(audio: np.ndarray, model) -> tuple[np.ndarray, list[int]]:
    starts = []
    scores = []
    for window in make_windows(audio):
        starts.append(window.start)
        scores.append(model.run(window.samples[None, None])[0])
    return np.stack(scores), starts
