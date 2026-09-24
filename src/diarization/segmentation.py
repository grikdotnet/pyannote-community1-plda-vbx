"""Sliding segmentation windows and powerset speaker activity."""

import numpy as np

from .audio import SAMPLE_RATE


CHUNK_SAMPLES = 10 * SAMPLE_RATE
STEP_SAMPLES = SAMPLE_RATE
FRAME_DURATION = 0.0619375
FRAME_STEP = 0.016875
POWERSET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    dtype=np.float32,
)


def decode_powerset(scores: np.ndarray) -> np.ndarray:
    if scores.shape[-1] != 7:
        raise ValueError("segmentation must have seven powerset classes")
    return POWERSET[np.argmax(scores, axis=-1)]


def segment_audio(audio: np.ndarray, model) -> tuple[np.ndarray, list[int]]:
    starts = []
    scores = []
    start = 0
    while True:
        starts.append(start)
        chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        part = audio[start:start + CHUNK_SAMPLES]
        chunk[:len(part)] = part
        scores.append(model.run(chunk[None, None])[0])
        if start + CHUNK_SAMPLES > len(audio):
            break
        start += STEP_SAMPLES
    return np.stack(scores), starts
