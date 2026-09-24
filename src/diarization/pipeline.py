"""Orchestrate NCNN segmentation, speaker embeddings, VBx, and reconstruction."""

from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, compute_fbank
from .embedding import EmbeddingModel
from .models import NCNNModel
from .plda import PLDA
from .rttm import Segment
from .segmentation import CHUNK_SAMPLES, FRAME_DURATION, FRAME_STEP, decode_powerset, segment_audio
from .vbx import cluster_embeddings


class Diarizer:
    def __init__(self, root: Path, minimum_speakers: int = 1, maximum_speakers: int | None = None):
        if minimum_speakers < 1 or (maximum_speakers is not None and maximum_speakers < minimum_speakers):
            raise ValueError("speaker bounds must satisfy 1 <= minimum <= maximum")
        self.segmentation = NCNNModel(root / "models" / "segmentation")
        self.embedding = EmbeddingModel(root)
        self.plda = PLDA(root / "models")
        self.minimum_speakers = minimum_speakers
        self.maximum_speakers = maximum_speakers

    def diarize(self, audio: np.ndarray) -> list[Segment]:
        scores, starts = segment_audio(audio, self.segmentation)
        activity = decode_powerset(scores)
        chunks, frames, slots = activity.shape
        embeddings = np.full((chunks, slots, 256), np.nan, dtype=np.float32)
        clean = activity * (activity.sum(axis=-1, keepdims=True) < 2)
        minimum_frames = int(np.ceil(frames * 1680 / CHUNK_SAMPLES))
        for chunk_index, start in enumerate(starts):
            if not np.any(activity[chunk_index]):
                continue
            samples = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            part = audio[start:start + CHUNK_SAMPLES]
            samples[:len(part)] = part
            features = compute_fbank(samples)
            encoded = self.embedding.encode(features)
            for slot in range(slots):
                full_mask = activity[chunk_index, :, slot]
                if not np.any(full_mask):
                    continue
                clean_mask = clean[chunk_index, :, slot]
                selected = clean_mask if clean_mask.sum() > minimum_frames else full_mask
                embeddings[chunk_index, slot] = self.embedding.project(encoded, selected)
        enough_clean = clean.sum(axis=1) >= 0.2 * frames
        labels, centroids = cluster_embeddings(
            embeddings, enough_clean, activity, self.plda,
            minimum=self.minimum_speakers, maximum=self.maximum_speakers,
        )
        overlap_rate = np.mean(activity.sum(axis=-1) >= 2)
        weak_local_separation = overlap_rate < 0.01
        turns = self._short_window_assignments(audio, centroids) if weak_local_separation and len(centroids) > 1 else None
        count_threshold = 0.6 if weak_local_separation else 0.1
        return reconstruct(activity, labels, starts, len(audio) / SAMPLE_RATE, turns, count_threshold)

    def _short_window_assignments(self, audio: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Resolve turns hidden inside a ten-second local speaker slot."""
        normalized = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-10)
        assignments = np.full(int(np.ceil(len(audio) / SAMPLE_RATE)), -1, dtype=np.int32)
        for second in range(len(assignments)):
            start = max(0, second - 1) * SAMPLE_RATE
            end = min(len(audio), (second + 1) * SAMPLE_RATE)
            if end - start < 16000:
                continue
            fbank = compute_fbank(audio[start:end])
            if len(fbank) < 9:
                continue
            embedding = self.embedding.embed(fbank, np.ones(1, dtype=np.float32))
            embedding /= max(np.linalg.norm(embedding), 1e-10)
            similarity = normalized @ embedding
            order = np.argsort(similarity)
            if similarity[order[-1]] >= 0.2 and similarity[order[-1]] - similarity[order[-2]] >= 0.08:
                assignments[second] = order[-1]
        return assignments


def reconstruct(activity: np.ndarray, labels: np.ndarray, starts: list[int], duration: float,
                turns: np.ndarray | None = None, count_threshold: float = 0.5) -> list[Segment]:
    clusters = int(labels.max()) + 1
    if clusters <= 0:
        return []
    frame_count = int(np.ceil(duration / FRAME_STEP)) + 1
    accumulated = np.zeros((frame_count, clusters), dtype=np.float32)
    counts = np.zeros(frame_count, dtype=np.float32)
    coverage = np.zeros(frame_count, dtype=np.float32)
    for chunk_index, start in enumerate(starts):
        offset = round((start / SAMPLE_RATE) / FRAME_STEP)
        for frame in range(activity.shape[1]):
            global_frame = offset + frame
            if global_frame >= frame_count or global_frame * FRAME_STEP + FRAME_DURATION / 2 >= duration:
                break
            local = activity[chunk_index, frame]
            counts[global_frame] += local.sum()
            coverage[global_frame] += 1
            for slot in np.flatnonzero(local):
                cluster = labels[chunk_index, slot]
                if cluster >= 0:
                    accumulated[global_frame, cluster] += 1
    count = np.floor(counts / np.maximum(coverage, 1) + 1 - count_threshold).astype(int)
    count = np.minimum(count, clusters)
    binary = np.zeros_like(accumulated, dtype=bool)
    for frame, speaker_count in enumerate(count):
        if speaker_count and np.any(accumulated[frame]):
            selected = np.argsort(-accumulated[frame])[:speaker_count]
            binary[frame, selected] = accumulated[frame, selected] > 0
            if speaker_count == 1 and turns is not None:
                second = min(int(frame * FRAME_STEP), len(turns) - 1)
                if turns[second] >= 0:
                    binary[frame] = False
                    binary[frame, turns[second]] = True
    segments = []
    for cluster in range(clusters):
        active = np.flatnonzero(binary[:, cluster])
        if not len(active):
            continue
        boundaries = np.flatnonzero(np.diff(active) > 1)
        runs = np.split(active, boundaries + 1)
        for run in runs:
            start_time = max(0, run[0] * FRAME_STEP)
            end_time = min(duration, (run[-1] + 1) * FRAME_STEP)
            segments.append(Segment(start_time, end_time, f"speaker_{cluster:02d}"))
    return sorted(segments, key=lambda value: (value.start, value.speaker))
