"""Orchestrate NCNN segmentation, speaker embeddings, VBx, and reconstruction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, compute_fbank
from .embedding import EmbeddingModel, EmbeddingResult, assemble_embeddings, embed_window
from .models import NCNNModel
from .plda import PLDA
from .rttm import Segment
from .segmentation import (
    FRAME_STEP, SegmentationResult, assemble_segmentation, make_windows, recording_frame_mask,
    segment_window,
)
from .vbx import cluster_embeddings


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    centroids: np.ndarray


class Diarizer:
    def __init__(self, models_dir: Path, minimum_speakers: int = 1, maximum_speakers: int | None = None,
                 gpu_index: int | None = None):
        if minimum_speakers < 1 or (maximum_speakers is not None and maximum_speakers < minimum_speakers):
            raise ValueError("speaker bounds must satisfy 1 <= minimum <= maximum")
        self.segmentation = NCNNModel(models_dir / "segmentation", gpu_index=gpu_index)
        self.embedding = EmbeddingModel(models_dir, gpu_index=gpu_index)
        self.plda = PLDA(models_dir)
        self.minimum_speakers = minimum_speakers
        self.maximum_speakers = maximum_speakers

    def segment(self, audio: np.ndarray) -> SegmentationResult:
        results = [segment_window(window, self.segmentation) for window in make_windows(audio)]
        return assemble_segmentation(results, len(audio))

    def extract_embeddings(self, audio: np.ndarray, segmentation: SegmentationResult) -> EmbeddingResult:
        if segmentation.duration_samples != len(audio):
            raise ValueError("segmentation duration does not match audio")
        if len(segmentation.starts) != len(segmentation.activity):
            raise ValueError("segmentation windows do not match audio")
        results = []
        for window in make_windows(audio):
            if window.index >= len(segmentation.starts) or window.start != segmentation.starts[window.index]:
                raise ValueError("segmentation windows do not match audio")
            results.append(embed_window(window, segmentation.activity[window.index], self.embedding))
        if len(results) != len(segmentation.starts):
            raise ValueError("segmentation windows do not match audio")
        return assemble_embeddings(results, segmentation)

    def cluster(self, segmentation: SegmentationResult, embeddings: EmbeddingResult) -> ClusteringResult:
        if embeddings.vectors.shape != (len(segmentation.starts), segmentation.activity.shape[2], 256):
            raise ValueError("embeddings do not match segmentation")
        if embeddings.train_mask.shape != embeddings.vectors.shape[:2]:
            raise ValueError("embedding train mask does not match vectors")
        valid_frames = recording_frame_mask(segmentation.starts, segmentation.activity.shape[1],
                                            segmentation.duration_samples)
        activity = segmentation.activity * valid_frames[:, :, None]
        labels, centroids = cluster_embeddings(
            embeddings.vectors, embeddings.train_mask, activity, self.plda,
            minimum=self.minimum_speakers, maximum=self.maximum_speakers,
        )
        return ClusteringResult(labels, centroids)

    def reconstruct(self, audio: np.ndarray, segmentation: SegmentationResult,
                    clustering: ClusteringResult) -> list[Segment]:
        if segmentation.duration_samples != len(audio):
            raise ValueError("segmentation duration does not match audio")
        activity = segmentation.activity
        if clustering.labels.shape != (len(segmentation.starts), activity.shape[2]):
            raise ValueError("cluster labels do not match segmentation")
        if clustering.centroids.ndim != 2 or clustering.centroids.shape[1] != 256:
            raise ValueError("cluster centroids must have 256 dimensions")
        overlap_rate = np.mean(activity.sum(axis=-1) >= 2)
        weak_local_separation = overlap_rate < 0.01
        turns = self._short_window_assignments(audio, clustering.centroids) if weak_local_separation and len(clustering.centroids) > 1 else None
        count_threshold = 0.6 if weak_local_separation else 0.1
        return reconstruct(activity, clustering.labels, list(segmentation.starts), len(audio) / SAMPLE_RATE,
                           turns, count_threshold, minimum=self.minimum_speakers)

    def diarize(self, audio: np.ndarray) -> list[Segment]:
        segmentation = self.segment(audio)
        embeddings = self.extract_embeddings(audio, segmentation)
        clustering = self.cluster(segmentation, embeddings)
        return self.reconstruct(audio, segmentation, clustering)

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
                turns: np.ndarray | None = None, count_threshold: float = 0.5,
                *, minimum: int = 1) -> list[Segment]:
    clusters = int(labels.max()) + 1
    if clusters <= 0:
        return []
    frame_count = int(np.ceil(duration / FRAME_STEP)) + 1
    accumulated = np.zeros((frame_count, clusters), dtype=np.float32)
    counts = np.zeros(frame_count, dtype=np.float32)
    coverage = np.zeros(frame_count, dtype=np.float32)
    valid_frames = recording_frame_mask(starts, activity.shape[1], round(duration * SAMPLE_RATE))
    for chunk_index, start in enumerate(starts):
        offset = round((start / SAMPLE_RATE) / FRAME_STEP)
        for frame in range(activity.shape[1]):
            global_frame = offset + frame
            if global_frame >= frame_count or not valid_frames[chunk_index, frame]:
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
    supported = np.any(accumulated > 0, axis=0)
    required = min(minimum, int(supported.sum()))
    for cluster in np.flatnonzero(supported):
        if int(binary.any(axis=0).sum()) >= required:
            break
        if binary[:, cluster].any():
            continue
        candidates = np.flatnonzero(accumulated[:, cluster] > 0)
        if not len(candidates):
            continue
        frame = int(candidates[np.argmax(accumulated[candidates, cluster])])
        assigned = np.flatnonzero(binary[frame])
        if len(assigned) >= count[frame]:
            removable = [int(other) for other in assigned if binary[:, other].sum() > 1]
            if removable:
                binary[frame, removable[-1]] = False
        binary[frame, cluster] = True
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
