"""Masked statistics pooling and projection after NCNN frame encoding."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio import compute_fbank
from .models import NCNNModel
from .segmentation import AudioWindow, CHUNK_SAMPLES, SegmentationResult


@dataclass(frozen=True)
class WindowEmbedding:
    index: int
    start: int
    vectors: np.ndarray
    train_mask: np.ndarray


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    train_mask: np.ndarray


def masked_stats_pool(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    weights = np.asarray(mask, dtype=np.float32)[None, :]
    if features.ndim != 2 or features.shape[1] != weights.shape[1]:
        raise ValueError("mask length must equal the number of feature frames")
    if np.any(weights < 0) or np.any(weights > 1):
        raise ValueError("mask weights must be between zero and one")
    total = weights.sum() + 1e-8
    mean = (features * weights).sum(axis=1) / total
    variance = (((features - mean[:, None]) ** 2) * weights).sum(axis=1)
    variance /= total - (weights * weights).sum() / total + 1e-8
    return np.concatenate((mean, np.sqrt(np.maximum(variance, 0)))).astype(np.float32)


class EmbeddingModel:
    def __init__(self, models_dir: Path, gpu_index: int | None = None):
        self.encoder = NCNNModel(models_dir / "embedding_encoder", gpu_index=gpu_index)
        self.weight = np.load(models_dir / "resnet_seg_1_weight.npy")
        self.bias = np.load(models_dir / "resnet_seg_1_bias.npy")

    def encode(self, fbank: np.ndarray) -> np.ndarray:
        return self.encoder.run(fbank[None])[0]

    def project(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.full(256, np.nan, dtype=np.float32)
        indices = np.minimum(
            np.floor(np.arange(features.shape[1]) * len(mask) / features.shape[1]).astype(int),
            len(mask) - 1,
        )
        statistics = masked_stats_pool(features, mask[indices])
        return statistics @ self.weight.T + self.bias

    def embed(self, fbank: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.project(self.encode(fbank), mask)


def embed_window(window: AudioWindow, activity: np.ndarray, model: EmbeddingModel) -> WindowEmbedding:
    """Extract local speaker vectors for one independently scheduled window."""
    if activity.ndim != 2 or activity.shape[1] != 3:
        raise ValueError("window activity must have shape (frames, 3)")
    frames, slots = activity.shape
    vectors = np.full((slots, 256), np.nan, dtype=np.float32)
    clean = activity * (activity.sum(axis=-1, keepdims=True) < 2)
    train_mask = clean.sum(axis=0) >= 0.2 * frames
    if np.any(activity):
        features = compute_fbank(window.samples)
        encoded = model.encode(features)
        minimum_frames = int(np.ceil(frames * 1680 / CHUNK_SAMPLES))
        for slot in range(slots):
            full_mask = activity[:, slot]
            if not np.any(full_mask):
                continue
            clean_mask = clean[:, slot]
            selected = clean_mask if clean_mask.sum() > minimum_frames else full_mask
            vectors[slot] = model.project(encoded, selected)
    return WindowEmbedding(window.index, window.start, vectors, train_mask)


def assemble_embeddings(results: Sequence[WindowEmbedding], segmentation: SegmentationResult) -> EmbeddingResult:
    """Validate a complete set of embeddings and restore window order."""
    starts = segmentation.starts
    slots = segmentation.activity.shape[2]
    ordered: list[WindowEmbedding | None] = [None] * len(starts)
    for result in results:
        if not 0 <= result.index < len(starts) or result.start != starts[result.index]:
            raise ValueError("embedding window index or start does not match segmentation")
        if ordered[result.index] is not None:
            raise ValueError("duplicate embedding window index")
        if result.vectors.shape != (slots, 256) or result.train_mask.shape != (slots,):
            raise ValueError("embedding window shapes do not match segmentation")
        ordered[result.index] = result
    if any(result is None for result in ordered):
        raise ValueError("missing embedding window result")
    return EmbeddingResult(
        np.stack([result.vectors for result in ordered if result is not None]),
        np.stack([result.train_mask for result in ordered if result is not None]),
    )
