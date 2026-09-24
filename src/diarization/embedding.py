"""Masked statistics pooling and projection after NCNN frame encoding."""

from pathlib import Path

import numpy as np

from .models import NCNNModel


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
    def __init__(self, root: Path):
        self.encoder = NCNNModel(root / "models" / "embedding_encoder")
        models = root / "models"
        self.weight = np.load(models / "resnet_seg_1_weight.npy")
        self.bias = np.load(models / "resnet_seg_1_bias.npy")

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
