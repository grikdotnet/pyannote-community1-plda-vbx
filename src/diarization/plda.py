"""PLDA normalization and scoring, independent of model inference.

The transform follows Community-1's PLDA array contract. The LDA-space score
equation is from the Apache-2.0 BUTSpeechFIT/VBx reference in ``reference/vbx/VBx``.
"""

from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-10)


def score_in_lda_space(enroll: np.ndarray, test: np.ndarray, psi: np.ndarray) -> np.ndarray:
    inverse_total = 1 / (1 + psi)
    inverse_twice = 1 / (1 + 2 * psi)
    gamma = -0.25 * (inverse_twice + 1 - 2 * inverse_total)
    cross = -0.5 * (inverse_twice - 1)
    offset = -0.5 * (np.log1p(2 * psi).sum() - 2 * np.log1p(psi).sum())
    return (enroll * cross) @ test.T + (enroll**2 @ gamma)[:, None] + (test**2 @ gamma) + offset


class PLDA:
    def __init__(self, directory: Path):
        with np.load(directory / "xvec_transform.npz") as transform:
            self.mean1 = transform["mean1"]
            self.lda = transform["lda"]
            self.mean2 = transform["mean2"]
        with np.load(directory / "plda.npz") as model:
            self.mean = model["mu"]
            source_transform = model["tr"]
            source_psi = model["psi"]
        if self.mean1.shape != (256,) or self.lda.shape != (256, 128) or self.mean2.shape != (128,):
            raise ValueError("invalid 256-to-128 x-vector transform")
        if self.mean.shape != (128,) or source_transform.shape != (128, 128) or source_psi.shape != (128,):
            raise ValueError("invalid PLDA arrays")
        within = np.linalg.inv(source_transform.T @ source_transform)
        between = np.linalg.inv((source_transform.T / source_psi) @ source_transform)
        eigenvalues, eigenvectors = eigh(between, within)
        self.psi = eigenvalues[::-1]
        self.transform_matrix = eigenvectors.T[::-1]

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        centered = l2_normalize(embeddings - self.mean1) * np.sqrt(256)
        projected = centered @ self.lda - self.mean2
        normalized = l2_normalize(projected) * np.sqrt(128)
        return (normalized - self.mean) @ self.transform_matrix.T
