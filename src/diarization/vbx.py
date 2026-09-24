"""Deterministic VBx HMM refinement.

Equations are adapted from the Apache-2.0 BUTSpeechFIT/VBx implementation
included under ``reference/vbx/VBx/VBx.py``. Runtime imports no code from that folder.
"""

import numpy as np
from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


def forward_backward(log_likelihood: np.ndarray, transition: np.ndarray, prior: np.ndarray):
    log_transition = np.log(transition + 1e-8)
    forward = np.empty_like(log_likelihood)
    backward = np.empty_like(log_likelihood)
    forward[0] = log_likelihood[0] + np.log(prior + 1e-8)
    backward[-1] = 0
    for frame in range(1, len(log_likelihood)):
        forward[frame] = log_likelihood[frame] + logsumexp(forward[frame - 1] + log_transition.T, axis=1)
    for frame in range(len(log_likelihood) - 2, -1, -1):
        backward[frame] = logsumexp(log_transition + log_likelihood[frame + 1] + backward[frame + 1], axis=1)
    total = logsumexp(forward[-1])
    return np.exp(forward + backward - total), total, forward, backward


def refine(features: np.ndarray, psi: np.ndarray, initial: np.ndarray, prior: np.ndarray,
           *, loop_probability: float = 0.9, fa: float = 0.07, fb: float = 0.8,
           max_iterations: int = 20, tolerance: float = 1e-4):
    features = np.asarray(features, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    gamma = np.asarray(initial, dtype=np.float64).copy()
    pi = np.asarray(prior, dtype=np.float64).copy()
    dimension = features.shape[1]
    constant = -0.5 * ((features**2).sum(axis=1, keepdims=True) + dimension * np.log(2 * np.pi))
    rho = features * np.sqrt(psi)
    previous = -np.inf
    for iteration in range(max_iterations):
        inverse = 1 / (1 + fa / fb * gamma.sum(axis=0, keepdims=True).T * psi)
        alpha = fa / fb * inverse * (gamma.T @ rho)
        likelihood = fa * (rho @ alpha.T - 0.5 * (inverse + alpha**2) @ psi + constant)
        transition = np.eye(len(pi)) * loop_probability + (1 - loop_probability) * pi
        gamma, total, forward, backward = forward_backward(likelihood, transition, pi)
        objective = total + fb * 0.5 * np.sum(np.log(inverse) - inverse - alpha**2 + 1)
        pi = gamma[0] + (1 - loop_probability) * pi * np.sum(
            np.exp(logsumexp(forward[:-1], axis=1, keepdims=True) + likelihood[1:] + backward[1:] - total), axis=0
        )
        pi /= pi.sum()
        if iteration > 0 and objective - previous < tolerance:
            break
        previous = objective
    return gamma, pi


def cluster_embeddings(embeddings: np.ndarray, train_mask: np.ndarray, activity: np.ndarray,
                       plda, minimum: int = 1, maximum: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """AHC initialization, VBx refinement, then unique local-to-global assignment."""
    chunks, slots, dimension = embeddings.shape
    valid = np.isfinite(embeddings).all(axis=-1) & (activity.sum(axis=1) > 0)
    training = train_mask & valid
    if training.sum() < 2:
        training = valid
    labels = np.full((chunks, slots), -2, dtype=np.int32)
    examples = embeddings[training]
    if not len(examples):
        return labels, np.empty((0, dimension), dtype=np.float32)
    if len(examples) == 1:
        labels[valid] = 0
        return labels, examples.copy()
    normalized = examples / np.maximum(np.linalg.norm(examples, axis=1, keepdims=True), 1e-10)
    tree = linkage(normalized, method="centroid", metric="euclidean")
    groups = fcluster(tree, 0.6, criterion="distance") - 1
    group_count = len(np.unique(groups))
    desired = max(minimum, min(group_count, maximum if maximum is not None else group_count))
    desired = min(desired, len(examples))
    if desired != group_count:
        groups = cut_tree(tree, n_clusters=desired).ravel()
    _, groups = np.unique(groups, return_inverse=True)
    initial = np.full((len(groups), groups.max() + 1), 1e-6, dtype=np.float64)
    initial[np.arange(len(groups)), groups] = 1
    initial /= initial.sum(axis=1, keepdims=True)
    transformed = plda.transform(examples)
    posterior, prior = refine(transformed, plda.psi, initial, np.full(initial.shape[1], 1 / initial.shape[1]))
    active = np.flatnonzero(prior > 1e-7)
    requested = min(minimum, len(prior))
    if len(active) < requested:
        active = np.argsort(-prior)[:requested]
    weights = posterior[:, active]
    centroids = (weights.T @ examples) / np.maximum(weights.sum(axis=0)[:, None], 1e-8)
    centroid_norm = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-10)
    for chunk in range(chunks):
        slot_indices = np.flatnonzero(valid[chunk])
        if not len(slot_indices):
            continue
        vectors = embeddings[chunk, slot_indices].copy()
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-10)
        similarity = vectors @ centroid_norm.T
        rows, columns = linear_sum_assignment(similarity, maximize=True)
        labels[chunk, slot_indices[rows]] = columns
    # Stable IDs: first active local speaker occurrence determines the number.
    first: dict[int, tuple[int, int, int]] = {}
    for chunk, slot in np.argwhere(labels >= 0):
        cluster = int(labels[chunk, slot])
        frames = np.flatnonzero(activity[chunk, :, slot])
        key = (int(chunk), int(frames[0]) if len(frames) else 10**9, int(slot))
        first[cluster] = min(first.get(cluster, key), key)
    order = {old: new for new, old in enumerate(sorted(first, key=lambda value: first[value]))}
    result = np.full_like(labels, -2)
    for old, new in order.items():
        result[labels == old] = new
    ordered_centroids = np.stack([centroids[old] for old in sorted(order, key=lambda old: order[old])])
    return result, ordered_centroids
