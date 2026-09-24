import importlib.util
from pathlib import Path

import numpy as np

from diarization.plda import PLDA, score_in_lda_space
from diarization.vbx import cluster_embeddings, refine


ROOT = Path(__file__).resolve().parents[1]


def reference_module(filename):
    spec = importlib.util.spec_from_file_location("local_reference", ROOT / "reference" / "vbx" / "VBx" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plda_transform_contract_and_scores_match_reference():
    plda = PLDA(ROOT / "models")
    embeddings = np.random.default_rng(44).normal(size=(5, 256)).astype(np.float32)
    features = plda.transform(embeddings)
    assert features.shape == (5, 128)
    assert np.isfinite(features).all()
    x = np.array([[0.2, -0.3], [-0.1, 0.4]], dtype=np.float64)
    psi = np.array([0.4, 1.2])
    reference = reference_module("diarization_lib.py").PLDA_scoring_in_LDA_space(x, x, psi)
    np.testing.assert_allclose(score_in_lda_space(x, x, psi), reference, atol=1e-12)


def test_vbx_refinement_matches_pinned_local_reference():
    data = np.array([[0.1, 0.5], [0.2, 0.4], [1.4, -0.5], [1.5, -0.4]], dtype=np.float64)
    psi = np.array([0.4, 1.2])
    gamma = np.array([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]])
    pi = np.array([0.5, 0.5])
    expected_gamma, expected_pi, _ = reference_module("VBx.py").VBx(
        data, psi, loopProb=0.9, Fa=0.07, Fb=0.8, pi=pi.copy(), gamma=gamma.copy(), maxIters=4
    )
    actual_gamma, actual_pi = refine(data, psi, gamma, pi, loop_probability=0.9, fa=0.07, fb=0.8, max_iterations=4)
    np.testing.assert_allclose(actual_gamma, expected_gamma, atol=1e-10)
    np.testing.assert_allclose(actual_pi, expected_pi, atol=1e-10)
    np.testing.assert_array_equal(actual_gamma.argmax(axis=1), expected_gamma.argmax(axis=1))


def test_cluster_count_obeys_speaker_bounds():
    rng = np.random.default_rng(9)
    centers = rng.normal(size=(2, 256)).astype(np.float32)
    embeddings = centers[None].repeat(4, axis=0) + rng.normal(0, 0.01, (4, 2, 256))
    activity = np.ones((4, 10, 2), dtype=np.float32)
    labels, centroids = cluster_embeddings(
        embeddings.astype(np.float32), np.ones((4, 2), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )
    assert centroids.shape == (2, 256)
    assert set(labels.ravel()) == {0, 1}
