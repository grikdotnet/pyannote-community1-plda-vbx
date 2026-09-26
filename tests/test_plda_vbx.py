import importlib.util
from pathlib import Path

import numpy as np
import pytest

from diarization.plda import PLDA, score_in_lda_space
from diarization.pipeline import reconstruct
from diarization.vbx import cluster_embeddings, refine


ROOT = Path(__file__).resolve().parents[1]


def three_slot_embeddings(chunks=1):
    embeddings = np.zeros((chunks, 3, 256), dtype=np.float32)
    embeddings[:, 0, 0] = 1
    embeddings[:, 1, 0] = 0.99
    embeddings[:, 1, 1] = 0.01
    embeddings[:, 2, 2] = 1
    return embeddings


def slot_activity(ranges, chunks=1):
    activity = np.zeros((chunks, 30, 3), dtype=np.float32)
    for slot, (start, end) in enumerate(ranges):
        activity[:, start:end, slot] = 1
    return activity


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


def test_orthogonal_observations_honor_exactly_two_speakers():
    embeddings = np.zeros((1, 3, 256), dtype=np.float32)
    embeddings[0, np.arange(3), np.arange(3)] = 1
    activity = slot_activity([(0, 8), (10, 18), (20, 28)])
    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 3), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )
    turns = reconstruct(activity, labels, [0], 0.6)
    assert len(centroids) == 2
    assert {turn.speaker for turn in turns} == {"speaker_00", "speaker_01"}


def test_identical_observations_in_separate_windows_honor_minimum():
    embeddings = np.zeros((2, 1, 256), dtype=np.float32)
    embeddings[:, 0, 0] = 1
    activity = np.ones((2, 10, 1), dtype=np.float32)
    labels, centroids = cluster_embeddings(
        embeddings, np.ones((2, 1), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )
    assert len(centroids) == 2
    assert set(labels.ravel()) == {0, 1}
    turns = reconstruct(activity, labels, [0, 16000], 2.0, minimum=2)
    assert {turn.speaker for turn in turns} == {"speaker_00", "speaker_01"}


def test_valid_slots_outside_training_mask_honor_minimum():
    embeddings = np.zeros((1, 3, 256), dtype=np.float32)
    embeddings[0, np.arange(3), np.arange(3)] = 1
    activity = slot_activity([(0, 8), (10, 18), (20, 28)])
    labels, centroids = cluster_embeddings(
        embeddings, np.array([[True, False, False]]), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )
    assert len(centroids) == 2
    assert set(labels.ravel()) == {0, 1}


def test_minimum_is_capped_by_valid_observations():
    embeddings = np.zeros((1, 2, 256), dtype=np.float32)
    embeddings[0, 0, 0] = 1
    activity = np.zeros((1, 10, 2), dtype=np.float32)
    activity[0, :5, 0] = 1
    activity[0, 5:, 1] = 1
    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 2), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )
    assert len(centroids) == 1
    assert labels.tolist() == [[0, -2]]


def test_sequential_local_speakers_keep_all_speech_with_two_global_speakers():
    embeddings = three_slot_embeddings()
    activity = slot_activity([(0, 8), (10, 18), (20, 28)])

    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 3), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )
    turns = reconstruct(activity, labels, [0], 0.6)

    assert centroids.shape == (2, 256)
    assert (labels >= 0).all()
    assert labels[0, 0] == labels[0, 1]
    assert [(turn.start, turn.end) for turn in turns] == [
        (0, 0.135), (0.16875, 0.30375), (0.3375, 0.47250000000000003),
    ]


@pytest.mark.parametrize("maximum", [None, 3])
def test_temporal_conflict_creates_speaker_when_capacity_remains(maximum):
    embeddings = three_slot_embeddings()
    activity = slot_activity([(0, 20), (0, 30), (10, 30)])

    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 3), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=maximum,
    )

    assert labels.tolist() == [[0, 1, 2]]
    assert centroids.shape == (3, 256)


def test_temporal_conflict_forces_best_speaker_at_limit():
    embeddings = three_slot_embeddings()
    activity = slot_activity([(0, 20), (0, 30), (10, 30)])

    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 3), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )

    assert labels.tolist() == [[0, 0, 1]]
    assert centroids.shape == (2, 256)


def test_temporal_conflict_skips_best_speaker_when_another_is_available():
    embeddings = three_slot_embeddings()
    activity = slot_activity([(0, 15), (10, 20), (20, 30)])

    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 3), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )

    assert labels.tolist() == [[0, 1, 1]]
    assert centroids.shape == (2, 256)


def test_overlapping_windows_can_reuse_global_speaker_id():
    embeddings = three_slot_embeddings(chunks=2)
    activity = slot_activity([(0, 8), (10, 18), (20, 28)], chunks=2)

    labels, centroids = cluster_embeddings(
        embeddings, np.ones((2, 3), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )

    assert labels.tolist() == [[0, 0, 1], [0, 0, 1]]
    assert centroids.shape == (2, 256)


def test_silent_and_nonfinite_local_slots_remain_unlabeled():
    embeddings = np.zeros((1, 4, 256), dtype=np.float32)
    embeddings[0, 0, 0] = 1
    embeddings[0, 1, 2] = 1
    embeddings[0, 2] = np.nan
    embeddings[0, 3, 3] = 1
    activity = np.zeros((1, 30, 4), dtype=np.float32)
    activity[0, 0:10, 0] = 1
    activity[0, 10:20, 1] = 1
    activity[0, 20:30, 2] = 1

    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 4), dtype=bool), activity,
        PLDA(ROOT / "models"), minimum=2, maximum=2,
    )

    assert labels.tolist() == [[0, 1, -2, -2]]
    assert centroids.shape == (2, 256)


def test_new_speaker_centroid_is_available_in_later_window():
    embeddings = three_slot_embeddings(chunks=2)
    embeddings[1] = 0
    embeddings[1, 0] = embeddings[0, 1]
    activity = slot_activity([(0, 20), (0, 30), (10, 30)], chunks=2)
    activity[1] = 0
    activity[1, 0:10, 0] = 1
    train_mask = np.zeros((2, 3), dtype=bool)
    train_mask[0] = True

    labels, centroids = cluster_embeddings(
        embeddings, train_mask, activity,
        PLDA(ROOT / "models"), minimum=2, maximum=3,
    )

    assert labels.tolist() == [[0, 1, 2], [1, -2, -2]]
    assert centroids.shape == (3, 256)


def test_equal_plda_scores_choose_a_deterministic_speaker():
    class FlatPLDA:
        psi = np.ones(2, dtype=np.float64)

        def transform(self, values):
            return np.zeros((len(values), 2), dtype=np.float64)

    embeddings = three_slot_embeddings()
    activity = slot_activity([(0, 8), (10, 18), (20, 28)])

    labels, centroids = cluster_embeddings(
        embeddings, np.ones((1, 3), dtype=bool), activity,
        FlatPLDA(), minimum=2, maximum=2,
    )

    assert labels.tolist() == [[0, 1, 1]]
    assert centroids.shape == (2, 256)
