from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading

import numpy as np
import pytest

from diarization.audio import read_wav
from diarization.embedding import EmbeddingModel, EmbeddingResult, assemble_embeddings, embed_window
from diarization.models import NCNNModel
from diarization.pipeline import Diarizer, reconstruct
from diarization.rttm import format_rttm
from diarization.segmentation import SegmentationResult, assemble_segmentation, make_windows, segment_window


ROOT = Path(__file__).resolve().parents[1]


def test_reconstruction_retains_two_speech_supported_ids_in_overlapping_windows():
    activity = np.zeros((2, 100, 1), dtype=np.float32)
    activity[0, 59:100, 0] = 1
    activity[1, :41, 0] = 1
    labels = np.array([[0], [1]], dtype=np.int32)

    turns = reconstruct(activity, labels, [0, 16000], 2.0, minimum=2)

    assert {turn.speaker for turn in turns} == {"speaker_00", "speaker_01"}
    assert all(0.99 <= turn.start < turn.end <= 1.69 for turn in turns)


def test_reconstruction_retains_sparse_speech_diluted_by_window_coverage():
    activity = np.zeros((3, 180, 1), dtype=np.float32)
    activity[0, 120, 0] = 1
    activity[1, 62, 0] = 1
    labels = np.array([[0], [1], [-2]], dtype=np.int32)

    turns = reconstruct(activity, labels, [0, 16000, 32000], 4.0,
                        count_threshold=0.6, minimum=2)

    assert {turn.speaker for turn in turns} == {"speaker_00", "speaker_01"}
    assert all(2.0 <= turn.start < turn.end < 2.1 for turn in turns)


def test_padded_activity_does_not_count_as_valid_observation():
    activity = np.zeros((1, 100, 2), dtype=np.float32)
    activity[0, :10, 0] = 1
    activity[0, 80:90, 1] = 1
    vectors = np.zeros((1, 2, 256), dtype=np.float32)
    vectors[0, 0, 0] = 1
    vectors[0, 1, 1] = 1
    segmentation = SegmentationResult(activity, (0,), 16000)
    embeddings = EmbeddingResult(vectors, np.ones((1, 2), dtype=bool))
    diarizer = Diarizer(ROOT / "models", minimum_speakers=2, maximum_speakers=2)

    clustering = diarizer.cluster(segmentation, embeddings)
    turns = diarizer.reconstruct(np.zeros(16000, dtype=np.float32), segmentation, clustering)

    assert clustering.labels.tolist() == [[0, -2]]
    assert {turn.speaker for turn in turns} == {"speaker_00"}


def test_final_maximum_caps_returned_speaker_ids():
    activity = np.zeros((1, 30, 3), dtype=np.float32)
    for slot in range(3):
        activity[0, slot * 10:(slot + 1) * 10, slot] = 1
    vectors = np.zeros((1, 3, 256), dtype=np.float32)
    vectors[0, np.arange(3), np.arange(3)] = 1
    diarizer = Diarizer(ROOT / "models", minimum_speakers=2, maximum_speakers=2)
    segmentation = SegmentationResult(activity, (0,), 16000)
    clustering = diarizer.cluster(segmentation, EmbeddingResult(vectors, np.ones((1, 3), dtype=bool)))
    turns = diarizer.reconstruct(np.zeros(16000, dtype=np.float32), segmentation, clustering)
    assert {turn.speaker for turn in turns} == {"speaker_00", "speaker_01"}


@pytest.mark.parametrize("samples,starts", [
    (0, [0]), (159999, [0]), (160000, [0, 16000]),
    (160001, [0, 16000]), (176000, [0, 16000, 32000]),
])
def test_window_schedule_preserves_padding_and_stride(samples, starts):
    windows = list(make_windows(np.ones(samples, dtype=np.float32)))
    assert [window.index for window in windows] == list(range(len(starts)))
    assert [window.start for window in windows] == starts
    assert all(window.samples.shape == (160000,) for window in windows)
    assert windows[-1].samples[-1] == (1 if samples >= starts[-1] + 160000 else 0)


def test_stages_match_diarize_and_preserve_speaker_ids():
    audio = read_wav(ROOT / "fixture" / "dev00.wav")
    diarizer = Diarizer(ROOT / "models")
    segmentation = diarizer.segment(audio)
    embeddings = diarizer.extract_embeddings(audio, segmentation)
    clustering = diarizer.cluster(segmentation, embeddings)
    assert np.mean(segmentation.activity.sum(axis=-1) >= 2) < 0.01
    assert len(clustering.centroids) > 1  # Exercises short-window reassignment.
    explicit = diarizer.reconstruct(audio, segmentation, clustering)
    wrapped = diarizer.diarize(audio)

    assert explicit == wrapped
    assert format_rttm("dev00", explicit) == format_rttm("dev00", wrapped)
    assert {segment.speaker for segment in explicit} == {"speaker_00", "speaker_01"}


def test_shuffled_window_results_assemble_like_sequential_stages():
    audio = read_wav(ROOT / "fixture" / "dev00.wav")[:32000]
    diarizer = Diarizer(ROOT / "models")
    windows = list(make_windows(audio))
    segmentation = diarizer.segment(audio)
    window_segments = [segment_window(window, diarizer.segmentation) for window in windows]
    assembled_segments = assemble_segmentation(window_segments[::-1], len(audio))
    np.testing.assert_array_equal(assembled_segments.activity, segmentation.activity)
    assert assembled_segments.starts == segmentation.starts

    embeddings = diarizer.extract_embeddings(audio, segmentation)
    window_embeddings = [
        embed_window(window, window_segments[window.index].activity, diarizer.embedding)
        for window in windows
    ]
    assembled_embeddings = assemble_embeddings(window_embeddings[::-1], assembled_segments)
    np.testing.assert_array_equal(assembled_embeddings.vectors, embeddings.vectors)
    np.testing.assert_array_equal(assembled_embeddings.train_mask, embeddings.train_mask)
    with pytest.raises(ValueError, match="duplicate"):
        assemble_embeddings([*window_embeddings, window_embeddings[0]], assembled_segments)
    with pytest.raises(ValueError, match="missing"):
        assemble_embeddings(window_embeddings[:-1], assembled_segments)
    with pytest.raises(ValueError, match="start"):
        assemble_embeddings([replace(window_embeddings[0], start=-1), *window_embeddings[1:]], assembled_segments)
    clustered = diarizer.cluster(assembled_segments, assembled_embeddings)
    assert diarizer.reconstruct(audio, assembled_segments, clustered) == diarizer.reconstruct(
        audio, segmentation, diarizer.cluster(segmentation, embeddings)
    )


def test_assembly_rejects_missing_and_duplicate_window_results():
    audio = np.zeros(16000, dtype=np.float32)
    windows = list(make_windows(audio))
    model = NCNNModel(ROOT / "models" / "segmentation")
    results = [segment_window(window, model) for window in windows]
    with pytest.raises(ValueError, match="duplicate"):
        assemble_segmentation([results[0], results[0], *results[1:]], len(audio))
    with pytest.raises(ValueError, match="missing"):
        assemble_segmentation(results[:-1], len(audio))


def test_empty_activity_skips_embeddings_and_returns_no_segments():
    class SilentSegmentation:
        def run(self, data):
            scores = np.zeros((1, 589, 7), dtype=np.float32)
            scores[..., 0] = 1
            return scores

    class UnusedEmbedding:
        def encode(self, features):
            raise AssertionError("silent windows must skip embedding inference")

    diarizer = object.__new__(Diarizer)
    diarizer.segmentation = SilentSegmentation()
    diarizer.embedding = UnusedEmbedding()
    diarizer.plda = None
    diarizer.minimum_speakers = 2
    diarizer.maximum_speakers = None
    audio = np.zeros(16000, dtype=np.float32)
    segmentation = diarizer.segment(audio)
    embeddings = diarizer.extract_embeddings(audio, segmentation)
    clustering = diarizer.cluster(segmentation, embeddings)
    assert np.isnan(embeddings.vectors).all()
    assert not embeddings.train_mask.any()
    assert not len(clustering.centroids)
    assert diarizer.reconstruct(audio, segmentation, clustering) == diarizer.diarize(audio) == []


def test_threaded_windows_with_worker_owned_models_match_sequential():
    audio = read_wav(ROOT / "fixture" / "dev00.wav")[:16000]
    windows = list(make_windows(audio))
    diarizer = Diarizer(ROOT / "models")
    sequential_segmentation = diarizer.segment(audio)
    sequential_embeddings = diarizer.extract_embeddings(audio, sequential_segmentation)
    local = threading.local()

    def worker_model(name):
        model = getattr(local, name, None)
        if model is None:
            model = NCNNModel(ROOT / "models" / "segmentation") if name == "segmentation" else EmbeddingModel(ROOT / "models")
            setattr(local, name, model)
        return model

    with ThreadPoolExecutor(max_workers=2) as pool:
        segments = list(pool.map(lambda window: segment_window(window, worker_model("segmentation")), windows))
        threaded_segmentation = assemble_segmentation(segments[::-1], len(audio))
        embeddings = list(pool.map(
            lambda window: embed_window(window, segments[window.index].activity, worker_model("embedding")), windows
        ))
    threaded_embeddings = assemble_embeddings(embeddings[::-1], threaded_segmentation)
    np.testing.assert_array_equal(threaded_segmentation.activity, sequential_segmentation.activity)
    np.testing.assert_array_equal(threaded_embeddings.vectors, sequential_embeddings.vectors)
    np.testing.assert_array_equal(threaded_embeddings.train_mask, sequential_embeddings.train_mask)
