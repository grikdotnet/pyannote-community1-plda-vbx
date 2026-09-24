from pathlib import Path

import numpy as np

from diarization.rttm import Segment, format_rttm, score_rttm
from diarization.segmentation import decode_powerset, segment_audio


ROOT = Path(__file__).resolve().parents[1]


def test_powerset_decoding_keeps_two_simultaneous_speakers():
    logits = np.eye(7, dtype=np.float32)[None]
    decoded = decode_powerset(logits)
    np.testing.assert_array_equal(decoded[0, 4], [1, 1, 0])
    np.testing.assert_array_equal(decoded[0, 5], [1, 0, 1])
    np.testing.assert_array_equal(decoded[0, 6], [0, 1, 1])


def test_segmentation_windows_have_model_frame_count():
    audio = np.zeros(160000, dtype=np.float32)
    class FakeModel:
        def run(self, data):
            assert data.shape == (1, 1, 160000)
            return np.zeros((1, 589, 7), dtype=np.float32)
    scores, starts = segment_audio(audio, FakeModel())
    assert scores.shape == (2, 589, 7)
    assert starts == [0, 16000]


def test_rttm_round_trip_and_overlap_scoring():
    truth = [Segment(0, 2, "A"), Segment(1, 3, "B")]
    candidate = [Segment(0, 2, "first"), Segment(1, 3, "second")]
    text = format_rttm("sample", candidate)
    assert len(text.splitlines()) == 2
    assert all(line.startswith("SPEAKER sample 1 ") for line in text.splitlines())
    result = score_rttm(truth, candidate)
    assert result["der"] == 0
    missing = score_rttm(truth, [candidate[0]])
    assert missing["miss"] == 2
    assert missing["der"] == 0.5
    extra = candidate + [Segment(3, 4, "third")]
    assert score_rttm(truth, extra)["false_alarm"] == 1
    assert score_rttm(truth, extra, uem=(0, 3))["false_alarm"] == 0
