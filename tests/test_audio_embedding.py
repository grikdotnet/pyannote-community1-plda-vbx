from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf

from diarization.audio import compute_fbank, read_wav
from diarization.embedding import EmbeddingModel, masked_stats_pool


ROOT = Path(__file__).resolve().parents[1]


def test_wav_reader_accepts_both_fixture_encodings(tmp_path):
    for subtype in ("PCM_16", "FLOAT"):
        path = tmp_path / f"{subtype}.wav"
        sf.write(path, np.array([0.0, 0.25, -0.25], dtype=np.float32), 16000, subtype=subtype)
        result = read_wav(path)
        np.testing.assert_allclose(result, [0, 0.25, -0.25], atol=1 / 32768)


def test_wav_reader_rejects_stereo_and_wrong_rate(tmp_path):
    for samples, rate in ((np.zeros((100, 2)), 16000), (np.zeros(100), 8000)):
        path = tmp_path / "bad.wav"
        sf.write(path, samples, rate)
        with pytest.raises(ValueError):
            read_wav(path)


def test_masked_pool_uses_unbiased_weighted_variance():
    features = np.array([[1, 3, 100], [2, 4, 200]], dtype=np.float32)
    pooled = masked_stats_pool(features, np.array([1, 1, 0], dtype=np.float32))
    np.testing.assert_allclose(pooled, [2, 3, np.sqrt(2), np.sqrt(2)], atol=1e-5)


def test_all_active_split_embedding_matches_full_onnx():
    audio = read_wav(ROOT / "fixture" / "dev00.wav")[:160000]
    fbank = compute_fbank(audio)
    assert fbank.shape == (998, 80)
    actual = EmbeddingModel(ROOT / "models").embed(fbank, np.ones(125, dtype=np.float32))
    reference = ort.InferenceSession(
        str(ROOT / "reference" / "FredrikKarlssonSpeech-pyannote-onnx" / "embedding" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    ).run(None, {"fbank": fbank[None]})[0][0]
    assert actual.shape == (256,)
    np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
