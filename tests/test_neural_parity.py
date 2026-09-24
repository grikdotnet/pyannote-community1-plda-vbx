from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from diarization.models import NCNNModel


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("segmentation", "reference/FredrikKarlssonSpeech-pyannote-onnx/segmentation/model.onnx", "waveform", (1, 1, 160000)),
    ("segmentation", "reference/FredrikKarlssonSpeech-pyannote-onnx/segmentation/model.onnx", "waveform", (1, 1, 80000)),
    ("embedding_encoder", "reference/pyannote-community-1-onnx-split/embedding_encoder.onnx", "fbank_features", (1, 998, 80)),
    ("embedding_encoder", "reference/pyannote-community-1-onnx-split/embedding_encoder.onnx", "fbank_features", (1, 500, 80)),
)


@pytest.mark.parametrize("name,onnx_path,input_name,shape", CASES)
def test_ncnn_matches_onnx(name, onnx_path, input_name, shape):
    rng = np.random.default_rng(144)
    data = rng.normal(0, 0.1, shape).astype(np.float32)
    reference = ort.InferenceSession(str(ROOT / onnx_path), providers=["CPUExecutionProvider"]).run(None, {input_name: data})[0]
    actual = NCNNModel(ROOT / "models" / name).run(data)
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all()
    max_absolute = np.max(np.abs(actual - reference))
    max_relative = np.max(np.abs(actual - reference) / np.maximum(np.abs(reference), 1e-8))
    print(f"{name} {shape}: max absolute={max_absolute:.8g}, max relative={max_relative:.8g}")
    np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
