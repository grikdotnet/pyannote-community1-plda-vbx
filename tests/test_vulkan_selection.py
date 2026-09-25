from pathlib import Path
from types import SimpleNamespace

import pytest

from diarization import models
from diarization.cli import main
from diarization.pipeline import Diarizer


class FakeNet:
    created = []

    def __init__(self):
        self.opt = SimpleNamespace(num_threads=None, use_vulkan_compute=False)
        self.calls = []
        self.created.append(self)

    def set_vulkan_device(self, index):
        self.calls.append(("device", index))

    def load_param(self, path):
        self.calls.append(("param", path))
        return 0

    def load_model(self, path):
        self.calls.append(("model", path))
        return 0


@pytest.fixture
def fake_ncnn(monkeypatch):
    FakeNet.created = []
    monkeypatch.setattr(models.ncnn, "Net", FakeNet)
    monkeypatch.setattr(models.ncnn, "get_gpu_count", lambda: 2)
    return FakeNet


def test_ncnn_model_selects_vulkan_before_loading(tmp_path, fake_ncnn):
    models.NCNNModel(tmp_path / "segmentation", gpu_index=1)
    net = fake_ncnn.created[0]
    assert net.opt.use_vulkan_compute is True
    assert net.calls[0] == ("device", 1)
    assert [call[0] for call in net.calls] == ["device", "param", "model"]


def test_ncnn_model_defaults_to_cpu(tmp_path, fake_ncnn):
    models.NCNNModel(tmp_path / "segmentation")
    net = fake_ncnn.created[0]
    assert net.opt.use_vulkan_compute is False
    assert [call[0] for call in net.calls] == ["param", "model"]


@pytest.mark.parametrize("index", [-1, 2])
def test_ncnn_model_rejects_invalid_device(tmp_path, fake_ncnn, index):
    with pytest.raises(ValueError, match="Vulkan GPU index"):
        models.NCNNModel(tmp_path / "segmentation", gpu_index=index)
    assert not fake_ncnn.created or fake_ncnn.created[0].calls == []


def test_ncnn_model_reports_missing_vulkan_binding(tmp_path, fake_ncnn, monkeypatch):
    monkeypatch.delattr(models.ncnn, "get_gpu_count")
    with pytest.raises(RuntimeError, match="no Vulkan support"):
        models.NCNNModel(tmp_path / "segmentation", gpu_index=0)


def test_diarizer_uses_same_device_for_both_models(monkeypatch, tmp_path):
    indices = []

    class StubModel:
        def __init__(self, prefix: Path, gpu_index=None):
            indices.append((prefix.name, gpu_index))

    monkeypatch.setattr("diarization.pipeline.NCNNModel", StubModel)
    monkeypatch.setattr("diarization.embedding.NCNNModel", StubModel)
    monkeypatch.setattr("diarization.embedding.np.load", lambda path: None)
    monkeypatch.setattr("diarization.pipeline.PLDA", lambda path: None)
    Diarizer(tmp_path, gpu_index=1)
    assert indices == [("segmentation", 1), ("embedding_encoder", 1)]


def test_cli_passes_gpu_index(monkeypatch, tmp_path):
    selected = []
    closed = []

    class StubDiarizer:
        def __init__(self, root, minimum_speakers, maximum_speakers, gpu_index):
            selected.append(gpu_index)

        def diarize(self, audio):
            return []

    monkeypatch.setattr("diarization.cli.Diarizer", StubDiarizer)
    monkeypatch.setattr("diarization.cli.read_wav", lambda path: None)
    monkeypatch.setattr("diarization.cli.close_vulkan", lambda: closed.append(True))
    output = tmp_path / "output.rttm"
    main([str(tmp_path / "input.wav"), str(output), "--gpu-index", "1"])
    assert selected == [1]
    assert closed == [True]
    assert output.exists()
