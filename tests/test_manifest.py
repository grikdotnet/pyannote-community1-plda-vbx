import json
from pathlib import Path

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_describes_local_model_contracts():
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text())
    assets = manifest["assets"]
    for name in ("segmentation", "embedding_reference", "embedding_encoder", "projection_weight", "projection_bias", "plda", "xvec_transform"):
        entry = assets[name]
        assert (ROOT / entry["path"]).is_file()
        assert len(entry["sha256"]) == 64
        assert entry["license"] == "CC-BY-4.0"

    assert assets["segmentation"]["inputs"] == {"waveform": ["batch", 1, "samples"]}
    assert assets["embedding_encoder"]["inputs"] == {"fbank_features": ["batch_size", "num_frames", 80]}
    assert np.load(ROOT / assets["projection_weight"]["path"]).shape == (256, 5120)
    assert np.load(ROOT / assets["projection_bias"]["path"]).shape == (256,)
    assert assets["projection_weight"]["path"] == "models/resnet_seg_1_weight.npy"
    assert assets["projection_bias"]["path"] == "models/resnet_seg_1_bias.npy"
    assert assets["plda"]["path"] == "models/plda.npz"
    assert assets["xvec_transform"]["path"] == "models/xvec_transform.npz"
    assert {key: list(np.load(ROOT / assets["plda"]["path"])[key].shape) for key in ("mu", "tr", "psi")} == manifest["plda_contract"]
    assert {key: list(np.load(ROOT / assets["xvec_transform"]["path"])[key].shape) for key in ("mean1", "lda", "mean2")} == manifest["transform_contract"]
    assert onnx.load(ROOT / assets["segmentation"]["path"]).graph.output[0].name == "scores"
    assert set(manifest["reference_code"]) == {"reference/vbx/VBx/VBx.py", "reference/vbx/VBx/diarization_lib.py"}
    split_script = ROOT / "reference" / "pyannote-community-1-onnx-split" / "split_pyannote_embedding.py"
    compile(split_script.read_text(encoding="utf-8"), str(split_script), "exec")
