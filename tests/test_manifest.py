import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def test_manifest_describes_local_model_contracts():
    manifest = json.loads((MODELS / "manifest.json").read_text())
    assets = manifest["assets"]
    assert manifest["path_base"] == "."
    assert set(assets) == {"projection_weight", "projection_bias", "plda", "xvec_transform"}
    for name in assets:
        entry = assets[name]
        assert (MODELS / entry["path"]).is_file()
        assert len(entry["sha256"]) == 64
        assert entry["license"] == "CC-BY-4.0"

    assert np.load(MODELS / assets["projection_weight"]["path"]).shape == (256, 5120)
    assert np.load(MODELS / assets["projection_bias"]["path"]).shape == (256,)
    assert {key: list(np.load(MODELS / assets["plda"]["path"])[key].shape) for key in ("mu", "tr", "psi")} == manifest["plda_contract"]
    assert {key: list(np.load(MODELS / assets["xvec_transform"]["path"])[key].shape) for key in ("mean1", "lda", "mean2")} == manifest["transform_contract"]

    paths = {entry["path"]: entry["sha256"] for entry in assets.values()}
    for pair in manifest["conversion"].values():
        for entry in pair.values():
            paths[entry["path"]] = entry["sha256"]
    actual = {path.name for path in MODELS.iterdir() if path.suffix in {".param", ".bin", ".npy", ".npz"}}
    assert set(paths) == actual
    for relative, expected_hash in paths.items():
        path = MODELS / relative
        assert Path(relative).name == relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    assert set(manifest["reference_models"]) == {"segmentation", "embedding_reference", "embedding_encoder"}
    assert manifest["reference_models"]["segmentation"]["inputs"] == {"waveform": ["batch", 1, "samples"]}
    assert manifest["reference_models"]["embedding_encoder"]["inputs"] == {"fbank_features": ["batch_size", "num_frames", 80]}
    assert all("url" in entry and "path" not in entry for entry in manifest["reference_models"].values())
    assert "reference_code" not in manifest
