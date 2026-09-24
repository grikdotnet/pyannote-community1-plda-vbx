"""Generate or verify the local model inventory without downloading weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifest.json"
SOURCES = {
    "fredrik": {
        "url": "https://huggingface.co/FredrikKarlssonSpeech/pyannote-speaker-diarization-onnx",
        "revision": "0a7a3bf63c16c6718a7411a3e3fe01598fc46550",
    },
    "split": {
        "url": "https://huggingface.co/welcomyou/pyannote-community-1-onnx-split",
        "revision": "cde44c2db938c8abb755853b9a87cb3179c47803",
    },
    "community_plda": {
        "url": "https://huggingface.co/pyannote/speaker-diarization-community-1/tree/main/plda",
        "revision": None,
        "status": "local files supplied; upstream revision not recorded",
    },
    "vbx_reference": {
        "url": "https://github.com/BUTSpeechFIT/VBx",
        "revision": None,
        "status": "local source pinned by SHA-256; git revision unavailable",
    },
}
PATHS = {
    "segmentation": ("reference/FredrikKarlssonSpeech-pyannote-onnx/segmentation/model.onnx", "fredrik"),
    "embedding_reference": ("reference/FredrikKarlssonSpeech-pyannote-onnx/embedding/model.onnx", "fredrik"),
    "embedding_encoder": ("reference/pyannote-community-1-onnx-split/embedding_encoder.onnx", "split"),
    "projection_weight": ("models/resnet_seg_1_weight.npy", "split"),
    "projection_bias": ("models/resnet_seg_1_bias.npy", "split"),
    "plda": ("models/plda.npz", "community_plda"),
    "xvec_transform": ("models/xvec_transform.npz", "community_plda"),
}
REFERENCE_PATHS = ("reference/vbx/VBx/VBx.py", "reference/vbx/VBx/diarization_lib.py")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_contract(values):
    return {
        value.name: [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
        for value in values
    }


def inventory():
    assets = {}
    for name, (relative, source) in PATHS.items():
        path = ROOT / relative
        entry = {
            "path": relative,
            "sha256": sha256(path),
            "source": source,
            "license": "CC-BY-4.0",
        }
        if path.suffix == ".onnx":
            graph = onnx.load(path).graph
            entry["inputs"] = tensor_contract(graph.input)
            entry["outputs"] = tensor_contract(graph.output)
        elif path.suffix == ".npy":
            entry["shape"] = list(np.load(path, mmap_mode="r").shape)
        else:
            with np.load(path) as data:
                entry["arrays"] = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files}
        assets[name] = entry
    return {
        "sources": SOURCES,
        "assets": assets,
        "reference_code": {
            relative: {"sha256": sha256(ROOT / relative), "license": "Apache-2.0"}
            for relative in REFERENCE_PATHS
        },
        "plda_contract": {key: assets["plda"]["arrays"][key]["shape"] for key in ("mu", "tr", "psi")},
        "transform_contract": {key: assets["xvec_transform"]["arrays"][key]["shape"] for key in ("mean1", "lda", "mean2")},
        "tool_versions": {name: importlib.metadata.version(name) for name in ("numpy", "onnx", "onnxruntime", "ncnn", "pnnx", "kaldi-native-fbank")},
        "conversion": {
            name: {
                extension: {
                    "path": f"models/{name}.{extension}",
                    "sha256": sha256(ROOT / f"models/{name}.{extension}"),
                }
                for extension in ("param", "bin")
            }
            if (ROOT / f"models/{name}.param").is_file() and (ROOT / f"models/{name}.bin").is_file()
            else "pending"
            for name in ("segmentation", "embedding_encoder")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = inventory()
    if args.check:
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key in ("sources", "assets", "reference_code", "plda_contract", "transform_contract", "tool_versions", "conversion"):
            if recorded[key] != current[key]:
                raise SystemExit(f"manifest mismatch: {key}")
        print("model manifest verified")
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(MANIFEST)


if __name__ == "__main__":
    main()
