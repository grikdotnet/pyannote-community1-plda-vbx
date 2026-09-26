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
MODELS = ROOT / "models"
MANIFEST = MODELS / "manifest.json"
SOURCES = {
    "fredrik": {
        "url": "https://huggingface.co/FredrikKarlssonSpeech/pyannote-speaker-diarization-onnx",
        "revision": "0a7a3bf63c16c6718a7411a3e3fe01598fc46550",
    },
    "split": {
        "url": "https://huggingface.co/welcomyou/pyannote-community-1-onnx-split",
        "revision": "cde44c2db938c8abb755853b9a87cb3179c47803",
    },
    "but_fit_plda": {
        "url": "https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md/tree/6285693ddd5b38e8229acb93f864f3d04a82bee1/plda",
        "revision": "6285693ddd5b38e8229acb93f864f3d04a82bee1",
        "license": "CC-BY-4.0",
        "license_url": "https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md/blob/6285693ddd5b38e8229acb93f864f3d04a82bee1/plda/LICENSE",
        "redistributed_via": "https://huggingface.co/pyannote/speaker-diarization-community-1/tree/main/plda",
    },
}
REFERENCE_ONNX = {
    "segmentation": ("reference/FredrikKarlssonSpeech-pyannote-onnx/segmentation/model.onnx", "fredrik", "segmentation/model.onnx"),
    "embedding_reference": ("reference/FredrikKarlssonSpeech-pyannote-onnx/embedding/model.onnx", "fredrik", "embedding/model.onnx"),
    "embedding_encoder": ("reference/pyannote-community-1-onnx-split/embedding_encoder.onnx", "split", "embedding_encoder.onnx"),
}
ASSET_PATHS = {
    "projection_weight": ("resnet_seg_1_weight.npy", "split"),
    "projection_bias": ("resnet_seg_1_bias.npy", "split"),
    "plda": ("plda.npz", "but_fit_plda"),
    "xvec_transform": ("xvec_transform.npz", "but_fit_plda"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_contract(values):
    return {
        value.name: [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
        for value in values
    }


def inventory():
    references = {}
    for name, (relative, source, upstream_path) in REFERENCE_ONNX.items():
        path = ROOT / relative
        graph = onnx.load(path).graph
        references[name] = {
            "url": f"{SOURCES[source]['url']}/blob/{SOURCES[source]['revision']}/{upstream_path}",
            "sha256": sha256(path),
            "source": source,
            "license": "CC-BY-4.0",
            "inputs": tensor_contract(graph.input),
            "outputs": tensor_contract(graph.output),
        }

    assets = {}
    for name, (relative, source) in ASSET_PATHS.items():
        path = MODELS / relative
        entry = {
            "path": relative,
            "sha256": sha256(path),
            "source": source,
            "license": "CC-BY-4.0",
        }
        if path.suffix == ".npy":
            entry["shape"] = list(np.load(path, mmap_mode="r").shape)
        else:
            with np.load(path) as data:
                entry["arrays"] = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files}
        assets[name] = entry
    return {
        "path_base": ".",
        "sources": SOURCES,
        "reference_models": references,
        "assets": assets,
        "plda_contract": {key: assets["plda"]["arrays"][key]["shape"] for key in ("mu", "tr", "psi")},
        "transform_contract": {key: assets["xvec_transform"]["arrays"][key]["shape"] for key in ("mean1", "lda", "mean2")},
        "tool_versions": {name: importlib.metadata.version(name) for name in ("numpy", "onnx", "onnxruntime", "ncnn", "pnnx", "kaldi-native-fbank")},
        "conversion": {
            name: {
                extension: {
                    "path": f"{name}.{extension}",
                    "sha256": sha256(MODELS / f"{name}.{extension}"),
                }
                for extension in ("param", "bin")
            }
            if (MODELS / f"{name}.param").is_file() and (MODELS / f"{name}.bin").is_file()
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
        if recorded != current:
            raise SystemExit("manifest mismatch")
        print("model manifest verified")
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(MANIFEST)


if __name__ == "__main__":
    main()
