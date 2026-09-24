"""Reproduce the pinned FP32 pnnx conversions in the local models folder."""

import subprocess
import sys
import tempfile
import importlib.metadata
from pathlib import Path

from finish_segmentation_ncnn import finish


ROOT = Path(__file__).resolve().parents[1]
PNNX = Path(sys.executable).parent / ("pnnx.exe" if sys.platform == "win32" else "pnnx")
EXPECTED_VERSIONS = {
    "pnnx": "20260526",
    "ncnn": "1.0.20260526",
    "onnx": "1.19.0",
    "onnxruntime": "1.23.0",
}


def convert(source: Path, name: str, first: str, second: str, temporary: Path):
    output = ROOT / "models" / name
    command = [
        str(PNNX), str(source), f"inputshape={first}", f"inputshape2={second}", "fp16=0",
        f"ncnnparam={output}.param", f"ncnnbin={output}.bin",
        f"pnnxparam={temporary / name}.pnnx.param", f"pnnxbin={temporary / name}.pnnx.bin",
        f"pnnxpy={temporary / name}_pnnx.py", f"pnnxonnx={temporary / name}.pnnx.onnx",
        f"ncnnpy={temporary / name}_ncnn.py",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} {expected} required, found {actual}")
    if not PNNX.is_file():
        raise FileNotFoundError(f"pnnx executable missing: {PNNX}")
    with tempfile.TemporaryDirectory() as work:
        temporary = Path(work)
        convert(ROOT / "reference/FredrikKarlssonSpeech-pyannote-onnx/segmentation/model.onnx",
                "segmentation", "[1,1,160000]", "[1,1,80000]", temporary)
        finish()
        convert(ROOT / "reference/pyannote-community-1-onnx-split/embedding_encoder.onnx",
                "embedding_encoder", "[1,998,80]", "[1,500,80]", temporary)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "model_manifest.py")], check=True)


if __name__ == "__main__":
    main()
