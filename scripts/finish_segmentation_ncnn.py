"""Replace pnnx's unsupported 1D instance norms with native NCNN norms.

pnnx writes the four affine arrays into its weight file but leaves the layer
type as nn.InstanceNorm1d. NCNN's InstanceNorm expects a three-dimensional Mat
and reads gamma before beta, whereas pnnx writes beta before gamma.
"""

from pathlib import Path

import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
PARAM = ROOT / "models" / "segmentation.param"
WEIGHTS = ROOT / "models" / "segmentation.bin"
SOURCE = ROOT / "reference" / "FredrikKarlssonSpeech-pyannote-onnx" / "segmentation" / "model.onnx"


def finish():
    lines = PARAM.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "7767517"
    layers, blobs = map(int, lines[1].split())
    replacements = []
    channels = [1, 80, 60, 60]
    for line in lines[2:]:
        fields = line.split()
        if fields[0] != "nn.InstanceNorm1d":
            replacements.append(line)
            continue
        index = int(fields[1].split("_")[1])
        count = channels[index]
        source_blob, target_blob = fields[4:6]
        before = f"norm{index}_3d_input"
        after = f"norm{index}_3d_output"
        replacements.extend([
            f"Reshape norm{index}_expand 1 1 {source_blob} {before} 0=0 1=1 2={count}",
            f"InstanceNorm norm{index}_native 1 1 {before} {after} 0={count} 1=0.00001 2=1",
            f"Reshape norm{index}_squeeze 1 1 {after} {target_blob} 0=0 1={count}",
        ])
    assert len(replacements) == layers + 8
    PARAM.write_text("\n".join([lines[0], f"{layers + 8} {blobs + 8}", *replacements]) + "\n", encoding="utf-8")

    graph = onnx.load(SOURCE).graph
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    data = WEIGHTS.read_bytes()
    for node in graph.node:
        if node.op_type != "InstanceNormalization":
            continue
        gamma = initializers[node.input[1]].astype("<f4").tobytes()
        beta = initializers[node.input[2]].astype("<f4").tobytes()
        old = beta + gamma
        assert data.count(old) == 1, f"affine block not found for {node.name}"
        data = data.replace(old, gamma + beta, 1)
    WEIGHTS.write_bytes(data)


if __name__ == "__main__":
    finish()
