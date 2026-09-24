"""NCNN inference boundary. The public arrays follow the ONNX contracts."""

from pathlib import Path

import ncnn
import numpy as np


class NCNNModel:
    def __init__(self, prefix: Path):
        self.prefix = Path(prefix)
        self.net = ncnn.Net()
        self.net.opt.num_threads = 1
        if self.net.load_param(str(self.prefix.with_suffix(".param"))) != 0:
            raise ValueError(f"cannot load NCNN parameters: {self.prefix}")
        if self.net.load_model(str(self.prefix.with_suffix(".bin"))) != 0:
            raise ValueError(f"cannot load NCNN weights: {self.prefix}")

    def run(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.shape[0] != 1:
            raise ValueError("NCNNModel currently accepts batch size 1")
        extractor = self.net.create_extractor()
        if extractor.input("in0", ncnn.Mat(np.ascontiguousarray(data[0])).clone()) != 0:
            raise RuntimeError("NCNN input failed")
        status, output = extractor.extract("out0")
        if status != 0:
            raise RuntimeError(f"NCNN inference failed: {status}")
        return np.array(output)[None]
