"""NCNN inference boundary. The public arrays follow the ONNX contracts."""

from pathlib import Path

import ncnn
import numpy as np


def close_vulkan() -> None:
    """Release NCNN's process-wide Vulkan instance after GPU models are gone."""
    if hasattr(ncnn, "destroy_gpu_instance"):
        ncnn.destroy_gpu_instance()


class NCNNModel:
    def __init__(self, prefix: Path, gpu_index: int | None = None):
        self.prefix = Path(prefix)
        self.net = ncnn.Net()
        self.net.opt.num_threads = 1
        if gpu_index is not None:
            if not hasattr(ncnn, "get_gpu_count") or not hasattr(self.net, "set_vulkan_device"):
                raise RuntimeError("installed NCNN Python binding has no Vulkan support")
            gpu_count = ncnn.get_gpu_count()
            if not 0 <= gpu_index < gpu_count:
                raise ValueError(f"Vulkan GPU index {gpu_index} is unavailable (found {gpu_count} devices)")
            self.net.set_vulkan_device(gpu_index)
            self.net.opt.use_vulkan_compute = True
            # Keep the converted FP32 models close to their CPU/ONNX outputs.
            self.net.opt.use_fp16_packed = False
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_arithmetic = False
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
