# NCNN Pyannote Community-1 diarization

This pipeline reads 16 kHz mono PCM16 or float32 WAV and writes
overlap-aware RTTM. Neural inference uses NCNN.

## Run

On Windows, create a virtual environment and install the package into it
(skip the first command if `.venv` already exists):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -e .
```

Run with that same interpreter:

```powershell
.\.venv\Scripts\python.exe -m diarization.cli fixture/dev00.wav dev00.rttm
.\.venv\Scripts\python.exe -m diarization.cli fixture/ES2005a.wav ES2005a.rttm --min-speakers 1 --max-speakers 8
.\.venv\Scripts\python.exe -m diarization.cli fixture/dev00.wav dev00-gpu.rttm --gpu-index 0
```

Speaker count is estimated automatically. The optional bounds constrain the
number of global speaker clusters. Each RTTM line uses the input WAV stem as
the recording ID.

Inference uses the CPU by default. To use a Vulkan GPU, install an NCNN Python
build with Vulkan support and a working Vulkan driver, then call with `--gpu-index`. 
When Vulkan is initialized, NCNN writes a capability listing for every detected
device to stderr. Vulkan accelerates the segmentation and embedding encoder networks; audio
features, embedding pooling, PLDA, and VBx remain on the CPU.
When using `Diarizer` directly with a GPU, release references to its models and
call `diarization.models.close_vulkan()` before process exit (CLI comand does this).

## Reproduce and verify

Source revisions, licenses, and the pnnx conversion step are in [model provenance](models/README.md). 

Scripts used:
```
python scripts/convert_models.py
python scripts/model_manifest.py --check
mypy src/diarization --ignore-missing-imports --check-untyped-defs
```

The numerical parity tests compare segmentation scores and embedding frame
features against the FP32 ONNX graphs at two shapes. The CLI tests score the
fixtures with zero collar and overlap included, and report miss, false alarm,
and confusion separately.
