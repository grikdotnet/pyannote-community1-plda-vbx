# NCNN Pyannote Community-1 diarization with PLDA + VBx

This is an implementation of a full speaker diarization pipeline for local
inference. It takes a WAV recording through Pyannote Community-1 segmentation,
speaker embedding extraction, and PLDA/VBx clustering, then reconstructs
speaker turns and writes an RTTM file. The neural models run with NCNN, while
audio processing, embedding pooling, PLDA, and VBx run locally in the pipeline.

## Pipeline steps

1. **Chunking:** Split the 16 kHz audio into overlapping 10-second windows, one second apart, padding the final window when needed.
2. **Segmentation:** Run the Community-1 segmentation model on each window and decode its scores into local speaker activity, including overlap.
3. **Embeddings:** Run the speaker embedding encoder, pool and project its frame features for each local speaker.
4. **PLDA:** Transform the embeddings into the PLDA space for speaker comparison.
5. **VBx clustering:** Initialize global speaker groups with hierarchical clustering, refine them with VBx, and assign local speakers to global identities.
6. **Reconstruction:** Combine the overlapping windows into timed speaker turns, which can be written as RTTM.

## Weights parity

The NCNN backend produced byte-identical RTTM files to the backend with weights numerically validated against the original PyTorch checkpoints (see [evaluation details](docs/evaluation.md)).

## Use as a Python library

Install the package in your application's environment as shown under
[Command-line Run](#command-line-run), then point `Diarizer` at this repository's
root directory so it can find `models/`:

```python
from pathlib import Path

from diarization.audio import read_wav
from diarization.pipeline import Diarizer
from diarization.rttm import format_rttm

root = Path(r"path/to/pyannote-ncnn-plda")
wav_path = Path("meeting.wav")

diarizer = Diarizer(root)
segments = diarizer.diarize(read_wav(wav_path))
for segment in segments:
    print(segment.start, segment.end, segment.speaker)

Path("meeting.rttm").write_text(
    format_rttm(wav_path.stem, segments), encoding="utf-8"
)
```

`read_wav` accepts a 16 kHz mono WAV with 16-bit PCM or 32-bit float samples.
If your application already has audio samples, pass a mono, 16 kHz `float32`
NumPy array directly to `diarize`. It returns speaker turns as `Segment`
objects; writing RTTM is optional. `Diarizer` also accepts `minimum_speakers`,
`maximum_speakers`, and `gpu_index` arguments.

## GPU Acceleration

Inference uses the CPU by default. To use a Vulkan GPU from Python, pass its index when creating the diarizer:

```python
diarizer = Diarizer(root, gpu_index=0)
```

This requires a Vulkan-enabled NCNN Python build and a working driver. The selected GPU runs these NCNN models:

- Segmentation model
- Embedding encoder

Chunking, audio features, embedding pooling and projection, PLDA, VBx clustering, and speaker-turn reconstruction run on the CPU. When Vulkan is initialized, NCNN writes a capability listing for every detected
device to stderr. 

When using `Diarizer` directly with a GPU, release references to its models and call `diarization.models.close_vulkan()` before process exit (the CLI does this).

## Threading

The sample `Diarizer` processes windows sequentially. A multi-threaded
implementation could:

a) process independent windows concurrently for segmentation and embedding extraction, then collect their results in time order before the recording-wide PLDA/VBx clustering and reconstruction; 
b) run steps in separate threads, connected with queues.

This requires explicit scheduling across threads.

## Command-line Run

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

## Sources

Source revisions, licenses, and the pnnx conversion step are in [model provenance](models/README.md). 

Scripts used:
```
python scripts/convert_models.py
python scripts/model_manifest.py --check
mypy src/diarization --ignore-missing-imports --check-untyped-defs
```
