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
[Command-line Run](#command-line-run), then point `Diarizer` at the directory
that directly contains the model files:

```python
from pathlib import Path

from diarization.audio import read_wav
from diarization.pipeline import Diarizer
from diarization.rttm import format_rttm

models_dir = Path(r"path/to/pyannote-ncnn-plda/models")
wav_path = Path("meeting.wav")

diarizer = Diarizer(models_dir)
audio = read_wav(wav_path)
segmentation = diarizer.segment(audio)
embeddings = diarizer.extract_embeddings(audio, segmentation)
clustering = diarizer.cluster(segmentation, embeddings)
segments = diarizer.reconstruct(audio, segmentation, clustering)
for segment in segments:
    print(segment.start, segment.end, segment.speaker)

Path("meeting.rttm").write_text(
    format_rttm(wav_path.stem, segments), encoding="utf-8"
)
```

`read_wav` accepts a 16 kHz mono WAV with 16-bit PCM or 32-bit float samples.
If the application already has audio samples, pass a mono, 16 kHz `float32`
NumPy array directly to the stages. `segment` returns activity and window starts;
`extract_embeddings` returns local speaker vectors and their training mask;
`cluster` returns global labels and centroids; `reconstruct` returns speaker
turns as `Segment` objects. `diarizer.diarize(audio)` remains a convenience call
that runs the four stages in order. Writing RTTM is optional. `Diarizer` also accepts `minimum_speakers`,
`maximum_speakers`, and `gpu_index` arguments.

## GPU Acceleration

Inference uses the CPU by default. To use a Vulkan GPU from Python, pass its index when creating the diarizer:

```python
diarizer = Diarizer(models_dir, gpu_index=0)
```

This requires a Vulkan-enabled NCNN Python build and a working driver. The selected GPU runs these NCNN models:

- Segmentation model
- Embedding encoder

Chunking, audio features, embedding pooling and projection, PLDA, VBx clustering, and speaker-turn reconstruction run on the CPU. When Vulkan is initialized, NCNN writes a capability listing for every detected
device to stderr. 

When using `Diarizer` directly with a GPU, release references to its models and call `diarization.models.close_vulkan()` before process exit (the CLI does this).

## Threading

The four whole-recording stages depend on each other in order. PLDA/VBx clustering needs all the embeddings. Reconstruction uses the speaker groups found by clustering to build timed speaker turns. If speakers rarely talk at the same time, reconstruction may use the embedding model again to tell them apart in short stretches.

| Calls | Run in parallel in threads? | Requirement |
| --- | --- | --- |
| `segment_window` per window | Yes | One segmentation `NCNNModel` per worker. |
| `embed_window` per window | Yes | Each window's activity is ready; one `EmbeddingModel` per worker. |
| `segment`, `extract_embeddings`, `cluster`, `reconstruct` on whole recording | No | Call in that order, or use `diarize`. |
| Complete recordings | Yes | A separate `Diarizer` per recording. |

Use one NCNN model instance per worker thread. Do not share `Diarizer`, `NCNNModel`, or `EmbeddingModel` among threads. This example continues the library example above:

```python
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import local

from diarization.embedding import EmbeddingModel, assemble_embeddings, embed_window
from diarization.models import NCNNModel
from diarization.segmentation import assemble_segmentation, make_windows, segment_window

worker = local()

def segment_job(window):
    if not hasattr(worker, "segmentation"):
        worker.segmentation = NCNNModel(models_dir / "segmentation")
    return segment_window(window, worker.segmentation)

def embed_job(window):
    if not hasattr(worker, "embedding"):
        worker.embedding = EmbeddingModel(models_dir)
    return embed_window(window, activity_by_index[window.index], worker.embedding)

def run_bounded(pool, job):
    windows = iter(make_windows(audio))
    while batch := list(islice(windows, 8)):
        yield from pool.map(job, batch)

with ThreadPoolExecutor(max_workers=2) as pool:
    window_segments = list(run_bounded(pool, segment_job))
    segmentation = assemble_segmentation(window_segments, len(audio))
    activity_by_index = {item.index: item.activity for item in window_segments}
    window_embeddings = list(run_bounded(pool, embed_job))

embeddings = assemble_embeddings(window_embeddings, segmentation)
clustering = diarizer.cluster(segmentation, embeddings)
segments = diarizer.reconstruct(audio, segmentation, clustering)
```

The assembly functions accept results in any completion order and reject
duplicate, missing, or mismatched window indices. `make_windows` yields padded
windows lazily; the batches above limit the number held by the executor.
Treat the returned arrays as read-only while other workers use them.
Separate recordings can run in parallel with separate `Diarizer` instances.

When using Vulkan, finish all workers and release their model instances before
calling `close_vulkan()`. See [NCNN's extractor guidance](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result) and [concurrent allocator guidance](https://github.com/Tencent/ncnn/wiki/custom-allocator).

## Command-line Run

On Windows, create a virtual environment and install the package into it
(skip the first command if `.venv` already exists):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -e .
```

Run with that same interpreter:

```powershell
.\.venv\Scripts\python.exe -m diarization.cli fixture/dev00.wav dev00.rttm --models-dir models
.\.venv\Scripts\python.exe -m diarization.cli fixture/ES2005a.wav ES2005a.rttm --models-dir models --min-speakers 1 --max-speakers 8
.\.venv\Scripts\python.exe -m diarization.cli fixture/dev00.wav dev00-gpu.rttm --models-dir models --gpu-index 0
```

Speaker count is estimated automatically. The optional bounds constrain the
number of global speaker clusters. Each RTTM line uses the input WAV stem as
the recording ID.

## Sources

Find sources and licenses in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). 

Scripts used for conversion:
```
python scripts/convert_models.py
python scripts/model_manifest.py --check
mypy src/diarization --ignore-missing-imports --check-untyped-defs
```
