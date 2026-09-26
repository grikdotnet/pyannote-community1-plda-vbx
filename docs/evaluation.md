## AMI recording comparison

We validated the NCNN models against FP32 ONNX exports from [FredrikKarlssonSpeech](https://huggingface.co/FredrikKarlssonSpeech/pyannote-speaker-diarization-onnx).
The ONNX model card reports numerical validation against the original PyTorch checkpoints.

Three AMI recordings—`IS1008a`, `ES2011a`, and `TS3004a`—from the [AMI Corpus Mirror](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/) were processed with both the default NCNN backend and a reference ONNX backend. The recordings total **56 minutes 43 seconds**, including **40 minutes 45 seconds during which at least one speaker is annotated as speaking**. Both backends used the same PCM16 WAV files.

The backends share masked pooling, projection, PLDA/VBx, reconstruction, and RTTM scoring. Tests also check that the split ONNX encoder’s all-active embedding matches the full FredrikKarlssonSpeech ONNX embedding model.

For all three recordings, the ONNX and NCNN backends produced **byte-for-byte identical RTTM files**.

| Meeting | NCNN DER | ONNX DER | RTTM byte-identical |
| --- | ---: | ---: | --- |
| IS1008a | 14.74% | 14.74% | yes |
| ES2011a | 21.20% | 21.20% | yes |
| TS3004a | 25.85% | 25.85% | yes |

Both runs use the `only_words` references and one full-length UEM per meeting,
with no collar and overlap included. 

## FP32 neural parity

The fixed-input test uses `atol=1e-4, rtol=1e-3`. Relative error divides by
`max(abs(reference), 1e-8)`, so nearly zero reference values can yield a
large relative maximum while still satisfying the combined tolerance.

| Graph and input shape | Maximum absolute error | Maximum relative error |
| --- | ---: | ---: |
| Segmentation `(1, 1, 160000)` | 2.1458e-5 | 1.2965e-5 |
| Segmentation `(1, 1, 80000)` | 1.1921e-5 | 8.0298e-6 |
| Embedding encoder `(1, 998, 80)` | 4.3288e-6 | 4.9422e-2 |
| Embedding encoder `(1, 500, 80)` | 3.6061e-6 | 1.1797e-2 |

`pytest -q` runs numerical parity, pooling, PLDA/VBx, WAV/RTTM, and both
fixture CLI checks. `scripts/model_manifest.py --check` verifies source and
converted artifact hashes. The clean runtime virtual environment was created
with `uv venv .runtime-venv` and `uv pip install --python
.runtime-venv/Scripts/python.exe -e .`; it contains no Torch, ONNX, ONNX
Runtime, or `pyannote.audio`.
