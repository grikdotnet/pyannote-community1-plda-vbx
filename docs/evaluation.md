# Fixture verification

The scorer in `diarization.rttm.score_rttm` integrates exact RTTM interval
boundaries, uses a global optimal speaker mapping, applies no collar, and
includes simultaneous speakers in the reference denominator. No speech
regions are excluded inside the stated evaluation map.

| Recording | Evaluation map (s) | Reference speaker-seconds | Miss (s) | False alarm (s) | Confusion (s) | DER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev00` | full 0–30 | 28.4970 | 2.0898 | 0.3566 | 3.9480 | 22.44% |
| `ES2005a` | annotated 0–306.608 | 332.3770 | 58.2221 | 14.8463 | 6.1569 | 23.84% |
| `ES2005a` | full WAV, no UEM | 332.3770 | 58.2221 | 47.8538 | 6.1569 | 33.77% |

The ES2005a WAV lasts 477.877 seconds, but its supplied RTTM and the local
VBx VAD reference end at about 306.6 seconds. The main benchmark uses the
explicit interval covered by that annotation. The full-WAV diagnostic shows
the consequence of treating the unannotated tail as negative ground truth;
the CLI still processes the entire WAV.

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
