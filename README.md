# NCNN Pyannote Community-1 diarization

This pipeline reads 16 kHz mono PCM16 or float32 WAV and writes
overlap-aware RTTM. Neural inference uses NCNN.

## Run

Install the runtime package with `uv pip install -e .`, then run:

```powershell
python -m diarization.cli fixture/dev00.wav dev00.rttm
python -m diarization.cli fixture/ES2005a.wav ES2005a.rttm --min-speakers 1 --max-speakers 8
```

Speaker count is estimated automatically. The optional bounds constrain the
number of global speaker clusters. Each RTTM line uses the input WAV stem as
the recording ID.

## Reproduce and verify

See [model provenance](models/README.md) for source revisions, licenses, and
the pnnx conversion step. In a development environment with the pinned
optional packages and pnnx installed, run:

```powershell
python scripts/convert_models.py
python scripts/model_manifest.py --check
pytest -q
mypy src/diarization --ignore-missing-imports --check-untyped-defs
```

The numerical parity tests compare segmentation scores and embedding frame
features against the FP32 ONNX graphs at two shapes. The CLI tests score the
fixtures with zero collar and overlap included, and report miss, false alarm,
and confusion separately. ES2005a is evaluated on its annotated
`[0, 306.608]` second interval; its 477.877-second WAV contains an
unannotated tail. The full-WAV score is reported separately in `PLAN.md`.
