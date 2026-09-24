# Repository guidance

This repository has two goals:

1. Convert pyannote Community-1 ONNX model components to NCNN.
2. Build a standalone diarization pipeline with PLDA and VBx behavior referenced from [BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx).

Start with [PLAN.md](PLAN.md) before implementing a stage. Keep its success criteria and verification steps current as the project develops.

## Current assets

`pyannote-community-1-onnx-split/` contains the downloaded split embedding encoder and projection weights:

- `embedding_encoder.onnx`
- `resnet_seg_1_weight.npy`
- `resnet_seg_1_bias.npy`
- `split_pyannote_embedding.py`, which reproduces the split from a full embedding ONNX model

The split encoder produces frame features. The `.npy` files hold the projection after masked statistics pooling. Read the directory's README for tensor details. Do not assume these files include segmentation, the complete Community-1 pipeline, or PLDA parameters.

## Working rules

- Create or update a plan before changing implementation files. Define a check for each planned result.
- Verify converted NCNN outputs against ONNX on fixed inputs before wiring them into the pipeline.
- Keep model inference, PLDA/VBx algorithms, and audio/RTTM input and output code in separate modules.
- Document the provenance and license of any additional model weights or code brought into this repository.

## Agent skills

### Issue tracker

Issues and specs live as local Markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout: root `CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.
