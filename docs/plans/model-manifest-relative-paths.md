# Plan: Make the model manifest relative to `models/`

## Summary

Make `models/manifest.json` describe files as they appear when `models/` is the root of a Hugging Face model repository. Preserve ONNX provenance as external references without listing files outside the bundle as local paths.

## Success Criteria

- Every manifest `path` resolves inside `models/` -> verify: resolve each against the manifest directory, reject `..` traversal, and confirm the file exists.
- The manifest describes all eight runtime weight files -> verify: compare recorded relative paths with the `.param`, `.bin`, `.npy`, and `.npz` files in `models/`.
- ONNX provenance remains available without claiming the files are bundled -> verify: reference entries retain source URLs, hashes, and tensor contracts but have no local `path`.
- Manifest generation remains reproducible -> verify: regenerate and run `scripts/model_manifest.py --check` and the focused manifest test.
- Runtime behavior is unchanged -> verify: inspect the diff for model weights and `src/` changes.

## Implementation Steps

1. Update the focused manifest test to require model-root-relative paths and a complete eight-file inventory -> verify: it fails on the current `models/` and `reference/` paths.
2. Update `scripts/model_manifest.py` to separate bundled files from external ONNX references -> verify: generated paths resolve inside `models/` and all external references use URLs.
3. Regenerate `models/manifest.json` and align the model card wording -> verify: metadata distinguishes local assets from upstream reference models.
4. Run the focused test, manifest checker, and diff check -> verify: all pass without model weight or runtime edits.

## Architecture Classification

`scripts/model_manifest.py` is a development and release inventory tool. `models/manifest.json` is metadata. No runtime module or class changes; inference paths and outputs are preserved.

## Tests / Verification

Run the focused manifest test before and after the generator change, then run the generator checker and inspect the diff. No broad inference test is needed because weights and runtime loading code are unchanged.

## Non-Goals

No model conversion, weight editing, Hub upload, or pipeline algorithm changes.

## Assumptions

The contents of `models/` will be uploaded at the Hub repository root. ONNX source graphs and VBx reference code stay in the development repository and are not part of this model bundle.
