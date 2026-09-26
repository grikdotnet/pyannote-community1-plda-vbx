# Plan: Enforce final speaker-count bounds

## Summary

Fix [issue 02](../../.scratch/diarization-clustering/issues/02-enforce-minimum-speaker-bound.md). `minimum_speakers` and `maximum_speakers` constrain the number of distinct speaker IDs in the turns returned by `Diarizer.diarize()`, not only the initial clustering groups. With enough valid speaker observations, the final count is within both bounds. With fewer observations than the minimum, return the largest feasible count; silence returns no speakers. A valid observation has speech within the real recording and a usable embedding. Similar embeddings and valid slots excluded by the clean-speech training mask still count.

## Success Criteria

- A feasible minimum is honored after hierarchy cutting, VBx refinement, and local-slot assignment -> verify: three orthogonal embeddings in one window with `minimum=maximum=2` produce two used global IDs and two centroids.
- The minimum applies to the final turns, even if two cluster IDs compete for the same recording frames -> verify: a two-window overlap fixture that currently reconstructs only `speaker_00` returns turns containing two distinct IDs on speech-supported frames.
- Sparse speech diluted by overlapping window coverage still counts -> verify: a three-window fixture whose count threshold initially removes both active frames returns two speech-supported IDs when the minimum is two.
- Similar-sounding observations still satisfy an explicit minimum -> verify: two identical valid embeddings in separate windows with `minimum=maximum=2` produce two IDs in the returned turns.
- Valid observations outside the clean-speech training mask count toward feasibility -> verify: a fixture with one trainable slot and at least two other valid slots returns two speakers when the minimum is two.
- The minimum is capped by the number of valid observations with speech inside the recording -> verify: zero valid observations return zero speakers, one returns one when the minimum is two, and padded-only activity does not increase the count.
- The configured maximum remains a hard cap on returned speaker IDs -> verify: existing issue 01 conflict tests and a final-turn count test with `maximum=2` never return a third ID.
- Existing speech and model behavior is preserved -> verify: issue 01's sequential-slot test retains every supported turn, fixed-input NCNN/ONNX parity tests pass, and existing full-pipeline fixtures stay within their DER limits.

## Implementation Steps

1. Add failing algorithm tests in `tests/test_plda_vbx.py` for the centroid-linkage inversion, identical observations across windows, too few trainable slots, and fewer valid observations than the requested minimum -> verify: each test fails on the current count behavior for the expected reason.
2. Add failing pipeline and reconstruction tests in `tests/test_pipeline_stages.py` for padded-only activity, two cluster IDs collapsing to one returned ID, silence, and the final maximum -> verify: tests expose the current mismatch between clustering labels and returned turns.
3. In `src/diarization/vbx.py`, produce exactly the feasible requested number of initial groups when a speaker bound changes the automatic group count, including nonmonotonic centroid-linkage trees -> verify: the three-orthogonal-vector test has exactly two initial groups and two used final groups. Preserve the current automatic distance-threshold behavior when no bound forces a different count.
4. In `src/diarization/vbx.py`, use valid lower-quality observations when the training subset is too small, and keep at least the feasible minimum represented after VBx and local assignment -> verify: the identical-embedding and training-mask tests return the required number of used IDs with deterministic label and centroid order.
5. In `src/diarization/pipeline.py` and, if a shared timing helper is needed, `src/diarization/segmentation.py`, exclude activity beyond the real recording from speaker-count feasibility and reconstruction -> verify: the padded-only fixture contributes no speaker and the existing window-timing tests pass.
6. In the reconstruction path in `src/diarization/pipeline.py`, retain enough speech-supported turns for each required identity to appear in the returned result without emitting speech in silence -> verify: the two-window overlap and sparse three-window tests contain two speaker IDs, and every emitted interval overlaps detected speech within the recording.
7. Update `README.md` to state that the bounds apply to returned speakers and that the minimum is capped by valid observations -> verify: the Python and CLI documentation describe the same behavior as the regression tests.
8. Run focused and repository-wide verification -> verify: `pytest tests/test_plda_vbx.py tests/test_pipeline_stages.py -q`, `pytest -q`, `mypy src/diarization --ignore-missing-imports --check-untyped-defs`, and `git diff --check` pass.

## Architecture Classification

- `src/diarization/vbx.py`: PLDA/VBx clustering and local-to-global speaker assignment; enforces a feasible count in the labels and centroids it returns.
- `src/diarization/pipeline.py`: stage orchestration and speaker-turn reconstruction; applies real-recording feasibility and enforces the count in returned turns.
- `src/diarization/segmentation.py` (only if needed): segmentation frame timing used to distinguish recording audio from padded windows; no model inference change.
- `tests/test_plda_vbx.py` and `tests/test_pipeline_stages.py`: algorithm and end-to-end boundary regressions.
- `README.md`, `CONTEXT.md`, and `docs/adr/0002-final-speaker-count-bounds.md`: public behavior, domain vocabulary, and the decision behind a strict user-specified count.

## Tests / Verification

First run the new focused tests and confirm the orthogonal-vector case returns one speaker instead of two, the identical two-window case loses a final ID, and the reconstruction fixture drops one ID despite two labels. After implementation, rerun those tests and inspect both the clustering result and returned `Segment` speakers. Run the existing issue 01 regressions to ensure no valid speech slot is lost and the maximum still caps identities. Then run the full suite, mypy, and diff validation listed above. Use fixed embeddings and activity masks for the new behavior checks; retain existing real-audio fixtures as regression coverage rather than treating their DER as proof of the new boundary cases.

## Non-Goals

- Changing NCNN or ONNX graphs, weights, audio feature extraction, or PLDA/VBx refinement equations.
- Reworking the separate issue 01 policy for preserving speech when the maximum forces a merge.
- Guaranteeing that an explicit speaker minimum improves speaker identification accuracy or DER; two very similar observations may be split to honor the bound.
- Creating a speaker or a turn without a valid observation and detected speech inside the recording.

## Assumptions

- The final count is the number of distinct speaker IDs present in returned turns or RTTM lines, not the number of internal centroids alone.
- One valid local speaker slot is one observation, even if another slot has an identical embedding or describes the same real person. The minimum can deliberately split such observations.
- A valid observation has a usable embedding and detected speech within the real recording; the clean-speech training mask controls estimation quality, not whether the observation counts.
- If the number of valid observations is below `minimum_speakers`, the target count is that smaller number; if there are none, return no speaker turns. When sufficient observations exist, `minimum_speakers <= final_count <= maximum_speakers` (with no upper bound when `maximum_speakers=None`).
- Speaker IDs and tie handling remain deterministic. Existing issue 01 behavior retains every valid speech slot and treats a configured maximum as a hard cap.
