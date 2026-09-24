# Speaker diarization

This context names the speech and speaker concepts used by the standalone diarization pipeline.

## Language

**Local speaker**:
A speaker identity within one segmentation window. The same person may receive a different local identity in another window.

**Global speaker**:
A speaker identity reconciled across a recording.

**Speaker embedding**:
A fixed-length representation of the voice associated with a candidate speaker.

**Overlap**:
A time interval in which more than one speaker is active.
