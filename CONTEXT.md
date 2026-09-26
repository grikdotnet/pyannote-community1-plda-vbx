# Speaker diarization

This context names the speech and speaker concepts used by the standalone diarization pipeline.

## Language

**Local speaker**:
A speaker identity within one segmentation window. The same person may receive a different local identity in another window.

**Global speaker**:
A speaker identity reconciled across a recording.

**Final speaker count**:
The number of distinct global speaker identities present in the returned speaker turns for a recording.

**Speaker embedding**:
A fixed-length representation of the voice associated with a candidate speaker.

**Valid speaker observation**:
A local speaker candidate with detected speech inside the recording and a usable voice embedding. It can count toward the speaker minimum even when it is less suitable for estimating speaker groups.

**Overlap**:
A time interval in which more than one speaker is active.

**Temporal conflict**:
Two local speakers within the same segmentation window are active at the same time. Local speakers from different windows may describe the same global speaker during overlapping recording time.
