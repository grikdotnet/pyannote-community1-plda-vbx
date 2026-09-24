"""RTTM serialization and no-collar, overlap-inclusive DER scoring."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str


def format_rttm(recording: str, segments: list[Segment]) -> str:
    rows = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end, item.speaker)):
        if segment.end <= segment.start:
            continue
        rows.append(
            f"SPEAKER {recording} 1 {segment.start:.4f} {segment.end - segment.start:.4f} "
            f"<NA> <NA> {segment.speaker} <NA> <NA>"
        )
    return "\n".join(rows) + ("\n" if rows else "")


def parse_rttm(path: Path, recording: str | None = None) -> list[Segment]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "SPEAKER" and (recording is None or fields[1] == recording):
            start, duration = float(fields[3]), float(fields[4])
            segments.append(Segment(start, start + duration, fields[7]))
    return segments


def score_rttm(reference: list[Segment], hypothesis: list[Segment],
               uem: tuple[float, float] | None = None) -> dict[str, float]:
    if uem is not None:
        start, end = uem
        if end <= start:
            raise ValueError("evaluation end must exceed start")
        reference = [Segment(max(item.start, start), min(item.end, end), item.speaker)
                     for item in reference if item.start < end and item.end > start]
        hypothesis = [Segment(max(item.start, start), min(item.end, end), item.speaker)
                      for item in hypothesis if item.start < end and item.end > start]
    boundaries = sorted({time for segment in reference + hypothesis for time in (segment.start, segment.end)})
    if len(boundaries) < 2:
        return {"miss": 0.0, "false_alarm": 0.0, "confusion": 0.0, "reference": 0.0, "der": 0.0}
    truth_ids = sorted({item.speaker for item in reference})
    hypothesis_ids = sorted({item.speaker for item in hypothesis})
    overlap = np.zeros((len(truth_ids), len(hypothesis_ids)))
    intervals = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        mid = (start + end) / 2
        truth = {item.speaker for item in reference if item.start <= mid < item.end}
        predicted = {item.speaker for item in hypothesis if item.start <= mid < item.end}
        intervals.append((end - start, truth, predicted))
        for first in truth:
            for second in predicted:
                overlap[truth_ids.index(first), hypothesis_ids.index(second)] += end - start
    mapping = {}
    if overlap.size:
        rows, columns = linear_sum_assignment(overlap, maximize=True)
        mapping = {hypothesis_ids[column]: truth_ids[row] for row, column in zip(rows, columns)}
    miss = false_alarm = confusion = total = 0.0
    for duration, truth, predicted in intervals:
        count_truth, count_predicted = len(truth), len(predicted)
        matched = len(truth & {mapping.get(speaker) for speaker in predicted})
        miss += duration * max(0, count_truth - count_predicted)
        false_alarm += duration * max(0, count_predicted - count_truth)
        confusion += duration * (min(count_truth, count_predicted) - matched)
        total += duration * count_truth
    return {
        "miss": miss,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "reference": total,
        "der": (miss + false_alarm + confusion) / total if total else 0.0,
    }
