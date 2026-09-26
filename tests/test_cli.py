import subprocess
import sys
from pathlib import Path

import pytest

from diarization.rttm import parse_rttm, score_rttm


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("recording,truth,uem", [
    ("dev00", "debug.development.rttm", None),
    ("ES2005a", "ES2005a.rttm", (0, 306.608)),
])
def test_cli_meets_overlap_inclusive_der(recording, truth, uem, tmp_path):
    output = tmp_path / f"{recording}.rttm"
    subprocess.run(
        [sys.executable, "-m", "diarization.cli", str(ROOT / "fixture" / f"{recording}.wav"),
         str(output), "--models-dir", str(ROOT / "models")],
        check=True,
        cwd=ROOT,
    )
    segments = parse_rttm(output, recording)
    assert segments
    components = score_rttm(parse_rttm(ROOT / "fixture" / truth, recording), segments, uem=uem)
    print(recording, components)
    assert components["der"] <= 0.25
