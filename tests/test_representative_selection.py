from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    _select_representative_polymer_ids,
)


class TestRepresentativeSelection(unittest.TestCase):
    def test_selects_fixed_plus_top_bottom(self) -> None:
        available = [
            "GOx",
            "GOx with DMSO",
            "GOx with EtOH",
            "PMPC",
            "P1",
            "P2",
            "P3",
            "P4",
        ]
        scores = pd.DataFrame(
            [
                {"polymer_id": "GOx with DMSO", "score": 99.0},
                {"polymer_id": "P1", "score": 10.0},
                {"polymer_id": "P2", "score": 9.0},
                {"polymer_id": "P3", "score": 1.0},
                {"polymer_id": "P4", "score": 8.0},
            ]
        )
        selected = _select_representative_polymer_ids(
            available_polymer_ids=available,
            score_df=scores,
            score_col="score",
            reference_polymer_id="GOX",
            top_n=2,
            bottom_n=1,
            target_total=6,
        )
        self.assertEqual(
            selected,
            ["GOx", "GOx with EtOH", "PMPC", "P1", "P2", "P3"],
        )

    def test_backfills_when_fixed_controls_missing(self) -> None:
        available = ["GOx", "P1", "P2", "P3"]
        scores = pd.DataFrame(
            [
                {"polymer_id": "P1", "score": 3.0},
                {"polymer_id": "P2", "score": 2.0},
                {"polymer_id": "P3", "score": 1.0},
            ]
        )
        selected = _select_representative_polymer_ids(
            available_polymer_ids=available,
            score_df=scores,
            score_col="score",
            reference_polymer_id="GOX",
            top_n=2,
            bottom_n=1,
            target_total=6,
        )
        self.assertEqual(selected, ["GOx", "P1", "P2", "P3"])

    def test_excludes_gox_controls_and_pmpc_from_variable_slots(self) -> None:
        available = [
            "GOx",
            "GOx with DMSO",
            "GOx with EtOH",
            "PMPC",
            "P1",
            "P2",
            "P3",
            "P4",
        ]
        scores = pd.DataFrame(
            [
                {"polymer_id": "GOx with DMSO", "score": 100.0},
                {"polymer_id": "PMPC", "score": 90.0},
                {"polymer_id": "P1", "score": 8.0},
                {"polymer_id": "P2", "score": 7.0},
                {"polymer_id": "P3", "score": 1.0},
                {"polymer_id": "P4", "score": 6.0},
            ]
        )
        selected = _select_representative_polymer_ids(
            available_polymer_ids=available,
            score_df=scores,
            score_col="score",
            reference_polymer_id="GOX",
            top_n=2,
            bottom_n=1,
            target_total=6,
        )
        self.assertEqual(selected[:3], ["GOx", "GOx with EtOH", "PMPC"])
        self.assertEqual(selected[3:], ["P1", "P2", "P3"])


if __name__ == "__main__":
    unittest.main()
