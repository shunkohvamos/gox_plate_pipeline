from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.loader import attach_row_map  # noqa: E402


class AttachRowMapTests(unittest.TestCase):
    def test_wildcard_plate_fallback_with_exact_override(self) -> None:
        df = pd.DataFrame(
            {
                "plate_id": ["plate1", "plate2", "plate2"],
                "well": ["A1", "A1", "B1"],
                "row": ["A", "A", "B"],
                "signal": [10.0, 20.0, 30.0],
            }
        )
        row_map = pd.DataFrame(
            {
                "plate_id": ["", "plate2"],
                "row": ["A", "B"],
                "polymer_id": ["P_ALL_A", "P2_B"],
                "sample_name": ["all_a", "plate2_b"],
            }
        )

        out = attach_row_map(df, row_map)

        # A row is filled by wildcard for both plate1 and plate2.
        self.assertEqual(out.loc[(out["plate_id"] == "plate1") & (out["row"] == "A"), "polymer_id"].iloc[0], "P_ALL_A")
        self.assertEqual(out.loc[(out["plate_id"] == "plate2") & (out["row"] == "A"), "polymer_id"].iloc[0], "P_ALL_A")

        # Exact mapping should override wildcard for plate2/B.
        self.assertEqual(out.loc[(out["plate_id"] == "plate2") & (out["row"] == "B"), "polymer_id"].iloc[0], "P2_B")


if __name__ == "__main__":
    unittest.main()
