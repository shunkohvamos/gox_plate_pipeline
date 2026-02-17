from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.summary import (  # noqa: E402
    build_polymer_heat_summary,
    filter_well_table_for_summary,
)


class TestSummaryOutlierFilter(unittest.TestCase):
    def test_robust_group_outlier_is_excluded_before_summary(self) -> None:
        df = pd.DataFrame(
            [
                # Context at heat=20
                {"polymer_id": "GOX", "heat_min": 20.0, "status": "ok", "abs_activity": 9.0, "REA_percent": 54.0, "plate_id": "p1", "well": "A1"},
                {"polymer_id": "GOX", "heat_min": 20.0, "status": "ok", "abs_activity": 8.5, "REA_percent": 52.0, "plate_id": "p1", "well": "A2"},
                # Target polymer with one extreme replicate outlier.
                {"polymer_id": "P1", "heat_min": 20.0, "status": "ok", "abs_activity": 10.0, "REA_percent": 60.0, "plate_id": "p1", "well": "B1"},
                {"polymer_id": "P1", "heat_min": 20.0, "status": "ok", "abs_activity": 11.0, "REA_percent": 61.0, "plate_id": "p1", "well": "B2"},
                {"polymer_id": "P1", "heat_min": 20.0, "status": "ok", "abs_activity": 200.0, "REA_percent": 900.0, "plate_id": "p1", "well": "B3"},
            ]
        )

        filtered, events = filter_well_table_for_summary(
            df,
            run_id="R-test",
            apply_outlier_filter=True,
            outlier_min_samples=3,
            outlier_z_threshold=3.5,
            outlier_ratio_low=0.33,
            outlier_ratio_high=3.0,
            outlier_pair_ratio_threshold=3.0,
            outlier_min_keep=1,
        )
        summary = build_polymer_heat_summary(filtered, run_id="R-test")
        p1 = summary[(summary["polymer_id"] == "P1") & (summary["heat_min"] == 20.0)].iloc[0]

        self.assertEqual(int(p1["n"]), 2)
        self.assertAlmostEqual(float(p1["mean_abs_activity"]), 10.5, places=6)
        self.assertAlmostEqual(float(p1["mean_REA_percent"]), 60.5, places=6)
        self.assertEqual(len(events), 1)
        self.assertEqual(str(events.iloc[0]["method"]), "robust_group")
        self.assertEqual(str(events.iloc[0]["well"]), "B3")

    def test_pair_ratio_context_excludes_farther_of_two_replicates(self) -> None:
        df = pd.DataFrame(
            [
                # Run-wide heat=20 context (other polymer IDs)
                {"polymer_id": "GOX", "heat_min": 20.0, "status": "ok", "abs_activity": 9.0, "REA_percent": 55.0, "plate_id": "p1", "well": "A1"},
                {"polymer_id": "Q1", "heat_min": 20.0, "status": "ok", "abs_activity": 2.5, "REA_percent": 16.0, "plate_id": "p1", "well": "A2"},
                # Target polymer with n=2 and large disagreement
                {"polymer_id": "P2", "heat_min": 20.0, "status": "ok", "abs_activity": 2.0, "REA_percent": 14.0, "plate_id": "p1", "well": "C1"},
                {"polymer_id": "P2", "heat_min": 20.0, "status": "ok", "abs_activity": 20.0, "REA_percent": 140.0, "plate_id": "p1", "well": "C2"},
            ]
        )

        filtered, events = filter_well_table_for_summary(
            df,
            run_id="R-test",
            apply_outlier_filter=True,
            outlier_min_samples=3,
            outlier_z_threshold=3.5,
            outlier_ratio_low=0.33,
            outlier_ratio_high=3.0,
            outlier_pair_ratio_threshold=3.0,
            outlier_min_keep=1,
        )
        summary = build_polymer_heat_summary(filtered, run_id="R-test")
        p2 = summary[(summary["polymer_id"] == "P2") & (summary["heat_min"] == 20.0)].iloc[0]

        self.assertEqual(int(p2["n"]), 1)
        self.assertAlmostEqual(float(p2["mean_abs_activity"]), 2.0, places=6)
        self.assertAlmostEqual(float(p2["mean_REA_percent"]), 14.0, places=6)
        self.assertEqual(len(events), 1)
        self.assertEqual(str(events.iloc[0]["method"]), "pair_ratio_context")
        self.assertEqual(str(events.iloc[0]["well"]), "C2")


if __name__ == "__main__":
    unittest.main()
