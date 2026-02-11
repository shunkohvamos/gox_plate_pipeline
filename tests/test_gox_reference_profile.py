from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.polymer_timeseries import resolve_gox_reference_profile  # noqa: E402


def _write_summary_simple(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class TestGOxReferenceProfile(unittest.TestCase):
    def test_same_run_gox_is_preferred(self) -> None:
        df = pd.DataFrame(
            [
                {"polymer_id": "GOx", "heat_min": 0, "abs_activity": 100.0},
                {"polymer_id": "GOx", "heat_min": 20, "abs_activity": 80.0},
                {"polymer_id": "P1", "heat_min": 0, "abs_activity": 120.0},
                {"polymer_id": "P1", "heat_min": 20, "abs_activity": 90.0},
            ]
        )
        ref = resolve_gox_reference_profile(run_id="R0", summary_df=df, at_time_min=20.0)
        self.assertEqual(ref["source"], "same_run_gox")
        self.assertEqual(ref["reference_run_id"], "R0")
        self.assertAlmostEqual(float(ref["gox_abs_activity_at_time"]), 80.0, places=6)

    def test_nearest_round_run_is_used_when_shift_is_large(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed_dir = root / "processed"
            run_map = root / "run_round_map.tsv"
            run_map.write_text("run_id\tround_id\nRA\tR1\nRB\tR1\nRC\tR1\n", encoding="utf-8")

            # Target run (RA): no GOx, high-scale profile.
            _write_summary_simple(
                processed_dir / "RA" / "fit" / "summary_simple.csv",
                [
                    {"run_id": "RA", "polymer_id": "P1", "heat_min": 0, "abs_activity": 120.0, "REA_percent": 100.0},
                    {"run_id": "RA", "polymer_id": "P1", "heat_min": 20, "abs_activity": 100.0, "REA_percent": 83.3},
                    {"run_id": "RA", "polymer_id": "P2", "heat_min": 0, "abs_activity": 70.0, "REA_percent": 100.0},
                    {"run_id": "RA", "polymer_id": "P2", "heat_min": 20, "abs_activity": 50.0, "REA_percent": 71.4},
                ],
            )

            # Round run RB: GOx exists but non-GOx profile is far from RA.
            _write_summary_simple(
                processed_dir / "RB" / "fit" / "summary_simple.csv",
                [
                    {"run_id": "RB", "polymer_id": "GOx", "heat_min": 0, "abs_activity": 360.0, "REA_percent": 100.0},
                    {"run_id": "RB", "polymer_id": "GOx", "heat_min": 20, "abs_activity": 300.0, "REA_percent": 83.3},
                    {"run_id": "RB", "polymer_id": "P1", "heat_min": 20, "abs_activity": 30.0, "REA_percent": 25.0},
                    {"run_id": "RB", "polymer_id": "P2", "heat_min": 20, "abs_activity": 15.0, "REA_percent": 21.4},
                ],
            )

            # Round run RC: GOx exists and non-GOx profile is close to RA.
            _write_summary_simple(
                processed_dir / "RC" / "fit" / "summary_simple.csv",
                [
                    {"run_id": "RC", "polymer_id": "GOx", "heat_min": 0, "abs_activity": 150.0, "REA_percent": 100.0},
                    {"run_id": "RC", "polymer_id": "GOx", "heat_min": 20, "abs_activity": 120.0, "REA_percent": 80.0},
                    {"run_id": "RC", "polymer_id": "P1", "heat_min": 20, "abs_activity": 95.0, "REA_percent": 79.2},
                    {"run_id": "RC", "polymer_id": "P2", "heat_min": 20, "abs_activity": 52.0, "REA_percent": 74.3},
                ],
            )

            target_df = pd.read_csv(processed_dir / "RA" / "fit" / "summary_simple.csv")
            ref = resolve_gox_reference_profile(
                run_id="RA",
                summary_df=target_df,
                at_time_min=20.0,
                processed_dir=processed_dir,
                run_round_map_path=run_map,
            )
            self.assertEqual(ref["source"], "nearest_round_run_gox")
            self.assertEqual(ref["reference_run_id"], "RC")
            self.assertAlmostEqual(float(ref["gox_abs_activity_at_time"]), 120.0, places=6)


if __name__ == "__main__":
    unittest.main()
