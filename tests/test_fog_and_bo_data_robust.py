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

from gox_plate_pipeline.bo_data import build_bo_learning_data_from_round_averaged  # noqa: E402
from gox_plate_pipeline.fog import build_fog_plate_aware  # noqa: E402


def _decay_rows(run_id: str, plate_id: str, polymer_id: str, rea_values: list[float]) -> list[dict]:
    times = [0.0, 5.0, 10.0, 20.0, 40.0]
    rows: list[dict] = []
    for i, (t, rea) in enumerate(zip(times, rea_values), start=1):
        rows.append(
            {
                "run_id": run_id,
                "plate_id": plate_id,
                "polymer_id": polymer_id,
                "well": f"A{i}",
                "heat_min": t,
                "REA_percent": rea,
                "abs_activity": rea,
                "status": "ok",
            }
        )
    return rows


class TestFogAndBODataRobust(unittest.TestCase):
    def test_same_plate_guard_fallbacks_on_extreme_gox(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            processed = base / "processed"
            run_round_map = {"run_a": "R1", "run_b": "R1"}

            run_a_fit = processed / "run_a" / "fit"
            run_b_fit = processed / "run_b" / "fit"
            run_a_fit.mkdir(parents=True, exist_ok=True)
            run_b_fit.mkdir(parents=True, exist_ok=True)

            rows_a: list[dict] = []
            rows_a += _decay_rows("run_a", "plate1", "GOX", [100, 80, 60, 40, 20])
            rows_a += _decay_rows("run_a", "plate1", "P1", [100, 85, 70, 50, 25])
            rows_a += _decay_rows("run_a", "plate3", "GOX", [100, 15, 5, 1, 0.6])  # extreme low t50
            rows_a += _decay_rows("run_a", "plate3", "P1", [100, 85, 70, 50, 25])
            pd.DataFrame(rows_a).to_csv(run_a_fit / "rates_with_rea.csv", index=False)

            rows_b: list[dict] = []
            rows_b += _decay_rows("run_b", "plate2", "GOX", [100, 78, 58, 38, 18])
            rows_b += _decay_rows("run_b", "plate2", "P1", [100, 84, 69, 49, 24])
            pd.DataFrame(rows_b).to_csv(run_b_fit / "rates_with_rea.csv", index=False)

            per_row, round_avg, _, warning = build_fog_plate_aware(
                run_round_map,
                processed,
                exclude_outlier_gox=False,
                gox_guard_same_plate=True,
                gox_round_fallback_stat="median",
            )

            self.assertFalse(per_row.empty)
            self.assertIn("same_round", set(per_row["denominator_source"].astype(str)))
            plate3 = per_row[(per_row["run_id"] == "run_a") & (per_row["plate_id"] == "plate3") & (per_row["polymer_id"] == "P1")]
            self.assertEqual(len(plate3), 1)
            self.assertEqual(str(plate3.iloc[0]["denominator_source"]), "same_round")
            self.assertGreater(float(plate3.iloc[0]["gox_t50_used_min"]), 2.0)
            self.assertTrue(len(warning.guarded_same_plate) >= 1)

            for col in ["robust_fog", "robust_log_fog", "log_fog_mad"]:
                self.assertIn(col, round_avg.columns)

    def test_bo_learning_prefers_robust_log_fog_if_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            fog_path = base / "fog_round_averaged.csv"
            pd.DataFrame(
                [
                    {
                        "round_id": "R1",
                        "polymer_id": "P1",
                        "mean_fog": 10.0,
                        "mean_log_fog": 2.302585093,
                        "robust_fog": 1.2,
                        "robust_log_fog": 0.1823215568,
                        "log_fog_mad": 0.03,
                        "n_observations": 5,
                        "run_ids": "a,b",
                    }
                ]
            ).to_csv(fog_path, index=False)

            catalog_df = pd.DataFrame(
                [{"polymer_id": "P1", "frac_MPC": 0.5, "frac_BMA": 0.3, "frac_MTAC": 0.2, "x": 0.6, "y": 0.5}]
            )
            learning_df, excluded_df = build_bo_learning_data_from_round_averaged(catalog_df, fog_path)

            self.assertTrue(excluded_df.empty)
            self.assertEqual(len(learning_df), 1)
            self.assertAlmostEqual(float(learning_df.iloc[0]["log_fog"]), 0.1823215568, places=8)
            self.assertEqual(str(learning_df.iloc[0]["objective_source"]), "robust_round_aggregated")
            self.assertAlmostEqual(float(learning_df.iloc[0]["log_fog_mad"]), 0.03, places=8)


if __name__ == "__main__":
    unittest.main()

