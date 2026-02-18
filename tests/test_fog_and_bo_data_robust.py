from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_data import (  # noqa: E402
    build_bo_learning_data_from_round_averaged,
    build_round_coverage_summary,
)
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
                        "mean_fog_native_constrained": 1.1,
                        "mean_log_fog_native_constrained": 0.0953101798,
                        "robust_fog_native_constrained": 1.05,
                        "robust_log_fog_native_constrained": 0.0487901642,
                        "mean_fog_native_soft": 0.95,
                        "mean_log_fog_native_soft": -0.0512932944,
                        "robust_fog_native_soft": 0.90,
                        "robust_log_fog_native_soft": -0.1053605157,
                        "robust_objective_loglinear_main": 0.321,
                        "robust_objective_loglinear_main_exp": 1.378307,
                        "native_feasible_fraction": 0.75,
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
            self.assertAlmostEqual(
                float(learning_df.iloc[0]["log_fog_native_constrained"]),
                0.0487901642,
                places=8,
            )
            self.assertAlmostEqual(
                float(learning_df.iloc[0]["log_fog_native_soft"]),
                -0.1053605157,
                places=8,
            )
            self.assertAlmostEqual(float(learning_df.iloc[0]["fog_native_soft"]), 0.90, places=8)
            self.assertAlmostEqual(float(learning_df.iloc[0]["native_feasible_fraction"]), 0.75, places=8)
            self.assertEqual(str(learning_df.iloc[0]["objective_source"]), "robust_round_aggregated")
            self.assertAlmostEqual(float(learning_df.iloc[0]["log_fog_mad"]), 0.03, places=8)
            self.assertAlmostEqual(float(learning_df.iloc[0]["objective_loglinear_main"]), 0.321, places=8)
            self.assertAlmostEqual(float(learning_df.iloc[0]["objective_loglinear_main_exp"]), 1.378307, places=6)

    def test_round_coverage_summary_and_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            fog_path = base / "fog_round_averaged.csv"
            pd.DataFrame(
                [
                    {
                        "round_id": "R9",
                        "polymer_id": "P1",
                        "mean_fog": 1.2,
                        "mean_log_fog": 0.1823215568,
                    }
                ]
            ).to_csv(fog_path, index=False)
            catalog_df = pd.DataFrame(
                [{"polymer_id": "P1", "frac_MPC": 0.5, "frac_BMA": 0.3, "frac_MTAC": 0.2}]
            )
            run_round_map = {"run_a": "R1", "run_b": "R2"}

            with self.assertRaisesRegex(ValueError, "not found in run_round_map"):
                build_bo_learning_data_from_round_averaged(
                    catalog_df,
                    fog_path,
                    run_round_map=run_round_map,
                    strict_round_coverage=True,
                )

            learning_df, excluded_df = build_bo_learning_data_from_round_averaged(
                catalog_df,
                fog_path,
                run_round_map=run_round_map,
                strict_round_coverage=False,
            )
            self.assertTrue(excluded_df.empty)
            self.assertEqual(len(learning_df), 1)
            coverage = build_round_coverage_summary(learning_df["round_id"].tolist(), run_round_map)
            self.assertIn("R9", coverage["round_ids_missing_in_map"])
            self.assertIn("R1", coverage["round_ids_in_map_but_unused"])
            self.assertIn("R2", coverage["round_ids_in_map_but_unused"])

    def test_plate_aware_supports_custom_reference_polymer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            processed = base / "processed"
            run_round_map = {"run_c": "R2"}

            run_c_fit = processed / "run_c" / "fit"
            run_c_fit.mkdir(parents=True, exist_ok=True)

            rows_c: list[dict] = []
            rows_c += _decay_rows("run_c", "plate1", "BETA-GAL", [100, 82, 66, 44, 22])
            rows_c += _decay_rows("run_c", "plate1", "P1", [100, 80, 64, 42, 20])
            pd.DataFrame(rows_c).to_csv(run_c_fit / "rates_with_rea.csv", index=False)

            per_row, round_avg, trace_df, _warning = build_fog_plate_aware(
                run_round_map,
                processed,
                reference_polymer_id="BETA-GAL",
            )

            self.assertFalse(per_row.empty)
            self.assertFalse(round_avg.empty)
            self.assertFalse(trace_df.empty)
            self.assertTrue((per_row["reference_polymer_id"].astype(str) == "BETA-GAL").all())
            self.assertIn("gox_t50_used_min", per_row.columns)
            self.assertTrue(np.isfinite(pd.to_numeric(per_row["gox_t50_used_min"], errors="coerce")).all())
            self.assertTrue((trace_df["reference_polymer_id"].astype(str) == "BETA-GAL").all())


if __name__ == "__main__":
    unittest.main()
