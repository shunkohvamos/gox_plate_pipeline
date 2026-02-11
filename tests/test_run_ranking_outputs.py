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

from gox_plate_pipeline.fog import write_run_ranking_outputs  # noqa: E402


class TestRunRankingOutputs(unittest.TestCase):
    def test_write_run_ranking_outputs_creates_ranked_csv_and_png(self) -> None:
        fog_df = pd.DataFrame(
            [
                {"run_id": "R1", "polymer_id": "GOx", "t50_min": 10.0, "t50_censored": 0, "fog": 1.0, "log_fog": 0.0},
                {"run_id": "R1", "polymer_id": "P1", "t50_min": 20.0, "t50_censored": 0, "fog": 2.0, "log_fog": 0.693},
                {"run_id": "R1", "polymer_id": "P2", "t50_min": 15.0, "t50_censored": 0, "fog": 1.5, "log_fog": 0.405},
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            outputs = write_run_ranking_outputs(fog_df, "R1", out_dir)

            self.assertTrue(outputs["t50_ranking_csv"].is_file())
            self.assertTrue(outputs["fog_ranking_csv"].is_file())
            self.assertTrue(outputs["t50_ranking_png"].is_file())
            self.assertTrue(outputs["fog_ranking_png"].is_file())

            t50_tbl = pd.read_csv(outputs["t50_ranking_csv"])
            fog_tbl = pd.read_csv(outputs["fog_ranking_csv"])

            self.assertEqual(t50_tbl.iloc[0]["polymer_id"], "P1")
            self.assertEqual(int(t50_tbl.iloc[0]["rank_t50"]), 1)
            self.assertEqual(fog_tbl.iloc[0]["polymer_id"], "P1")
            self.assertEqual(int(fog_tbl.iloc[0]["rank_fog"]), 1)
            self.assertTrue((t50_tbl["run_id"] == "R1").all())
            self.assertTrue((fog_tbl["run_id"] == "R1").all())

    def test_write_run_ranking_outputs_skips_png_when_no_valid_rows(self) -> None:
        fog_df = pd.DataFrame(
            [
                {"run_id": "R2", "polymer_id": "GOx", "t50_min": float("nan"), "t50_censored": 1, "fog": float("nan")},
                {"run_id": "R2", "polymer_id": "P1", "t50_min": float("nan"), "t50_censored": 1, "fog": float("nan")},
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            outputs = write_run_ranking_outputs(fog_df, "R2", out_dir)
            self.assertTrue(outputs["t50_ranking_csv"].is_file())
            self.assertTrue(outputs["fog_ranking_csv"].is_file())
            self.assertNotIn("t50_ranking_png", outputs)
            self.assertNotIn("fog_ranking_png", outputs)

    def test_write_run_ranking_outputs_penalizes_low_abs_activity(self) -> None:
        fog_df = pd.DataFrame(
            [
                {
                    "run_id": "R3",
                    "polymer_id": "GOx",
                    "t50_min": 10.0,
                    "t50_censored": 0,
                    "fog": 1.0,
                    "log_fog": 0.0,
                    "abs_activity_at_0": 100.0,
                },
                {
                    # Higher raw t50, but very low absolute activity.
                    "run_id": "R3",
                    "polymer_id": "P_low_abs",
                    "t50_min": 20.0,
                    "t50_censored": 0,
                    "fog": 2.0,
                    "log_fog": 0.693,
                    "abs_activity_at_0": 20.0,
                },
                {
                    # Lower raw t50 than P_low_abs, but high absolute activity.
                    "run_id": "R3",
                    "polymer_id": "P_high_abs",
                    "t50_min": 16.0,
                    "t50_censored": 0,
                    "fog": 1.6,
                    "log_fog": 0.470,
                    "abs_activity_at_0": 95.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            outputs = write_run_ranking_outputs(fog_df, "R3", out_dir)
            t50_tbl = pd.read_csv(outputs["t50_ranking_csv"])
            fog_tbl = pd.read_csv(outputs["fog_ranking_csv"])

            # Activity-adjusted scores:
            #   P_low_abs: 20 * 0.20 = 4
            #   P_high_abs: 16 * 0.95 = 15.2
            self.assertEqual(t50_tbl.iloc[0]["polymer_id"], "P_high_abs")
            self.assertEqual(int(t50_tbl.iloc[0]["rank_t50"]), 1)
            self.assertEqual(fog_tbl.iloc[0]["polymer_id"], "P_high_abs")
            self.assertEqual(int(fog_tbl.iloc[0]["rank_fog"]), 1)


if __name__ == "__main__":
    unittest.main()
