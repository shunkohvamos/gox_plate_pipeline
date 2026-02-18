from __future__ import annotations

import math
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
                {
                    "run_id": "R1",
                    "polymer_id": "GOx",
                    "t50_min": 10.0,
                    "t50_censored": 0,
                    "fog": 1.0,
                    "log_fog": 0.0,
                    "abs_activity_at_0": 100.0,
                },
                {
                    "run_id": "R1",
                    "polymer_id": "P1",
                    "t50_min": 20.0,
                    "t50_censored": 0,
                    "fog": 2.0,
                    "log_fog": 0.693,
                    "abs_activity_at_0": 120.0,
                },
                {
                    "run_id": "R1",
                    "polymer_id": "P2",
                    "t50_min": 15.0,
                    "t50_censored": 0,
                    "fog": 1.5,
                    "log_fog": 0.405,
                    "abs_activity_at_0": 95.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            outputs = write_run_ranking_outputs(fog_df, "R1", out_dir)

            self.assertTrue(outputs["t50_ranking_csv"].is_file())
            self.assertTrue(outputs["fog_ranking_csv"].is_file())
            self.assertTrue(outputs["fog_native_constrained_ranking_csv"].is_file())
            self.assertTrue(outputs["objective_native_soft_ranking_csv"].is_file())
            self.assertTrue(outputs["objective_loglinear_main_ranking_csv"].is_file())
            self.assertTrue(outputs["supp_rank_correlation_csv"].is_file())
            self.assertTrue(outputs["t50_ranking_png"].is_file())
            self.assertTrue(outputs["fog_ranking_png"].is_file())
            self.assertTrue(outputs["fog_native_constrained_ranking_png"].is_file())
            self.assertTrue(outputs["objective_native_soft_ranking_png"].is_file())
            self.assertTrue(outputs["objective_loglinear_main_ranking_png"].is_file())
            self.assertTrue(outputs["mainA_log_u0_vs_log_fog_iso_score_png"].is_file())
            self.assertTrue(outputs["mainB_u0_vs_fog_tradeoff_with_pareto_png"].is_file())
            self.assertTrue(outputs["mainE_u0_vs_fog_loglog_regression_png"].is_file())
            self.assertTrue(outputs["mainF_u0_vs_t50_loglog_regression_png"].is_file())
            self.assertTrue(outputs["supp_weight_sensitivity_png"].is_file())
            self.assertTrue(outputs["supp_threshold_sensitivity_png"].is_file())
            self.assertIn("fog_native_constrained_tradeoff_png", outputs)
            self.assertTrue(outputs["fog_native_constrained_tradeoff_png"].is_file())
            self.assertIn("objective_native_soft_tradeoff_png", outputs)
            self.assertTrue(outputs["objective_native_soft_tradeoff_png"].is_file())
            self.assertIn("objective_activity_bonus_penalty_profile_ranks_csv", outputs)
            self.assertTrue(outputs["objective_activity_bonus_penalty_profile_ranks_csv"].is_file())
            self.assertIn("objective_activity_bonus_penalty_proxy_curves_grid_png", outputs)
            self.assertTrue(outputs["objective_activity_bonus_penalty_proxy_curves_grid_png"].is_file())
            self.assertIn("objective_activity_bonus_penalty_profile_tradeoff_grid_png", outputs)
            self.assertTrue(outputs["objective_activity_bonus_penalty_profile_tradeoff_grid_png"].is_file())
            self.assertIn("objective_activity_bonus_penalty_profile_rank_heatmap_png", outputs)
            self.assertTrue(outputs["objective_activity_bonus_penalty_profile_rank_heatmap_png"].is_file())
            self.assertIn("figure_guide_md", outputs)
            self.assertTrue(outputs["figure_guide_md"].is_file())
            self.assertIn("fog_ranking_legacy_csv", outputs)
            self.assertTrue(outputs["fog_ranking_legacy_csv"].is_file())
            self.assertIn("fog_ranking_legacy_png", outputs)
            self.assertTrue(outputs["fog_ranking_legacy_png"].is_file())

            t50_tbl = pd.read_csv(outputs["t50_ranking_csv"])
            fog_tbl = pd.read_csv(outputs["fog_ranking_csv"])
            fog_native_tbl = pd.read_csv(outputs["fog_native_constrained_ranking_csv"])
            objective_tbl = pd.read_csv(outputs["objective_native_soft_ranking_csv"])
            objective_loglinear_tbl = pd.read_csv(outputs["objective_loglinear_main_ranking_csv"])

            self.assertEqual(t50_tbl.iloc[0]["polymer_id"], "P1")
            self.assertEqual(int(t50_tbl.iloc[0]["rank_t50"]), 1)
            self.assertEqual(fog_tbl.iloc[0]["polymer_id"], "P1")
            self.assertEqual(int(fog_tbl.iloc[0]["rank_fog"]), 1)
            self.assertEqual(fog_native_tbl.iloc[0]["polymer_id"], "P1")
            self.assertEqual(int(fog_native_tbl.iloc[0]["rank_fog_native_constrained"]), 1)
            self.assertEqual(objective_tbl.iloc[0]["polymer_id"], "P1")
            self.assertEqual(int(objective_tbl.iloc[0]["rank_objective_native_soft"]), 1)
            self.assertEqual(objective_loglinear_tbl.iloc[0]["polymer_id"], "P1")
            self.assertEqual(int(objective_loglinear_tbl.iloc[0]["rank_objective_loglinear_main"]), 1)
            self.assertTrue((t50_tbl["run_id"] == "R1").all())
            self.assertTrue((fog_tbl["run_id"] == "R1").all())
            self.assertIn("/csv/", str(outputs["fog_ranking_csv"]).replace("\\", "/"))

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
            self.assertTrue(outputs["fog_native_constrained_ranking_csv"].is_file())
            self.assertTrue(outputs["objective_native_soft_ranking_csv"].is_file())
            self.assertTrue(outputs["objective_loglinear_main_ranking_csv"].is_file())
            self.assertTrue(outputs["supp_rank_correlation_csv"].is_file())
            self.assertNotIn("t50_ranking_png", outputs)
            self.assertNotIn("fog_ranking_png", outputs)
            self.assertNotIn("fog_native_constrained_ranking_png", outputs)
            self.assertNotIn("objective_native_soft_ranking_png", outputs)
            self.assertNotIn("objective_loglinear_main_ranking_png", outputs)
            self.assertNotIn("fog_native_constrained_tradeoff_png", outputs)
            self.assertNotIn("objective_native_soft_tradeoff_png", outputs)
            self.assertNotIn("objective_activity_bonus_penalty_proxy_curves_grid_png", outputs)
            self.assertNotIn("objective_activity_bonus_penalty_profile_tradeoff_grid_png", outputs)
            self.assertNotIn("objective_activity_bonus_penalty_profile_rank_heatmap_png", outputs)
            self.assertNotIn("mainA_log_u0_vs_log_fog_iso_score_png", outputs)
            self.assertNotIn("mainB_u0_vs_fog_tradeoff_with_pareto_png", outputs)
            self.assertNotIn("mainE_u0_vs_fog_loglog_regression_png", outputs)
            self.assertNotIn("mainF_u0_vs_t50_loglog_regression_png", outputs)
            self.assertNotIn("supp_weight_sensitivity_png", outputs)
            self.assertNotIn("supp_threshold_sensitivity_png", outputs)

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
            fog_native_tbl = pd.read_csv(outputs["fog_native_constrained_ranking_csv"])
            objective_tbl = pd.read_csv(outputs["objective_native_soft_ranking_csv"])
            objective_loglinear_tbl = pd.read_csv(outputs["objective_loglinear_main_ranking_csv"])

            # Activity-adjusted scores:
            #   P_low_abs: 20 * 0.20 = 4
            #   P_high_abs: 16 * 0.95 = 15.2
            self.assertEqual(t50_tbl.iloc[0]["polymer_id"], "P_high_abs")
            self.assertEqual(int(t50_tbl.iloc[0]["rank_t50"]), 1)
            self.assertEqual(fog_tbl.iloc[0]["polymer_id"], "P_high_abs")
            self.assertEqual(int(fog_tbl.iloc[0]["rank_fog"]), 1)
            self.assertEqual(fog_native_tbl.iloc[0]["polymer_id"], "P_high_abs")
            self.assertNotIn("P_low_abs", set(fog_native_tbl["polymer_id"].astype(str)))
            self.assertEqual(objective_tbl.iloc[0]["polymer_id"], "P_high_abs")
            self.assertEqual(objective_loglinear_tbl.iloc[0]["polymer_id"], "P_high_abs")
            self.assertGreater(
                float(objective_tbl.loc[objective_tbl["polymer_id"] == "P_high_abs", "fog_native_soft"].iloc[0]),
                float(objective_tbl.loc[objective_tbl["polymer_id"] == "P_low_abs", "fog_native_soft"].iloc[0]),
            )

    def test_loglinear_objective_formula_and_tie_break(self) -> None:
        fog_df = pd.DataFrame(
            [
                {
                    "run_id": "R5",
                    "polymer_id": "GOx",
                    "t50_min": 10.0,
                    "t50_censored": 0,
                    "fog": 1.0,
                    "log_fog": 0.0,
                    "abs_activity_at_0": 100.0,
                },
                {
                    # U0*=1.0, FoG*=1.0 -> S=0
                    "run_id": "R5",
                    "polymer_id": "P_equal",
                    "t50_min": 10.0,
                    "t50_censored": 0,
                    "fog": 1.0,
                    "log_fog": 0.0,
                    "abs_activity_at_0": 100.0,
                },
                {
                    # Same score as P_tie_high_u0: FoG*=1.5, U0*=0.8 -> log(1.2)
                    "run_id": "R5",
                    "polymer_id": "P_tie_low_u0",
                    "t50_min": 15.0,
                    "t50_censored": 0,
                    "fog": 1.5,
                    "log_fog": math.log(1.5),
                    "abs_activity_at_0": 80.0,
                },
                {
                    # Same score as P_tie_low_u0: FoG*=1.2, U0*=1.0 -> log(1.2)
                    "run_id": "R5",
                    "polymer_id": "P_tie_high_u0",
                    "t50_min": 12.0,
                    "t50_censored": 0,
                    "fog": 1.2,
                    "log_fog": math.log(1.2),
                    "abs_activity_at_0": 100.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            outputs = write_run_ranking_outputs(fog_df, "R5", out_dir)
            objective_loglinear_tbl = pd.read_csv(outputs["objective_loglinear_main_ranking_csv"])
            score_equal = float(
                objective_loglinear_tbl.loc[
                    objective_loglinear_tbl["polymer_id"] == "P_equal",
                    "objective_loglinear_main",
                ].iloc[0]
            )
            self.assertAlmostEqual(score_equal, 0.0, places=8)

            tie_high_rank = int(
                objective_loglinear_tbl.loc[
                    objective_loglinear_tbl["polymer_id"] == "P_tie_high_u0",
                    "rank_objective_loglinear_main",
                ].iloc[0]
            )
            tie_low_rank = int(
                objective_loglinear_tbl.loc[
                    objective_loglinear_tbl["polymer_id"] == "P_tie_low_u0",
                    "rank_objective_loglinear_main",
                ].iloc[0]
            )
            self.assertLess(tie_high_rank, tie_low_rank)

    def test_write_run_ranking_outputs_with_custom_reference_polymer(self) -> None:
        fog_df = pd.DataFrame(
            [
                {
                    "run_id": "R4",
                    "polymer_id": "BETA-GAL",
                    "reference_polymer_id": "BETA-GAL",
                    "t50_min": 8.0,
                    "t50_censored": 0,
                    "fog": 1.0,
                    "log_fog": 0.0,
                    "abs_activity_at_0": 110.0,
                    "native_activity_rel_at_0": 1.0,
                    "native_activity_feasible": 1,
                },
                {
                    "run_id": "R4",
                    "polymer_id": "P1",
                    "reference_polymer_id": "BETA-GAL",
                    "t50_min": 12.0,
                    "t50_censored": 0,
                    "fog": 1.5,
                    "log_fog": 0.4054651081,
                    "abs_activity_at_0": 95.0,
                    "native_activity_rel_at_0": 0.86,
                    "native_activity_feasible": 1,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            outputs = write_run_ranking_outputs(
                fog_df,
                "R4",
                out_dir,
                reference_polymer_id="BETA-GAL",
            )
            self.assertIn("fog_native_constrained_tradeoff_png", outputs)
            self.assertTrue(outputs["fog_native_constrained_tradeoff_png"].is_file())
            fog_tbl = pd.read_csv(outputs["fog_ranking_csv"])
            self.assertIn("reference_polymer_id", fog_tbl.columns)
            self.assertTrue((fog_tbl["reference_polymer_id"].astype(str) == "BETA-GAL").all())
            self.assertEqual(str(fog_tbl.iloc[0]["polymer_id"]), "P1")

    def test_solvent_control_policy_etoh_preferred_with_fallback_and_dmso_forced_to_gox(self) -> None:
        base_rows = [
            {
                "run_id": "R6",
                "polymer_id": "GOx",
                "t50_min": 10.0,
                "t50_censored": 0,
                "fog": 1.0,
                "log_fog": 0.0,
                "abs_activity_at_0": 100.0,
            },
            {
                "run_id": "R6",
                "polymer_id": "GOx with DMSO",
                "t50_min": 8.0,
                "t50_censored": 0,
                "fog": 0.8,
                "log_fog": math.log(0.8),
                "abs_activity_at_0": 90.0,
            },
            {
                "run_id": "R6",
                "polymer_id": "GOx with EtOH",
                "t50_min": 7.0,
                "t50_censored": 0,
                "fog": 0.7,
                "log_fog": math.log(0.7),
                "abs_activity_at_0": 85.0,
            },
            {
                "run_id": "R6",
                "polymer_id": "PMBTA-2",
                "t50_min": 12.0,
                "t50_censored": 0,
                "fog": 1.2,
                "log_fog": math.log(1.2),
                "abs_activity_at_0": 95.0,
                "solvent_group": "DMSO",
            },
            {
                "run_id": "R6",
                "polymer_id": "PMPC",
                "t50_min": 9.8,
                "t50_censored": 0,
                "fog": 0.98,
                "log_fog": math.log(0.98),
                "abs_activity_at_0": 88.0,
                "solvent_group": "ETOH",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "with_etoh"
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs = write_run_ranking_outputs(pd.DataFrame(base_rows), "R6", out_dir)
            obj_tbl = pd.read_csv(outputs["objective_activity_bonus_penalty_ranking_csv"])
            dmso_row = obj_tbl[obj_tbl["polymer_id"] == "PMBTA-2"].iloc[0]
            etoh_row = obj_tbl[obj_tbl["polymer_id"] == "PMPC"].iloc[0]
            self.assertEqual(str(dmso_row["solvent_control_polymer_id"]).strip(), "GOx")
            self.assertEqual(str(etoh_row["solvent_control_polymer_id"]).strip(), "GOx with EtOH")

        rows_no_etoh_ctrl = [r for r in base_rows if str(r["polymer_id"]) != "GOx with EtOH"]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "without_etoh"
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs = write_run_ranking_outputs(pd.DataFrame(rows_no_etoh_ctrl), "R6", out_dir)
            obj_tbl = pd.read_csv(outputs["objective_activity_bonus_penalty_ranking_csv"])
            etoh_like_row = obj_tbl[obj_tbl["polymer_id"] == "PMPC"].iloc[0]
            self.assertEqual(str(etoh_like_row["solvent_control_polymer_id"]).strip(), "GOx")


if __name__ == "__main__":
    unittest.main()
