from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.bo_engine import BOConfig, run_bo  # noqa: E402


class TestBOEngine(unittest.TestCase):
    def test_sparse_design_uses_isotropic_kernel_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            learning_path = base / "bo_learning.csv"
            fog_path = base / "fog_plate_aware.csv"
            out_root = base / "bo_runs"

            # Sparse start set (7 points): PMBTA-1..5 + PMPC + PMTAC.
            learning_df = pd.DataFrame(
                [
                    {"polymer_id": "PMPC", "round_id": "R1", "frac_MPC": 1.0, "frac_BMA": 0.0, "frac_MTAC": 0.0, "log_fog": 0.10},
                    {"polymer_id": "PMTAC", "round_id": "R1", "frac_MPC": 0.0, "frac_BMA": 0.0, "frac_MTAC": 1.0, "log_fog": -0.05},
                    {"polymer_id": "PMBTA-1", "round_id": "R1", "frac_MPC": 0.80, "frac_BMA": 0.04, "frac_MTAC": 0.16, "log_fog": 0.00},
                    {"polymer_id": "PMBTA-2", "round_id": "R1", "frac_MPC": 0.80, "frac_BMA": 0.16, "frac_MTAC": 0.04, "log_fog": 0.08},
                    {"polymer_id": "PMBTA-3", "round_id": "R1", "frac_MPC": 0.30, "frac_BMA": 0.14, "frac_MTAC": 0.56, "log_fog": 0.12},
                    {"polymer_id": "PMBTA-4", "round_id": "R1", "frac_MPC": 0.30, "frac_BMA": 0.56, "frac_MTAC": 0.14, "log_fog": 0.11},
                    {"polymer_id": "PMBTA-5", "round_id": "R1", "frac_MPC": 0.55, "frac_BMA": 0.225, "frac_MTAC": 0.225, "log_fog": 0.06},
                ]
            )
            learning_df.to_csv(learning_path, index=False)

            fog_df = pd.DataFrame(
                [
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "PMPC", "t50_min": 10.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.0, "log_fog": 0.0},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "PMTAC", "t50_min": 9.5, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 0.95, "log_fog": -0.0512932944},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "PMBTA-1", "t50_min": 10.2, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.02, "log_fog": 0.0198026273},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "PMBTA-2", "t50_min": 10.7, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.07, "log_fog": 0.0676586485},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "PMBTA-3", "t50_min": 11.4, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.14, "log_fog": 0.1304536099},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "PMBTA-4", "t50_min": 11.3, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.13, "log_fog": 0.1222176327},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "PMBTA-5", "t50_min": 10.8, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.08, "log_fog": 0.0769610411},
                ]
            )
            fog_df.to_csv(fog_path, index=False)

            cfg = BOConfig(
                n_suggestions=4,
                candidate_step=0.10,
                min_component=0.0,
                min_distance_between=0.02,
                min_distance_to_train=0.0,
                write_plots=False,
            )
            outputs = run_bo(
                bo_learning_path=learning_path,
                fog_plate_aware_path=fog_path,
                out_root_dir=out_root,
                bo_run_id="bo_sparse",
                config=cfg,
            )

            with open(outputs["bo_summary"], encoding="utf-8") as f:
                summary = json.load(f)
            self.assertFalse(summary["config"]["use_bma_mtac_coords"])
            self.assertTrue(summary["config"]["force_isotropic_applied"])
            self.assertEqual(summary["gp_hyperparams"]["kernel_mode"], "isotropic")

    def test_run_bo_rejects_round_anchor_correction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            learning_path = base / "bo_learning.csv"
            fog_path = base / "fog_plate_aware.csv"
            out_root = base / "bo_runs"

            learning_df = pd.DataFrame(
                [
                    {"polymer_id": "P1", "round_id": "R1", "frac_MPC": 0.80, "frac_BMA": 0.10, "frac_MTAC": 0.10, "log_fog": 0.05},
                    {"polymer_id": "P2", "round_id": "R1", "frac_MPC": 0.60, "frac_BMA": 0.20, "frac_MTAC": 0.20, "log_fog": 0.10},
                    {"polymer_id": "P3", "round_id": "R2", "frac_MPC": 0.40, "frac_BMA": 0.30, "frac_MTAC": 0.30, "log_fog": 0.15},
                ]
            )
            learning_df.to_csv(learning_path, index=False)

            fog_df = pd.DataFrame(
                [
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "P1", "t50_min": 12.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.2, "log_fog": 0.1823215568},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "P2", "t50_min": 13.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.3, "log_fog": 0.2623642645},
                    {"round_id": "R2", "run_id": "runB", "plate_id": "plate1", "polymer_id": "P3", "t50_min": 14.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.4, "log_fog": 0.3364722366},
                ]
            )
            fog_df.to_csv(fog_path, index=False)

            cfg = BOConfig(
                n_suggestions=3,
                candidate_step=0.20,
                min_component=0.0,
                min_distance_between=0.01,
                min_distance_to_train=0.0,
                write_plots=False,
                apply_round_anchor_correction=True,
            )

            with self.assertRaisesRegex(ValueError, "not allowed"):
                run_bo(
                    bo_learning_path=learning_path,
                    fog_plate_aware_path=fog_path,
                    out_root_dir=out_root,
                    bo_run_id="bo_forbidden_correction",
                    config=cfg,
                )

    def test_run_bo_writes_core_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            learning_path = base / "bo_learning.csv"
            fog_path = base / "fog_plate_aware.csv"
            out_root = base / "bo_runs"

            learning_df = pd.DataFrame(
                [
                    # R1
                    {"polymer_id": "P1", "round_id": "R1", "frac_MPC": 0.80, "frac_BMA": 0.10, "frac_MTAC": 0.10, "log_fog": 0.05},
                    {"polymer_id": "P2", "round_id": "R1", "frac_MPC": 0.60, "frac_BMA": 0.20, "frac_MTAC": 0.20, "log_fog": 0.10},
                    {"polymer_id": "P3", "round_id": "R1", "frac_MPC": 0.40, "frac_BMA": 0.30, "frac_MTAC": 0.30, "log_fog": 0.15},
                    # R2
                    {"polymer_id": "P1", "round_id": "R2", "frac_MPC": 0.80, "frac_BMA": 0.10, "frac_MTAC": 0.10, "log_fog": 0.00},
                    {"polymer_id": "P2", "round_id": "R2", "frac_MPC": 0.60, "frac_BMA": 0.20, "frac_MTAC": 0.20, "log_fog": 0.08},
                    {"polymer_id": "P3", "round_id": "R2", "frac_MPC": 0.40, "frac_BMA": 0.30, "frac_MTAC": 0.30, "log_fog": 0.12},
                ]
            )
            learning_df.to_csv(learning_path, index=False)

            fog_df = pd.DataFrame(
                [
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "P1", "t50_min": 12.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.2, "log_fog": 0.1823215568},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "P2", "t50_min": 13.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.3, "log_fog": 0.2623642645},
                    {"round_id": "R1", "run_id": "runA", "plate_id": "plate1", "polymer_id": "P3", "t50_min": 14.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.4, "log_fog": 0.3364722366},
                    {"round_id": "R2", "run_id": "runB", "plate_id": "plate1", "polymer_id": "P1", "t50_min": 11.5, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.15, "log_fog": 0.1397619424},
                    {"round_id": "R2", "run_id": "runB", "plate_id": "plate1", "polymer_id": "P2", "t50_min": 12.8, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.28, "log_fog": 0.2468600779},
                    {"round_id": "R2", "run_id": "runB", "plate_id": "plate1", "polymer_id": "P3", "t50_min": 13.9, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.39, "log_fog": 0.3293037471},
                ]
            )
            fog_df.to_csv(fog_path, index=False)

            cfg = BOConfig(
                n_suggestions=3,
                exploration_ratio=0.34,
                candidate_step=0.20,
                min_component=0.0,
                min_distance_between=0.01,
                min_distance_to_train=0.0,
                write_plots=False,
            )

            outputs = run_bo(
                bo_learning_path=learning_path,
                fog_plate_aware_path=fog_path,
                out_root_dir=out_root,
                bo_run_id="bo_test",
                config=cfg,
            )

            required_keys = [
                "training_data",
                "candidate_log",
                "suggestions",
                "next_experiment_top5",
                "fog_rank_all",
                "fog_rank_by_round",
                "t50_rank_all",
                "t50_rank_by_round",
                "bo_summary",
                "manifest",
            ]
            for key in required_keys:
                self.assertIn(key, outputs)
                self.assertTrue(Path(outputs[key]).is_file(), f"missing output: {key}")

            suggestions = pd.read_csv(outputs["suggestions"])
            self.assertGreaterEqual(len(suggestions), 1)
            self.assertLessEqual(len(suggestions), 3)
            self.assertTrue((suggestions["selected"] == 1).all())
            self.assertTrue((suggestions["constraint_sum_ok"]).all())
            self.assertTrue((suggestions["constraint_bounds_ok"]).all())
            self.assertIn("selection_reason", suggestions.columns)

            next_exp = pd.read_csv(outputs["next_experiment_top5"])
            self.assertGreaterEqual(len(next_exp), 1)
            self.assertLessEqual(len(next_exp), 3)
            self.assertIn("pred_fog_mean", next_exp.columns)
            self.assertIn("pred_t50_min_mean_vs_last_round_gox", next_exp.columns)
            self.assertIn("last_round_gox_t50_min_median", next_exp.columns)
            self.assertIn("prob_fog_gt_1", next_exp.columns)
            self.assertIn("pred_fog_lower95", next_exp.columns)
            self.assertIn("pred_t50_min_lower95_vs_last_round_gox", next_exp.columns)
            self.assertIn("priority_rank", next_exp.columns)
            self.assertIn("priority_score", next_exp.columns)
            self.assertIn("recommended_top3", next_exp.columns)
            self.assertTrue((next_exp["priority_rank"].diff().fillna(1) >= 0).all())

    def test_run_bo_robust_heteroskedastic_control_replicate(self) -> None:
        """Integration: run_bo with robust columns, heteroskedastic, control, replicate."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            learning_path = base / "bo_learning.csv"
            fog_path = base / "fog_plate_aware.csv"
            out_root = base / "bo_runs"

            learning_df = pd.DataFrame(
                [
                    {"polymer_id": "PMPC", "round_id": "R1", "frac_MPC": 1.0, "frac_BMA": 0.0, "frac_MTAC": 0.0, "log_fog": 0.1, "n_observations": 2, "log_fog_mad": 0.02},
                    {"polymer_id": "PMTAC", "round_id": "R1", "frac_MPC": 0.0, "frac_BMA": 0.0, "frac_MTAC": 1.0, "log_fog": 0.05, "n_observations": 2, "log_fog_mad": 0.01},
                    {"polymer_id": "P1", "round_id": "R1", "frac_MPC": 0.5, "frac_BMA": 0.3, "frac_MTAC": 0.2, "log_fog": 0.2, "n_observations": 3, "log_fog_mad": 0.03},
                    {"polymer_id": "P2", "round_id": "R1", "frac_MPC": 0.4, "frac_BMA": 0.35, "frac_MTAC": 0.25, "log_fog": 0.15, "n_observations": 2, "log_fog_mad": 0.02},
                    {"polymer_id": "P3", "round_id": "R1", "frac_MPC": 0.6, "frac_BMA": 0.2, "frac_MTAC": 0.2, "log_fog": 0.25, "n_observations": 2, "log_fog_mad": 0.025},
                ]
            )
            learning_df.to_csv(learning_path, index=False)

            fog_df = pd.DataFrame(
                [
                    {"round_id": "R1", "run_id": "r1", "plate_id": "p1", "polymer_id": "PMPC", "t50_min": 10.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.1, "log_fog": 0.095},
                    {"round_id": "R1", "run_id": "r1", "plate_id": "p1", "polymer_id": "PMTAC", "t50_min": 9.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 0.9, "log_fog": -0.105},
                    {"round_id": "R1", "run_id": "r1", "plate_id": "p1", "polymer_id": "P1", "t50_min": 12.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.2, "log_fog": 0.182},
                    {"round_id": "R1", "run_id": "r1", "plate_id": "p1", "polymer_id": "P2", "t50_min": 11.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.1, "log_fog": 0.095},
                    {"round_id": "R1", "run_id": "r1", "plate_id": "p1", "polymer_id": "P3", "t50_min": 13.0, "gox_t50_used_min": 10.0, "denominator_source": "same_plate", "fog": 1.3, "log_fog": 0.262},
                ]
            )
            fog_df.to_csv(fog_path, index=False)

            cfg = BOConfig(
                n_suggestions=8,
                exploration_ratio=0.35,
                anchor_fraction=0.12,
                replicate_fraction=0.12,
                candidate_step=0.15,
                min_component=0.0,
                min_distance_between=0.02,
                min_distance_to_train=0.0,
                write_plots=False,
                enable_heteroskedastic_noise=True,
                use_exact_anchor_compositions=True,
                replicate_source="exploit",
            )

            outputs = run_bo(
                bo_learning_path=learning_path,
                fog_plate_aware_path=fog_path,
                out_root_dir=out_root,
                bo_run_id="bo_integration",
                config=cfg,
            )

            self.assertIn("suggestions", outputs)
            suggestions = pd.read_csv(outputs["suggestions"])
            self.assertGreaterEqual(len(suggestions), 1)
            self.assertTrue((suggestions["selected"] == 1).all())
            reasons = suggestions["selection_reason"].astype(str)
            self.assertTrue(
                reasons.str.contains("anchor_|exploit|explore|replicate", case=False, regex=True).any(),
                "expected at least one of anchor_, exploit, explore, replicate",
            )

            with open(outputs["bo_summary"], encoding="utf-8") as f:
                summary = json.load(f)
            self.assertIn("gp_hyperparams", summary)
            self.assertIn("obs_noise_rel_min", summary["gp_hyperparams"])
            self.assertIn("obs_noise_rel_median", summary["gp_hyperparams"])

            with open(outputs["manifest"], encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertIn("extra", manifest)
            self.assertIn("bo_learning_path", manifest["extra"])
            self.assertIn("fog_plate_aware_path", manifest["extra"])


if __name__ == "__main__":
    unittest.main()
