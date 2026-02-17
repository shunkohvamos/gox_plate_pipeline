from __future__ import annotations

import json
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

from gox_plate_pipeline.bo_engine import propose_batch_next_points, run_pure_regression_bo  # noqa: E402


def _sample_learning_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"polymer_id": "A", "round_id": "R1", "frac_MPC": 0.80, "frac_BMA": 0.10, "frac_MTAC": 0.10, "log_fog": 0.02, "log_fog_corrected": 0.02, "log_fog_native_constrained": 0.02},
            {"polymer_id": "B", "round_id": "R1", "frac_MPC": 0.60, "frac_BMA": 0.25, "frac_MTAC": 0.15, "log_fog": 0.18, "log_fog_corrected": 0.18, "log_fog_native_constrained": 0.18},
            {"polymer_id": "C", "round_id": "R1", "frac_MPC": 0.35, "frac_BMA": 0.45, "frac_MTAC": 0.20, "log_fog": 0.30, "log_fog_corrected": 0.30, "log_fog_native_constrained": 0.30},
            {"polymer_id": "A", "round_id": "R2", "frac_MPC": 0.80, "frac_BMA": 0.10, "frac_MTAC": 0.10, "log_fog": 0.01, "log_fog_corrected": 0.01, "log_fog_native_constrained": 0.01},
            {"polymer_id": "D", "round_id": "R2", "frac_MPC": 0.50, "frac_BMA": 0.10, "frac_MTAC": 0.40, "log_fog": 0.12, "log_fog_corrected": 0.12, "log_fog_native_constrained": 0.12},
            {"polymer_id": "E", "round_id": "R2", "frac_MPC": 0.20, "frac_BMA": 0.60, "frac_MTAC": 0.20, "log_fog": 0.28, "log_fog_corrected": 0.28, "log_fog_native_constrained": 0.28},
        ]
    )


def _pairwise_min_distance(df: pd.DataFrame) -> float:
    frac = df[["frac_MPC", "frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    if frac.shape[0] < 2:
        return float("nan")
    diff = frac[:, None, :] - frac[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    i, j = np.triu_indices(frac.shape[0], k=1)
    return float(np.min(d[i, j]))


class TestPureRegressionBO(unittest.TestCase):
    def test_propose_batch_returns_expected_columns_and_q(self) -> None:
        learning_df = _sample_learning_df()
        cand, _ = propose_batch_next_points(
            learning_df,
            q=4,
            acquisition="ei",
            seed=123,
            n_random_candidates=4000,
            composition_constraints={
                "min_mpc": 0.05,
                "max_mpc": 0.95,
                "min_bma": 0.05,
                "max_bma": 0.95,
                "min_mtac": 0.05,
                "max_mtac": 0.95,
            },
        )
        self.assertEqual(len(cand), 4)
        required = {
            "x",
            "y",
            "pred_mean",
            "pred_std",
            "acq_value",
            "frac_MPC",
            "frac_BMA",
            "frac_MTAC",
            "min_dist_to_known",
            "acquisition",
            "ucb_kappa_used",
            "diversity_threshold",
            "diversity_relaxed",
            "selection_order",
        }
        self.assertTrue(required.issubset(set(cand.columns)))
        sums = cand["frac_MPC"] + cand["frac_BMA"] + cand["frac_MTAC"]
        self.assertTrue(np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-6))
        self.assertTrue((cand["selection_order"].astype(int).to_numpy() == np.arange(1, 5)).all())

    def test_ucb_beta_is_supported(self) -> None:
        learning_df = _sample_learning_df()
        cand, _ = propose_batch_next_points(
            learning_df,
            q=3,
            acquisition="ucb",
            seed=7,
            n_random_candidates=3000,
            ucb_kappa=1.0,
            ucb_beta=9.0,
        )
        self.assertEqual(len(cand), 3)
        self.assertTrue(np.allclose(cand["ucb_kappa_used"].to_numpy(dtype=float), 3.0))

    def test_default_objective_missing_raises_error(self) -> None:
        learning_df = _sample_learning_df().drop(columns=["log_fog_native_constrained"])
        with self.assertRaises(ValueError):
            propose_batch_next_points(
                learning_df,
                q=3,
                acquisition="ei",
                seed=11,
                n_random_candidates=2500,
            )

    def test_infeasible_constraints_raise(self) -> None:
        learning_df = _sample_learning_df()
        with self.assertRaises(ValueError):
            propose_batch_next_points(
                learning_df,
                q=2,
                acquisition="ei",
                seed=1,
                n_random_candidates=200,
                composition_constraints={
                    "min_mpc": 0.40,
                    "max_mpc": 0.95,
                    "min_bma": 0.40,
                    "max_bma": 0.95,
                    "min_mtac": 0.40,
                    "max_mtac": 0.95,
                },
            )

    def test_diversity_in_fraction_space_spreads_batch(self) -> None:
        learning_df = _sample_learning_df()
        div_params = {"min_fraction_distance": 0.12}
        cand_on, _ = propose_batch_next_points(
            learning_df,
            q=4,
            acquisition="ei",
            seed=99,
            n_random_candidates=12000,
            diversity_params=div_params,
            enable_diversity=True,
        )
        cand_off, _ = propose_batch_next_points(
            learning_df,
            q=4,
            acquisition="ei",
            seed=99,
            n_random_candidates=12000,
            diversity_params=div_params,
            enable_diversity=False,
        )
        min_on = _pairwise_min_distance(cand_on)
        min_off = _pairwise_min_distance(cand_off)
        self.assertGreaterEqual(min_on, min_off - 1e-9)

    def test_run_pure_regression_bo_writes_manifest_and_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            learning_df = _sample_learning_df()
            learning_path = base / "bo_learning.csv"
            learning_df.to_csv(learning_path, index=False)
            out_dir = base / "out"
            outputs = run_pure_regression_bo(
                learning_df=learning_df,
                out_dir=out_dir,
                q=3,
                acquisition="ucb",
                seed=42,
                run_id="pure_test",
                write_plots=False,
                ucb_beta=4.0,
                learning_input_path=learning_path,
            )
            for key in ["candidates", "candidates_no_diversity", "summary", "manifest"]:
                self.assertIn(key, outputs)
                self.assertTrue(Path(outputs[key]).is_file(), f"missing output: {key}")

            cand = pd.read_csv(outputs["candidates"])
            self.assertTrue((cand["run_id"] == "pure_test").all())
            self.assertTrue((cand["bo_run_id"] == "pure_test").all())
            self.assertEqual(len(cand), 3)

            with open(outputs["summary"], encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["run_id"], "pure_test")
            self.assertEqual(summary["acquisition"], "ucb")
            self.assertAlmostEqual(float(summary["config"]["ucb_kappa_used"]), 2.0, places=6)

            with open(outputs["manifest"], encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["run_id"], "pure_test")
            self.assertIn("input_files", manifest)
            self.assertGreaterEqual(len(manifest["input_files"]), 1)


if __name__ == "__main__":
    unittest.main()
