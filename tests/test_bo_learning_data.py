"""
Minimal tests for BO learning data:
- Composition sum = 1 in BO catalog.
- BO learning CSV only contains polymer_ids from the BO catalog.
- Excluded report contains non-catalog and missing log_fog rows.
"""
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

from gox_plate_pipeline.bo_data import (  # noqa: E402
    load_bo_catalog,
    build_bo_learning_data,
    xy_from_frac,
    frac_from_xy,
    BO_X_COLS,
    BO_Y_COL,
)
from gox_plate_pipeline.fog import build_fog_summary  # noqa: E402
import numpy as np  # noqa: E402


class TestBOLearningData(unittest.TestCase):
    def test_catalog_sum_equal_one(self) -> None:
        """Composition columns must sum to 1."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("polymer_id,frac_MPC,frac_BMA,frac_MTAC\n")
            f.write("P1,0.5,0.4,0.1\n")
            f.write("P2,0.4,0.35,0.25\n")
            path = Path(f.name)
        try:
            cat = load_bo_catalog(path, validate_sum=True)
            self.assertEqual(cat.shape[0], 2)
            for _, row in cat.iterrows():
                self.assertAlmostEqual(row["frac_MPC"] + row["frac_BMA"] + row["frac_MTAC"], 1.0, places=5)
        finally:
            path.unlink(missing_ok=True)

    def test_catalog_tsv_loads(self) -> None:
        """BO catalog can be TSV (tab-separated); composition sum = 1."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("polymer_id\tfrac_MPC\tfrac_BMA\tfrac_MTAC\n")
            f.write("P1\t0.5\t0.4\t0.1\n")
            f.write("P2\t0.4\t0.35\t0.25\n")
            path = Path(f.name)
        try:
            cat = load_bo_catalog(path, validate_sum=True)
            self.assertEqual(cat.shape[0], 2)
            for _, row in cat.iterrows():
                self.assertAlmostEqual(row["frac_MPC"] + row["frac_BMA"] + row["frac_MTAC"], 1.0, places=5)
        finally:
            path.unlink(missing_ok=True)

    def test_catalog_sum_not_one_raises(self) -> None:
        """Catalog with sum != 1 raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("polymer_id,frac_MPC,frac_BMA,frac_MTAC\n")
            f.write("P1,0.5,0.5,0.0\n")  # sum=1 ok
            f.write("P2,0.5,0.5,0.1\n")  # sum=1.1
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_bo_catalog(path, validate_sum=True)
            self.assertIn("sum must be 1", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_learning_only_catalog_polymers(self) -> None:
        """BO learning CSV must not contain polymer_ids outside the BO catalog."""
        catalog_df = pd.DataFrame({
            "polymer_id": ["PMBTA-1"],
            "frac_MPC": [0.5],
            "frac_BMA": [0.4],
            "frac_MTAC": [0.1],
        })

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("run_id,polymer_id,t50_min,t50_censored,gox_t50_same_run_min,fog,log_fog,fog_missing_reason,n_points,input_t50_file,input_tidy\n")
            f.write("R1,GOx,10.0,0,10.0,1.0,0.0,,7,f,t\n")
            f.write("R1,PMBTA-1,11.0,0,10.0,1.1,0.095,,7,f,t\n")
            f.write("R1,PMBTA-2,12.0,0,10.0,1.2,0.182,,7,f,t\n")
            fog_path = Path(f.name)
        try:
            learning_df, excluded_df = build_bo_learning_data(catalog_df, [fog_path])

            # Only PMBTA-1 is in catalog; learning must have only PMBTA-1
            self.assertEqual(set(learning_df["polymer_id"]), {"PMBTA-1"})
            self.assertEqual(len(learning_df), 1)
            self.assertEqual(learning_df[BO_Y_COL].iloc[0], 0.095)

            # GOx and PMBTA-2 must be in excluded (not_in_bo_catalog)
            self.assertEqual(len(excluded_df), 2)
            reasons = set(excluded_df["reason"])
            self.assertIn("not_in_bo_catalog", reasons)
        finally:
            fog_path.unlink(missing_ok=True)

    def test_excluded_when_log_fog_missing(self) -> None:
        """Rows with missing log_fog are excluded and reported."""
        catalog_df = pd.DataFrame({
            "polymer_id": ["PMBTA-1", "PMBTA-2"],
            "frac_MPC": [0.5, 0.5],
            "frac_BMA": [0.4, 0.35],
            "frac_MTAC": [0.1, 0.15],
        })

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("run_id,polymer_id,t50_min,t50_censored,gox_t50_same_run_min,fog,log_fog,fog_missing_reason,n_points,input_t50_file,input_tidy\n")
            f.write("R1,PMBTA-1,11.0,0,10.0,1.1,0.095,,7,f,t\n")
            f.write("R1,PMBTA-2,,1,,,,\"no_bare_gox_in_run\",7,f,t\n")
            fog_path = Path(f.name)
        try:
            learning_df, excluded_df = build_bo_learning_data(catalog_df, [fog_path])

            self.assertEqual(len(learning_df), 1)
            self.assertEqual(learning_df["polymer_id"].iloc[0], "PMBTA-1")

            self.assertEqual(len(excluded_df), 1)
            self.assertEqual(excluded_df["polymer_id"].iloc[0], "PMBTA-2")
            self.assertIn("no_bare_gox_in_run", excluded_df["reason"].iloc[0])
        finally:
            fog_path.unlink(missing_ok=True)

    def test_fog_nan_when_no_gox_in_run(self) -> None:
        """Run without GOx must yield FoG and log_fog NaN (fog_missing_reason = no_bare_gox_in_run)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("run_id,polymer_id,polymer_label,n_points,y0_REA_percent,t50_linear_min,t50_exp_min,fit_model,fit_k_per_min,fit_tau_min,fit_plateau,fit_r2,rea_connector\n")
            f.write("R1,PMBTA-1,PMBTA-1,7,100.0,12.0,11.5,exp,0.06,16.7,,0.98,exp\n")
            t50_path = Path(f.name)
        try:
            fog_df = build_fog_summary(t50_path, "R1")
            self.assertEqual(len(fog_df), 1)
            self.assertEqual(fog_df["polymer_id"].iloc[0], "PMBTA-1")
            self.assertTrue(pd.isna(fog_df["gox_t50_same_run_min"].iloc[0]))
            self.assertTrue(pd.isna(fog_df["fog"].iloc[0]))
            self.assertTrue(pd.isna(fog_df["log_fog"].iloc[0]))
            self.assertEqual(fog_df["fog_missing_reason"].iloc[0], "no_bare_gox_in_run")
        finally:
            t50_path.unlink(missing_ok=True)

    def test_xy_blank_computed_from_frac(self) -> None:
        """When x,y are blank in catalog, they are computed from frac."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("polymer_id,frac_MPC,frac_BMA,frac_MTAC\n")
            f.write("P1,0.5,0.4,0.1\n")  # y=0.5, x=0.4/0.5=0.8
            path = Path(f.name)
        try:
            cat = load_bo_catalog(path, validate_sum=True)
            self.assertIn("x", cat.columns)
            self.assertIn("y", cat.columns)
            self.assertAlmostEqual(cat["y"].iloc[0], 0.5, places=5)
            self.assertAlmostEqual(cat["x"].iloc[0], 0.8, places=5)
        finally:
            path.unlink(missing_ok=True)

    def test_xy_wrong_raises(self) -> None:
        """When x,y are provided but inconsistent with frac, loader raises."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("polymer_id,frac_MPC,frac_BMA,frac_MTAC,x,y\n")
            f.write("P1,0.5,0.4,0.1,0.5,0.5\n")  # correct would be x=0.8, y=0.5
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_bo_catalog(path, validate_sum=True, validate_xy_consistency=True)
            self.assertIn("inconsistent", str(ctx.exception).lower())
        finally:
            path.unlink(missing_ok=True)

    def test_y_zero_x_nan_inverse_ok(self) -> None:
        """y==0: x is NaN; frac_from_xy(0, 0) gives frac_MPC=1, frac_BMA=0, frac_MTAC=0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("polymer_id,frac_MPC,frac_BMA,frac_MTAC,x,y\n")
            f.write("PMPC,1.0,0.0,0.0,,0\n")  # x blank, y=0
            path = Path(f.name)
        try:
            cat = load_bo_catalog(path, validate_sum=True)
            self.assertTrue(np.isnan(cat["x"].iloc[0]))
            self.assertAlmostEqual(cat["y"].iloc[0], 0.0, places=5)
        finally:
            path.unlink(missing_ok=True)
        # inverse: y==0 -> frac_MPC=1, frac_BMA=0, frac_MTAC=0; x ignored
        fMPC, fBMA, fMTAC = frac_from_xy(0.0, 0.0)
        self.assertAlmostEqual(float(fMPC), 1.0, places=5)
        self.assertAlmostEqual(float(fBMA), 0.0, places=5)
        self.assertAlmostEqual(float(fMTAC), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
