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

from gox_plate_pipeline.fog import build_fog_summary  # noqa: E402
from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    fit_exponential_decay,
    t50_linear,
    value_at_time_linear,
)


class TestT50DefinitionModes(unittest.TestCase):
    def test_fit_exponential_decay_t50_modes(self) -> None:
        t = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0], dtype=float)
        y_true = 130.0 * np.exp(-0.05 * t)

        fit_half = fit_exponential_decay(t, y_true, y0=130.0, min_points=4, t50_definition="y0_half")
        fit_rea50 = fit_exponential_decay(t, y_true, y0=130.0, min_points=4, t50_definition="rea50")

        self.assertIsNotNone(fit_half)
        self.assertIsNotNone(fit_rea50)
        assert fit_half is not None
        assert fit_rea50 is not None
        self.assertEqual(fit_half.model, "exp")
        self.assertEqual(fit_rea50.model, "exp")

        expected_half = float(np.log(2.0) / fit_half.k)
        expected_rea50 = float(np.log(fit_rea50.y0 / 50.0) / fit_rea50.k)
        self.assertAlmostEqual(float(fit_half.t50), expected_half, places=6)
        self.assertAlmostEqual(float(fit_rea50.t50), expected_rea50, places=6)
        self.assertGreater(float(fit_rea50.t50), float(fit_half.t50))

    def test_fit_exponential_decay_with_fixed_y0_for_rea(self) -> None:
        t = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0], dtype=float)
        y_true = 100.0 * np.exp(-0.05 * t)

        fit = fit_exponential_decay(
            t,
            y_true,
            y0=95.0,  # ignored when fixed_y0 is provided
            fixed_y0=100.0,
            min_points=4,
            t50_definition="rea50",
        )

        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertEqual(fit.model, "exp")
        self.assertAlmostEqual(float(fit.y0), 100.0, places=9)
        expected_t50 = float(np.log(100.0 / 50.0) / fit.k)
        self.assertAlmostEqual(float(fit.t50), expected_t50, places=6)

    def test_t50_linear_fixed_target(self) -> None:
        t = np.array([0.0, 10.0, 20.0], dtype=float)
        y = np.array([40.0, 30.0, 20.0], dtype=float)
        # Already below REA=50 at the first observed point.
        t50 = t50_linear(t, y, y0=100.0, target_value=50.0)
        self.assertEqual(float(t50), 0.0)

    def test_value_at_time_linear_for_rea20(self) -> None:
        t = np.array([0.0, 10.0, 30.0], dtype=float)
        y = np.array([100.0, 80.0, 40.0], dtype=float)
        rea20 = value_at_time_linear(t, y, at_time_min=20.0)
        self.assertAlmostEqual(float(rea20), 60.0, places=6)

    def test_build_fog_summary_uses_target_column_for_censoring(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "run_id,polymer_id,polymer_label,n_points,y0_REA_percent,t50_definition,t50_target_rea_percent,"
                "t50_linear_min,t50_exp_min,fit_model,fit_k_per_min,fit_tau_min,fit_plateau,fit_r2,rea_connector\n"
            )
            # Plateau below target(70): not censored when t50 exists.
            f.write("R1,P1,P1,7,140,y0_half,70,6.0,5.0,exp_plateau,0.1,10,60,0.99,exp\n")
            # Plateau above/equal target(50): censored.
            f.write("R1,P2,P2,7,140,rea50,50,6.0,5.0,exp_plateau,0.1,10,60,0.99,exp\n")
            t50_path = Path(f.name)
        try:
            fog_df = build_fog_summary(t50_path, "R1")
            p1 = fog_df[fog_df["polymer_id"] == "P1"].iloc[0]
            p2 = fog_df[fog_df["polymer_id"] == "P2"].iloc[0]
            self.assertEqual(int(p1["t50_censored"]), 0)
            self.assertEqual(int(p2["t50_censored"]), 1)
        finally:
            t50_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
