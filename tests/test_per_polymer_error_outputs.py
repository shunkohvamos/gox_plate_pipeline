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

from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    plot_per_polymer_timeseries_with_error_band,
)


class TestPerPolymerErrorOutputs(unittest.TestCase):
    def test_writes_per_polymer_and_all_with_error_plots(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "R-test"
            fit_dir = root / "fit"
            fit_dir.mkdir(parents=True, exist_ok=True)
            stats_path = fit_dir / "summary_stats.csv"
            pd.DataFrame(
                [
                    {"polymer_id": "GOx", "heat_min": 0.0, "n": 3, "mean_abs_activity": 10.0, "sem_abs_activity": 0.3, "mean_REA_percent": 100.0, "sem_REA_percent": 1.0, "include_in_all_polymers": True},
                    {"polymer_id": "GOx", "heat_min": 20.0, "n": 3, "mean_abs_activity": 4.0, "sem_abs_activity": 0.2, "mean_REA_percent": 40.0, "sem_REA_percent": 1.5, "include_in_all_polymers": True},
                    {"polymer_id": "GOx", "heat_min": 60.0, "n": 3, "mean_abs_activity": 1.0, "sem_abs_activity": 0.1, "mean_REA_percent": 10.0, "sem_REA_percent": 0.8, "include_in_all_polymers": True},
                    {"polymer_id": "P1", "heat_min": 0.0, "n": 3, "mean_abs_activity": 12.0, "sem_abs_activity": 0.4, "mean_REA_percent": 100.0, "sem_REA_percent": 1.2, "include_in_all_polymers": True},
                    {"polymer_id": "P1", "heat_min": 20.0, "n": 3, "mean_abs_activity": 7.0, "sem_abs_activity": 0.3, "mean_REA_percent": 58.0, "sem_REA_percent": 1.6, "include_in_all_polymers": True},
                    {"polymer_id": "P1", "heat_min": 60.0, "n": 3, "mean_abs_activity": 2.0, "sem_abs_activity": 0.2, "mean_REA_percent": 17.0, "sem_REA_percent": 0.9, "include_in_all_polymers": True},
                ]
            ).to_csv(stats_path, index=False)

            t50_dir = fit_dir / "t50" / "csv"
            t50_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"polymer_id": "GOx", "native_activity_rel_at_0": 1.0, "t50_exp_min": 22.0, "t50_linear_min": 21.0},
                    {"polymer_id": "P1", "native_activity_rel_at_0": 1.15, "t50_exp_min": 35.0, "t50_linear_min": 33.0},
                ]
            ).to_csv(t50_dir / f"t50__{run_id}.csv", index=False)

            out_dir = plot_per_polymer_timeseries_with_error_band(
                summary_stats_path=stats_path,
                run_id=run_id,
                out_fit_dir=fit_dir,
                color_map_path=root / "polymer_colors.yml",
                reference_polymer_id="GOX",
                native_activity_min_rel=0.70,
                t50_definition="y0_half",
            )

            self.assertIsNotNone(out_dir)
            assert out_dir is not None
            self.assertTrue((out_dir / f"GOx__{run_id}.png").is_file())
            self.assertTrue((out_dir / f"P1__{run_id}.png").is_file())
            self.assertTrue((fit_dir / f"all_polymers_with_error__{run_id}.png").is_file())
            self.assertTrue((fit_dir / f"all_polymers_with_error_decision_split__{run_id}.png").is_file())

    def test_returns_none_without_replicates_and_removes_stale_all_plot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "R-no-rep"
            fit_dir = root / "fit"
            fit_dir.mkdir(parents=True, exist_ok=True)
            stats_path = fit_dir / "summary_stats.csv"
            pd.DataFrame(
                [
                    {"polymer_id": "GOx", "heat_min": 0.0, "n": 1, "mean_abs_activity": 10.0, "sem_abs_activity": 0.0, "mean_REA_percent": 100.0, "sem_REA_percent": 0.0},
                    {"polymer_id": "P1", "heat_min": 0.0, "n": 1, "mean_abs_activity": 9.0, "sem_abs_activity": 0.0, "mean_REA_percent": 100.0, "sem_REA_percent": 0.0},
                ]
            ).to_csv(stats_path, index=False)

            stale_all = fit_dir / f"all_polymers_with_error__{run_id}.png"
            stale_all.write_bytes(b"stale")
            stale_all_decision = fit_dir / f"all_polymers_with_error_decision_split__{run_id}.png"
            stale_all_decision.write_bytes(b"stale")

            out_dir = plot_per_polymer_timeseries_with_error_band(
                summary_stats_path=stats_path,
                run_id=run_id,
                out_fit_dir=fit_dir,
                color_map_path=root / "polymer_colors.yml",
                reference_polymer_id="GOX",
            )

            self.assertIsNone(out_dir)
            self.assertFalse(stale_all.exists())
            self.assertFalse(stale_all_decision.exists())


if __name__ == "__main__":
    unittest.main()
