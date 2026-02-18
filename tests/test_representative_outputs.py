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
    plot_per_polymer_timeseries,
    plot_per_polymer_timeseries_across_runs_with_error_bars,
)


def _base_summary_rows(scale: float = 1.0) -> list[dict[str, object]]:
    polymers = {
        "GOx": ([10.0, 6.0, 2.0], [100.0, 60.0, 20.0]),
        "GOx with EtOH": ([9.8, 5.8, 1.8], [100.0, 55.0, 15.0]),
        "PMPC": ([10.5, 6.2, 2.1], [100.0, 62.0, 23.0]),
        "P1": ([11.5, 8.5, 4.0], [100.0, 76.0, 44.0]),
        "P2": ([10.8, 7.1, 3.0], [100.0, 66.0, 31.0]),
        "P3": ([9.5, 4.5, 1.1], [100.0, 42.0, 9.0]),
    }
    heat = [0.0, 20.0, 60.0]
    rows: list[dict[str, object]] = []
    for pid, (abs_vals, rea_vals) in polymers.items():
        for t, aa, rea in zip(heat, abs_vals, rea_vals):
            rows.append(
                {
                    "polymer_id": pid,
                    "heat_min": t,
                    "abs_activity": float(aa) * float(scale),
                    "REA_percent": float(rea),
                    "include_in_all_polymers": True,
                    "all_polymers_pair": False,
                }
            )
    return rows


class TestRepresentativeOutputs(unittest.TestCase):
    def test_run_representative_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "R-rep"
            fit_dir = root / "fit"
            fit_dir.mkdir(parents=True, exist_ok=True)
            summary_simple_path = fit_dir / "summary_simple.csv"
            pd.DataFrame(_base_summary_rows(scale=1.0)).to_csv(summary_simple_path, index=False)

            t50_csv = plot_per_polymer_timeseries(
                summary_simple_path=summary_simple_path,
                run_id=run_id,
                out_fit_dir=fit_dir,
                color_map_path=root / "polymer_colors.yml",
                reference_polymer_id="GOX",
                t50_definition="y0_half",
            )

            self.assertTrue(Path(t50_csv).is_file())
            self.assertTrue((fit_dir / f"representative_4__{run_id}.png").is_file())
            self.assertTrue((fit_dir / f"representative_objective_loglinear_main__{run_id}.png").is_file())

    def test_group_representative_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "processed"
            run_ids = ["260101-1", "260101-2"]
            for idx, rid in enumerate(run_ids):
                fit_dir = processed / rid / "fit"
                fit_dir.mkdir(parents=True, exist_ok=True)
                scale = 1.0 + 0.05 * idx
                pd.DataFrame(_base_summary_rows(scale=scale)).to_csv(
                    fit_dir / "summary_simple.csv",
                    index=False,
                )

            out_dir = plot_per_polymer_timeseries_across_runs_with_error_bars(
                run_id=run_ids[0],
                processed_dir=processed,
                out_fit_dir=root / "across_plots",
                color_map_path=root / "polymer_colors.yml",
                same_date_runs=run_ids,
                group_label="260101",
                reference_polymer_id="GOX",
                apply_outlier_filter=False,
                dpi=120,
                t50_definition="y0_half",
            )

            self.assertIsNotNone(out_dir)
            assert out_dir is not None
            self.assertTrue((out_dir / "representative_t50_with_error__260101.png").is_file())
            self.assertTrue(
                (out_dir / "representative_objective_loglinear_main_with_error__260101.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
