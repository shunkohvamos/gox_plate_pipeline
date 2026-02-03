from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.fitting.pipeline import compute_rates_and_rea  # noqa: E402


class FullRangeOutlierSkipTests(unittest.TestCase):
    def test_single_outlier_keeps_full_range_for_two_c4_c5_like_traces(self) -> None:
        t = np.arange(6, 607, 30, dtype=float)  # 21 points

        y_c4_like = 100.0 + 2.5 * np.arange(t.size, dtype=float)
        y_c4_like[11] += 120.0  # one strong outlier

        y_c5_like = 120.0 + 1.6 * np.arange(t.size, dtype=float)
        y_c5_like[7] += 100.0  # one strong outlier

        rows: list[dict] = []
        for well, y in [("A4", y_c4_like), ("A5", y_c5_like)]:
            for ti, yi in zip(t, y):
                rows.append(
                    {
                        "plate_id": "plate1",
                        "well": well,
                        "time_s": float(ti),
                        "signal": float(yi),
                        "polymer_id": "P1",
                    }
                )

        tidy = pd.DataFrame(rows)
        selected, _ = compute_rates_and_rea(
            tidy=tidy,
            heat_times=[0, 5, 10, 15, 20, 40, 60],
        )

        for well, expected_skip in [("A4", "11"), ("A5", "7")]:
            row = selected[selected["well"] == well].iloc[0]
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["select_method_used"], "full_range_outlier_skip")
            self.assertEqual(float(row["t_start"]), 6.0)
            self.assertEqual(float(row["t_end"]), 606.0)
            self.assertEqual(int(row["n"]), 20)
            self.assertEqual(str(row["skip_indices"]), expected_skip)
            self.assertEqual(int(row["skip_count"]), 1)


if __name__ == "__main__":
    unittest.main()
