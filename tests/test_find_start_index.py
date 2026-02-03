from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.fitting.core import _auto_mono_eps, _find_start_index  # noqa: E402


class FindStartIndexTests(unittest.TestCase):
    def test_decreasing_without_lag_starts_at_zero(self) -> None:
        # Representative decreasing trace with small fluctuations (B1-like).
        y = np.array(
            [
                544, 544, 545, 543, 537, 536, 535, 531, 529, 524, 525,
                523, 518, 515, 518, 516, 510, 512, 506, 508, 507,
            ],
            dtype=float,
        )
        t = np.arange(y.size, dtype=float) * 30.0 + 6.0
        eps = _auto_mono_eps(y)
        idx = _find_start_index(t, y, eps, max_shift=5, window=3, allow_down_steps=1)
        self.assertEqual(idx, 0)

    def test_increasing_with_true_lag_skips_early_points(self) -> None:
        # True lag phase: early flat/noisy points then sustained rise.
        y = np.array(
            [100, 101, 99, 100, 100, 120, 150, 180, 210, 240],
            dtype=float,
        )
        t = np.arange(y.size, dtype=float) * 30.0 + 6.0
        eps = _auto_mono_eps(y)
        idx = _find_start_index(t, y, eps, max_shift=5, window=3, allow_down_steps=1)
        self.assertGreaterEqual(idx, 1)

    def test_clean_increasing_trace_can_start_at_zero(self) -> None:
        y = np.array([100, 105, 110, 116, 121, 126, 133, 140], dtype=float)
        t = np.arange(y.size, dtype=float) * 30.0 + 6.0
        eps = _auto_mono_eps(y)
        idx = _find_start_index(t, y, eps, max_shift=5, window=3, allow_down_steps=1)
        self.assertEqual(idx, 0)


if __name__ == "__main__":
    unittest.main()
