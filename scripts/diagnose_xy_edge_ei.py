#!/usr/bin/env python3
"""
Diagnose EI / mean / std behavior at the (x,y) edge (small y = MPC≈1).

When y is small, x = BMA/(BMA+MTAC) has little physical meaning (both BMA and MTAC
are tiny). If the GP still varies strongly in x there, the EI terrain can look
jagged or show spurious structure near that edge.

This script:
- Loads BO training data (or uses a minimal fixture)
- Fits GP on (x, y)
- Evaluates EI and mean along slices at fixed y (y=0.05, 0.2, 0.5)
- Reports variation in EI and mean along x (e.g. std) to confirm edge behavior.

Usage:
  uv run python scripts/diagnose_xy_edge_ei.py [path/to/bo_training_data__*.csv]
  If no path given, uses a minimal synthetic dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gox_plate_pipeline.bo_engine import GPModel2D, _ei  # noqa: E402


def main() -> None:
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1])
        if not path.is_file():
            print(f"File not found: {path}")
            sys.exit(1)
        df = pd.read_csv(path)
        # use first occurrence per polymer for unique design
        learning = (
            df.groupby("polymer_id", as_index=False)
            .first()
            .dropna(subset=["x", "y", "log_fog_corrected"])
        )
        if "log_fog_corrected" not in learning.columns:
            learning = learning.rename(columns={"log_fog": "log_fog_corrected"})
        X = learning[["x", "y"]].to_numpy(dtype=float)
        y = learning["log_fog_corrected"].to_numpy(dtype=float)
        # drop rows with NaN in x (y=0)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] < 3:
            print("Not enough rows after dropping NaN; need at least 3.")
            sys.exit(1)
        obs_noise_rel = np.ones(len(X), dtype=float)
        if "obs_noise_rel" in df.columns:
            ohr = df.groupby("polymer_id")["obs_noise_rel"].first().to_numpy(dtype=float)
            ohr = ohr[valid[: len(ohr)]]
            if len(ohr) == len(X):
                obs_noise_rel = ohr
        print(f"Loaded {X.shape[0]} design points from {path}")
    else:
        # Minimal synthetic data
        np.random.seed(42)
        rows = [
            (0.2, 0.2, 0.1),
            (0.8, 0.2, 0.2),
            (0.2, 0.7, 0.5),
            (0.8, 0.7, 0.4),
            (0.5, 0.45, 0.3),
        ]
        X = np.array([[r[0], r[1]] for r in rows], dtype=float)
        y = np.array([r[2] for r in rows], dtype=float)
        obs_noise_rel = np.ones(len(X), dtype=float)
        print("Using minimal synthetic (x,y) data (5 points)")

    gp = GPModel2D.fit(X, y, random_state=42, obs_noise_rel=obs_noise_rel)
    best = float(np.nanmax(y))

    print()
    print("GP length_scale_x =", gp.length_scale[0])
    print("GP length_scale_y =", gp.length_scale[1])
    print()

    # Slices: fixed y, x in [0.02, 0.98]
    x_slice = np.linspace(0.02, 0.98, 101)
    for y_fix in [0.05, 0.2, 0.5]:
        grid = np.column_stack([x_slice, np.full_like(x_slice, y_fix)])
        mu, std = gp.predict(grid)
        ei = _ei(mu, std, best, 0.01)
        mean_std = np.nanstd(mu)
        ei_std = np.nanstd(ei)
        print(f"Slice y = {y_fix} (MPC = {1 - y_fix:.2f}):")
        print(f"  std(pred_mean) along x: {mean_std:.6f}")
        print(f"  std(EI) along x:         {ei_std:.6f}")
        if y_fix <= 0.1 and (mean_std > 0.05 or ei_std > 0.01):
            print("  -> Notable variation in x at small y (edge); may cause jagged EI.")
        print()

    print("If at small y (e.g. 0.05) std(EI) or std(mean) along x is large,")
    print("the (x,y) coordinate system is giving spurious structure near MPC≈1.")


if __name__ == "__main__":
    main()
