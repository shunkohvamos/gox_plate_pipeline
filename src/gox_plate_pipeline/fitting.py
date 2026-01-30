# src/gox_plate_pipeline/fitting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FitResult:
    slope: float
    intercept: float
    r2: float
    n: int
    t_start: float
    t_end: float


class FitSelectionError(RuntimeError):
    """Raised when no acceptable fitting window can be selected for a well."""


def add_well_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'col' (1-12) extracted from 'well' like 'A1'.
    Keeps original columns.
    """
    out = df.copy()
    col = out["well"].astype(str).str.extract(r"^[A-H](\d{1,2})$", expand=False)
    out["col"] = pd.to_numeric(col, errors="coerce").astype("Int64")
    return out


def add_heat_time(df: pd.DataFrame, heat_times: list[float]) -> pd.DataFrame:
    """
    Map 'col' -> heat_min using heat_times list.
    col=1 maps to heat_times[0], col=2 -> heat_times[1], ...
    If col is outside the range, heat_min will be NaN.
    """
    out = add_well_coordinates(df)

    def _map(c: object) -> float:
        if pd.isna(c):
            return np.nan
        ci = int(c)
        if 1 <= ci <= len(heat_times):
            return float(heat_times[ci - 1])
        return np.nan

    out["heat_min"] = out["col"].apply(_map)
    return out


def _fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Ordinary least squares linear fit: y = a*x + b
    Returns slope/intercept and R^2.
    """
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return FitResult(
        slope=float(a),
        intercept=float(b),
        r2=float(r2),
        n=int(len(x)),
        t_start=float(np.min(x)),
        t_end=float(np.max(x)),
    )


def generate_candidate_windows(
    time_s: np.ndarray,
    min_points: int = 6,
    max_points: int = 12,
) -> list[tuple[int, int]]:
    """
    Candidate windows defined by index ranges [i0:i1] (inclusive endpoints) in data order.
    Returns list of (start_idx, end_idx) inclusive.

    Notes:
      - Larger min_points reduces arbitrariness.
      - max_points limits window length (still scan multiple sizes).
    """
    n = len(time_s)
    if n < min_points:
        return []

    windows: list[tuple[int, int]] = []
    last_start = n - min_points
    for i0 in range(0, last_start + 1):
        i1_min = i0 + min_points - 1
        i1_max = min(n - 1, i0 + max_points - 1)
        for i1 in range(i1_min, i1_max + 1):
            windows.append((i0, i1))
    return windows


def fit_initial_rate_one_well(
    df_well: pd.DataFrame,
    min_points: int = 6,
    max_points: int = 12,
) -> pd.DataFrame:
    """
    Return candidate fits for one well.

    df_well must have columns:
      - time_s (seconds)
      - signal (numeric)
      - well (string) (optional but recommended for plotting)
    """
    d = df_well.sort_values("time_s").reset_index(drop=True)
    t = d["time_s"].to_numpy(dtype=float)
    y = d["signal"].to_numpy(dtype=float)

    wins = generate_candidate_windows(t, min_points=min_points, max_points=max_points)
    if not wins:
        return pd.DataFrame(
            columns=["t_start", "t_end", "n", "slope", "intercept", "r2", "start_idx", "end_idx"]
        )

    cand = []
    for i0, i1 in wins:
        xw = t[i0 : i1 + 1]
        yw = y[i0 : i1 + 1]
        fr = _fit_linear(xw, yw)
        cand.append(
            {
                "t_start": fr.t_start,
                "t_end": fr.t_end,
                "n": fr.n,
                "slope": fr.slope,
                "intercept": fr.intercept,
                "r2": fr.r2,
                "start_idx": i0,
                "end_idx": i1,
            }
        )
    return pd.DataFrame(cand)


def select_fit(
    cands: pd.DataFrame,
    method: str = "initial_positive",
    r2_min: float = 0.98,
    slope_min: float = 0.0,
    max_t_end: Optional[float] = 240.0,
) -> pd.Series:
    """
    Select one fit from candidates.

    Hard rules:
      - Exclude negative slopes (slope < slope_min). For initial rates, slope_min should be >= 0.
      - Optionally restrict to early region: t_end <= max_t_end (seconds). Set max_t_end=None to disable.

    Arbitrariness control:
      - Prefer larger n when fits are otherwise comparable.

    method:
      - initial_positive (recommended):
          1) apply hard filters
          2) among r2 >= r2_min: choose earliest t_end; tie: higher r2; tie: larger n; tie: earlier t_start
          3) if none satisfy r2_min: fallback to best r2; tie: larger n; tie: earlier t_end
      - best_r2:
          choose max r2 after hard filters; tie: larger n; tie: earlier t_end
    """
    if cands.empty:
        raise FitSelectionError("No candidate windows were generated.")

    c = cands.copy()

    # --- hard filters ---
    c = c[c["slope"] >= float(slope_min)].copy()
    if max_t_end is not None:
        c = c[c["t_end"] <= float(max_t_end)].copy()

    if c.empty:
        raise FitSelectionError(
            f"No candidates left after filtering (slope_min={slope_min}, max_t_end={max_t_end})."
        )

    if method == "initial_positive":
        ok = c[c["r2"] >= float(r2_min)].copy()
        if not ok.empty:
            ok = ok.sort_values(["t_end", "r2", "n", "t_start"], ascending=[True, False, False, True])
            return ok.iloc[0]

        # fallback: best r2, but prefer longer windows to reduce arbitrariness
        c = c.sort_values(["r2", "n", "t_end"], ascending=[False, False, True])
        return c.iloc[0]

    if method == "best_r2":
        c = c.sort_values(["r2", "n", "t_end"], ascending=[False, False, True])
        return c.iloc[0]

    raise ValueError(f"Unknown selection method: {method}")


def _enforce_final_safety(
    sel: pd.Series,
    slope_min: float = 0.0,
    max_t_end: Optional[float] = 240.0,
) -> None:
    """
    Final safety gate (must not silently pass):
      - slope must be >= slope_min
      - if max_t_end is set: t_end must be <= max_t_end
    """
    slope = float(sel["slope"])
    t_end = float(sel["t_end"])

    if slope < float(slope_min):
        raise FitSelectionError(
            f"Final safety triggered: selected slope {slope:.6g} < slope_min {slope_min:.6g}"
        )

    if max_t_end is not None and t_end > float(max_t_end):
        raise FitSelectionError(
            f"Final safety triggered: selected t_end {t_end:.6g} > max_t_end {max_t_end:.6g}"
        )


def plot_fit_for_well(
    df_well: pd.DataFrame,
    selected: pd.Series,
    out_png: Path,
) -> None:
    """
    Diagnostic plot for one well:
      - scatter of all points
      - fitted line over selected window
      - shaded selected time window
      - title includes slope and R2
    """
    d = df_well.sort_values("time_s").reset_index(drop=True)

    t = d["time_s"].to_numpy(dtype=float)
    y = d["signal"].to_numpy(dtype=float)

    t0 = float(selected["t_start"])
    t1 = float(selected["t_end"])
    slope = float(selected["slope"])
    intercept = float(selected["intercept"])
    r2 = float(selected["r2"])

    plt.figure()
    plt.scatter(t, y)

    # fitted line on selected window
    tt = np.array([t0, t1], dtype=float)
    yy = slope * tt + intercept
    plt.plot(tt, yy)

    plt.axvspan(t0, t1, alpha=0.2)

    well_name = str(d["well"].iloc[0]) if "well" in d.columns and len(d) > 0 else "NA"
    plt.title(f"Well {well_name} | slope={slope:.4g} | R2={r2:.4f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal (a.u.)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def compute_rates_and_rea(
    tidy: pd.DataFrame,
    heat_times: list[float],
    min_points: int = 6,
    max_points: int = 12,
    select_method: str = "initial_positive",
    r2_min: float = 0.98,
    slope_min: float = 0.0,
    max_t_end: Optional[float] = 240.0,
    plot_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-well initial rates (absolute activity) and REA (%).

    Absolute activity:
      - slope from selected linear window per well (no row-averaging)

    REA (%):
      - for each (plate_id, polymer_id), normalize each heat_min by baseline at heat_min==0
      - REA_percent = 100 * abs_activity / baseline_abs_activity

    Outputs:
      selected_rates:
        one row per well with slope, R2, selected window, identifiers, and status
      rea_table:
        selected_rates + baseline_abs_activity + REA_percent

    Notes:
      - Negative slopes are excluded at selection stage (slope_min >= 0).
      - Late windows are excluded via max_t_end (default 240s).
      - Final safety gate re-checks slope and t_end; failures are marked excluded.
    """
    df = add_heat_time(tidy, heat_times=heat_times)

    selected_rows: list[dict] = []

    group_cols = ["plate_id", "well"]
    if not all(c in df.columns for c in group_cols):
        raise ValueError(f"tidy must contain columns {group_cols}, got: {df.columns.tolist()}")

    for (plate_id, well), g in df.groupby(group_cols, sort=False):
        base = {
            "plate_id": plate_id,
            "well": well,
            "row": g["row"].iloc[0] if "row" in g.columns else "",
            "col": int(g["col"].iloc[0]) if "col" in g.columns and pd.notna(g["col"].iloc[0]) else np.nan,
            "heat_min": float(g["heat_min"].iloc[0]) if "heat_min" in g.columns else np.nan,
            "polymer_id": g["polymer_id"].iloc[0] if "polymer_id" in g.columns else "",
            "sample_name": g["sample_name"].iloc[0] if "sample_name" in g.columns else "",
            "source_file": g["source_file"].iloc[0] if "source_file" in g.columns else "",
        }

        try:
            cands = fit_initial_rate_one_well(g, min_points=min_points, max_points=max_points)

            sel = select_fit(
                cands,
                method=select_method,
                r2_min=r2_min,
                slope_min=slope_min,
                max_t_end=max_t_end,
            )

            # --- FINAL SAFETY (must never silently pass) ---
            _enforce_final_safety(sel, slope_min=slope_min, max_t_end=max_t_end)

            row = {
                **base,
                "status": "ok",
                "abs_activity": float(sel["slope"]),
                "slope": float(sel["slope"]),
                "intercept": float(sel["intercept"]),
                "r2": float(sel["r2"]),
                "n": int(sel["n"]),
                "t_start": float(sel["t_start"]),
                "t_end": float(sel["t_end"]),
                "select_method": select_method,
            }

            if plot_dir is not None:
                out_png = plot_dir / f"{plate_id}" / f"{well}.png"
                plot_fit_for_well(g, sel, out_png=out_png)

        except Exception as e:
            # Keep the well but mark as excluded; downstream can filter by status.
            row = {
                **base,
                "status": "excluded",
                "abs_activity": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "r2": np.nan,
                "n": np.nan,
                "t_start": np.nan,
                "t_end": np.nan,
                "select_method": select_method,
                "exclude_reason": str(e),
            }

        selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)

    # --- baseline (heat_min == 0) per (plate_id, polymer_id) ---
    baseline = (
        selected[(selected["status"] == "ok") & (selected["heat_min"] == 0)]
        .groupby(["plate_id", "polymer_id"], dropna=False)["abs_activity"]
        .median()
        .rename("baseline_abs_activity")
        .reset_index()
    )

    rea = selected.merge(baseline, on=["plate_id", "polymer_id"], how="left")

    # REA (%)
    rea["REA_percent"] = 100.0 * rea["abs_activity"] / rea["baseline_abs_activity"]

    return selected, rea
