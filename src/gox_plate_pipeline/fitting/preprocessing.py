# src/gox_plate_pipeline/fitting/preprocessing.py
"""
Data preprocessing functions for fitting pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_well_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'row' (A-H) and 'col' (1-12) extracted from 'well' like 'A1'.
    Keeps original columns.
    """
    out = df.copy()
    row = out["well"].astype(str).str.extract(r"^([A-H])", expand=False)
    col = out["well"].astype(str).str.extract(r"^[A-H](\d{1,2})$", expand=False)

    if "row" not in out.columns:
        out["row"] = row
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
