# src/gox_plate_pipeline/bo_data.py
"""
BO reference data: join BO catalog (BMA ternary composition) with FoG summary.

- BO catalog: polymer_id, frac_MPC, frac_BMA, frac_MTAC (order [MPC, BMA, MTAC], sum=1).
  Optional: x, y (2D projection), round_id. x,y follow:
  x = [BMA] / ([BMA] + [MTAC]),  y = ([BMA] + [MTAC]) / ([MPC] + [BMA] + [MTAC]).
- BO learning CSV: X = frac_MPC, frac_BMA, frac_MTAC; y = log_fog; lineage kept.
- Only polymers in the BO catalog are included (BMA-only; no EHMA/LMA/MPTSSi etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import yaml


# Column order for BO design variables (fixed). Ternary order: [MPC, BMA, MTAC].
BO_X_COLS = ["frac_MPC", "frac_BMA", "frac_MTAC"]
BO_Y_COL = "log_fog"
COMPOSITION_SUM_TOL = 1e-6
XY_TOL = 1e-6  # tolerance for x,y vs recomputed from frac
BO_XY_COLS = ["x", "y"]  # optional in catalog; x,y definition below


def xy_from_frac(
    frac_MPC: Union[float, np.ndarray],
    frac_BMA: Union[float, np.ndarray],
    frac_MTAC: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute x, y from ternary fractions (order [MPC, BMA, MTAC]).

    x = frac_BMA / (frac_BMA + frac_MTAC)
    y = frac_BMA + frac_MTAC
    When (frac_BMA + frac_MTAC) == 0, x is NaN and y is 0.
    """
    frac_BMA = np.asarray(frac_BMA, dtype=float)
    frac_MTAC = np.asarray(frac_MTAC, dtype=float)
    y = frac_BMA + frac_MTAC
    denom = frac_BMA + frac_MTAC
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(denom > 0, frac_BMA / denom, np.nan).astype(float)
    return x, y


def frac_from_xy(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Inverse: compute frac_MPC, frac_BMA, frac_MTAC from x, y.

    frac_MPC = 1 - y,  frac_BMA = x*y,  frac_MTAC = (1-x)*y.
    When y == 0: frac_MPC=1, frac_BMA=0, frac_MTAC=0; x is ignored.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    frac_MPC = np.where(y == 0, 1.0, 1.0 - y)
    frac_BMA = np.where(y == 0, 0.0, x * y)
    frac_MTAC = np.where(y == 0, 0.0, (1.0 - x) * y)
    return frac_MPC, frac_BMA, frac_MTAC


def load_bo_catalog(
    path: Path,
    *,
    validate_sum: bool = True,
    validate_xy_consistency: bool = True,
) -> pd.DataFrame:
    """
    Load BO catalog (BMA-only): polymer_id, frac_MPC, frac_BMA, frac_MTAC; optional x, y.

    Order of composition columns is [MPC, BMA, MTAC]. If validate_sum, each frac in [0,1] and sum=1.
    round_id is not in the catalog; it is assigned per run via run_round_map when building BO learning data.
    Optional x, y: if blank, computed from frac; if provided, recomputed and checked (mismatch → error when validate_xy_consistency).
    x = frac_BMA / (frac_BMA + frac_MTAC),  y = frac_BMA + frac_MTAC; when y==0, x is NaN.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    for c in ["polymer_id"] + BO_X_COLS:
        if c not in df.columns:
            raise ValueError(f"BO catalog must have {c}, got: {list(df.columns)}")

    df = df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    for c in BO_X_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # frac in [0,1]
    for c in BO_X_COLS:
        bad = (df[c] < -COMPOSITION_SUM_TOL) | (df[c] > 1.0 + COMPOSITION_SUM_TOL)
        if bad.any():
            raise ValueError(
                f"BO catalog: {c} must be in [0,1]. Offending: {df.loc[bad, ['polymer_id', c]].to_dict('records')}"
            )

    if validate_sum:
        s = df[BO_X_COLS].sum(axis=1)
        bad = np.abs(s - 1.0) > COMPOSITION_SUM_TOL
        if bad.any():
            raise ValueError(
                f"BO catalog: composition sum must be 1. Offending rows: {df.loc[bad, ['polymer_id'] + BO_X_COLS].to_dict('records')}"
            )

    # optional x, y, round_id
    has_x = "x" in df.columns
    has_y = "y" in df.columns
    has_round_id = "round_id" in df.columns
    if has_x:
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
    if has_y:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if has_round_id:
        df["round_id"] = df["round_id"].astype(str).str.strip()

    # compute x,y from frac where blank
    x_computed, y_computed = xy_from_frac(
        df["frac_MPC"].values,
        df["frac_BMA"].values,
        df["frac_MTAC"].values,
    )
    if not has_x:
        df["x"] = x_computed
    else:
        x_blank = pd.isna(df["x"])
        df.loc[x_blank, "x"] = np.asarray(x_computed).flat[x_blank.values]
    if not has_y:
        df["y"] = y_computed
    else:
        y_blank = pd.isna(df["y"])
        df.loc[y_blank, "y"] = np.asarray(y_computed).flat[y_blank.values]

    # consistency check: loaded x,y must match recomputed from frac
    # If inconsistent, warn and auto-correct (overwrite with computed values) instead of raising error.
    if validate_xy_consistency:
        xc = np.ravel(np.asarray(x_computed))
        yc = np.ravel(np.asarray(y_computed))
        y_ok = np.abs(df["y"].values - yc) <= XY_TOL
        # when y==0, x is NaN; require df["x"] NaN. when y>0, require |df["x"] - xc| <= tol
        y_zero = yc <= COMPOSITION_SUM_TOL
        x_ok = np.where(
            y_zero,
            np.isnan(df["x"].values),
            np.isfinite(xc) & (np.abs(df["x"].values - xc) <= XY_TOL),
        )
        if not (x_ok.all() and y_ok.all()):
            bad_idx = ~(x_ok & y_ok)
            bad_rows = df.loc[bad_idx, ['polymer_id'] + BO_X_COLS + ['x', 'y']].copy()
            bad_rows['x_computed'] = xc[bad_idx]
            bad_rows['y_computed'] = yc[bad_idx]
            import warnings
            warnings.warn(
                f"BO catalog: x,y inconsistent with frac for {bad_idx.sum()} row(s). "
                f"Auto-correcting to computed values from frac. "
                f"Offending rows: {bad_rows.to_dict('records')}",
                UserWarning,
                stacklevel=2,
            )
            # Auto-correct: overwrite x, y with computed values
            df.loc[bad_idx, "x"] = xc[bad_idx]
            df.loc[bad_idx, "y"] = yc[bad_idx]

    out_cols = ["polymer_id"] + (["round_id"] if has_round_id else []) + BO_X_COLS + ["x", "y"]
    subset = ["polymer_id", "round_id"] if has_round_id else ["polymer_id"]
    return df[[c for c in out_cols if c in df.columns]].drop_duplicates(subset=subset).reset_index(drop=True)


# Sentinel for "not used for BO" in run-round TSV; when loading, these are skipped (run not in map).
ROUND_NOT_USED_VALUES = ("", "—", "NA", "nan", "NaN", "（BOに使用しない）", "（未使用）")


def _is_round_used(round_id: str) -> bool:
    """True if round_id is a valid round (not empty / not-used sentinel)."""
    s = str(round_id).strip()
    if not s:
        return False
    return s.lower() not in (v.lower() for v in ROUND_NOT_USED_VALUES if v)


def load_run_round_map(path: Path) -> Dict[str, str]:
    """
    Load run_id → round_id mapping for BO.

    - YAML: run_round_map: { "251118": "R1", "260201": "R2" } or list of { run_id, round_id }.
    - CSV/TSV: run_id, round_id (header required). Rows with empty or "—"/NA round_id are skipped
      (those runs are not used for BO).
    Returns dict run_id -> round_id (both str). Only runs with a valid round_id are included.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Run-round map not found: {path}")

    out: Dict[str, str] = {}
    suf = path.suffix.lower()
    if suf in (".yml", ".yaml"):
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if obj is None:
            return out
        if isinstance(obj, dict):
            if "run_round_map" in obj:
                obj = obj["run_round_map"]
            for k, v in obj.items():
                vstr = str(v).strip()
                if _is_round_used(vstr):
                    out[str(k).strip()] = vstr
        elif isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    rid = row.get("run_id")
                    oid = row.get("round_id")
                    if rid is not None and oid is not None and _is_round_used(str(oid)):
                        out[str(rid).strip()] = str(oid).strip()
        return out
    if suf in (".csv", ".tsv"):
        sep = "\t" if suf == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if "run_id" not in df.columns or "round_id" not in df.columns:
            raise ValueError(f"File must have run_id and round_id, got: {list(df.columns)}")
        for _, r in df.iterrows():
            rid = str(r["run_id"]).strip()
            oid = str(r["round_id"]).strip() if pd.notna(r["round_id"]) else ""
            if _is_round_used(oid):
                out[rid] = oid
        return out
    raise ValueError(f"Unsupported run-round map format: {path.suffix}. Use .yml, .yaml, .csv, or .tsv.")


def collect_fog_summary_paths(
    processed_dir: Path,
    run_ids: Optional[List[str]] = None,
) -> List[Path]:
    """
    Collect fog_summary__{run_id}.csv under processed_dir.

    Expects layout: processed_dir/{run_id}/fit/fog_summary__{run_id}.csv.
    """
    processed_dir = Path(processed_dir)
    if not processed_dir.is_dir():
        return []

    paths: List[Path] = []
    for run_dir in sorted(processed_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rid = run_dir.name
        if run_ids is not None and rid not in run_ids:
            continue
        fit_dir = run_dir / "fit"
        p = fit_dir / f"fog_summary__{rid}.csv"
        if p.is_file():
            paths.append(p)

    return sorted(paths)


def build_bo_learning_data(
    catalog_df: pd.DataFrame,
    fog_summary_paths: List[Path],
    *,
    lineage_columns: Optional[List[str]] = None,
    run_round_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join BO catalog with FoG summaries to produce BO learning CSV and exclusion report.

    - Only polymer_ids present in catalog_df are included (BMA-only).
    - When run_round_map is provided: each run_id is assigned a round_id from the map.
      Catalog is matched by polymer_id only (catalog does not contain round_id; round_id comes from run_round_map).
      Learning rows get round_id from the run (via run_round_map).
    - Rows with missing log_fog (or fog) are excluded; reason is recorded in exclusion report.
    - X columns order: frac_MPC, frac_BMA, frac_MTAC. y column: log_fog.
    - Lineage columns from fog summary are kept (run_id, input_tidy, etc.).

    Returns:
        learning_df: rows with valid log_fog, columns [polymer_id] + [round_id?] + BO_X_COLS + [BO_Y_COL] + lineage.
        excluded_df: excluded rows with columns polymer_id, run_id, reason, (round_id if map given).
    """
    catalog_df = catalog_df.copy()
    catalog_ids = set(catalog_df["polymer_id"].astype(str).str.strip())
    # round_id in learning data comes only from run_round_map (per-run), not from catalog.

    if lineage_columns is None:
        lineage_columns = ["run_id", "input_t50_file", "input_tidy"]

    learning_rows: list = []
    excluded_rows: list = []

    for path in fog_summary_paths:
        path = Path(path)
        if not path.is_file():
            continue
        fog = pd.read_csv(path)
        fog["polymer_id"] = fog["polymer_id"].astype(str).str.strip()

        run_id = fog["run_id"].iloc[0] if "run_id" in fog.columns and len(fog) else path.stem.replace("fog_summary__", "")
        round_id: Optional[str] = None
        if run_round_map is not None:
            round_id = run_round_map.get(str(run_id).strip())
            if round_id is None:
                continue  # run not in map; skip this run

        for _, row in fog.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            # Only catalog polymers (BMA-only)
            if pid not in catalog_ids:
                excluded_rows.append({
                    "polymer_id": pid,
                    "run_id": run_id,
                    **({"round_id": round_id} if round_id is not None else {}),
                    "reason": "not_in_bo_catalog",
                })
                continue

            # Match catalog by polymer_id only (round_id is from run_round_map per run, not from catalog)
            cat_match = catalog_df[catalog_df["polymer_id"] == pid]
            if cat_match.empty:
                continue  # already excluded by catalog_ids check above
            cat_row = cat_match.iloc[0]

            log_fog = row.get(BO_Y_COL)
            fog_val = row.get("fog")
            if pd.isna(log_fog) or (fog_val is not None and (pd.isna(fog_val) or fog_val <= 0)):
                r = row.get("fog_missing_reason")
                reason = (str(r).strip() if pd.notna(r) and str(r).strip() else "log_fog_missing_or_invalid")
                excluded_rows.append({
                    "polymer_id": pid,
                    "run_id": run_id,
                    **({"round_id": round_id} if round_id is not None else {}),
                    "reason": reason,
                })
                continue

            new_row = {
                "polymer_id": pid,
                "frac_MPC": float(cat_row["frac_MPC"]),
                "frac_BMA": float(cat_row["frac_BMA"]),
                "frac_MTAC": float(cat_row["frac_MTAC"]),
                BO_Y_COL: float(log_fog),
            }
            if round_id is not None:
                new_row["round_id"] = round_id
            for c in ["x", "y"]:
                if c in cat_row.index and pd.notna(cat_row.get(c)):
                    new_row[c] = cat_row[c]
            for c in lineage_columns:
                if c in fog.columns:
                    new_row[c] = row.get(c, "")
            learning_rows.append(new_row)

    learning_df = pd.DataFrame(learning_rows)
    excluded_df = pd.DataFrame(excluded_rows)

    if not learning_df.empty:
        # Enforce column order: polymer_id, round_id?, X cols, y, lineage
        cols = ["polymer_id"] + (["round_id"] if "round_id" in learning_df.columns else []) + BO_X_COLS + [BO_Y_COL] + [c for c in lineage_columns if c in learning_df.columns]
        learning_df = learning_df[[c for c in cols if c in learning_df.columns]]

    return learning_df, excluded_df


def build_bo_learning_data_from_round_averaged(
    catalog_df: pd.DataFrame,
    fog_round_averaged_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build BO learning CSV from round-averaged FoG (one row per round_id, polymer_id).

    - Reads fog_round_averaged CSV (round_id, polymer_id, mean_fog, mean_log_fog, n_observations, run_ids).
    - Only polymer_ids in catalog_df are included. y column = mean_log_fog.
    - Rows with missing or invalid mean_log_fog are excluded.
    """
    catalog_df = catalog_df.copy()
    catalog_ids = set(catalog_df["polymer_id"].astype(str).str.strip())

    path = Path(fog_round_averaged_path)
    if not path.is_file():
        raise FileNotFoundError(f"Round-averaged FoG not found: {path}")
    df = pd.read_csv(path)
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()

    for c in ["round_id", "polymer_id", "mean_fog", "mean_log_fog"]:
        if c not in df.columns:
            raise ValueError(f"Round-averaged FoG must have {c}, got: {list(df.columns)}")

    learning_rows: list = []
    excluded_rows: list = []

    for _, row in df.iterrows():
        pid = str(row["polymer_id"]).strip()
        if pid not in catalog_ids:
            excluded_rows.append({
                "round_id": row["round_id"],
                "polymer_id": pid,
                "reason": "not_in_bo_catalog",
            })
            continue

        mean_fog = row.get("mean_fog")
        mean_log_fog = row.get("mean_log_fog")
        if pd.isna(mean_log_fog) or (pd.notna(mean_fog) and (pd.isna(mean_fog) or float(mean_fog) <= 0)):
            excluded_rows.append({
                "round_id": row["round_id"],
                "polymer_id": pid,
                "reason": "mean_log_fog_missing_or_invalid",
            })
            continue

        cat_row = catalog_df[catalog_df["polymer_id"] == pid].iloc[0]
        new_row = {
            "polymer_id": pid,
            "round_id": row["round_id"],
            "frac_MPC": float(cat_row["frac_MPC"]),
            "frac_BMA": float(cat_row["frac_BMA"]),
            "frac_MTAC": float(cat_row["frac_MTAC"]),
            BO_Y_COL: float(mean_log_fog),
        }
        for c in ["x", "y"]:
            if c in cat_row.index and pd.notna(cat_row.get(c)):
                new_row[c] = cat_row[c]
        if "run_ids" in row:
            new_row["run_ids"] = row["run_ids"]
        if "n_observations" in row:
            new_row["n_observations"] = row["n_observations"]
        learning_rows.append(new_row)

    learning_df = pd.DataFrame(learning_rows)
    excluded_df = pd.DataFrame(excluded_rows)

    if not learning_df.empty:
        cols = ["polymer_id", "round_id"] + BO_X_COLS + [BO_Y_COL] + [c for c in ["run_ids", "n_observations"] if c in learning_df.columns]
        learning_df = learning_df[[c for c in cols if c in learning_df.columns]]

    return learning_df, excluded_df


def write_bo_learning_csv(learning_df: pd.DataFrame, out_path: Path) -> None:
    """Write BO learning CSV (X order: frac_MPC, frac_BMA, frac_MTAC; y: log_fog)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    learning_df.to_csv(out_path, index=False)


def write_exclusion_report(excluded_df: pd.DataFrame, out_path: Path) -> None:
    """Write exclusion report CSV (polymer_id, run_id, reason)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_df.to_csv(out_path, index=False)
