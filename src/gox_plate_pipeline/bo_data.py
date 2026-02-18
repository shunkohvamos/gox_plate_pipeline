# src/gox_plate_pipeline/bo_data.py
"""
BO reference data: join BO catalog (BMA ternary composition) with FoG summary.

- BO catalog: polymer_id, frac_MPC, frac_BMA, frac_MTAC (order [MPC, BMA, MTAC], sum=1).
  Optional: x, y (2D projection), round_id. x,y follow:
  x = [BMA] / ([BMA] + [MTAC]),  y = ([BMA] + [MTAC]) / ([MPC] + [BMA] + [MTAC]).
- BO learning CSV: X = frac_MPC, frac_BMA, frac_MTAC; baseline y = log_fog,
  and optional objective columns (e.g. log_fog_activity_bonus_penalty) are preserved.
- Only polymers in the BO catalog are included (BMA-only; no EHMA/LMA/MPTSSi etc.).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import yaml


# Column order for BO design variables (fixed). Ternary order: [MPC, BMA, MTAC].
BO_X_COLS = ["frac_MPC", "frac_BMA", "frac_MTAC"]
BO_Y_COL = "log_fog"
ABS_OBJECTIVE_COL = "fog_activity_bonus_penalty"
ABS_OBJECTIVE_LOG_COL = "log_fog_activity_bonus_penalty"
ABS_OBJECTIVE_OLD_COL = "fog_abs_bonus_penalty"
ABS_OBJECTIVE_LOG_OLD_COL = "log_fog_abs_bonus_penalty"
ABS_OBJECTIVE_LEGACY_COL = "fog_abs_modulated"
ABS_OBJECTIVE_LOG_LEGACY_COL = "log_fog_abs_modulated"
OBJECTIVE_LOGLINEAR_MAIN_COL = "objective_loglinear_main"
OBJECTIVE_LOGLINEAR_MAIN_EXP_COL = "objective_loglinear_main_exp"
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


def build_round_coverage_summary(
    observed_round_ids: Iterable[object],
    run_round_map: Dict[str, str],
) -> Dict[str, object]:
    """
    Build round-coverage summary for auditability.

    Returns:
      {
        "n_map_entries": int,
        "n_map_round_ids": int,
        "n_observed_round_ids": int,
        "map_round_ids": [...],
        "observed_round_ids": [...],
        "round_ids_missing_in_map": [...],
        "round_ids_in_map_but_unused": [...],
      }
    """
    map_round_ids = sorted(
        {
            str(v).strip()
            for v in run_round_map.values()
            if _is_round_used(str(v))
        }
    )
    observed = sorted({str(x).strip() for x in observed_round_ids if str(x).strip()})
    observed_set = set(observed)
    map_set = set(map_round_ids)
    missing_in_map = sorted(observed_set - map_set)
    unused_in_data = sorted(map_set - observed_set)
    return {
        "n_map_entries": int(len(run_round_map)),
        "n_map_round_ids": int(len(map_round_ids)),
        "n_observed_round_ids": int(len(observed)),
        "map_round_ids": map_round_ids,
        "observed_round_ids": observed,
        "round_ids_missing_in_map": missing_in_map,
        "round_ids_in_map_but_unused": unused_in_data,
    }


def file_fingerprint(path: Path) -> Dict[str, object]:
    """Compute file fingerprint metadata (path/hash/mtime/size) for provenance sidecars."""
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "error": "file not found"}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "sha256": h.hexdigest(),
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": int(st.st_size),
    }


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
    - Rows with use_for_bo=False (from metadata TSV) are excluded; reason is "excluded_by_use_for_bo_flag".
    - When run_round_map is provided: each run_id is assigned a round_id from the map.
      Catalog is matched by polymer_id only (catalog does not contain round_id; round_id comes from run_round_map).
      Learning rows get round_id from the run (via run_round_map).
    - Rows with missing log_fog (or fog) are excluded; reason is recorded in exclusion report.
    - X columns order: frac_MPC, frac_BMA, frac_MTAC. y column: log_fog.
      Additional objective candidates (when present) are passed through:
      log_fog_native_constrained, fog_native_constrained,
      log_fog_native_soft, fog_native_soft.
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
            
            # Exclude if use_for_bo is False (explicitly marked as not for BO)
            use_for_bo = row.get("use_for_bo", True)
            if pd.notna(use_for_bo) and not bool(use_for_bo):
                excluded_rows.append({
                    "polymer_id": pid,
                    "run_id": run_id,
                    **({"round_id": round_id} if round_id is not None else {}),
                    "reason": "excluded_by_use_for_bo_flag",
                })
                continue
            
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
            # Keep optional constrained-objective / native-activity fields for BO objective switching.
            for metric_col in [
                ABS_OBJECTIVE_COL,
                ABS_OBJECTIVE_LOG_COL,
                ABS_OBJECTIVE_OLD_COL,
                ABS_OBJECTIVE_LOG_OLD_COL,
                ABS_OBJECTIVE_LEGACY_COL,
                ABS_OBJECTIVE_LOG_LEGACY_COL,
                "fog_solvent_balanced",
                "log_fog_solvent_balanced",
                "fog_vs_solvent_control",
                "abs0_vs_solvent_control",
                OBJECTIVE_LOGLINEAR_MAIN_COL,
                OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
                "abs_activity_down_penalty",
                "abs_activity_up_bonus",
                "abs_activity_balance_factor",
                "solvent_activity_down_penalty",
                "solvent_activity_up_bonus",
                "solvent_activity_balance_factor",
                "solvent_group",
                "solvent_control_polymer_id",
                "fog_native_constrained",
                "log_fog_native_constrained",
                "fog_native_soft",
                "log_fog_native_soft",
                "native_activity_soft_penalty",
                "native_activity_rel_at_0",
                "native_activity_min_rel_threshold",
                "native_activity_feasible",
                "fog_constraint_reason",
            ]:
                if metric_col in fog.columns:
                    val = row.get(metric_col)
                    if metric_col in {"fog_constraint_reason", "solvent_group", "solvent_control_polymer_id"}:
                        new_row[metric_col] = "" if pd.isna(val) else str(val)
                    elif metric_col == "native_activity_feasible":
                        vv = pd.to_numeric(val, errors="coerce")
                        new_row[metric_col] = int(vv) if np.isfinite(vv) else 0
                    else:
                        vv = pd.to_numeric(val, errors="coerce")
                        new_row[metric_col] = float(vv) if np.isfinite(vv) else np.nan
            solvent_present = np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_COL, np.nan), errors="coerce")
            ) or np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_OLD_COL, np.nan), errors="coerce")
            ) or np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LEGACY_COL, np.nan), errors="coerce")
            ) or np.isfinite(
                pd.to_numeric(new_row.get("fog_solvent_balanced", np.nan), errors="coerce")
            )
            if (not solvent_present) and ("fog_vs_solvent_control" in row):
                fog_rel_v = pd.to_numeric(row.get("fog_vs_solvent_control"), errors="coerce")
                abs_rel_v = pd.to_numeric(row.get("abs0_vs_solvent_control"), errors="coerce")
                if np.isfinite(fog_rel_v) and fog_rel_v > 0.0 and np.isfinite(abs_rel_v):
                    clipped = float(np.clip(abs_rel_v, 0.0, 1.0) ** 2.0)
                    up_bonus = float(1.0 + 0.35 * np.clip(abs_rel_v - 1.05, 0.0, 0.30))
                    balance = clipped * up_bonus
                    solvent_score = float(fog_rel_v) * balance
                    if solvent_score > 0.0:
                        new_row["abs_activity_down_penalty"] = clipped
                        new_row["abs_activity_up_bonus"] = up_bonus
                        new_row["abs_activity_balance_factor"] = balance
                        new_row["solvent_activity_down_penalty"] = clipped
                        new_row["solvent_activity_up_bonus"] = up_bonus
                        new_row["solvent_activity_balance_factor"] = balance
                        new_row[ABS_OBJECTIVE_COL] = solvent_score
                        new_row[ABS_OBJECTIVE_LOG_COL] = float(np.log(solvent_score))
                        new_row["fog_solvent_balanced"] = solvent_score
                        new_row["log_fog_solvent_balanced"] = float(np.log(solvent_score))
            if (ABS_OBJECTIVE_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_OLD_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_COL] = float(
                    pd.to_numeric(new_row[ABS_OBJECTIVE_OLD_COL], errors="coerce")
                )
            if (ABS_OBJECTIVE_LOG_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LOG_OLD_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_LOG_COL] = float(
                    pd.to_numeric(new_row[ABS_OBJECTIVE_LOG_OLD_COL], errors="coerce")
                )
            if (ABS_OBJECTIVE_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LEGACY_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_COL] = float(
                    pd.to_numeric(new_row[ABS_OBJECTIVE_LEGACY_COL], errors="coerce")
                )
            if (ABS_OBJECTIVE_LOG_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LOG_LEGACY_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_LOG_COL] = float(
                    pd.to_numeric(new_row[ABS_OBJECTIVE_LOG_LEGACY_COL], errors="coerce")
                )
            if (ABS_OBJECTIVE_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get("fog_solvent_balanced", np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_COL] = float(pd.to_numeric(new_row["fog_solvent_balanced"], errors="coerce"))
            if (ABS_OBJECTIVE_LOG_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get("log_fog_solvent_balanced", np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_LOG_COL] = float(pd.to_numeric(new_row["log_fog_solvent_balanced"], errors="coerce"))
            if (ABS_OBJECTIVE_LEGACY_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_LEGACY_COL] = float(pd.to_numeric(new_row[ABS_OBJECTIVE_COL], errors="coerce"))
            if (ABS_OBJECTIVE_LOG_LEGACY_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LOG_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_LOG_LEGACY_COL] = float(
                    pd.to_numeric(new_row[ABS_OBJECTIVE_LOG_COL], errors="coerce")
                )
            if (ABS_OBJECTIVE_OLD_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_OLD_COL] = float(pd.to_numeric(new_row[ABS_OBJECTIVE_COL], errors="coerce"))
            if (ABS_OBJECTIVE_LOG_OLD_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LOG_COL, np.nan), errors="coerce")
            ):
                new_row[ABS_OBJECTIVE_LOG_OLD_COL] = float(
                    pd.to_numeric(new_row[ABS_OBJECTIVE_LOG_COL], errors="coerce")
                )
            if ("fog_solvent_balanced" not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_COL, np.nan), errors="coerce")
            ):
                new_row["fog_solvent_balanced"] = float(pd.to_numeric(new_row[ABS_OBJECTIVE_COL], errors="coerce"))
            if ("log_fog_solvent_balanced" not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LOG_COL, np.nan), errors="coerce")
            ):
                new_row["log_fog_solvent_balanced"] = float(pd.to_numeric(new_row[ABS_OBJECTIVE_LOG_COL], errors="coerce"))
            if (OBJECTIVE_LOGLINEAR_MAIN_COL not in new_row) and ("fog_vs_solvent_control" in row):
                fog_rel_v = pd.to_numeric(row.get("fog_vs_solvent_control"), errors="coerce")
                abs_rel_v = pd.to_numeric(row.get("abs0_vs_solvent_control"), errors="coerce")
                if np.isfinite(fog_rel_v) and fog_rel_v > 0.0 and np.isfinite(abs_rel_v) and abs_rel_v > 0.0:
                    score_loglinear = float(np.log(fog_rel_v) + np.log(abs_rel_v))
                    new_row[OBJECTIVE_LOGLINEAR_MAIN_COL] = score_loglinear
                    new_row[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = float(np.exp(score_loglinear))
            if (OBJECTIVE_LOGLINEAR_MAIN_EXP_COL not in new_row) and np.isfinite(
                pd.to_numeric(new_row.get(OBJECTIVE_LOGLINEAR_MAIN_COL, np.nan), errors="coerce")
            ):
                new_row[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = float(
                    np.exp(float(pd.to_numeric(new_row[OBJECTIVE_LOGLINEAR_MAIN_COL], errors="coerce")))
                )
            solvent_present = np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_COL, np.nan), errors="coerce")
            ) or np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_OLD_COL, np.nan), errors="coerce")
            ) or np.isfinite(
                pd.to_numeric(new_row.get(ABS_OBJECTIVE_LEGACY_COL, np.nan), errors="coerce")
            ) or np.isfinite(
                pd.to_numeric(new_row.get("fog_solvent_balanced", np.nan), errors="coerce")
            )
            soft_present = np.isfinite(
                pd.to_numeric(new_row.get("fog_native_soft"), errors="coerce")
            )
            if (not soft_present) and ("fog" in row):
                fog_v = pd.to_numeric(row.get("fog"), errors="coerce")
                native_v = pd.to_numeric(row.get("native_activity_rel_at_0"), errors="coerce")
                if np.isfinite(fog_v) and fog_v > 0.0 and np.isfinite(native_v):
                    native_soft = float(np.clip(native_v, 0.0, 1.0) ** 2.0)
                    fog_soft = float(fog_v) * native_soft
                    if fog_soft > 0.0:
                        new_row["native_activity_soft_penalty"] = native_soft
                        new_row["fog_native_soft"] = fog_soft
                        new_row["log_fog_native_soft"] = float(np.log(fog_soft))
            if (not solvent_present) and np.isfinite(pd.to_numeric(new_row.get("fog_native_soft"), errors="coerce")):
                soft_fallback = float(new_row["fog_native_soft"])
                new_row[ABS_OBJECTIVE_COL] = soft_fallback
                new_row[ABS_OBJECTIVE_OLD_COL] = soft_fallback
                new_row[ABS_OBJECTIVE_LEGACY_COL] = soft_fallback
                new_row["fog_solvent_balanced"] = soft_fallback
                log_soft_fallback = pd.to_numeric(new_row.get("log_fog_native_soft"), errors="coerce")
                if np.isfinite(log_soft_fallback):
                    new_row[ABS_OBJECTIVE_LOG_COL] = float(log_soft_fallback)
                    new_row[ABS_OBJECTIVE_LOG_OLD_COL] = float(log_soft_fallback)
                    new_row[ABS_OBJECTIVE_LOG_LEGACY_COL] = float(log_soft_fallback)
                    new_row["log_fog_solvent_balanced"] = float(log_soft_fallback)
                elif float(new_row[ABS_OBJECTIVE_COL]) > 0.0:
                    log_soft = float(np.log(float(new_row[ABS_OBJECTIVE_COL])))
                    new_row[ABS_OBJECTIVE_LOG_COL] = log_soft
                    new_row[ABS_OBJECTIVE_LOG_OLD_COL] = log_soft
                    new_row[ABS_OBJECTIVE_LOG_LEGACY_COL] = log_soft
                    new_row["log_fog_solvent_balanced"] = log_soft
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
        cols = (
            ["polymer_id"]
            + (["round_id"] if "round_id" in learning_df.columns else [])
            + BO_X_COLS
            + [BO_Y_COL]
            + [
                c
                for c in [
                    ABS_OBJECTIVE_COL,
                    ABS_OBJECTIVE_LOG_COL,
                    ABS_OBJECTIVE_OLD_COL,
                    ABS_OBJECTIVE_LOG_OLD_COL,
                    ABS_OBJECTIVE_LEGACY_COL,
                    ABS_OBJECTIVE_LOG_LEGACY_COL,
                    "fog_solvent_balanced",
                    "log_fog_solvent_balanced",
                    OBJECTIVE_LOGLINEAR_MAIN_COL,
                    OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
                    "fog_vs_solvent_control",
                    "abs0_vs_solvent_control",
                    "abs_activity_down_penalty",
                    "abs_activity_up_bonus",
                    "abs_activity_balance_factor",
                    "solvent_activity_down_penalty",
                    "solvent_activity_up_bonus",
                    "solvent_activity_balance_factor",
                    "solvent_group",
                    "solvent_control_polymer_id",
                    "fog_native_constrained",
                    "log_fog_native_constrained",
                    "fog_native_soft",
                    "log_fog_native_soft",
                    "native_activity_soft_penalty",
                    "native_activity_rel_at_0",
                    "native_activity_min_rel_threshold",
                    "native_activity_feasible",
                    "fog_constraint_reason",
                ]
                if c in learning_df.columns
            ]
            + [c for c in lineage_columns if c in learning_df.columns]
        )
        learning_df = learning_df[[c for c in cols if c in learning_df.columns]]

    return learning_df, excluded_df


def build_bo_learning_data_from_round_averaged(
    catalog_df: pd.DataFrame,
    fog_round_averaged_path: Path,
    *,
    run_round_map: Optional[Dict[str, str]] = None,
    strict_round_coverage: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build BO learning CSV from round-averaged FoG (one row per round_id, polymer_id).

    - Reads fog_round_averaged CSV (round_id, polymer_id, mean_fog, mean_log_fog, ...).
    - If robust columns exist (`robust_fog`, `robust_log_fog`), they are preferred as BO objective.
      Otherwise, falls back to mean columns.
    - Only polymer_ids in catalog_df are included.
    - Rows with missing or invalid objective log_fog are excluded.
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
    if run_round_map is not None:
        coverage = build_round_coverage_summary(df["round_id"].tolist(), run_round_map)
        if strict_round_coverage and coverage["round_ids_missing_in_map"]:
            miss = ", ".join(coverage["round_ids_missing_in_map"])
            raise ValueError(
                "Round IDs in round-averaged FoG are not found in run_round_map: "
                f"{miss}. Update run_round_map or pass strict_round_coverage=False."
            )
    has_robust_cols = ("robust_fog" in df.columns) and ("robust_log_fog" in df.columns)
    fog_col = "robust_fog" if has_robust_cols else "mean_fog"
    log_col = "robust_log_fog" if has_robust_cols else "mean_log_fog"
    fog_native_col = (
        "robust_fog_native_constrained"
        if has_robust_cols and "robust_fog_native_constrained" in df.columns
        else ("mean_fog_native_constrained" if "mean_fog_native_constrained" in df.columns else None)
    )
    log_native_col = (
        "robust_log_fog_native_constrained"
        if has_robust_cols and "robust_log_fog_native_constrained" in df.columns
        else ("mean_log_fog_native_constrained" if "mean_log_fog_native_constrained" in df.columns else None)
    )
    fog_native_soft_col = (
        "robust_fog_native_soft"
        if has_robust_cols and "robust_fog_native_soft" in df.columns
        else ("mean_fog_native_soft" if "mean_fog_native_soft" in df.columns else None)
    )
    log_native_soft_col = (
        "robust_log_fog_native_soft"
        if has_robust_cols and "robust_log_fog_native_soft" in df.columns
        else ("mean_log_fog_native_soft" if "mean_log_fog_native_soft" in df.columns else None)
    )
    if fog_native_soft_col is None:
        fog_native_soft_col = fog_native_col
    if log_native_soft_col is None:
        log_native_soft_col = log_native_col
    loglinear_col = (
        "robust_objective_loglinear_main"
        if has_robust_cols and "robust_objective_loglinear_main" in df.columns
        else ("mean_objective_loglinear_main" if "mean_objective_loglinear_main" in df.columns else None)
    )
    loglinear_exp_col = (
        "robust_objective_loglinear_main_exp"
        if has_robust_cols and "robust_objective_loglinear_main_exp" in df.columns
        else ("mean_objective_loglinear_main_exp" if "mean_objective_loglinear_main_exp" in df.columns else None)
    )
    fog_solvent_balanced_col = (
        "robust_fog_activity_bonus_penalty"
        if has_robust_cols and "robust_fog_activity_bonus_penalty" in df.columns
        else (
            "mean_fog_activity_bonus_penalty"
            if "mean_fog_activity_bonus_penalty" in df.columns
            else (
                "robust_fog_abs_bonus_penalty"
                if has_robust_cols and "robust_fog_abs_bonus_penalty" in df.columns
                else (
                    "mean_fog_abs_bonus_penalty"
                    if "mean_fog_abs_bonus_penalty" in df.columns
                    else (
                        "robust_fog_abs_modulated"
                        if has_robust_cols and "robust_fog_abs_modulated" in df.columns
                        else (
                            "mean_fog_abs_modulated"
                            if "mean_fog_abs_modulated" in df.columns
                            else (
                                "robust_fog_solvent_balanced"
                                if has_robust_cols and "robust_fog_solvent_balanced" in df.columns
                                else ("mean_fog_solvent_balanced" if "mean_fog_solvent_balanced" in df.columns else None)
                            )
                        )
                    )
                )
            )
        )
    )
    log_solvent_balanced_col = (
        "robust_log_fog_activity_bonus_penalty"
        if has_robust_cols and "robust_log_fog_activity_bonus_penalty" in df.columns
        else (
            "mean_log_fog_activity_bonus_penalty"
            if "mean_log_fog_activity_bonus_penalty" in df.columns
            else (
                "robust_log_fog_abs_bonus_penalty"
                if has_robust_cols and "robust_log_fog_abs_bonus_penalty" in df.columns
                else (
                    "mean_log_fog_abs_bonus_penalty"
                    if "mean_log_fog_abs_bonus_penalty" in df.columns
                    else (
                        "robust_log_fog_abs_modulated"
                        if has_robust_cols and "robust_log_fog_abs_modulated" in df.columns
                        else (
                            "mean_log_fog_abs_modulated"
                            if "mean_log_fog_abs_modulated" in df.columns
                            else (
                                "robust_log_fog_solvent_balanced"
                                if has_robust_cols and "robust_log_fog_solvent_balanced" in df.columns
                                else ("mean_log_fog_solvent_balanced" if "mean_log_fog_solvent_balanced" in df.columns else None)
                            )
                        )
                    )
                )
            )
        )
    )
    if fog_solvent_balanced_col is None:
        fog_solvent_balanced_col = fog_native_soft_col
    if log_solvent_balanced_col is None:
        log_solvent_balanced_col = log_native_soft_col
    objective_source = "robust_round_aggregated" if has_robust_cols else "mean_round_aggregated"

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

        fog_val = row.get(fog_col)
        log_fog_val = row.get(log_col)
        if pd.isna(log_fog_val) or (pd.notna(fog_val) and (pd.isna(fog_val) or float(fog_val) <= 0)):
            excluded_rows.append({
                "round_id": row["round_id"],
                "polymer_id": pid,
                "reason": f"{log_col}_missing_or_invalid",
            })
            continue

        cat_row = catalog_df[catalog_df["polymer_id"] == pid].iloc[0]
        new_row = {
            "polymer_id": pid,
            "round_id": row["round_id"],
            "frac_MPC": float(cat_row["frac_MPC"]),
            "frac_BMA": float(cat_row["frac_BMA"]),
            "frac_MTAC": float(cat_row["frac_MTAC"]),
            BO_Y_COL: float(log_fog_val),
            "objective_source": objective_source,
        }
        if fog_native_col is not None:
            fog_native_val = pd.to_numeric(row.get(fog_native_col), errors="coerce")
            new_row["fog_native_constrained"] = float(fog_native_val) if np.isfinite(fog_native_val) else np.nan
        if log_native_col is not None:
            log_native_val = pd.to_numeric(row.get(log_native_col), errors="coerce")
            new_row["log_fog_native_constrained"] = (
                float(log_native_val) if np.isfinite(log_native_val) else np.nan
            )
        if fog_native_soft_col is not None:
            fog_native_soft_val = pd.to_numeric(row.get(fog_native_soft_col), errors="coerce")
            new_row["fog_native_soft"] = (
                float(fog_native_soft_val) if np.isfinite(fog_native_soft_val) else np.nan
            )
        if log_native_soft_col is not None:
            log_native_soft_val = pd.to_numeric(row.get(log_native_soft_col), errors="coerce")
            new_row["log_fog_native_soft"] = (
                float(log_native_soft_val) if np.isfinite(log_native_soft_val) else np.nan
            )
        if loglinear_col is not None:
            loglinear_val = pd.to_numeric(row.get(loglinear_col), errors="coerce")
            new_row[OBJECTIVE_LOGLINEAR_MAIN_COL] = (
                float(loglinear_val) if np.isfinite(loglinear_val) else np.nan
            )
        if loglinear_exp_col is not None:
            loglinear_exp_val = pd.to_numeric(row.get(loglinear_exp_col), errors="coerce")
            new_row[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = (
                float(loglinear_exp_val) if np.isfinite(loglinear_exp_val) else np.nan
            )
        elif np.isfinite(pd.to_numeric(new_row.get(OBJECTIVE_LOGLINEAR_MAIN_COL), errors="coerce")):
            new_row[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = float(
                np.exp(float(pd.to_numeric(new_row[OBJECTIVE_LOGLINEAR_MAIN_COL], errors="coerce")))
            )
        if fog_solvent_balanced_col is not None:
            fog_solvent_val = pd.to_numeric(row.get(fog_solvent_balanced_col), errors="coerce")
            fog_solvent_val_f = float(fog_solvent_val) if np.isfinite(fog_solvent_val) else np.nan
            new_row[ABS_OBJECTIVE_COL] = fog_solvent_val_f
            new_row[ABS_OBJECTIVE_OLD_COL] = fog_solvent_val_f
            new_row[ABS_OBJECTIVE_LEGACY_COL] = fog_solvent_val_f
            new_row["fog_solvent_balanced"] = fog_solvent_val_f
        if log_solvent_balanced_col is not None:
            log_solvent_val = pd.to_numeric(row.get(log_solvent_balanced_col), errors="coerce")
            log_solvent_val_f = float(log_solvent_val) if np.isfinite(log_solvent_val) else np.nan
            new_row[ABS_OBJECTIVE_LOG_COL] = log_solvent_val_f
            new_row[ABS_OBJECTIVE_LOG_OLD_COL] = log_solvent_val_f
            new_row[ABS_OBJECTIVE_LOG_LEGACY_COL] = log_solvent_val_f
            new_row["log_fog_solvent_balanced"] = log_solvent_val_f
        if "native_feasible_fraction" in row:
            nat_frac = pd.to_numeric(row.get("native_feasible_fraction"), errors="coerce")
            new_row["native_feasible_fraction"] = float(nat_frac) if np.isfinite(nat_frac) else np.nan
        for c in ["x", "y"]:
            if c in cat_row.index and pd.notna(cat_row.get(c)):
                new_row[c] = cat_row[c]
        if "run_ids" in row:
            new_row["run_ids"] = row["run_ids"]
        if "n_observations" in row:
            new_row["n_observations"] = row["n_observations"]
        if "log_fog_mad" in row and pd.notna(row.get("log_fog_mad")):
            new_row["log_fog_mad"] = row["log_fog_mad"]
        learning_rows.append(new_row)

    learning_df = pd.DataFrame(learning_rows)
    excluded_df = pd.DataFrame(excluded_rows)

    if not learning_df.empty:
        cols = ["polymer_id", "round_id"] + BO_X_COLS + [BO_Y_COL] + [
            c
            for c in [
                "log_fog_native_constrained",
                "fog_native_constrained",
                "log_fog_native_soft",
                "fog_native_soft",
                OBJECTIVE_LOGLINEAR_MAIN_COL,
                OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
                ABS_OBJECTIVE_LOG_COL,
                ABS_OBJECTIVE_COL,
                ABS_OBJECTIVE_LOG_OLD_COL,
                ABS_OBJECTIVE_OLD_COL,
                ABS_OBJECTIVE_LOG_LEGACY_COL,
                ABS_OBJECTIVE_LEGACY_COL,
                "log_fog_solvent_balanced",
                "fog_solvent_balanced",
                "native_feasible_fraction",
                "objective_source",
                "run_ids",
                "n_observations",
                "log_fog_mad",
            ]
            if c in learning_df.columns
        ]
        learning_df = learning_df[[c for c in cols if c in learning_df.columns]]

    return learning_df, excluded_df


def write_bo_learning_csv(learning_df: pd.DataFrame, out_path: Path) -> None:
    """Write BO learning CSV (X order: frac_MPC, frac_BMA, frac_MTAC)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    learning_df.to_csv(out_path, index=False)


def write_exclusion_report(excluded_df: pd.DataFrame, out_path: Path) -> None:
    """Write exclusion report CSV (polymer_id, run_id, reason)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_df.to_csv(out_path, index=False)
