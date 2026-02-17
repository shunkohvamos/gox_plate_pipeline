# src/gox_plate_pipeline/summary.py
"""
Polymer × heat aggregation for BO: summary table, lineage, and run manifest.

Input: well-level result table (e.g. rates_with_rea) with polymer_id, heat_min,
       status, abs_activity; optional REA_percent. File names are not hardcoded.
Output: single BO file bo_output__{run_id}.json (summary + lineage + manifest, traceable to raw).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["polymer_id", "heat_min", "status", "abs_activity", "plate_id", "well"]
OPTIONAL_REA = "REA_percent"
SUMMARY_OUTLIER_EVENT_COLUMNS = [
    "run_id",
    "polymer_id",
    "heat_min",
    "plate_id",
    "well",
    "abs_activity",
    "REA_percent",
    "group_n",
    "group_n_keep",
    "method",
    "trigger",
    "pair_ratio_abs",
    "pair_ratio_rea",
    "heat_abs_median",
    "heat_rea_median",
]


def _well_sort_key(well: str) -> tuple:
    """ウェル名 A1, A2, ..., H7 を (行序, 列序) に変換（A=0, B=1, ..., H=7）。"""
    well = str(well).strip()
    if not well:
        return (999, 999)
    row = well[0].upper()
    try:
        col = int(well[1:]) if len(well) > 1 else 0
    except ValueError:
        col = 0
    row_ord = ord(row) - ord("A") if "A" <= row <= "Z" else 999
    return (row_ord, col)


def _polymer_id_order_by_well(well_df: pd.DataFrame) -> List[str]:
    """
    生データのウェル順（A1, A2, ..., A7, B1, ..., H7）で polymer_id の初出順を返す。
    """
    if "well" not in well_df.columns or "polymer_id" not in well_df.columns:
        return []
    # (plate_id, well, polymer_id) をユニークに（1 well に 1 polymer_id ではないので先頭を採用）
    sub = well_df[["plate_id", "well", "polymer_id"]].drop_duplicates()
    sub = sub.copy()
    sub["_row_ord"] = sub["well"].apply(lambda w: _well_sort_key(w)[0])
    sub["_col_ord"] = sub["well"].apply(lambda w: _well_sort_key(w)[1])
    sub = sub.sort_values(["plate_id", "_row_ord", "_col_ord"])
    seen: set = set()
    order: List[str] = []
    for pid in sub["polymer_id"].astype(str):
        if pid not in seen:
            seen.add(pid)
            order.append(pid)
    return order


def _sort_summary_by_well_order(summary: pd.DataFrame, well_df: pd.DataFrame) -> pd.DataFrame:
    """summary の polymer_id をウェル A→H 順にソートし、続けて heat_min でソート。"""
    if summary.empty or "polymer_id" not in summary.columns or "heat_min" not in summary.columns:
        return summary.reset_index(drop=True)
    order = _polymer_id_order_by_well(well_df)
    if not order:
        return summary.sort_values(["polymer_id", "heat_min"]).reset_index(drop=True)
    summary = summary.copy()
    summary["polymer_id"] = pd.Categorical(
        summary["polymer_id"].astype(str),
        categories=order,
        ordered=True,
    )
    return summary.sort_values(["polymer_id", "heat_min"]).reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    try:
        root = Path(repo_root or Path.cwd())
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _robust_outlier_mask(
    values: np.ndarray,
    *,
    min_samples: int = 3,
    z_threshold: float = 3.5,
    ratio_low: Optional[float] = 0.33,
    ratio_high: Optional[float] = 3.0,
    min_keep: int = 1,
) -> np.ndarray:
    """
    Robust 1D outlier mask using MAD-based z-score and optional ratio bounds.

    If exclusions would leave fewer than min_keep observations, no points are excluded.
    """
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=bool)
    finite = np.isfinite(arr)
    idx = np.flatnonzero(finite)
    vals = arr[finite]
    n = int(vals.size)
    if n < max(1, int(min_samples)):
        return out

    med = float(np.nanmedian(vals))
    abs_dev = np.abs(vals - med)
    mad = float(np.nanmedian(abs_dev))
    scale = 1.4826 * mad

    z_mask = np.zeros(n, dtype=bool)
    if np.isfinite(scale) and scale > 1e-12 and np.isfinite(float(z_threshold)) and float(z_threshold) > 0.0:
        z_mask = (abs_dev / scale) > float(z_threshold)

    ratio_mask = np.zeros(n, dtype=bool)
    rl = float(ratio_low) if ratio_low is not None else None
    rh = float(ratio_high) if ratio_high is not None else None
    if (
        rl is not None
        and rh is not None
        and rl > 0.0
        and rh > 0.0
        and np.isfinite(med)
        and med > 1e-12
    ):
        ratio = vals / med
        ratio_mask = (ratio < rl) | (ratio > rh)

    cand = z_mask | ratio_mask
    if not np.any(cand):
        return out
    if (n - int(np.count_nonzero(cand))) < max(1, int(min_keep)):
        return out
    out[idx] = cand
    return out


def _safe_ratio_max(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size < 2:
        return np.nan
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0:
        return np.nan
    return hi / lo


def _pair_outlier_idx(
    *,
    abs_vals: np.ndarray,
    rea_vals: Optional[np.ndarray],
    heat_abs_median: Optional[float],
    heat_rea_median: Optional[float],
    pair_ratio_threshold: float,
) -> Optional[int]:
    """
    For n=2 groups with large disagreement, choose one row to exclude.

    We keep the replicate that is closer in scale and consistency to the same run:
    same-heat medians across all polymer_id (abs_activity and REA_percent) are used
    as the run-scale reference; the point farther from this reference is excluded,
    so the one adopted is more in line with data quality/scale of other polymers
    in the same run. Tie-breaker: group median.
    """
    if abs_vals.size != 2:
        return None
    ratio_abs = _safe_ratio_max(abs_vals)
    ratio_rea = _safe_ratio_max(rea_vals) if rea_vals is not None else np.nan
    use_abs = bool(np.isfinite(ratio_abs) and ratio_abs >= float(pair_ratio_threshold))
    use_rea = bool(np.isfinite(ratio_rea) and ratio_rea >= float(pair_ratio_threshold))
    if not (use_abs or use_rea):
        return None

    score = np.zeros(2, dtype=float)
    used_context = False
    if use_abs:
        if heat_abs_median is not None and np.isfinite(float(heat_abs_median)) and float(heat_abs_median) > 0.0:
            score += np.abs(np.log(np.clip(abs_vals, 1e-12, np.inf) / float(heat_abs_median)))
            used_context = True
    if use_rea and rea_vals is not None:
        if heat_rea_median is not None and np.isfinite(float(heat_rea_median)) and float(heat_rea_median) > 0.0:
            score += np.abs(np.log(np.clip(rea_vals, 1e-12, np.inf) / float(heat_rea_median)))
            used_context = True

    # Fallback when same-heat run context is unavailable.
    if not used_context:
        if use_abs:
            abs_med = float(np.nanmedian(abs_vals))
            if np.isfinite(abs_med) and abs_med > 0.0:
                score += np.abs(np.log(np.clip(abs_vals, 1e-12, np.inf) / abs_med))
        if use_rea and rea_vals is not None:
            rea_med = float(np.nanmedian(rea_vals))
            if np.isfinite(rea_med) and rea_med > 0.0:
                score += np.abs(np.log(np.clip(rea_vals, 1e-12, np.inf) / rea_med))

    if np.isclose(score[0], score[1]):
        return int(np.argmax(np.asarray(abs_vals, dtype=float)))
    return int(np.argmax(score))


def filter_well_table_for_summary(
    well_df: pd.DataFrame,
    *,
    run_id: str,
    apply_outlier_filter: bool = False,
    outlier_min_samples: int = 3,
    outlier_z_threshold: float = 3.5,
    outlier_ratio_low: float = 0.33,
    outlier_ratio_high: float = 3.0,
    outlier_pair_ratio_threshold: float = 3.0,
    outlier_min_keep: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optionally exclude extreme replicate wells before polymer×heat aggregation.

    Outliers are detected per (polymer_id, heat_min) group.
    For n >= outlier_min_samples: robust MAD+ratio rule.
    For n == 2 with strong disagreement: keep the row closer to same-heat run medians.

    For n==2 with strong disagreement (pair ratio >= threshold): one replicate is
    excluded and the other kept; we keep the one closer in scale to same-run
    same-heat data (other polymer_id). For n>=3, default outlier_min_keep=2:
    do not exclude if that would leave fewer than 2 points, so SEM (and error bars)
    remain defined. The all-data summary (summary_stats_all) still provides
    error bars when n=2 is reduced to n=1.
    """
    for c in REQUIRED_COLUMNS:
        if c not in well_df.columns:
            raise ValueError(f"Well table must contain {c}, got: {list(well_df.columns)}")

    if not apply_outlier_filter:
        return well_df.copy(), pd.DataFrame(columns=SUMMARY_OUTLIER_EVENT_COLUMNS)

    work = well_df.copy()
    work["polymer_id"] = work["polymer_id"].astype(str)
    work["heat_min"] = pd.to_numeric(work["heat_min"], errors="coerce")
    work["abs_activity"] = pd.to_numeric(work["abs_activity"], errors="coerce")
    has_rea = OPTIONAL_REA in work.columns
    if has_rea:
        work[OPTIONAL_REA] = pd.to_numeric(work[OPTIONAL_REA], errors="coerce")

    candidate_mask = (
        work["status"].astype(str).str.strip().str.lower().eq("ok")
        & np.isfinite(work["abs_activity"])
        & np.isfinite(work["heat_min"])
        & work["polymer_id"].str.strip().ne("")
    )
    cand = work.loc[candidate_mask].copy()
    if cand.empty:
        return work, pd.DataFrame(columns=SUMMARY_OUTLIER_EVENT_COLUMNS)

    heat_abs_med = (
        cand.groupby("heat_min", as_index=True)["abs_activity"]
        .median()
        .to_dict()
    )
    if has_rea:
        heat_rea_med = (
            cand.groupby("heat_min", as_index=True)[OPTIONAL_REA]
            .median()
            .to_dict()
        )
    else:
        heat_rea_med = {}

    drop_index: set[int] = set()
    outlier_rows: list[dict[str, Any]] = []
    min_keep = max(1, int(outlier_min_keep))
    for (pid, heat_min), sub in cand.groupby(["polymer_id", "heat_min"], sort=False, dropna=False):
        idx = sub.index.to_numpy(dtype=int)
        n_group = int(len(sub))
        if n_group <= 1:
            continue
        abs_vals = sub["abs_activity"].to_numpy(dtype=float)
        rea_vals = sub[OPTIONAL_REA].to_numpy(dtype=float) if has_rea else None

        out_mask = np.zeros(n_group, dtype=bool)
        method = ""
        trigger = ""
        pair_abs_ratio = np.nan
        pair_rea_ratio = np.nan
        if n_group == 2:
            pair_abs_ratio = _safe_ratio_max(abs_vals)
            pair_rea_ratio = _safe_ratio_max(rea_vals) if rea_vals is not None else np.nan
            out_idx = _pair_outlier_idx(
                abs_vals=abs_vals,
                rea_vals=rea_vals,
                heat_abs_median=float(heat_abs_med.get(float(heat_min), np.nan)),
                heat_rea_median=float(heat_rea_med.get(float(heat_min), np.nan)) if has_rea else None,
                pair_ratio_threshold=float(outlier_pair_ratio_threshold),
            )
            if out_idx is not None:
                out_mask[int(out_idx)] = True
                method = "pair_ratio_context"
                if np.isfinite(pair_abs_ratio) and pair_abs_ratio >= float(outlier_pair_ratio_threshold):
                    trigger = "abs_activity_pair_ratio"
                elif np.isfinite(pair_rea_ratio) and pair_rea_ratio >= float(outlier_pair_ratio_threshold):
                    trigger = "REA_percent_pair_ratio"
                else:
                    trigger = "pair_ratio"
        else:
            out_abs = _robust_outlier_mask(
                abs_vals,
                min_samples=int(outlier_min_samples),
                z_threshold=float(outlier_z_threshold),
                ratio_low=float(outlier_ratio_low),
                ratio_high=float(outlier_ratio_high),
                min_keep=min_keep,
            )
            if rea_vals is not None:
                out_rea = _robust_outlier_mask(
                    rea_vals,
                    min_samples=int(outlier_min_samples),
                    z_threshold=float(outlier_z_threshold),
                    ratio_low=float(outlier_ratio_low),
                    ratio_high=float(outlier_ratio_high),
                    min_keep=min_keep,
                )
            else:
                out_rea = np.zeros(n_group, dtype=bool)
            out_mask = out_abs | out_rea
            if np.any(out_mask):
                method = "robust_group"
                if bool(np.any(out_abs)) and bool(np.any(out_rea)):
                    trigger = "abs_and_rea"
                elif bool(np.any(out_abs)):
                    trigger = "abs_activity"
                else:
                    trigger = "REA_percent"

        n_keep = int(n_group - int(np.count_nonzero(out_mask)))
        if not np.any(out_mask):
            continue
        # For n>=3, require at least min_keep so SEM/error bars remain; for n==2 allow n_keep=1 when pair is divergent.
        if n_group >= 3 and n_keep < min_keep:
            continue

        sub_dropped = sub.loc[idx[out_mask]].copy()
        for _, rr in sub_dropped.iterrows():
            outlier_rows.append(
                {
                    "run_id": str(run_id),
                    "polymer_id": str(pid),
                    "heat_min": float(heat_min),
                    "plate_id": str(rr.get("plate_id", "")),
                    "well": str(rr.get("well", "")),
                    "abs_activity": float(rr.get("abs_activity", np.nan)),
                    "REA_percent": (
                        float(rr.get(OPTIONAL_REA, np.nan))
                        if has_rea else np.nan
                    ),
                    "group_n": int(n_group),
                    "group_n_keep": int(n_keep),
                    "method": method,
                    "trigger": trigger,
                    "pair_ratio_abs": float(pair_abs_ratio) if np.isfinite(pair_abs_ratio) else np.nan,
                    "pair_ratio_rea": float(pair_rea_ratio) if np.isfinite(pair_rea_ratio) else np.nan,
                    "heat_abs_median": float(heat_abs_med.get(float(heat_min), np.nan)),
                    "heat_rea_median": (
                        float(heat_rea_med.get(float(heat_min), np.nan))
                        if has_rea else np.nan
                    ),
                }
            )
        drop_index.update(int(i) for i in idx[out_mask].tolist())

    if not drop_index:
        return work, pd.DataFrame(columns=SUMMARY_OUTLIER_EVENT_COLUMNS)
    filtered = work.drop(index=sorted(drop_index)).copy()
    events_df = pd.DataFrame(outlier_rows, columns=SUMMARY_OUTLIER_EVENT_COLUMNS)
    return filtered, events_df


def build_polymer_heat_summary(well_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    Aggregate well-level results by polymer_id × heat_min.

    Uses only rows with status == 'ok' and finite abs_activity (and REA_percent if present).
    Returns summary with run_id, polymer_id, heat_min, n, mean, std, sem for abs_activity
    and REA_percent (if column exists).
    """
    for c in REQUIRED_COLUMNS:
        if c not in well_df.columns:
            raise ValueError(f"Well table must contain {c}, got: {list(well_df.columns)}")

    ok = well_df.loc[
        (well_df["status"].astype(str).str.strip().str.lower() == "ok")
        & np.isfinite(pd.to_numeric(well_df["abs_activity"], errors="coerce"))
    ].copy()

    ok["heat_min"] = pd.to_numeric(ok["heat_min"], errors="coerce")
    ok["abs_activity"] = pd.to_numeric(ok["abs_activity"], errors="coerce")
    has_rea = OPTIONAL_REA in ok.columns
    if has_rea:
        ok[OPTIONAL_REA] = pd.to_numeric(ok[OPTIONAL_REA], errors="coerce")

    grp = ok.groupby(["polymer_id", "heat_min"], dropna=False)

    summary_rows: List[Dict[str, Any]] = []
    for (polymer_id, heat_min), sub in grp:
        n = int(sub["abs_activity"].notna().sum())
        mean_aa = float(sub["abs_activity"].mean())
        std_aa = float(sub["abs_activity"].std()) if n > 1 else np.nan
        sem_aa = float(sub["abs_activity"].sem()) if n > 1 else np.nan
        row: Dict[str, Any] = {
            "run_id": run_id,
            "polymer_id": polymer_id,
            "heat_min": float(heat_min),
            "n": n,
            "mean_abs_activity": mean_aa,
            "std_abs_activity": std_aa,
            "sem_abs_activity": sem_aa,
        }
        if has_rea and OPTIONAL_REA in sub.columns and sub[OPTIONAL_REA].notna().any():
            row["mean_REA_percent"] = float(sub[OPTIONAL_REA].mean())
            row["std_REA_percent"] = float(sub[OPTIONAL_REA].std()) if n > 1 else np.nan
            row["sem_REA_percent"] = float(sub[OPTIONAL_REA].sem()) if n > 1 else np.nan
        else:
            row["mean_REA_percent"] = np.nan
            row["std_REA_percent"] = np.nan
            row["sem_REA_percent"] = np.nan
        
        # Preserve include_in_all_polymers flag (should be same for all rows with same polymer_id)
        if "include_in_all_polymers" in sub.columns:
            include_flag_val = sub["include_in_all_polymers"].iloc[0] if len(sub) > 0 else None
            if pd.isna(include_flag_val):
                row["include_in_all_polymers"] = True
            elif isinstance(include_flag_val, bool):
                row["include_in_all_polymers"] = include_flag_val
            else:
                # Handle string "True"/"False"
                s = str(include_flag_val).strip().upper()
                row["include_in_all_polymers"] = s in ("TRUE", "1", "YES")
        else:
            row["include_in_all_polymers"] = True
        
        # Preserve all_polymers_pair flag (should be same for all rows with same polymer_id)
        if "all_polymers_pair" in sub.columns:
            pair_flag_val = sub["all_polymers_pair"].iloc[0] if len(sub) > 0 else None
            if pd.isna(pair_flag_val):
                row["all_polymers_pair"] = False
            elif isinstance(pair_flag_val, bool):
                row["all_polymers_pair"] = pair_flag_val
            else:
                # Handle string "True"/"False"
                s = str(pair_flag_val).strip().upper()
                row["all_polymers_pair"] = s in ("TRUE", "1", "YES")
        else:
            row["all_polymers_pair"] = False
        
        summary_rows.append(row)

    out = pd.DataFrame(summary_rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "run_id",
                "polymer_id",
                "heat_min",
                "n",
                "mean_abs_activity",
                "std_abs_activity",
                "sem_abs_activity",
                "mean_REA_percent",
                "std_REA_percent",
                "sem_REA_percent",
                "include_in_all_polymers",
                "all_polymers_pair",
            ]
        )
    return out


def build_polymer_heat_lineage(well_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    One row per well: which wells went into each (polymer_id, heat_min) summary.

    Columns: run_id, polymer_id, heat_min, plate_id, well, abs_activity, REA_percent (if present),
    source_file (if present). Only includes rows with status == 'ok' and finite abs_activity.
    """
    for c in REQUIRED_COLUMNS:
        if c not in well_df.columns:
            raise ValueError(f"Well table must contain {c}, got: {list(well_df.columns)}")

    ok = well_df.loc[
        (well_df["status"].astype(str).str.strip().str.lower() == "ok")
        & np.isfinite(pd.to_numeric(well_df["abs_activity"], errors="coerce"))
    ].copy()

    cols = ["run_id", "polymer_id", "heat_min", "plate_id", "well", "abs_activity"]
    if OPTIONAL_REA in well_df.columns:
        cols.append(OPTIONAL_REA)
    if "source_file" in well_df.columns:
        cols.append("source_file")

    ok["run_id"] = run_id
    ok["heat_min"] = pd.to_numeric(ok["heat_min"], errors="coerce")
    available = [c for c in cols if c in ok.columns]
    return ok[available].copy()


def build_run_manifest_dict(
    run_id: str,
    input_paths: List[Path],
    *,
    git_root: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build manifest dict (run_id, timestamp, git commit, input file hashes/mtime/size).
    Used both for standalone run_manifest JSON and for the consolidated BO output file.
    """
    input_paths = [Path(p) for p in input_paths]
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(git_root),
        "input_files": [],
    }
    for p in input_paths:
        if not p.is_file():
            manifest["input_files"].append({"path": str(p), "error": "file not found"})
            continue
        stat = p.stat()
        manifest["input_files"].append({
            "path": str(p.resolve()),
            "sha256": _file_sha256(p),
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "size_bytes": stat.st_size,
        })
    if extra:
        manifest["extra"] = extra
    return manifest


def write_run_manifest(
    run_id: str,
    input_paths: List[Path],
    out_path: Path,
    *,
    git_root: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write run_manifest__{run_id}.json with run_id, timestamp, git commit, input file hashes/mtime/size.
    """
    manifest = build_run_manifest_dict(
        run_id, input_paths, git_root=git_root, extra=extra
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _json_default(o: Any) -> Any:
    """JSON encoder: unsupported types (fallback). NaN/Inf are cleaned in _clean_for_json."""
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _clean_for_json(obj: Any) -> Any:
    """Recursively replace float/numpy NaN/Inf with None so JSON is valid."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(x) for x in obj]
    if hasattr(obj, "item"):  # numpy scalar
        try:
            obj = obj.item()
        except (ValueError, AttributeError, TypeError):
            pass
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _write_summary_simple_csv(summary: pd.DataFrame, out_path: Path) -> None:
    """
    polymer_id, heat_min, abs_activity, REA_percent のみの簡易 CSV を出力（ウェル情報なし）。
    run_id は追跡性のために含める（core-rules: provenance）。
    include_in_all_polymers 列があれば含める（all_polymers プロット用）。
    all_polymers_pair 列があれば含める（all_polymers_pair プロット用）。
    """
    cols = ["run_id", "polymer_id", "heat_min", "mean_abs_activity", "mean_REA_percent"]
    if "include_in_all_polymers" in summary.columns:
        cols.append("include_in_all_polymers")
    if "all_polymers_pair" in summary.columns:
        cols.append("all_polymers_pair")
    available = [c for c in cols if c in summary.columns]
    simple = summary[available].copy()
    simple = simple.rename(columns={
        "mean_abs_activity": "abs_activity",
        "mean_REA_percent": "REA_percent",
    })
    simple.to_csv(out_path, index=False)


def _write_summary_stats_csv(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Full polymer×heat summary with n/mean/std/sem columns.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)


def aggregate_and_write(
    well_df: pd.DataFrame,
    run_id: str,
    out_dir: Path,
    *,
    input_paths_for_manifest: Optional[List[Path]] = None,
    git_root: Optional[Path] = None,
    bo_dir: Optional[Path] = None,
    summary_simple_path: Optional[Path] = None,
    summary_stats_path: Optional[Path] = None,
    summary_stats_all_path: Optional[Path] = None,
    summary_outlier_events_path: Optional[Path] = None,
    apply_summary_outlier_filter: bool = False,
    summary_outlier_min_samples: int = 3,
    summary_outlier_z_threshold: float = 3.5,
    summary_outlier_ratio_low: float = 0.33,
    summary_outlier_ratio_high: float = 3.0,
    summary_outlier_pair_ratio_threshold: float = 3.0,
    summary_outlier_min_keep: int = 2,
    extra_output_files: Optional[List[str]] = None,
) -> Path:
    """
    Build summary and lineage. BO 用 JSON は fit/bo/、簡易テーブルは fit/ 直下に出力。

    - bo_dir 指定時: bo_dir/bo_output.json を出力（ベイズ最適化用は後工程で利用）
    - summary_simple_path 指定時: そのパスに summary_simple.csv（polymer_id, heat_min, abs_activity, REA_percent）を出力
    - いずれも未指定時: out_dir/bo/bo_output__{run_id}.json のみ（legacy）
    Returns: path to bo_output.json.
    """
    out_dir = Path(out_dir)
    if bo_dir is not None:
        bo_dir = Path(bo_dir)
        bo_dir.mkdir(parents=True, exist_ok=True)
        out_path = bo_dir / "bo_output.json"
    else:
        bo_dir = out_dir / "bo"
        bo_dir.mkdir(parents=True, exist_ok=True)
        out_path = bo_dir / f"bo_output__{run_id}.json"

    well_df_for_summary, summary_outlier_events = filter_well_table_for_summary(
        well_df,
        run_id=run_id,
        apply_outlier_filter=bool(apply_summary_outlier_filter),
        outlier_min_samples=int(summary_outlier_min_samples),
        outlier_z_threshold=float(summary_outlier_z_threshold),
        outlier_ratio_low=float(summary_outlier_ratio_low),
        outlier_ratio_high=float(summary_outlier_ratio_high),
        outlier_pair_ratio_threshold=float(summary_outlier_pair_ratio_threshold),
        outlier_min_keep=int(summary_outlier_min_keep),
    )

    summary = build_polymer_heat_summary(well_df_for_summary, run_id)
    summary = _sort_summary_by_well_order(summary, well_df_for_summary)
    lineage = build_polymer_heat_lineage(well_df_for_summary, run_id)
    # All-data summary (no outlier exclusion) for error-bar plots that show SEM using all replicates.
    if summary_stats_all_path is not None:
        summary_all = build_polymer_heat_summary(well_df, run_id)
        summary_all = _sort_summary_by_well_order(summary_all, well_df)
    extra_files = ["bo_output.json"]
    if summary_simple_path is not None:
        extra_files.append(str(summary_simple_path.name))
    if summary_stats_path is not None:
        extra_files.append(str(summary_stats_path.name))
    if summary_stats_all_path is not None:
        extra_files.append(str(Path(summary_stats_all_path).name))
    if summary_outlier_events_path is not None:
        extra_files.append(str(Path(summary_outlier_events_path).name))
    if extra_output_files:
        # Keep as short relative names; paths are traceable via the run_id fit directory.
        for x in extra_output_files:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            extra_files.append(s)
        # stable + avoid duplicates
        extra_files = sorted(set(extra_files))
    manifest_extra: Dict[str, Any] = {"output_files": extra_files}
    if apply_summary_outlier_filter:
        manifest_extra["summary_outlier_filter"] = {
            "enabled": True,
            "min_samples": int(summary_outlier_min_samples),
            "z_threshold": float(summary_outlier_z_threshold),
            "ratio_low": float(summary_outlier_ratio_low),
            "ratio_high": float(summary_outlier_ratio_high),
            "pair_ratio_threshold": float(summary_outlier_pair_ratio_threshold),
            "min_keep": int(summary_outlier_min_keep),
            "n_excluded_rows": int(len(summary_outlier_events)),
        }

    manifest = build_run_manifest_dict(
        run_id,
        input_paths_for_manifest or [],
        git_root=git_root,
        extra=manifest_extra,
    )

    payload = {
        "summary": _clean_for_json(summary.to_dict(orient="records")),
        "lineage": _clean_for_json(lineage.to_dict(orient="records")),
        "manifest": _clean_for_json(manifest),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    if summary_simple_path is not None:
        summary_simple_path = Path(summary_simple_path)
        summary_simple_path.parent.mkdir(parents=True, exist_ok=True)
        _write_summary_simple_csv(summary, summary_simple_path)

    if summary_stats_path is not None:
        summary_stats_path = Path(summary_stats_path)
        summary_stats_path.parent.mkdir(parents=True, exist_ok=True)
        _write_summary_stats_csv(summary, summary_stats_path)

    if summary_stats_all_path is not None:
        summary_stats_all_path = Path(summary_stats_all_path)
        summary_stats_all_path.parent.mkdir(parents=True, exist_ok=True)
        _write_summary_stats_csv(summary_all, summary_stats_all_path)

    if summary_outlier_events_path is not None:
        summary_outlier_events_path = Path(summary_outlier_events_path)
        summary_outlier_events_path.parent.mkdir(parents=True, exist_ok=True)
        summary_outlier_events.to_csv(summary_outlier_events_path, index=False)

    return out_path
