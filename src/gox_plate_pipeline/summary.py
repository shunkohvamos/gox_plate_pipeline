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
    order = _polymer_id_order_by_well(well_df)
    if not order:
        return summary.sort_values(["polymer_id", "heat_min"])
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
        summary_rows.append(row)

    out = pd.DataFrame(summary_rows)
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
    """
    cols = ["polymer_id", "heat_min", "mean_abs_activity", "mean_REA_percent"]
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

    summary = build_polymer_heat_summary(well_df, run_id)
    summary = _sort_summary_by_well_order(summary, well_df)
    lineage = build_polymer_heat_lineage(well_df, run_id)
    extra_files = ["bo_output.json"]
    if summary_simple_path is not None:
        extra_files.append(str(summary_simple_path.name))
    if summary_stats_path is not None:
        extra_files.append(str(summary_stats_path.name))
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
    manifest = build_run_manifest_dict(
        run_id,
        input_paths_for_manifest or [],
        git_root=git_root,
        extra={"output_files": extra_files},
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

    return out_path
