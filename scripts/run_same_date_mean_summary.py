#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local src/ is used (avoid importing an older installed package)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Headless plotting backend
import matplotlib

matplotlib.use("Agg")

from gox_plate_pipeline.fog import write_run_ranking_outputs  # noqa: E402
from gox_plate_pipeline.bo_data import load_run_round_map  # noqa: E402
from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    extract_measurement_date_from_run_id,
    load_or_create_polymer_color_map,
    normalize_t50_definition,
    plot_per_polymer_timeseries_across_runs_with_error_bars,
)
from gox_plate_pipeline.summary import build_run_manifest_dict  # noqa: E402


def _normalize_t50_value(v: object) -> str:
    try:
        return normalize_t50_definition(str(v))
    except Exception:
        return ""


def _parse_bool(v: object, *, default: bool = True) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    if s in {"", "—", "-", "na", "n/a", "none", "null"}:
        return default
    return default


def _safe_group_stem(text: str) -> str:
    s = "" if text is None else str(text).strip()
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return out or "group"


def _load_run_group_table(path: Path | None) -> dict[str, dict[str, object]]:
    """
    Load run group TSV as:
      run_id -> {"group_id": str, "include": bool}
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}

    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    if "run_id" not in df.columns:
        return {}

    group_col = None
    for c in ["group_id", "analysis_group_id", "date_prefix", "round_id"]:
        if c in df.columns:
            group_col = c
            break
    include_col = None
    for c in ["include_in_group_mean", "include_in_same_date_mean", "include", "enabled", "use"]:
        if c in df.columns:
            include_col = c
            break

    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        rid = str(row.get("run_id", "")).strip()
        if not rid:
            continue
        gid = str(row.get(group_col, "")).strip() if group_col is not None else ""
        if not gid:
            gid = extract_measurement_date_from_run_id(rid)
        inc = _parse_bool(row.get(include_col), default=True) if include_col is not None else True
        out[rid] = {"group_id": gid, "include": bool(inc)}
    return out


def _collect_group_runs(
    *,
    run_id: str,
    explicit_runs: list[str] | None,
    run_group_table: dict[str, dict[str, object]] | None,
    group_id_override: str | None = None,
) -> tuple[list[str], str]:
    table = run_group_table or {}

    if explicit_runs:
        runs = sorted({str(x).strip() for x in explicit_runs if str(x).strip()})
        if run_id not in runs:
            runs.append(run_id)
        inferred_group = str(table.get(run_id, {}).get("group_id", "")).strip()
        if not inferred_group:
            inferred_group = extract_measurement_date_from_run_id(run_id)
        if group_id_override:
            inferred_group = str(group_id_override).strip()
        return sorted(set(runs)), inferred_group

    if group_id_override:
        gid = str(group_id_override).strip()
        if not gid:
            gid = extract_measurement_date_from_run_id(run_id)
        runs = [rid for rid, meta in table.items() if str(meta.get("group_id", "")).strip() == gid and bool(meta.get("include", True))]
        if not runs:
            runs = [run_id]
        return sorted(set(runs)), gid

    if table and run_id in table:
        gid = str(table[run_id].get("group_id", "")).strip() or extract_measurement_date_from_run_id(run_id)
        runs = [rid for rid, meta in table.items() if str(meta.get("group_id", "")).strip() == gid and bool(meta.get("include", True))]
        # If anchor run itself is excluded in the table, still keep it to avoid surprising no-op.
        if run_id not in runs:
            runs.append(run_id)
        return sorted(set(runs)), gid

    # Fallback (no run group table or anchor missing in table): anchor run only.
    gid = extract_measurement_date_from_run_id(run_id)
    return [run_id], gid


# Backward-compat helper (used by tests / older references)
def _load_same_date_include_map(path: Path | None) -> dict[str, bool]:
    table = _load_run_group_table(path)
    return {rid: bool(meta.get("include", True)) for rid, meta in table.items()}


# Backward-compat helper (used by tests / older references)
def _collect_same_date_runs(
    *,
    run_id: str,
    processed_dir: Path,
    explicit_runs: list[str] | None,
    same_date_include_map: dict[str, bool] | None = None,
) -> list[str]:
    del processed_dir  # no longer used in group-based collection
    table = {}
    for rid, inc in (same_date_include_map or {}).items():
        table[str(rid).strip()] = {
            "group_id": extract_measurement_date_from_run_id(str(rid)),
            "include": bool(inc),
        }
    runs, _gid = _collect_group_runs(
        run_id=run_id,
        explicit_runs=explicit_runs,
        run_group_table=table,
        group_id_override=None,
    )
    return runs


def _existing_paths_for_runs(processed_dir: Path, run_ids: list[str]) -> tuple[list[Path], list[Path]]:
    summary_paths: list[Path] = []
    fog_paths: list[Path] = []
    for rid in run_ids:
        s = processed_dir / rid / "fit" / "summary_simple.csv"
        f = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
        if s.is_file():
            summary_paths.append(s)
        if f.is_file():
            fog_paths.append(f)
    return summary_paths, fog_paths


def _normalize_polymer_token(value: object) -> str:
    return str(value).strip().upper()


def _is_excluded_run_top_polymer(polymer_id: object, *, reference_polymer_id: str = "GOX") -> bool:
    """
    Exclusion rule for run-top3 bars.
    - Exclude reference polymer itself (e.g., GOX)
    - Exclude GOX solvent-control variants (e.g., "GOx with DMSO", "GOx with EtOH")
    """
    pid = str(polymer_id).strip()
    if not pid:
        return True
    pid_norm = _normalize_polymer_token(pid)
    ref_norm = _normalize_polymer_token(reference_polymer_id)
    if pid_norm == ref_norm:
        return True
    if pid_norm.startswith(f"{ref_norm} WITH "):
        return True
    if pid_norm.startswith(f"{ref_norm}_WITH_"):
        return True
    return False


def _robust_outlier_mask(
    values: np.ndarray,
    *,
    min_samples: int = 4,
    z_threshold: float = 3.5,
    ratio_low: float | None = 0.33,
    ratio_high: float | None = 3.0,
    min_keep: int = 2,
) -> np.ndarray:
    """MAD-based outlier detection with optional multiplicative guard around median."""
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
    if np.isfinite(scale) and scale > 1e-12:
        z_mask = abs_dev > float(z_threshold) * scale

    ratio_mask = np.zeros(n, dtype=bool)
    rl = None if ratio_low is None else float(ratio_low)
    rh = None if ratio_high is None else float(ratio_high)
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


def _aggregate_same_date_fog(
    *,
    run_ids: list[str],
    processed_dir: Path,
    group_run_id: str,
    t50_definition: str,
    reference_polymer_id: str = "GOX",
    apply_outlier_filter: bool = True,
    outlier_min_runs: int = 4,
    outlier_z_threshold: float = 3.5,
    outlier_ratio_low: float = 0.33,
    outlier_ratio_high: float = 3.0,
    reference_abs0_outlier_low: float = 0.50,
    reference_abs0_outlier_high: float = 2.00,
    outlier_min_keep: int = 2,
    outlier_events: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rid in run_ids:
        p = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df = df.copy()
        df["source_run_id"] = rid
        if "t50_definition" in df.columns:
            df["_t50_norm"] = df["t50_definition"].map(_normalize_t50_value)
            matched = df["_t50_norm"] == t50_definition
            if matched.any():
                df = df.loc[matched].copy()
            else:
                continue
            df["t50_definition"] = df["_t50_norm"]
            df = df.drop(columns=["_t50_norm"])
        else:
            df["t50_definition"] = t50_definition
        rows.append(df)

    base_cols = [
        "run_id",
        "polymer_id",
        "t50_min",
        "t50_sem_min",
        "t50_censored",
        "fog",
        "fog_sem",
        "log_fog",
        "fog_native_constrained",
        "fog_native_constrained_sem",
        "log_fog_native_constrained",
        "native_activity_rel_at_0",
        "native_activity_rel_at_0_sem",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "functional_activity_at_20_rel",
        "functional_activity_at_20_rel_sem",
        "abs_activity_at_0",
        "abs_activity_at_0_sem",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0_ref_sem",
        "abs_activity_at_20",
        "abs_activity_at_20_sem",
        "gox_abs_activity_at_20_ref",
        "gox_abs_activity_at_20_ref_sem",
        "native_0",
        "t50_definition",
        "source_run_ids",
        "n_source_runs",
        "n_t50",
        "n_fog",
        "n_fog_native_constrained",
        "n_functional",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "use_for_bo",
        "fog_constraint_reason",
        "fog_missing_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=base_cols)

    combined = pd.concat(rows, ignore_index=True)
    combined["polymer_id"] = combined["polymer_id"].astype(str).str.strip()
    combined = combined[combined["polymer_id"].ne("")].copy()
    if combined.empty:
        return pd.DataFrame(columns=base_cols)

    combined["t50_min"] = pd.to_numeric(combined.get("t50_min", np.nan), errors="coerce")
    combined["fog"] = pd.to_numeric(combined.get("fog", np.nan), errors="coerce")
    combined["log_fog"] = pd.to_numeric(combined.get("log_fog", np.nan), errors="coerce")
    combined["fog_native_constrained"] = pd.to_numeric(
        combined.get("fog_native_constrained", np.nan), errors="coerce"
    )
    combined["log_fog_native_constrained"] = pd.to_numeric(
        combined.get("log_fog_native_constrained", np.nan), errors="coerce"
    )
    combined["abs_activity_at_0"] = pd.to_numeric(combined.get("abs_activity_at_0", np.nan), errors="coerce")
    combined["gox_abs_activity_at_0_ref"] = pd.to_numeric(
        combined.get("gox_abs_activity_at_0_ref", np.nan), errors="coerce"
    )
    combined["abs_activity_at_20"] = pd.to_numeric(combined.get("abs_activity_at_20", np.nan), errors="coerce")
    combined["gox_abs_activity_at_20_ref"] = pd.to_numeric(
        combined.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce"
    )
    combined["functional_activity_at_20_rel"] = pd.to_numeric(
        combined.get("functional_activity_at_20_rel", np.nan), errors="coerce"
    )
    combined["native_activity_rel_at_0"] = pd.to_numeric(
        combined.get("native_activity_rel_at_0", np.nan), errors="coerce"
    )
    combined["native_activity_min_rel_threshold"] = pd.to_numeric(
        combined.get("native_activity_min_rel_threshold", np.nan), errors="coerce"
    )
    combined["native_activity_feasible"] = pd.to_numeric(
        combined.get("native_activity_feasible", np.nan), errors="coerce"
    )
    combined["rea_at_20_percent"] = pd.to_numeric(combined.get("rea_at_20_percent", np.nan), errors="coerce")
    combined["t50_target_rea_percent"] = pd.to_numeric(combined.get("t50_target_rea_percent", np.nan), errors="coerce")
    if "fog_constraint_reason" in combined.columns:
        combined["fog_constraint_reason"] = combined["fog_constraint_reason"].astype(str)
    else:
        combined["fog_constraint_reason"] = ""
    if "use_for_bo" not in combined.columns:
        combined["use_for_bo"] = True

    # Run-level reference abs(0) outlier guard.
    # If reference baseline is extreme in a run, exclude that run from grouped FoG aggregation.
    if apply_outlier_filter:
        ref_norm = _normalize_polymer_token(reference_polymer_id)
        combined["_polymer_norm"] = combined["polymer_id"].map(_normalize_polymer_token)
        ref_rows = combined[
            (combined["_polymer_norm"] == ref_norm)
            & np.isfinite(combined["abs_activity_at_0"])
            & (combined["abs_activity_at_0"] > 0.0)
        ].copy()
        excluded_ref_runs: set[str] = set()
        if not ref_rows.empty:
            ref_per_run = (
                ref_rows.groupby("source_run_id", as_index=False)
                .agg(reference_abs0=("abs_activity_at_0", "median"))
            )
            ref_vals = ref_per_run["reference_abs0"].to_numpy(dtype=float)
            ref_min_samples = max(
                2,
                min(int(outlier_min_runs), int(np.count_nonzero(np.isfinite(ref_vals)))),
            )
            ref_mask = _robust_outlier_mask(
                ref_vals,
                min_samples=ref_min_samples,
                z_threshold=outlier_z_threshold,
                ratio_low=reference_abs0_outlier_low,
                ratio_high=reference_abs0_outlier_high,
                min_keep=max(2, int(outlier_min_keep)),
            )
            if np.any(ref_mask):
                excluded_ref_runs = {
                    str(x).strip()
                    for x in ref_per_run.loc[ref_mask, "source_run_id"].tolist()
                    if str(x).strip()
                }
                if excluded_ref_runs:
                    combined = combined[
                        ~combined["source_run_id"].astype(str).isin(excluded_ref_runs)
                    ].copy()
                    if outlier_events is not None:
                        for _, rr in ref_per_run.loc[ref_mask].iterrows():
                            outlier_events.append(
                                {
                                    "event_type": "reference_run_excluded",
                                    "group_run_id": group_run_id,
                                    "source_run_id": str(rr["source_run_id"]).strip(),
                                    "polymer_id": str(reference_polymer_id),
                                    "metric": "abs_activity_at_0",
                                    "value": float(rr["reference_abs0"]),
                                }
                            )
        if "_polymer_norm" in combined.columns:
            combined = combined.drop(columns=["_polymer_norm"])
        if combined.empty:
            return pd.DataFrame(columns=base_cols)

    out_rows: list[dict[str, object]] = []
    for pid, g in combined.groupby("polymer_id", sort=True):
        pid = str(pid).strip()
        if not pid:
            continue

        g = g.copy()
        # Polymer-level outlier guard on FoG objective values within the run group.
        if apply_outlier_filter:
            score_col: str | None = None
            score_vals = pd.Series(dtype=float)
            for cand in ["log_fog_native_constrained", "log_fog", "fog_native_constrained", "fog"]:
                if cand not in g.columns:
                    continue
                v = pd.to_numeric(g[cand], errors="coerce")
                if cand in {"fog_native_constrained", "fog"}:
                    v = v.where(v > 0.0)
                if int(np.count_nonzero(np.isfinite(v.to_numpy(dtype=float)))) >= max(1, int(outlier_min_runs)):
                    score_col = cand
                    score_vals = v
                    break
            if score_col is not None:
                score_arr = score_vals.to_numpy(dtype=float)
                finite_idx = np.flatnonzero(np.isfinite(score_arr))
                score_for_mask = score_arr[finite_idx]
                score_mask = _robust_outlier_mask(
                    score_for_mask,
                    min_samples=outlier_min_runs,
                    z_threshold=outlier_z_threshold,
                    ratio_low=(None if score_col.startswith("log_") else outlier_ratio_low),
                    ratio_high=(None if score_col.startswith("log_") else outlier_ratio_high),
                    min_keep=max(2, int(outlier_min_keep)),
                )
                if np.any(score_mask):
                    to_drop_idx = g.index[finite_idx[score_mask]]
                    if (len(g) - len(to_drop_idx)) >= max(2, int(outlier_min_keep)):
                        if outlier_events is not None:
                            dropped = g.loc[to_drop_idx, ["source_run_id"]].copy()
                            dropped["_score"] = score_arr[finite_idx[score_mask]]
                            for _, rr in dropped.iterrows():
                                outlier_events.append(
                                    {
                                        "event_type": "polymer_metric_outlier_excluded",
                                        "group_run_id": group_run_id,
                                        "source_run_id": str(rr["source_run_id"]).strip(),
                                        "polymer_id": pid,
                                        "metric": score_col,
                                        "value": float(rr["_score"]),
                                    }
                                )
                        g = g.drop(index=to_drop_idx).copy()
        if g.empty:
            continue

        src_runs = sorted({str(x).strip() for x in g["source_run_id"].tolist() if str(x).strip()})
        t50_vals = g["t50_min"]
        t50_vals = t50_vals[np.isfinite(t50_vals) & (t50_vals > 0)]
        fog_vals = g["fog"]
        fog_vals = fog_vals[np.isfinite(fog_vals) & (fog_vals > 0)]
        fog_native_vals = g["fog_native_constrained"]
        fog_native_vals = fog_native_vals[np.isfinite(fog_native_vals) & (fog_native_vals > 0)]
        rea20_vals = g["rea_at_20_percent"]
        rea20_vals = rea20_vals[np.isfinite(rea20_vals)]
        tgt_vals = g["t50_target_rea_percent"]
        tgt_vals = tgt_vals[np.isfinite(tgt_vals)]
        func_vals = g["functional_activity_at_20_rel"]
        func_vals = func_vals[np.isfinite(func_vals) & (func_vals > 0)]
        native_vals = g["native_activity_rel_at_0"]
        native_vals = native_vals[np.isfinite(native_vals)]
        native_thr_vals = g["native_activity_min_rel_threshold"]
        native_thr_vals = native_thr_vals[np.isfinite(native_thr_vals)]
        abs0_vals = g["abs_activity_at_0"]
        abs0_vals = abs0_vals[np.isfinite(abs0_vals) & (abs0_vals > 0)]
        gox_abs0_vals = g["gox_abs_activity_at_0_ref"]
        gox_abs0_vals = gox_abs0_vals[np.isfinite(gox_abs0_vals) & (gox_abs0_vals > 0)]
        abs20_vals = g["abs_activity_at_20"]
        abs20_vals = abs20_vals[np.isfinite(abs20_vals) & (abs20_vals > 0)]
        gox_abs20_vals = g["gox_abs_activity_at_20_ref"]
        gox_abs20_vals = gox_abs20_vals[np.isfinite(gox_abs20_vals) & (gox_abs20_vals > 0)]
        fog_reason = ""
        if "fog_constraint_reason" in g.columns:
            reasons = [str(x).strip() for x in g["fog_constraint_reason"].tolist() if str(x).strip()]
            fog_reason = ";".join(sorted(set(reasons)))
        use_flags = g["use_for_bo"].fillna(True).astype(bool)

        mean_t50 = float(t50_vals.mean()) if len(t50_vals) > 0 else np.nan
        mean_fog = float(fog_vals.mean()) if len(fog_vals) > 0 else np.nan
        mean_fog_native = float(fog_native_vals.mean()) if len(fog_native_vals) > 0 else np.nan
        mean_native = float(native_vals.mean()) if len(native_vals) > 0 else np.nan
        mean_native_thr = float(native_thr_vals.mean()) if len(native_thr_vals) > 0 else np.nan
        native_feasible = (
            int(mean_native >= mean_native_thr)
            if (np.isfinite(mean_native) and np.isfinite(mean_native_thr))
            else 0
        )
        out_rows.append(
            {
                "run_id": group_run_id,
                "polymer_id": pid,
                "t50_min": mean_t50,
                "t50_sem_min": float(t50_vals.sem()) if len(t50_vals) > 1 else np.nan,
                "t50_censored": int(0 if np.isfinite(mean_t50) else 1),
                "fog": mean_fog,
                "fog_sem": float(fog_vals.sem()) if len(fog_vals) > 1 else np.nan,
                "log_fog": float(np.log(mean_fog)) if np.isfinite(mean_fog) and mean_fog > 0 else np.nan,
                "fog_native_constrained": mean_fog_native,
                "fog_native_constrained_sem": float(fog_native_vals.sem()) if len(fog_native_vals) > 1 else np.nan,
                "log_fog_native_constrained": (
                    float(np.log(mean_fog_native))
                    if np.isfinite(mean_fog_native) and mean_fog_native > 0
                    else np.nan
                ),
                "native_activity_rel_at_0": mean_native,
                "native_activity_rel_at_0_sem": float(native_vals.sem()) if len(native_vals) > 1 else np.nan,
                "native_activity_min_rel_threshold": mean_native_thr,
                "native_activity_feasible": native_feasible,
                "functional_activity_at_20_rel": float(func_vals.mean()) if len(func_vals) > 0 else np.nan,
                "functional_activity_at_20_rel_sem": float(func_vals.sem()) if len(func_vals) > 1 else np.nan,
                "abs_activity_at_0": float(abs0_vals.mean()) if len(abs0_vals) > 0 else np.nan,
                "abs_activity_at_0_sem": float(abs0_vals.sem()) if len(abs0_vals) > 1 else np.nan,
                "gox_abs_activity_at_0_ref": float(gox_abs0_vals.mean()) if len(gox_abs0_vals) > 0 else np.nan,
                "gox_abs_activity_at_0_ref_sem": float(gox_abs0_vals.sem()) if len(gox_abs0_vals) > 1 else np.nan,
                "abs_activity_at_20": float(abs20_vals.mean()) if len(abs20_vals) > 0 else np.nan,
                "abs_activity_at_20_sem": float(abs20_vals.sem()) if len(abs20_vals) > 1 else np.nan,
                "gox_abs_activity_at_20_ref": float(gox_abs20_vals.mean()) if len(gox_abs20_vals) > 0 else np.nan,
                "gox_abs_activity_at_20_ref_sem": float(gox_abs20_vals.sem()) if len(gox_abs20_vals) > 1 else np.nan,
                "native_0": mean_native,
                "t50_definition": t50_definition,
                "source_run_ids": "|".join(src_runs),
                "n_source_runs": int(len(src_runs)),
                "n_t50": int(len(t50_vals)),
                "n_fog": int(len(fog_vals)),
                "n_fog_native_constrained": int(len(fog_native_vals)),
                "n_functional": int(len(func_vals)),
                "t50_target_rea_percent": float(tgt_vals.mean()) if len(tgt_vals) > 0 else np.nan,
                "rea_at_20_percent": float(rea20_vals.mean()) if len(rea20_vals) > 0 else np.nan,
                "use_for_bo": bool(use_flags.all()) if len(use_flags) > 0 else True,
                "fog_constraint_reason": fog_reason,
                "fog_missing_reason": "" if len(fog_vals) > 0 else "missing_in_all_source_runs",
            }
        )
    return pd.DataFrame(out_rows, columns=base_cols)


def _collect_run_top_t50(
    *,
    run_ids: list[str],
    processed_dir: Path,
    group_run_id: str,
    t50_definition: str,
    reference_polymer_id: str = "GOX",
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Collect top-N polymers by t50 for each source run.

    Returns one row per (source_run_id, rank_in_source_run) with:
      run_id (group output run_id), source_run_id, rank_in_source_run,
      polymer_id, t50_min, t50_definition.
    """
    n_top = max(1, int(top_n))
    rows: list[dict[str, object]] = []
    for rid in run_ids:
        p = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if df.empty or "polymer_id" not in df.columns:
            continue
        if "t50_definition" in df.columns:
            df = df.copy()
            df["_t50_norm"] = df["t50_definition"].map(_normalize_t50_value)
            matched = df["_t50_norm"] == t50_definition
            if matched.any():
                df = df.loc[matched].copy()
                df["t50_definition"] = df["_t50_norm"]
            else:
                continue
            df = df.drop(columns=["_t50_norm"])
        else:
            df = df.copy()
            df["t50_definition"] = t50_definition

        df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
        df = df[df["polymer_id"].ne("")].copy()
        if df.empty:
            continue
        df = df[
            ~df["polymer_id"].map(
                lambda x: _is_excluded_run_top_polymer(x, reference_polymer_id=reference_polymer_id)
            )
        ].copy()
        if df.empty:
            continue
        df["t50_min"] = pd.to_numeric(df.get("t50_min", np.nan), errors="coerce")
        df = df[np.isfinite(df["t50_min"]) & (df["t50_min"] > 0.0)].copy()
        if df.empty:
            continue
        df = df.sort_values(["t50_min", "polymer_id"], ascending=[False, True], kind="mergesort")
        df = df.head(n_top).reset_index(drop=True)
        df["rank_in_source_run"] = np.arange(1, len(df) + 1, dtype=int)
        for _, row in df.iterrows():
            rows.append(
                {
                    "run_id": group_run_id,
                    "source_run_id": rid,
                    "rank_in_source_run": int(row["rank_in_source_run"]),
                    "polymer_id": str(row["polymer_id"]).strip(),
                    "t50_min": float(row["t50_min"]),
                    "t50_definition": str(row.get("t50_definition", t50_definition)).strip() or t50_definition,
                }
            )
    out_cols = [
        "run_id",
        "source_run_id",
        "rank_in_source_run",
        "polymer_id",
        "t50_min",
        "t50_definition",
    ]
    return pd.DataFrame(rows, columns=out_cols)


def _plot_run_top_t50(
    top_t50_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map_path: Path,
    top_n: int = 3,
) -> bool:
    """
    Plot one grouped bar chart: top-N t50 values for each source run.
    """
    if top_t50_df.empty:
        return False
    data = top_t50_df.copy()
    required = {"source_run_id", "rank_in_source_run", "polymer_id", "t50_min"}
    if not required.issubset(set(data.columns)):
        return False
    data["source_run_id"] = data["source_run_id"].astype(str).str.strip()
    data["polymer_id"] = data["polymer_id"].astype(str).str.strip()
    data["rank_in_source_run"] = pd.to_numeric(data["rank_in_source_run"], errors="coerce")
    data["t50_min"] = pd.to_numeric(data["t50_min"], errors="coerce")
    data = data[
        data["source_run_id"].ne("")
        & data["polymer_id"].ne("")
        & np.isfinite(data["rank_in_source_run"])
        & np.isfinite(data["t50_min"])
        & (data["t50_min"] > 0.0)
    ].copy()
    if data.empty:
        return False

    n_top = max(1, int(top_n))
    data["rank_in_source_run"] = data["rank_in_source_run"].astype(int)
    data = data[data["rank_in_source_run"] <= n_top].copy()
    if data.empty:
        return False

    run_order = sorted(data["source_run_id"].unique().tolist())
    if not run_order:
        return False
    run_pos = {rid: i for i, rid in enumerate(run_order)}
    color_map = load_or_create_polymer_color_map(Path(color_map_path))
    default_color = "#4C78A8"

    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig_w = max(5.2, 0.62 * len(run_order) + 1.8)
    fig_h = 3.2
    bar_w = 0.78 / float(n_top)
    rank_offsets = {
        rank: (rank - (n_top + 1) / 2.0) * bar_w
        for rank in range(1, n_top + 1)
    }
    rank_alpha = {rank: max(0.55, 1.0 - 0.12 * (rank - 1)) for rank in range(1, n_top + 1)}

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ymax = float(np.nanmax(data["t50_min"].to_numpy(dtype=float)))
        y_lim = ymax * 1.40 if np.isfinite(ymax) and ymax > 0 else 1.0
        text_offset = y_lim * 0.03
        for rank in range(1, n_top + 1):
            subset = data[data["rank_in_source_run"] == rank].copy()
            if subset.empty:
                continue
            for _, row in subset.iterrows():
                rid = str(row["source_run_id"])
                x = float(run_pos[rid]) + float(rank_offsets[rank])
                value = float(row["t50_min"])
                pid = str(row["polymer_id"]).strip()
                color = str(color_map.get(pid, default_color))
                ax.bar(
                    x,
                    value,
                    width=bar_w * 0.92,
                    color=color,
                    edgecolor="0.2",
                    linewidth=0.45,
                    alpha=rank_alpha.get(rank, 1.0),
                    zorder=12,
                )
                ax.text(
                    x,
                    value + text_offset,
                    pid,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color="0.15",
                    zorder=18,
                    clip_on=False,
                )
        ax.set_ylim(0.0, y_lim)
        ax.set_xticks(np.arange(len(run_order), dtype=float))
        ax.set_xticklabels(run_order, rotation=35, ha="right")
        ax.set_xlabel("Source run ID")
        ax.set_ylabel(r"$t_{50}$ (min)")
        ax.set_title(f"Top {n_top} polymers by source run")
        ax.grid(axis="y", linestyle=":", alpha=0.30, zorder=1)
        legend_handles = [
            Patch(facecolor="0.35", edgecolor="0.2", alpha=rank_alpha.get(rank, 1.0), label=f"Rank {rank}")
            for rank in range(1, n_top + 1)
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            ncol=min(3, n_top),
        )
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _collect_all_round_runs(run_round_map_path: Path) -> list[str]:
    run_round_map = load_run_round_map(Path(run_round_map_path))
    return sorted({str(rid).strip() for rid in run_round_map.keys() if str(rid).strip()})


def _run_group_summary(
    *,
    anchor_run_id: str,
    plot_run_id: str,
    resolved_group_id: str,
    group_run_id: str,
    candidate_runs: list[str],
    processed_dir: Path,
    output_root: Path,
    t50_definition: str,
    reference_polymer_id: str,
    apply_outlier_filter: bool,
    outlier_min_runs: int,
    outlier_min_keep: int,
    outlier_z_threshold: float,
    outlier_ratio_low: float,
    outlier_ratio_high: float,
    reference_abs0_outlier_low: float,
    reference_abs0_outlier_high: float,
    dpi: int,
    run_group_tsv_path: Path | None,
    run_group_table: dict[str, dict[str, object]],
    group_scope: str,
    group_scope_detail: str,
    debug: bool = False,
    dry_run: bool = False,
) -> None:
    ranking_dir = output_root / "ranking"
    ranking_csv_dir = ranking_dir / "csv"
    plot_root = output_root / "plots"
    plot_group_label = _safe_group_stem(resolved_group_id)
    expected_plot_dir_name = f"per_polymer_across_runs__{plot_group_label}__{plot_run_id}"

    summary_paths, fog_paths = _existing_paths_for_runs(processed_dir, candidate_runs)
    runs_for_plot = sorted({p.parents[1].name for p in summary_paths})
    runs_for_fog = sorted({p.parents[1].name for p in fog_paths})

    if debug or dry_run:
        print("anchor run_id:", anchor_run_id)
        print("plot run_id:", plot_run_id)
        print("group_id:", resolved_group_id)
        print("output run_id:", group_run_id)
        print("group scope:", group_scope)
        if str(group_scope_detail).strip():
            print("group scope detail:", group_scope_detail)
        print("candidate runs:", candidate_runs)
        if run_group_tsv_path is not None:
            print("run-group TSV:", run_group_tsv_path)
            print("run-group entries:", len(run_group_table))
        else:
            print("run-group TSV: ignored")
        print("runs with summary_simple:", runs_for_plot)
        print("runs with fog_summary:", runs_for_fog)
        print("t50 definition:", t50_definition)
        print("reference polymer id:", reference_polymer_id)
        print("apply outlier filter:", apply_outlier_filter)
        print("outlier min runs:", int(outlier_min_runs))
        print("outlier min keep:", int(outlier_min_keep))
        print("outlier z threshold:", float(outlier_z_threshold))
        print("outlier ratio bounds:", (float(outlier_ratio_low), float(outlier_ratio_high)))
        print(
            "reference abs0 ratio bounds:",
            (float(reference_abs0_outlier_low), float(reference_abs0_outlier_high)),
        )
        print("output root:", output_root)

    if dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    ranking_dir.mkdir(parents=True, exist_ok=True)
    ranking_csv_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale per_polymer_across_runs directories for the same group label.
    # This prevents confusion when anchor run_id changes and old folders remain side-by-side.
    for old_plot_dir in plot_root.glob(f"per_polymer_across_runs__{plot_group_label}__*"):
        if not old_plot_dir.is_dir():
            continue
        if old_plot_dir.name == expected_plot_dir_name:
            continue
        shutil.rmtree(old_plot_dir, ignore_errors=True)

    # 1) Per-polymer plots across grouped runs (mean fit + SEM error bars)
    across_plot_dir = plot_per_polymer_timeseries_across_runs_with_error_bars(
        run_id=plot_run_id,
        processed_dir=processed_dir,
        out_fit_dir=plot_root,
        color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
        same_date_runs=runs_for_plot,
        group_label=plot_group_label,
        reference_polymer_id=reference_polymer_id,
        apply_outlier_filter=apply_outlier_filter,
        outlier_min_runs=int(outlier_min_runs),
        outlier_z_threshold=float(outlier_z_threshold),
        outlier_ratio_low=float(outlier_ratio_low),
        outlier_ratio_high=float(outlier_ratio_high),
        reference_abs0_outlier_low=float(reference_abs0_outlier_low),
        reference_abs0_outlier_high=float(reference_abs0_outlier_high),
        outlier_min_keep=int(outlier_min_keep),
        dpi=int(dpi),
    )
    if across_plot_dir is None:
        print("Skipped grouped plots: need at least 2 runs with summary_simple.csv.")
    else:
        print(f"Saved (grouped plots): {across_plot_dir}")
        fog_panel_dir = across_plot_dir / f"rea_comparison_fog_panel__{plot_group_label}"
        fog_grid_png = across_plot_dir / f"rea_comparison_fog_grid__{plot_group_label}.png"
        if fog_panel_dir.is_dir():
            print(f"Saved (grouped REA+FoG panels): {fog_panel_dir}")
        if fog_grid_png.is_file():
            print(f"Saved (grouped REA+FoG 5-col grid): {fog_grid_png}")

    # 2) Mean t50/FoG aggregation and ranking outputs
    outlier_events: list[dict[str, object]] = []
    agg_fog_df = _aggregate_same_date_fog(
        run_ids=runs_for_fog,
        processed_dir=processed_dir,
        group_run_id=group_run_id,
        t50_definition=t50_definition,
        reference_polymer_id=reference_polymer_id,
        apply_outlier_filter=apply_outlier_filter,
        outlier_min_runs=int(outlier_min_runs),
        outlier_z_threshold=float(outlier_z_threshold),
        outlier_ratio_low=float(outlier_ratio_low),
        outlier_ratio_high=float(outlier_ratio_high),
        reference_abs0_outlier_low=float(reference_abs0_outlier_low),
        reference_abs0_outlier_high=float(reference_abs0_outlier_high),
        outlier_min_keep=int(outlier_min_keep),
        outlier_events=outlier_events,
    )
    fog_summary_path = output_root / f"fog_summary__{group_run_id}.csv"
    agg_fog_df.to_csv(fog_summary_path, index=False)
    print(f"Saved (grouped mean FoG summary): {fog_summary_path}")

    outlier_report_path = output_root / f"outlier_constraints__{group_run_id}.csv"
    outlier_report_cols = [
        "event_type",
        "group_run_id",
        "source_run_id",
        "polymer_id",
        "metric",
        "value",
    ]
    outlier_report_df = pd.DataFrame(outlier_events, columns=outlier_report_cols)
    outlier_report_df.to_csv(outlier_report_path, index=False)
    print(f"Saved (outlier constraints report): {outlier_report_path}")

    top_t50_csv = ranking_csv_dir / f"run_top3_t50__{group_run_id}.csv"
    legacy_top_t50_csv = ranking_dir / f"run_top3_t50__{group_run_id}.csv"
    top_t50_png = ranking_dir / f"run_top3_t50__{group_run_id}.png"
    top_t50_grid_png = ranking_dir / f"run_top3_t50_grid__{group_run_id}.png"
    run_top_t50_df = _collect_run_top_t50(
        run_ids=runs_for_fog,
        processed_dir=processed_dir,
        group_run_id=group_run_id,
        t50_definition=t50_definition,
        reference_polymer_id=reference_polymer_id,
        top_n=3,
    )
    run_top_t50_df.to_csv(top_t50_csv, index=False)
    if legacy_top_t50_csv.is_file():
        legacy_top_t50_csv.unlink(missing_ok=True)
    print(f"Saved (run-wise top3 t50 table): {top_t50_csv}")
    wrote_top_t50_png = _plot_run_top_t50(
        run_top_t50_df,
        out_path=top_t50_png,
        color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
        top_n=3,
    )
    if wrote_top_t50_png:
        print(f"Saved (run-wise top3 t50 plot): {top_t50_png}")
    elif top_t50_png.exists():
        top_t50_png.unlink(missing_ok=True)
        print("Skipped run-wise top3 t50 plot: no plottable rows.")
    else:
        print("Skipped run-wise top3 t50 plot: no plottable rows.")
    if top_t50_grid_png.exists():
        top_t50_grid_png.unlink(missing_ok=True)
        print(f"Removed stale run-wise top3 t50 grid (disabled): {top_t50_grid_png}")

    if not agg_fog_df.empty:
        ranking_outputs = write_run_ranking_outputs(
            fog_df=agg_fog_df,
            run_id=group_run_id,
            out_dir=ranking_dir,
            color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
            reference_polymer_id=reference_polymer_id,
        )
        if "t50_ranking_csv" in ranking_outputs:
            print(f"Saved (mean t50 ranking): {ranking_outputs['t50_ranking_csv']}")
        if "fog_ranking_csv" in ranking_outputs:
            print(f"Saved (mean FoG ranking): {ranking_outputs['fog_ranking_csv']}")
        if "t50_ranking_png" in ranking_outputs:
            print(f"Saved (mean t50 ranking plot): {ranking_outputs['t50_ranking_png']}")
        if "fog_ranking_png" in ranking_outputs:
            print(f"Saved (mean FoG ranking plot): {ranking_outputs['fog_ranking_png']}")
    else:
        print("No rows were aggregated for grouped mean FoG; ranking outputs were skipped.")

    used_runs_path = output_root / f"source_runs__{group_run_id}.txt"
    with open(used_runs_path, "w", encoding="utf-8") as f:
        f.write("group_id\t" + str(resolved_group_id) + "\n")
        f.write("group_scope\t" + str(group_scope) + "\n")
        f.write("group_scope_detail\t" + str(group_scope_detail) + "\n")
        f.write("candidate_runs\t" + ",".join(candidate_runs) + "\n")
        f.write("runs_for_plot\t" + ",".join(runs_for_plot) + "\n")
        f.write("runs_for_fog\t" + ",".join(runs_for_fog) + "\n")
        f.write("t50_definition\t" + t50_definition + "\n")
        f.write("apply_outlier_filter\t" + str(apply_outlier_filter) + "\n")
        f.write("outlier_min_runs\t" + str(int(outlier_min_runs)) + "\n")
        f.write("outlier_min_keep\t" + str(int(outlier_min_keep)) + "\n")
        f.write("outlier_z_threshold\t" + str(float(outlier_z_threshold)) + "\n")
        f.write("outlier_ratio_low\t" + str(float(outlier_ratio_low)) + "\n")
        f.write("outlier_ratio_high\t" + str(float(outlier_ratio_high)) + "\n")
        f.write("reference_abs0_outlier_low\t" + str(float(reference_abs0_outlier_low)) + "\n")
        f.write("reference_abs0_outlier_high\t" + str(float(reference_abs0_outlier_high)) + "\n")
    print(f"Saved (source runs): {used_runs_path}")

    manifest_inputs: list[Path] = []
    manifest_inputs.extend(summary_paths)
    manifest_inputs.extend(fog_paths)
    manifest = build_run_manifest_dict(
        group_run_id,
        manifest_inputs,
        git_root=REPO_ROOT,
        extra={
            "anchor_run_id": anchor_run_id,
            "group_id": str(resolved_group_id),
            "group_scope": str(group_scope),
            "group_scope_detail": str(group_scope_detail),
            "candidate_runs": candidate_runs,
            "run_group_tsv": (str(run_group_tsv_path.resolve()) if run_group_tsv_path and run_group_tsv_path.exists() else ""),
            "runs_for_plot": runs_for_plot,
            "runs_for_fog": runs_for_fog,
            "t50_definition": t50_definition,
            "reference_polymer_id": reference_polymer_id,
            "apply_outlier_filter": apply_outlier_filter,
            "outlier_constraints": {
                "min_runs": int(outlier_min_runs),
                "min_keep": int(outlier_min_keep),
                "z_threshold": float(outlier_z_threshold),
                "ratio_low": float(outlier_ratio_low),
                "ratio_high": float(outlier_ratio_high),
                "reference_abs0_ratio_low": float(reference_abs0_outlier_low),
                "reference_abs0_ratio_high": float(reference_abs0_outlier_high),
                "n_events": int(len(outlier_events)),
            },
            "outputs": {
                "fog_summary": str(fog_summary_path.resolve()),
                "ranking_dir": str(ranking_dir.resolve()),
                "plots_dir": str(across_plot_dir.resolve()) if across_plot_dir is not None else "",
                "rea_fog_panel_dir": (
                    str((across_plot_dir / f"rea_comparison_fog_panel__{plot_group_label}").resolve())
                    if across_plot_dir is not None
                    else ""
                ),
                "rea_fog_grid_png": (
                    str((across_plot_dir / f"rea_comparison_fog_grid__{plot_group_label}.png").resolve())
                    if across_plot_dir is not None
                    else ""
                ),
                "outlier_constraints_report": str(outlier_report_path.resolve()),
                "run_top3_t50_csv": str(top_t50_csv.resolve()),
                "run_top3_t50_png": str(top_t50_png.resolve()) if wrote_top_t50_png else "",
            },
        },
    )
    manifest_path = output_root / f"run_manifest__{group_run_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved (manifest): {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Across run-group aggregation: per-polymer mean-fit plots with SEM error bars, "
            "and mean t50/FoG ranking bars."
        )
    )
    p.add_argument(
        "--run_id",
        required=False,
        default="",
        help="Anchor run_id. Required unless --all_rounds_only is set.",
    )
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed root containing {run_id}/fit outputs.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "across_runs",
        help="Output root for cross-run grouped outputs.",
    )
    p.add_argument(
        "--run_ids",
        nargs="*",
        default=None,
        help="Optional explicit run_id list. If omitted, runs are selected from run group TSV.",
    )
    p.add_argument(
        "--run_group_tsv",
        type=Path,
        default=REPO_ROOT / "meta" / "run_group_map.tsv",
        help=(
            "TSV to control run grouping and inclusion. "
            "Columns: run_id, group_id, include_in_group_mean."
        ),
    )
    p.add_argument(
        "--ignore_run_group_tsv",
        action="store_true",
        help="Ignore --run_group_tsv (then only --run_ids or anchor run is used).",
    )
    # Backward-compatible aliases (hidden)
    p.add_argument("--same_date_include_tsv", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ignore_same_date_include_tsv", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--group_id",
        default=None,
        help="Optional group_id override. Useful when selecting by group directly.",
    )
    p.add_argument(
        "--group_run_id",
        default=None,
        help="Optional output run_id label (default: {group_id}-group_mean).",
    )
    p.add_argument(
        "--run_round_map",
        type=Path,
        default=REPO_ROOT / "meta" / "bo_run_round_map.tsv",
        help="Path to run_id→round_id map. Used when --also_all_rounds/--all_rounds_only.",
    )
    p.add_argument(
        "--also_all_rounds",
        action="store_true",
        help="In addition to the selected group, also generate an all-rounds integrated summary.",
    )
    p.add_argument(
        "--all_rounds_only",
        action="store_true",
        help=(
            "Run only all-round-assigned summary mode (see --all_rounds_mode). "
            "Default mode is pooled across all valid round_id values."
        ),
    )
    p.add_argument(
        "--all_rounds_mode",
        type=str,
        default="pooled",
        choices=["pooled", "per_round", "both"],
        help=(
            "all-round-assigned summary mode: "
            "pooled=one integrated output across all valid rounds, "
            "per_round=one output per round_id, "
            "both=run pooled and per-round outputs."
        ),
    )
    p.add_argument(
        "--all_rounds_group_id",
        default="all_rounds",
        help=(
            "group_id label for pooled output when --all_rounds_mode is pooled/both "
            "(default: all_rounds)."
        ),
    )
    p.add_argument(
        "--all_rounds_group_run_id",
        default=None,
        help="Output run_id label for pooled output (default: {all_rounds_group_id}-group_mean).",
    )
    p.add_argument(
        "--per_round_group_prefix",
        default="round",
        help=(
            "group_id prefix used when --all_rounds_mode includes per_round. "
            "Per-round group_id becomes '{prefix}_{round_id}' (default: round)."
        ),
    )
    p.add_argument(
        "--t50_definition",
        type=str,
        default="y0_half",
        choices=["y0_half", "rea50"],
        help="t50 definition to aggregate from fog_summary files.",
    )
    p.add_argument(
        "--reference_polymer_id",
        type=str,
        default="GOX",
        help="Reference polymer ID used in ranking visualization/objective (default: GOX).",
    )
    p.add_argument(
        "--disable_outlier_filter",
        action="store_true",
        help="Disable robust outlier constraints for grouped mean plots/FoG aggregation.",
    )
    p.add_argument(
        "--outlier_min_runs",
        type=int,
        default=4,
        help="Minimum number of runs required before outlier filtering is applied (default: 4).",
    )
    p.add_argument(
        "--outlier_min_keep",
        type=int,
        default=2,
        help="Minimum runs to keep after outlier exclusion (default: 2).",
    )
    p.add_argument(
        "--outlier_z_threshold",
        type=float,
        default=3.5,
        help="MAD-based robust z-threshold for outlier detection (default: 3.5).",
    )
    p.add_argument(
        "--outlier_ratio_low",
        type=float,
        default=0.33,
        help="Lower multiplicative bound to median for outlier detection (default: 0.33).",
    )
    p.add_argument(
        "--outlier_ratio_high",
        type=float,
        default=3.0,
        help="Upper multiplicative bound to median for outlier detection (default: 3.0).",
    )
    p.add_argument(
        "--reference_abs0_outlier_low",
        type=float,
        default=0.5,
        help="Lower multiplicative bound for run-level reference abs(0) outlier guard (default: 0.5).",
    )
    p.add_argument(
        "--reference_abs0_outlier_high",
        type=float,
        default=2.0,
        help="Upper multiplicative bound for run-level reference abs(0) outlier guard (default: 2.0).",
    )
    p.add_argument("--dpi", type=int, default=600, help="PNG dpi.")
    p.add_argument("--dry_run", action="store_true", help="Print planned actions without writing files.")
    p.add_argument("--debug", action="store_true", help="Verbose logging.")
    args = p.parse_args()

    run_id = str(args.run_id).strip()
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    if not bool(args.all_rounds_only) and not run_id:
        raise ValueError("--run_id must be non-empty unless --all_rounds_only is set.")

    t50_definition = normalize_t50_definition(args.t50_definition)
    reference_polymer_id = str(args.reference_polymer_id).strip() or "GOX"
    apply_outlier_filter = not bool(args.disable_outlier_filter)
    if int(args.outlier_min_runs) < 2:
        raise ValueError("--outlier_min_runs must be >= 2.")
    if int(args.outlier_min_keep) < 2:
        raise ValueError("--outlier_min_keep must be >= 2.")
    if float(args.outlier_z_threshold) <= 0.0:
        raise ValueError("--outlier_z_threshold must be > 0.")
    if float(args.outlier_ratio_low) <= 0.0 or float(args.outlier_ratio_high) <= 0.0:
        raise ValueError("--outlier_ratio_low/high must be > 0.")
    if float(args.reference_abs0_outlier_low) <= 0.0 or float(args.reference_abs0_outlier_high) <= 0.0:
        raise ValueError("--reference_abs0_outlier_low/high must be > 0.")

    # Backward compatibility for old args
    ignore_group_tsv = bool(args.ignore_run_group_tsv) or bool(args.ignore_same_date_include_tsv)
    run_group_tsv_path = Path(args.run_group_tsv)
    if args.same_date_include_tsv is not None:
        run_group_tsv_path = Path(args.same_date_include_tsv)
    if ignore_group_tsv:
        run_group_tsv_path = None
    run_group_table = _load_run_group_table(run_group_tsv_path)

    if not bool(args.all_rounds_only):
        candidate_runs, resolved_group_id = _collect_group_runs(
            run_id=run_id,
            explicit_runs=args.run_ids,
            run_group_table=run_group_table,
            group_id_override=(str(args.group_id).strip() if args.group_id else None),
        )
        if not candidate_runs:
            raise ValueError("No candidate runs selected for grouped aggregation.")
        group_run_id = (
            str(args.group_run_id).strip()
            if args.group_run_id and str(args.group_run_id).strip()
            else f"{_safe_group_stem(resolved_group_id)}-group_mean"
        )
        if run_group_tsv_path is not None and run_group_tsv_path.is_file():
            group_scope = "group_runs_from_run_group_tsv"
            group_scope_detail = f"group_id={resolved_group_id}; source={run_group_tsv_path}"
        elif args.run_ids:
            group_scope = "explicit_run_ids"
            group_scope_detail = f"run_ids={','.join(sorted({str(x).strip() for x in args.run_ids if str(x).strip()}))}"
        else:
            group_scope = "anchor_only_fallback"
            group_scope_detail = f"anchor_run_id={run_id}"
        _run_group_summary(
            anchor_run_id=run_id,
            plot_run_id=run_id,
            resolved_group_id=resolved_group_id,
            group_run_id=group_run_id,
            candidate_runs=candidate_runs,
            processed_dir=processed_dir,
            output_root=Path(args.out_dir) / group_run_id,
            t50_definition=t50_definition,
            reference_polymer_id=reference_polymer_id,
            apply_outlier_filter=apply_outlier_filter,
            outlier_min_runs=int(args.outlier_min_runs),
            outlier_min_keep=int(args.outlier_min_keep),
            outlier_z_threshold=float(args.outlier_z_threshold),
            outlier_ratio_low=float(args.outlier_ratio_low),
            outlier_ratio_high=float(args.outlier_ratio_high),
            reference_abs0_outlier_low=float(args.reference_abs0_outlier_low),
            reference_abs0_outlier_high=float(args.reference_abs0_outlier_high),
            dpi=int(args.dpi),
            run_group_tsv_path=run_group_tsv_path,
            run_group_table=run_group_table,
            group_scope=group_scope,
            group_scope_detail=group_scope_detail,
            debug=bool(args.debug),
            dry_run=bool(args.dry_run),
        )

    if bool(args.also_all_rounds) or bool(args.all_rounds_only):
        run_round_map_path = Path(args.run_round_map)
        if not run_round_map_path.is_file():
            raise FileNotFoundError(f"run_round_map not found: {run_round_map_path}")
        run_round_table = _load_run_group_table(run_round_map_path)
        run_round_map = load_run_round_map(run_round_map_path)
        all_round_runs = sorted({str(rid).strip() for rid in run_round_map.keys() if str(rid).strip()})
        if not all_round_runs:
            raise ValueError("No run_id with valid round_id found in run_round_map.")
        round_ids = sorted({str(oid).strip() for oid in run_round_map.values() if str(oid).strip()})
        all_round_mode = str(args.all_rounds_mode).strip().lower()
        per_round_group_prefix = str(args.per_round_group_prefix).strip() or "round"

        if all_round_mode in {"pooled", "both"}:
            all_round_group_id = str(args.all_rounds_group_id).strip() or "all_rounds"
            all_round_group_run_id = (
                str(args.all_rounds_group_run_id).strip()
                if args.all_rounds_group_run_id and str(args.all_rounds_group_run_id).strip()
                else f"{_safe_group_stem(all_round_group_id)}-group_mean"
            )
            all_anchor_run_id = run_id if run_id else all_round_runs[0]
            _run_group_summary(
                anchor_run_id=all_anchor_run_id,
                plot_run_id=all_round_group_run_id,
                resolved_group_id=all_round_group_id,
                group_run_id=all_round_group_run_id,
                candidate_runs=all_round_runs,
                processed_dir=processed_dir,
                output_root=Path(args.out_dir) / all_round_group_run_id,
                t50_definition=t50_definition,
                reference_polymer_id=reference_polymer_id,
                apply_outlier_filter=apply_outlier_filter,
                outlier_min_runs=int(args.outlier_min_runs),
                outlier_min_keep=int(args.outlier_min_keep),
                outlier_z_threshold=float(args.outlier_z_threshold),
                outlier_ratio_low=float(args.outlier_ratio_low),
                outlier_ratio_high=float(args.outlier_ratio_high),
                reference_abs0_outlier_low=float(args.reference_abs0_outlier_low),
                reference_abs0_outlier_high=float(args.reference_abs0_outlier_high),
                dpi=int(args.dpi),
                run_group_tsv_path=run_round_map_path,
                run_group_table=run_round_table,
                group_scope="all_round_assigned_runs_pooled",
                group_scope_detail=(
                    f"pooled across all run_ids with valid round_id in {run_round_map_path}; "
                    f"round_ids={','.join(round_ids)}"
                ),
                debug=bool(args.debug),
                dry_run=bool(args.dry_run),
            )

        if all_round_mode in {"per_round", "both"}:
            rounds_to_runs: dict[str, list[str]] = {}
            for rid, oid in run_round_map.items():
                rid_s = str(rid).strip()
                oid_s = str(oid).strip()
                if not rid_s or not oid_s:
                    continue
                rounds_to_runs.setdefault(oid_s, []).append(rid_s)
            for round_id in sorted(rounds_to_runs.keys()):
                round_runs = sorted({str(x).strip() for x in rounds_to_runs.get(round_id, []) if str(x).strip()})
                if not round_runs:
                    continue
                round_group_id = f"{per_round_group_prefix}_{round_id}"
                round_group_run_id = f"{_safe_group_stem(round_group_id)}-group_mean"
                round_anchor_run_id = run_id if (run_id and run_id in round_runs) else round_runs[0]
                _run_group_summary(
                    anchor_run_id=round_anchor_run_id,
                    plot_run_id=round_group_run_id,
                    resolved_group_id=round_group_id,
                    group_run_id=round_group_run_id,
                    candidate_runs=round_runs,
                    processed_dir=processed_dir,
                    output_root=Path(args.out_dir) / round_group_run_id,
                    t50_definition=t50_definition,
                    reference_polymer_id=reference_polymer_id,
                    apply_outlier_filter=apply_outlier_filter,
                    outlier_min_runs=int(args.outlier_min_runs),
                    outlier_min_keep=int(args.outlier_min_keep),
                    outlier_z_threshold=float(args.outlier_z_threshold),
                    outlier_ratio_low=float(args.outlier_ratio_low),
                    outlier_ratio_high=float(args.outlier_ratio_high),
                    reference_abs0_outlier_low=float(args.reference_abs0_outlier_low),
                    reference_abs0_outlier_high=float(args.reference_abs0_outlier_high),
                    dpi=int(args.dpi),
                    run_group_tsv_path=run_round_map_path,
                    run_group_table=run_round_table,
                    group_scope="all_round_assigned_runs_per_round",
                    group_scope_detail=(
                        f"single round_id={round_id} from {run_round_map_path}; "
                        f"n_runs={len(round_runs)}"
                    ),
                    debug=bool(args.debug),
                    dry_run=bool(args.dry_run),
                )


if __name__ == "__main__":
    main()
