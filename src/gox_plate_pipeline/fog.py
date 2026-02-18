# src/gox_plate_pipeline/fog.py
"""
FoG (Fold over GOx) summary for BO: t50_polymer / t50_bare_GOx.

- Per-run: fog_summary__{run_id}.csv with FoG = t50_polymer / t50_GOx_same_run (denominator = same run GOx).
- Round-averaged: build_round_averaged_fog() averages FoG by (round_id, polymer_id). If a round has no GOx
  in any run, raises ValueError.
- Plate-aware FoG: build_fog_plate_aware() uses denominator rule "same plate GOx → same round GOx".
  Per-plate t50 is computed from rates_with_rea (REA vs heat_min per plate); then FoG = t50_polymer / gox_t50
  with gox_t50 from same (run, plate) if present, else mean GOx t50 in that round.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
import yaml


# t50 unit used in outputs (documented for BO and figures)
T50_UNIT = "min"

# Native-activity feasibility threshold for constrained FoG objective.
# Objective policy: maximize FoG only among polymers that keep sufficient baseline activity.
# Main-analysis default is 0.70 (sensitivity analyses should test 0.50-0.90).
NATIVE_ACTIVITY_MIN_REL_DEFAULT = 0.70
NATIVE_ACTIVITY_SOFT_PENALTY_EXPONENT = 2.0
DEFAULT_REFERENCE_POLYMER_ID = "GOX"
SOLVENT_BALANCED_PENALTY_EXPONENT = 2.0
SOLVENT_BALANCED_UP_BONUS_COEF = 0.35
SOLVENT_BALANCED_UP_BONUS_MAX_DELTA = 0.30
SOLVENT_BALANCED_UP_BONUS_DEADBAND = 0.05
ABS_ACTIVITY_OBJECTIVE_COL = "fog_activity_bonus_penalty"
ABS_ACTIVITY_OBJECTIVE_LOG_COL = "log_fog_activity_bonus_penalty"
ABS_ACTIVITY_OBJECTIVE_SEM_COL = "fog_activity_bonus_penalty_sem"
ABS_ACTIVITY_OBJECTIVE_RANK_COL = "rank_objective_fog_activity_bonus_penalty"
ABS_ACTIVITY_OBJECTIVE_OLD_COL = "fog_abs_bonus_penalty"
ABS_ACTIVITY_OBJECTIVE_LOG_OLD_COL = "log_fog_abs_bonus_penalty"
ABS_ACTIVITY_OBJECTIVE_SEM_OLD_COL = "fog_abs_bonus_penalty_sem"
ABS_ACTIVITY_OBJECTIVE_RANK_OLD_COL = "rank_objective_abs_bonus_penalty"
ABS_ACTIVITY_OBJECTIVE_LEGACY_COL = "fog_abs_modulated"
ABS_ACTIVITY_OBJECTIVE_LOG_LEGACY_COL = "log_fog_abs_modulated"
ABS_ACTIVITY_OBJECTIVE_SEM_LEGACY_COL = "fog_abs_modulated_sem"
ABS_ACTIVITY_OBJECTIVE_RANK_LEGACY_COL = "rank_objective_abs_modulated"
OBJECTIVE_LOGLINEAR_MAIN_COL = "objective_loglinear_main"
OBJECTIVE_LOGLINEAR_MAIN_EXP_COL = "objective_loglinear_main_exp"
OBJECTIVE_LOGLINEAR_MAIN_RANK_COL = "rank_objective_loglinear_main"
OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT = 1.0
OBJECTIVE_LOGLINEAR_TIE_EPS = 0.02
MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD = 0.75
OBJECTIVE_LOGLINEAR_WEIGHT_SWEEP: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
OBJECTIVE_LOGLINEAR_THRESHOLD_SWEEP: Tuple[float, ...] = (0.7, 0.8, 0.9)
OBJECTIVE_PROFILE_SPECS: Tuple[Tuple[str, str, float, float], ...] = (
    (
        "default",
        f"Default (exp={SOLVENT_BALANCED_PENALTY_EXPONENT:.1f}, bonus={SOLVENT_BALANCED_UP_BONUS_COEF:.2f})",
        SOLVENT_BALANCED_PENALTY_EXPONENT,
        SOLVENT_BALANCED_UP_BONUS_COEF,
    ),
    (
        "penalty_only",
        f"Penalty-only (exp={SOLVENT_BALANCED_PENALTY_EXPONENT:.1f}, bonus=0.00)",
        SOLVENT_BALANCED_PENALTY_EXPONENT,
        0.0,
    ),
    (
        "gentle_penalty",
        f"Gentler penalty (exp={SOLVENT_BALANCED_PENALTY_EXPONENT - 0.5:.1f}, bonus={SOLVENT_BALANCED_UP_BONUS_COEF:.2f})",
        SOLVENT_BALANCED_PENALTY_EXPONENT - 0.5,
        SOLVENT_BALANCED_UP_BONUS_COEF,
    ),
    (
        "strong_penalty",
        f"Stronger penalty (exp={SOLVENT_BALANCED_PENALTY_EXPONENT + 0.5:.1f}, bonus={SOLVENT_BALANCED_UP_BONUS_COEF:.2f})",
        SOLVENT_BALANCED_PENALTY_EXPONENT + 0.5,
        SOLVENT_BALANCED_UP_BONUS_COEF,
    ),
)
SOLVENT_GROUP_PBS = "PBS"
SOLVENT_GROUP_DMSO = "DMSO"
SOLVENT_GROUP_ETOH = "ETOH"
# Default solvent hints for current polymer panel. Unknown IDs fall back to PBS reference.
POLYMER_SOLVENT_GROUP_HINTS: Dict[str, str] = {
    "PMPC": SOLVENT_GROUP_ETOH,
    "PMTAC": SOLVENT_GROUP_PBS,
    "PMBTA-1": SOLVENT_GROUP_ETOH,
}
POLYMER_SOLVENT_GROUP_PREFIX_HINTS: Tuple[Tuple[str, str], ...] = (
    ("PMBTA-", SOLVENT_GROUP_DMSO),
)


def _ensure_csv_subdir(base_dir: Path) -> Path:
    csv_dir = Path(base_dir) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir


def _archive_legacy_file(legacy_path: Path, archive_path: Path) -> None:
    """Move legacy output into archive path without deleting historical artifacts."""
    try:
        if legacy_path.resolve() == archive_path.resolve():
            return
    except FileNotFoundError:
        pass
    if not legacy_path.is_file():
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    target = archive_path
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        idx = 1
        while True:
            candidate = target.with_name(f"{stem}__legacy_flat_{idx}{suffix}")
            if not candidate.exists():
                target = candidate
                break
            idx += 1
    legacy_path.replace(target)


def _normalize_polymer_id_token(value: object) -> str:
    return str(value).strip().upper()


def _is_reference_polymer_id(
    polymer_id: object,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    return _normalize_polymer_id_token(polymer_id) == _normalize_polymer_id_token(reference_polymer_id)


def _is_reference_like_polymer_id(
    polymer_id: object,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    True for the reference polymer itself and its solvent-control variants
    (e.g. "GOx with DMSO", "GOx with EtOH").
    """
    norm = _normalize_polymer_id_token(polymer_id)
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    if (not norm) or (not ref_norm):
        return False
    if norm == ref_norm:
        return True
    if norm.startswith(f"{ref_norm} WITH "):
        return True
    if norm.startswith(f"{ref_norm}_WITH_"):
        return True
    return False


def _exclude_reference_like_polymers(
    df: pd.DataFrame,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """Exclude reference-like rows from polymer-level comparison tables/plots."""
    if df.empty or ("polymer_id" not in df.columns):
        return df
    out = df.copy()
    out["polymer_id"] = out["polymer_id"].astype(str).str.strip()
    ref_like_mask = out["polymer_id"].map(
        lambda x: _is_reference_like_polymer_id(x, reference_polymer_id=reference_polymer_id)
    )
    return out[~ref_like_mask].copy()


def _missing_reference_reason(reference_polymer_id: str) -> str:
    if _is_reference_polymer_id(reference_polymer_id, DEFAULT_REFERENCE_POLYMER_ID):
        return "no_bare_gox_in_run"
    return "no_reference_polymer_in_run"


def _aggregate_reference_values(
    values: List[float],
    *,
    method: str = "median",
    trimmed_mean_proportion: float = 0.1,
) -> float:
    """Aggregate positive finite reference values with robust method."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return np.nan
    m = str(method).strip().lower()
    if m == "median":
        return float(np.nanmedian(arr))
    if m == "trimmed_mean":
        n = int(arr.size)
        if n >= 3:
            p = min(float(trimmed_mean_proportion), (n - 1) / (2 * n))
            return float(trim_mean(arr, proportiontocut=p))
        return float(np.nanmedian(arr))
    if m == "mean":
        return float(np.nanmean(arr))
    raise ValueError(f"Unknown reference aggregation method: {method!r}")


def _u_col_name(heat_min: float) -> str:
    """Column name for U(t) at a heat time."""
    h = float(heat_min)
    if math.isclose(h, round(h), rel_tol=0.0, abs_tol=1e-9):
        return f"U_{int(round(h))}"
    txt = f"{h:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
    return f"U_{txt}"


def _normalize_solvent_group(value: object) -> str:
    token = str(value).strip().upper()
    if token in {SOLVENT_GROUP_PBS, "WATER", "AQUEOUS"}:
        return SOLVENT_GROUP_PBS
    if token in {SOLVENT_GROUP_DMSO}:
        return SOLVENT_GROUP_DMSO
    if token in {SOLVENT_GROUP_ETOH, "ETHANOL"}:
        return SOLVENT_GROUP_ETOH
    return ""


def _sync_abs_objective_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep absolute-activity objective columns and legacy aliases in sync.
    """
    if df.empty:
        return df
    out = df.copy()
    alias_groups = [
        ["solvent_activity_down_penalty", "abs_activity_down_penalty"],
        ["solvent_activity_up_bonus", "abs_activity_up_bonus"],
        ["solvent_activity_balance_factor", "abs_activity_balance_factor"],
        [
            ABS_ACTIVITY_OBJECTIVE_COL,
            ABS_ACTIVITY_OBJECTIVE_OLD_COL,
            ABS_ACTIVITY_OBJECTIVE_LEGACY_COL,
            "fog_solvent_balanced",
        ],
        [
            ABS_ACTIVITY_OBJECTIVE_LOG_COL,
            ABS_ACTIVITY_OBJECTIVE_LOG_OLD_COL,
            ABS_ACTIVITY_OBJECTIVE_LOG_LEGACY_COL,
            "log_fog_solvent_balanced",
        ],
        [
            ABS_ACTIVITY_OBJECTIVE_SEM_COL,
            ABS_ACTIVITY_OBJECTIVE_SEM_OLD_COL,
            ABS_ACTIVITY_OBJECTIVE_SEM_LEGACY_COL,
            "fog_solvent_balanced_sem",
        ],
        [
            ABS_ACTIVITY_OBJECTIVE_RANK_COL,
            ABS_ACTIVITY_OBJECTIVE_RANK_OLD_COL,
            ABS_ACTIVITY_OBJECTIVE_RANK_LEGACY_COL,
            "rank_objective_solvent_balanced",
        ],
        [
            "mean_fog_activity_bonus_penalty",
            "mean_fog_abs_bonus_penalty",
            "mean_fog_abs_modulated",
            "mean_fog_solvent_balanced",
        ],
        [
            "mean_log_fog_activity_bonus_penalty",
            "mean_log_fog_abs_bonus_penalty",
            "mean_log_fog_abs_modulated",
            "mean_log_fog_solvent_balanced",
        ],
        [
            "robust_fog_activity_bonus_penalty",
            "robust_fog_abs_bonus_penalty",
            "robust_fog_abs_modulated",
            "robust_fog_solvent_balanced",
        ],
        [
            "robust_log_fog_activity_bonus_penalty",
            "robust_log_fog_abs_bonus_penalty",
            "robust_log_fog_abs_modulated",
            "robust_log_fog_solvent_balanced",
        ],
        [
            "log_fog_activity_bonus_penalty_mad",
            "log_fog_abs_bonus_penalty_mad",
            "log_fog_abs_modulated_mad",
            "log_fog_solvent_balanced_mad",
        ],
    ]
    for group in alias_groups:
        src = None
        for col in group:
            if col in out.columns:
                src = col
                break
        if src is None:
            continue
        for col in group:
            if col not in out.columns:
                out[col] = out[src]
    return out


def _load_polymer_solvent_maps(
    polymer_solvent_path: Optional[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load polymer-level solvent metadata map.

    Expected columns:
      - polymer_id (required)
      - stock_solvent (optional): PBS / DMSO / ETOH
      - objective_control_group (optional): overrides stock_solvent for objective control selection
    """
    if polymer_solvent_path is None:
        return {}, {}
    path = Path(polymer_solvent_path)
    if not path.is_file():
        return {}, {}
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        meta_df = pd.read_csv(path, sep=sep)
    except Exception:
        return {}, {}
    if "polymer_id" not in meta_df.columns:
        return {}, {}
    stock_col = None
    for cand in ["stock_solvent", "stock_solvent_group", "solvent_group", "solvent", "organic_solvent"]:
        if cand in meta_df.columns:
            stock_col = cand
            break
    control_col = None
    for cand in ["objective_control_group", "control_group", "objective_solvent_group"]:
        if cand in meta_df.columns:
            control_col = cand
            break
    stock_map: Dict[str, str] = {}
    control_map: Dict[str, str] = {}
    for _, row in meta_df.iterrows():
        pid_norm = _normalize_polymer_id_token(row.get("polymer_id", ""))
        if not pid_norm:
            continue
        stock_group = ""
        control_group = ""
        if stock_col is not None:
            stock_group = _normalize_solvent_group(row.get(stock_col, ""))
        if control_col is not None:
            control_group = _normalize_solvent_group(row.get(control_col, ""))
        if not control_group:
            control_group = stock_group
        if stock_group:
            stock_map[pid_norm] = stock_group
        if control_group:
            control_map[pid_norm] = control_group
    return stock_map, control_map


def _apply_polymer_solvent_maps(
    fog_df: pd.DataFrame,
    *,
    stock_map: Dict[str, str],
    control_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Apply polymer solvent metadata maps onto fog dataframe.
    """
    if fog_df.empty or ("polymer_id" not in fog_df.columns):
        return fog_df
    if (not stock_map) and (not control_map):
        return fog_df
    df = fog_df.copy()
    pid_norm = df["polymer_id"].astype(str).map(_normalize_polymer_id_token)
    stock_from_map = pid_norm.map(lambda k: stock_map.get(k, ""))
    control_from_map = pid_norm.map(lambda k: control_map.get(k, ""))
    control_from_map = np.where(
        pd.Series(control_from_map).astype(str).str.strip().ne(""),
        control_from_map,
        stock_from_map,
    )

    if "stock_solvent_group" in df.columns:
        current_stock = df["stock_solvent_group"].map(_normalize_solvent_group)
        df["stock_solvent_group"] = np.where(current_stock != "", current_stock, stock_from_map)
    else:
        df["stock_solvent_group"] = stock_from_map

    if "solvent_group" in df.columns:
        current_group = df["solvent_group"].map(_normalize_solvent_group)
        df["solvent_group"] = np.where(current_group != "", current_group, control_from_map)
    else:
        df["solvent_group"] = control_from_map
    return df


def _infer_solvent_group_from_polymer_id(polymer_id: object) -> str:
    norm = _normalize_polymer_id_token(polymer_id)
    if norm == "GOX":
        return SOLVENT_GROUP_PBS
    if norm.startswith("GOX WITH"):
        if "DMSO" in norm:
            return SOLVENT_GROUP_DMSO
        if ("ETOH" in norm) or ("ETHANOL" in norm):
            return SOLVENT_GROUP_ETOH
        return SOLVENT_GROUP_PBS
    if norm in POLYMER_SOLVENT_GROUP_HINTS:
        return POLYMER_SOLVENT_GROUP_HINTS[norm]
    for pref, grp in POLYMER_SOLVENT_GROUP_PREFIX_HINTS:
        if norm.startswith(pref):
            return grp
    return SOLVENT_GROUP_PBS


def _compute_solvent_balanced_activity_factor(
    abs0_rel_values: Any,
    *,
    exponent: float = SOLVENT_BALANCED_PENALTY_EXPONENT,
    up_bonus_coef: float = SOLVENT_BALANCED_UP_BONUS_COEF,
    up_bonus_max_delta: float = SOLVENT_BALANCED_UP_BONUS_MAX_DELTA,
    up_bonus_deadband: float = SOLVENT_BALANCED_UP_BONUS_DEADBAND,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute activity factor for solvent-matched objective.

    down_penalty = clip(U0*, 0, 1)^exponent
    up_bonus = 1 + up_bonus_coef * clip(U0* - (1 + deadband), 0, up_bonus_max_delta)
    factor = down_penalty * up_bonus
    """
    exp = float(exponent)
    if (not np.isfinite(exp)) or (exp <= 0.0):
        raise ValueError(f"exponent must be a positive finite number, got {exponent!r}")
    bonus_coef = float(up_bonus_coef)
    if (not np.isfinite(bonus_coef)) or (bonus_coef < 0.0):
        raise ValueError(f"up_bonus_coef must be non-negative finite, got {up_bonus_coef!r}")
    bonus_cap = float(up_bonus_max_delta)
    if (not np.isfinite(bonus_cap)) or (bonus_cap < 0.0):
        raise ValueError(f"up_bonus_max_delta must be non-negative finite, got {up_bonus_max_delta!r}")
    deadband = float(up_bonus_deadband)
    if (not np.isfinite(deadband)) or (deadband < 0.0):
        raise ValueError(f"up_bonus_deadband must be non-negative finite, got {up_bonus_deadband!r}")

    rel = np.asarray(pd.to_numeric(abs0_rel_values, errors="coerce"), dtype=float)
    clipped = np.clip(rel, 0.0, 1.0)
    down_penalty = np.power(clipped, exp)
    up_delta = np.clip(rel - (1.0 + deadband), 0.0, bonus_cap)
    up_bonus = 1.0 + bonus_coef * up_delta
    factor = down_penalty * up_bonus
    down_penalty = np.where(np.isfinite(rel), down_penalty, np.nan)
    up_bonus = np.where(np.isfinite(rel), up_bonus, np.nan)
    factor = np.where(np.isfinite(rel), factor, np.nan)
    return down_penalty, up_bonus, factor


def _compute_solvent_balanced_objective(
    fog_rel_values: Any,
    abs0_rel_values: Any,
    *,
    exponent: float = SOLVENT_BALANCED_PENALTY_EXPONENT,
    up_bonus_coef: float = SOLVENT_BALANCED_UP_BONUS_COEF,
    up_bonus_max_delta: float = SOLVENT_BALANCED_UP_BONUS_MAX_DELTA,
    up_bonus_deadband: float = SOLVENT_BALANCED_UP_BONUS_DEADBAND,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FoG-activity objective on solvent-matched axes:
      S = FoG* * activity_factor(U0*)

    where FoG* = t50 / t50_solvent_control, U0* = abs0 / abs0_solvent_control.
    """
    fog_rel = np.asarray(pd.to_numeric(fog_rel_values, errors="coerce"), dtype=float)
    down_penalty, up_bonus, factor = _compute_solvent_balanced_activity_factor(
        abs0_rel_values,
        exponent=exponent,
        up_bonus_coef=up_bonus_coef,
        up_bonus_max_delta=up_bonus_max_delta,
        up_bonus_deadband=up_bonus_deadband,
    )
    score = np.where(
        np.isfinite(fog_rel) & (fog_rel > 0.0) & np.isfinite(factor),
        fog_rel * factor,
        np.nan,
    )
    log_score = np.full_like(score, np.nan, dtype=float)
    ok = np.isfinite(score) & (score > 0.0)
    log_score[ok] = np.log(score[ok])
    return down_penalty, up_bonus, factor, score, log_score


def _compute_loglinear_main_objective(
    fog_rel_values: Any,
    abs0_rel_values: Any,
    *,
    weight_lambda: float = OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute primary log-linear objective:
      S_main = log(FoG*) + lambda * log(U0*)
      S_main_exp = exp(S_main)
    """
    lam = float(weight_lambda)
    if not np.isfinite(lam):
        raise ValueError(f"weight_lambda must be finite, got {weight_lambda!r}")
    fog_rel = np.asarray(pd.to_numeric(fog_rel_values, errors="coerce"), dtype=float)
    abs_rel = np.asarray(pd.to_numeric(abs0_rel_values, errors="coerce"), dtype=float)
    score = np.full_like(fog_rel, np.nan, dtype=float)
    valid = np.isfinite(fog_rel) & (fog_rel > 0.0) & np.isfinite(abs_rel) & (abs_rel > 0.0)
    if np.any(valid):
        score[valid] = np.log(fog_rel[valid]) + lam * np.log(abs_rel[valid])
    score_exp = np.where(np.isfinite(score), np.exp(score), np.nan)
    return score, score_exp


def _compute_pareto_front_mask(x_values: Any, y_values: Any) -> np.ndarray:
    """
    Pareto front mask for maximization on two metrics (x, y).
    """
    x = np.asarray(pd.to_numeric(x_values, errors="coerce"), dtype=float)
    y = np.asarray(pd.to_numeric(y_values, errors="coerce"), dtype=float)
    n = int(len(x))
    out = np.zeros(n, dtype=bool)
    valid = np.isfinite(x) & np.isfinite(y)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return out
    xv = x[idx]
    yv = y[idx]
    for local_i, global_i in enumerate(idx):
        dominated = (
            (xv >= xv[local_i])
            & (yv >= yv[local_i])
            & ((xv > xv[local_i]) | (yv > yv[local_i]))
        )
        if not np.any(dominated):
            out[global_i] = True
    return out


def _rank_by_loglinear_objective(
    df: pd.DataFrame,
    *,
    score_col: str,
    rank_col: str,
    u0_col: str = "abs0_vs_solvent_control",
    fog_col: str = "fog_vs_solvent_control",
    tie_eps: float = OBJECTIVE_LOGLINEAR_TIE_EPS,
) -> pd.DataFrame:
    """
    Rank rows by score descending, then deterministic tie-break:
      1) |Δscore| < tie_eps -> higher U0*
      2) then higher FoG*
      3) then smaller SEM (when available)
      4) then polymer_id ascending
    """
    out = df.copy()
    out[rank_col] = np.nan
    if out.empty:
        return out
    out["polymer_id"] = out.get("polymer_id", pd.Series([""] * len(out))).astype(str).str.strip()
    out["_score_tmp"] = pd.to_numeric(out.get(score_col, np.nan), errors="coerce")
    out["_u0_tmp"] = pd.to_numeric(out.get(u0_col, np.nan), errors="coerce")
    out["_fog_tmp"] = pd.to_numeric(out.get(fog_col, np.nan), errors="coerce")
    sem_candidates = [
        f"{score_col}_sem",
        OBJECTIVE_LOGLINEAR_MAIN_COL + "_sem",
        ABS_ACTIVITY_OBJECTIVE_SEM_COL,
        "fog_vs_solvent_control_sem",
        "fog_sem",
    ]
    sem_col = next((c for c in sem_candidates if c in out.columns), None)
    if sem_col is None:
        out["_sem_tmp"] = np.nan
    else:
        out["_sem_tmp"] = pd.to_numeric(out.get(sem_col, np.nan), errors="coerce")

    valid = (
        np.isfinite(out["_score_tmp"])
        & np.isfinite(out["_u0_tmp"])
        & (out["_u0_tmp"] > 0.0)
        & np.isfinite(out["_fog_tmp"])
        & (out["_fog_tmp"] > 0.0)
    )
    valid_df = out.loc[valid].copy()
    if valid_df.empty:
        return out.drop(columns=["_score_tmp", "_u0_tmp", "_fog_tmp", "_sem_tmp"], errors="ignore")

    valid_df = valid_df.sort_values(
        ["_score_tmp", "polymer_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index()
    tie_eps_f = float(tie_eps)
    tie_group = np.zeros(len(valid_df), dtype=int)
    if len(valid_df) > 0:
        group_id = 0
        anchor = float(valid_df.loc[0, "_score_tmp"])
        for i in range(1, len(valid_df)):
            score_i = float(valid_df.loc[i, "_score_tmp"])
            if (not np.isfinite(score_i)) or ((anchor - score_i) >= tie_eps_f):
                group_id += 1
                anchor = score_i
            tie_group[i] = group_id
    valid_df["_tie_group"] = tie_group
    valid_df["_sem_sort"] = np.where(
        np.isfinite(valid_df["_sem_tmp"]) & (valid_df["_sem_tmp"] >= 0.0),
        valid_df["_sem_tmp"],
        np.inf,
    )
    valid_df = valid_df.sort_values(
        ["_tie_group", "_u0_tmp", "_fog_tmp", "_sem_sort", "polymer_id"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    valid_df[rank_col] = np.arange(1, len(valid_df) + 1, dtype=int)
    out.loc[valid_df["index"].to_numpy(dtype=int), rank_col] = valid_df[rank_col].to_numpy(dtype=float)
    out = out.drop(columns=["_score_tmp", "_u0_tmp", "_fog_tmp", "_sem_tmp"], errors="ignore")
    return out


def _rank_corr_metrics(base_rank: pd.DataFrame, other_rank: pd.DataFrame, *, top_k: int = 5) -> Dict[str, float]:
    """
    Compute rank correlation metrics between two rank tables.
    """
    base = base_rank[["polymer_id", "rank"]].copy()
    other = other_rank[["polymer_id", "rank"]].copy()
    base["polymer_id"] = base["polymer_id"].astype(str).str.strip()
    other["polymer_id"] = other["polymer_id"].astype(str).str.strip()
    merged = base.merge(other, on="polymer_id", how="inner", suffixes=("_base", "_other"))
    n_common = int(len(merged))
    kendall_tau = np.nan
    spearman_rho = np.nan
    if n_common >= 2:
        kendall_tau = float(merged["rank_base"].corr(merged["rank_other"], method="kendall"))
        spearman_rho = float(merged["rank_base"].corr(merged["rank_other"], method="spearman"))
    base_top = set(base.sort_values("rank", ascending=True)["polymer_id"].head(int(top_k)).tolist())
    other_top = set(other.sort_values("rank", ascending=True)["polymer_id"].head(int(top_k)).tolist())
    topk_overlap = float(len(base_top & other_top)) / float(max(1, len(base_top)))
    return {
        "n_common": float(n_common),
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "topk_overlap": topk_overlap,
    }


def _compute_activity_objective_profile_ranks(
    run_df: pd.DataFrame,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Compute rank tables under a small profile family around the default objective.

    This is a sensitivity/audit helper to check how much ranking depends on
    penalty/bonus hyper-parameters (arbitrariness check).
    """
    empty_cols = [
        "polymer_id",
        "abs0_vs_solvent_control",
        "fog_vs_solvent_control",
        "fog_vs_solvent_control_sem",
        "profile_id",
        "profile_label",
        "penalty_exponent",
        "bonus_coef",
        "bonus_deadband",
        "bonus_max_delta",
        "score",
        "rank",
        "rank_default",
        "rank_delta_vs_default",
    ]
    if run_df.empty or ("polymer_id" not in run_df.columns):
        return pd.DataFrame(columns=empty_cols)
    df = _add_solvent_balanced_objective_columns(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    df["abs0_vs_solvent_control"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["fog_vs_solvent_control"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["fog_vs_solvent_control_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df = df[
        np.isfinite(df["abs0_vs_solvent_control"])
        & np.isfinite(df["fog_vs_solvent_control"])
        & (df["fog_vs_solvent_control"] > 0.0)
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    profile_rows: List[pd.DataFrame] = []
    for profile_id, profile_label, exp, bonus_coef in OBJECTIVE_PROFILE_SPECS:
        _, _, _, score, _ = _compute_solvent_balanced_objective(
            df["fog_vs_solvent_control"],
            df["abs0_vs_solvent_control"],
            exponent=float(exp),
            up_bonus_coef=float(bonus_coef),
            up_bonus_max_delta=SOLVENT_BALANCED_UP_BONUS_MAX_DELTA,
            up_bonus_deadband=SOLVENT_BALANCED_UP_BONUS_DEADBAND,
        )
        prof = df[
            [
                "polymer_id",
                "abs0_vs_solvent_control",
                "fog_vs_solvent_control",
                "fog_vs_solvent_control_sem",
            ]
        ].copy()
        prof["profile_id"] = profile_id
        prof["profile_label"] = profile_label
        prof["penalty_exponent"] = float(exp)
        prof["bonus_coef"] = float(bonus_coef)
        prof["bonus_deadband"] = float(SOLVENT_BALANCED_UP_BONUS_DEADBAND)
        prof["bonus_max_delta"] = float(SOLVENT_BALANCED_UP_BONUS_MAX_DELTA)
        prof["score"] = np.asarray(score, dtype=float)
        prof = prof[np.isfinite(prof["score"]) & (prof["score"] > 0.0)].copy()
        if prof.empty:
            continue
        prof = prof.sort_values(
            ["score", "fog_vs_solvent_control", "abs0_vs_solvent_control"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        prof["rank"] = np.arange(1, len(prof) + 1, dtype=int)
        profile_rows.append(prof)
    if not profile_rows:
        return pd.DataFrame(columns=empty_cols)
    out = pd.concat(profile_rows, axis=0, ignore_index=True)
    default_rank = (
        out[out["profile_id"] == "default"][["polymer_id", "rank"]]
        .rename(columns={"rank": "rank_default"})
        .copy()
    )
    out = out.merge(default_rank, on="polymer_id", how="left")
    out["rank_delta_vs_default"] = np.where(
        np.isfinite(pd.to_numeric(out["rank_default"], errors="coerce")),
        pd.to_numeric(out["rank"], errors="coerce") - pd.to_numeric(out["rank_default"], errors="coerce"),
        np.nan,
    )
    return out


def _add_solvent_balanced_objective_columns(
    fog_df: pd.DataFrame,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Add solvent-matched relative columns and FoG-activity objective columns.
    """
    df = fog_df.copy()
    if df.empty or ("polymer_id" not in df.columns):
        return df
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    if "run_id" not in df.columns:
        df["_solvent_tmp_run_id"] = "single_run"
        run_col = "_solvent_tmp_run_id"
    else:
        run_col = "run_id"
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    df["abs_activity_at_0"] = pd.to_numeric(df.get("abs_activity_at_0", np.nan), errors="coerce")
    df["t50_min"] = pd.to_numeric(df.get("t50_min", np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    if "stock_solvent_group" in df.columns:
        stock_explicit = df["stock_solvent_group"].map(_normalize_solvent_group)
    else:
        stock_explicit = pd.Series([""] * len(df), index=df.index, dtype=object)
    if "solvent_group" in df.columns:
        explicit_group = df["solvent_group"].map(_normalize_solvent_group)
    else:
        explicit_group = pd.Series([""] * len(df), index=df.index, dtype=object)
    inferred_group = df["polymer_id"].map(_infer_solvent_group_from_polymer_id)
    df["stock_solvent_group"] = np.where(stock_explicit != "", stock_explicit, inferred_group)
    df["solvent_group"] = np.where(explicit_group != "", explicit_group, inferred_group)

    ctrl_pid = pd.Series([""] * len(df), index=df.index, dtype=object)
    ctrl_abs0 = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)
    ctrl_t50 = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)
    ctrl_fog = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)

    def _best_index(sub: pd.DataFrame, mask: pd.Series) -> Optional[int]:
        cand = sub.loc[mask].copy()
        if cand.empty:
            return None
        qual = (
            np.isfinite(pd.to_numeric(cand.get("abs_activity_at_0", np.nan), errors="coerce")).astype(int)
            + np.isfinite(pd.to_numeric(cand.get("t50_min", np.nan), errors="coerce")).astype(int)
            + np.isfinite(pd.to_numeric(cand.get("fog", np.nan), errors="coerce")).astype(int)
        )
        cand["_qual"] = qual.to_numpy(dtype=int)
        cand = cand.sort_values("_qual", ascending=False, kind="mergesort")
        return int(cand.index[0])

    for _, sub in df.groupby(run_col, sort=False):
        sub = sub.copy()
        norm = sub["polymer_id"].map(_normalize_polymer_id_token)
        idx_ref = _best_index(sub, norm == ref_norm)
        idx_dmso = _best_index(sub, norm.str.contains("GOX WITH", regex=False) & norm.str.contains("DMSO", regex=False))
        idx_etoh = _best_index(
            sub,
            norm.str.contains("GOX WITH", regex=False)
            & (norm.str.contains("ETOH", regex=False) | norm.str.contains("ETHANOL", regex=False)),
        )
        # Policy:
        # - PBS: GOx baseline
        # - DMSO: treated as GOx-equivalent baseline (always GOx)
        # - ETOH: prefer GOx with EtOH; fallback to GOx when absent
        control_idx_by_group: Dict[str, Optional[int]] = {
            SOLVENT_GROUP_PBS: idx_ref,
            SOLVENT_GROUP_DMSO: idx_ref,
            SOLVENT_GROUP_ETOH: idx_etoh if idx_etoh is not None else idx_ref,
        }
        for idx in sub.index:
            grp = _normalize_solvent_group(df.at[idx, "solvent_group"]) or SOLVENT_GROUP_PBS
            cidx = control_idx_by_group.get(grp)
            if cidx is None:
                continue
            ctrl_pid.at[idx] = str(df.at[cidx, "polymer_id"]).strip()
            ctrl_abs0.at[idx] = float(pd.to_numeric(df.at[cidx, "abs_activity_at_0"], errors="coerce"))
            ctrl_t50.at[idx] = float(pd.to_numeric(df.at[cidx, "t50_min"], errors="coerce"))
            ctrl_fog.at[idx] = float(pd.to_numeric(df.at[cidx, "fog"], errors="coerce"))

    df["solvent_control_polymer_id"] = ctrl_pid
    df["solvent_control_abs_activity_at_0"] = ctrl_abs0.to_numpy(dtype=float)
    df["solvent_control_t50_min"] = ctrl_t50.to_numpy(dtype=float)
    df["solvent_control_fog"] = ctrl_fog.to_numpy(dtype=float)

    abs0_rel = np.where(
        np.isfinite(df["abs_activity_at_0"])
        & np.isfinite(df["solvent_control_abs_activity_at_0"])
        & (df["solvent_control_abs_activity_at_0"] > 0.0),
        df["abs_activity_at_0"] / df["solvent_control_abs_activity_at_0"],
        np.nan,
    )
    fog_rel = np.where(
        np.isfinite(df["t50_min"])
        & np.isfinite(df["solvent_control_t50_min"])
        & (df["solvent_control_t50_min"] > 0.0),
        df["t50_min"] / df["solvent_control_t50_min"],
        np.nan,
    )
    fallback_fog_rel = np.where(
        np.isfinite(df["fog"])
        & np.isfinite(df["solvent_control_fog"])
        & (df["solvent_control_fog"] > 0.0),
        df["fog"] / df["solvent_control_fog"],
        np.nan,
    )
    fog_rel = np.where(np.isfinite(fog_rel), fog_rel, fallback_fog_rel)

    down_penalty, up_bonus, balance_factor, score, log_score = _compute_solvent_balanced_objective(
        fog_rel,
        abs0_rel,
    )
    loglinear_main, loglinear_main_exp = _compute_loglinear_main_objective(
        fog_rel,
        abs0_rel,
        weight_lambda=OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT,
    )
    df["abs0_vs_solvent_control"] = abs0_rel
    df["fog_vs_solvent_control"] = fog_rel
    df["solvent_activity_down_penalty"] = down_penalty
    df["solvent_activity_up_bonus"] = up_bonus
    df["solvent_activity_balance_factor"] = balance_factor
    df["fog_solvent_balanced"] = score
    df["log_fog_solvent_balanced"] = log_score
    df[OBJECTIVE_LOGLINEAR_MAIN_COL] = loglinear_main
    df[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = loglinear_main_exp
    fog_sem = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df["fog_solvent_balanced_sem"] = np.where(
        np.isfinite(fog_sem) & np.isfinite(balance_factor),
        fog_sem * balance_factor,
        np.nan,
    )
    if "_solvent_tmp_run_id" in df.columns:
        df = df.drop(columns=["_solvent_tmp_run_id"])
    return _sync_abs_objective_alias_columns(df)


def _compute_native_soft_penalty(
    native_rel_values: Any,
    *,
    exponent: float = NATIVE_ACTIVITY_SOFT_PENALTY_EXPONENT,
) -> np.ndarray:
    """
    Soft penalty on native activity for objective shaping.

    penalty = clip(native_rel_at_0, 0, 1) ** exponent
    """
    exp = float(exponent)
    if not np.isfinite(exp) or exp <= 0.0:
        raise ValueError(f"exponent must be a positive finite number, got {exponent!r}")
    native_rel = np.asarray(pd.to_numeric(native_rel_values, errors="coerce"), dtype=float)
    clipped = np.clip(native_rel, 0.0, 1.0)
    penalty = np.power(clipped, exp)
    penalty = np.where(np.isfinite(native_rel), penalty, np.nan)
    return penalty


def _compute_penalized_fog_objective(
    fog_values: Any,
    native_rel_values: Any,
    *,
    exponent: float = NATIVE_ACTIVITY_SOFT_PENALTY_EXPONENT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (soft_penalty, penalized_fog, log_penalized_fog)."""
    fog = np.asarray(pd.to_numeric(fog_values, errors="coerce"), dtype=float)
    penalty = _compute_native_soft_penalty(native_rel_values, exponent=exponent)
    fog_soft = np.where(
        np.isfinite(fog) & (fog > 0.0) & np.isfinite(penalty),
        fog * penalty,
        np.nan,
    )
    log_fog_soft = np.full_like(fog_soft, np.nan, dtype=float)
    ok = np.isfinite(fog_soft) & (fog_soft > 0.0)
    log_fog_soft[ok] = np.log(fog_soft[ok])
    return penalty, fog_soft, log_fog_soft


def _format_penalty_exponent_label(exponent: float) -> str:
    exp = float(exponent)
    if math.isclose(exp, round(exp), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(exp)))
    return f"{exp:.2f}".rstrip("0").rstrip(".")


def _compute_t_theta_from_series(
    t: np.ndarray,
    u: np.ndarray,
    *,
    theta: float,
) -> Tuple[float, str]:
    """
    Compute t_theta where U(t) first falls below theta.

    Returns:
      (t_theta, censor_flag)
        - censor_flag = "" for interpolated crossing
        - "already_below" if first finite U is below theta
        - "never_cross" if never below theta (right-censored at max heat time)
        - "missing_profile" if no finite (t,u) points
    """
    tt = np.asarray(t, dtype=float)
    uu = np.asarray(u, dtype=float)
    mask = np.isfinite(tt) & np.isfinite(uu)
    tt = tt[mask]
    uu = uu[mask]
    if tt.size == 0:
        return np.nan, "missing_profile"
    order = np.argsort(tt)
    tt = tt[order]
    uu = uu[order]
    th = float(theta)
    if uu[0] < th:
        return float(tt[0]), "already_below"
    for i in range(int(tt.size) - 1):
        t0 = float(tt[i])
        t1 = float(tt[i + 1])
        u0 = float(uu[i])
        u1 = float(uu[i + 1])
        if (u0 >= th and u1 < th) or (u0 > th and u1 <= th):
            if math.isclose(u1, u0, rel_tol=0.0, abs_tol=1e-12):
                return float(t0), ""
            frac = (th - u0) / (u1 - u0)
            return float(t0 + frac * (t1 - t0)), ""
    return float(tt[-1]), "never_cross"


@dataclass
class FogWarningInfo:
    """Warning information collected during FoG calculation."""
    outlier_gox: List[Dict[str, Any]] = field(default_factory=list)  # List of outlier GOx t50 info
    guarded_same_plate: List[Dict[str, Any]] = field(default_factory=list)  # same_plate guard fallback details
    missing_rates_files: List[Dict[str, str]] = field(default_factory=list)  # List of missing rates_with_rea.csv
    reference_qc_fail_runs: List[Dict[str, Any]] = field(default_factory=list)  # run-level reference QC fails


def write_fog_warning_file(warning_info: FogWarningInfo, out_path: Path, exclude_outlier_gox: bool = False) -> None:
    """
    Write warning information to a markdown file.
    
    Args:
        warning_info: FogWarningInfo object containing warning details.
        out_path: Path to output warning file.
        exclude_outlier_gox: Whether outlier exclusion was enabled.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# FoG計算時の警告情報")
    lines.append("")
    lines.append(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    if (
        not warning_info.outlier_gox
        and not warning_info.guarded_same_plate
        and not warning_info.missing_rates_files
        and not warning_info.reference_qc_fail_runs
    ):
        lines.append("## 警告なし")
        lines.append("")
        lines.append("警告は検出されませんでした。")
    else:
        if warning_info.outlier_gox:
            lines.append("## ⚠️ 異常GOx t50値の検出")
            lines.append("")
            for i, outlier_info in enumerate(warning_info.outlier_gox, 1):
                lines.append(f"### {i}. Round {outlier_info['round_id']}")
                lines.append("")
                lines.append(f"- **GOx t50 中央値**: {outlier_info['median_gox_t50']:.3f} min")
                lines.append(f"- **異常値の閾値**: [{outlier_info['low_threshold']:.3f}, {outlier_info['high_threshold']:.3f}] min")
                lines.append("- **検出された異常値**:")
                for outlier in outlier_info['outliers']:
                    lines.append(f"  - {outlier['run_id']}/{outlier['plate_id']}: {outlier['gox_t50']:.3f} min")
                lines.append("")
                if outlier_info['excluded']:
                    lines.append("**処理**: 異常値はround平均GOx t50の計算から**除外されました**。")
                else:
                    lines.append("**処理**: 異常値はround平均GOx t50の計算に**含まれています**。")
                lines.append("")

        if warning_info.guarded_same_plate:
            lines.append("## ⚠️ same_plate分母ガードによるフォールバック")
            lines.append("")
            lines.append("異常なsame_plate GOx t50が検出されたため、分母をround代表値へフォールバックした行です。")
            lines.append("")
            for i, info in enumerate(warning_info.guarded_same_plate, 1):
                lines.append(
                    f"### {i}. Round {info['round_id']} / Run {info['run_id']} / Plate {info['plate_id']}"
                )
                lines.append("")
                lines.append(f"- **same_plate GOx t50**: {info['same_plate_gox_t50_min']:.3f} min")
                lines.append(f"- **round代表GOx t50**: {info['fallback_gox_t50_min']:.3f} min")
                lines.append(
                    f"- **ガード閾値**: [{info['guard_low_threshold_min']:.3f}, {info['guard_high_threshold_min']:.3f}] min"
                )
                lines.append("- **処理**: `denominator_source = same_round` へフォールバック")
                lines.append("")
        
        if warning_info.missing_rates_files:
            lines.append("## ⚠️ 見つからないrates_with_rea.csv")
            lines.append("")
            for i, missing_info in enumerate(warning_info.missing_rates_files, 1):
                lines.append(f"### {i}. Round {missing_info['round_id']}, Run {missing_info['run_id']}")
                lines.append("")
                lines.append(f"- **期待されるパス**: `{missing_info['expected_path']}`")
                lines.append("- **影響**: このrunはFoG計算からスキップされます。")
                lines.append("")

        if warning_info.reference_qc_fail_runs:
            lines.append("## ⚠️ 参照abs(0) QC fail run")
            lines.append("")
            lines.append("同一run内の参照abs_activity_at_0のばらつきまたは絶対値に問題があるため、QC failとなったrunです。")
            lines.append("")
            for i, info in enumerate(warning_info.reference_qc_fail_runs, 1):
                lines.append(f"### {i}. Round {info.get('round_id', '')}, Run {info.get('run_id', '')}")
                lines.append("")
                lines.append(f"- **n_reference_wells**: {info.get('n_reference_wells', '')}")
                lines.append(f"- **ref_abs0_median**: {info.get('ref_abs0_median', '')}")
                lines.append(f"- **ref_abs0_mad**: {info.get('ref_abs0_mad', '')}")
                lines.append(f"- **ref_abs0_rel_mad**: {info.get('ref_abs0_rel_mad', '')}")
                lines.append(f"- **reason**: {info.get('reason', '')}")
                lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## 「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」を実行した場合")
        lines.append("")
        lines.append("`--exclude_outlier_gox`フラグを有効にして実行すると、以下の変更が行われます：")
        lines.append("")
        lines.append("### 変更点")
        lines.append("")
        lines.append("1. **異常GOx t50値の除外**:")
        lines.append("   - 検出された異常GOx t50値がround平均GOx t50の計算から除外されます")
        lines.append("   - これにより、round平均GOx t50がより安定した値になります")
        lines.append("")
        lines.append("2. **FoG値への影響**:")
        lines.append("   - 異常GOx t50値を持つplateのポリマー:")
        lines.append("     - そのplateにGOxがある場合: `same_plate`のGOxを使うため、**影響なし**")
        lines.append("     - そのplateにGOxがない場合: `same_round`のGOxを使うため、**round平均GOx t50の変更の影響を受ける**")
        lines.append("   - 他のplateのポリマー:")
        lines.append("     - `same_round`のGOxを使う場合: **round平均GOx t50の変更の影響を受ける**")
        lines.append("     - `same_plate`のGOxを使う場合: **影響なし**")
        lines.append("")
        lines.append("3. **出力ファイルの変更**:")
        lines.append("   - `fog_plate_aware.csv`: 各ポリマーのFoG値が変更される可能性があります")
        lines.append("   - `fog_plate_aware_round_averaged.csv`: Round平均FoG値が変更される可能性があります")
        lines.append("")
        lines.append("### 実行方法")
        lines.append("")
        lines.append("VS Codeの「実行とデバッグ」パネルから、以下の設定を選択してください：")
        lines.append("")
        lines.append("- **「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」**")
        lines.append("")
        lines.append("または、コマンドラインから：")
        lines.append("")
        lines.append("```bash")
        lines.append("python scripts/build_fog_plate_aware.py \\")
        lines.append("  --run_round_map meta/bo/run_round_map.tsv \\")
        lines.append("  --processed_dir data/processed \\")
        lines.append("  --out_dir data/processed \\")
        lines.append("  --exclude_outlier_gox")
        lines.append("```")
        lines.append("")
        lines.append("### 注意事項")
        lines.append("")
        lines.append("- 異常値を除外すると、round平均GOx t50が変わるため、FoG値も変わります")
        lines.append("- 除外するかどうかは、データの品質と研究の目的に応じて判断してください")
        lines.append("- 異常値が多数ある場合、除外によりround平均GOx t50が大きく変わる可能性があります")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_fog_summary(
    t50_path: Path,
    run_id: str,
    *,
    manifest_path: Optional[Path] = None,
    input_tidy_path: Optional[Path] = None,
    row_map_path: Optional[Path] = None,
    polymer_solvent_path: Optional[Path] = None,
    native_activity_min_rel: float = NATIVE_ACTIVITY_MIN_REL_DEFAULT,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Build FoG summary from t50 CSV: FoG = t50_polymer / t50_bare_GOx (same run only).

    - t50 unit: minutes (T50_UNIT).
    - t50_censored: 1 if t50 is missing or REA plateau >= t50 target (did not reach target); 0 otherwise.
      Target uses t50_target_rea_percent when present (new schema), else legacy 50%.
    - gox_t50_same_run: bare GOx t50 in same run (min). NaN if no GOx in run.
    - fog: t50 / gox_t50_same_run. NaN if no GOx in run or t50 missing.
    - log_fog: log(fog) for BO; NaN when fog <= 0 or missing (handle consistently).
    - fog_missing_reason: e.g. "no_bare_gox_in_run" when GOx absent in run; empty otherwise.
    - use_for_bo: if row_map_path is provided, reads use_for_bo from metadata TSV (defaults to True).
      Controls whether this polymer should be included in Bayesian optimization learning data.
    - abs_activity_at_0: absolute activity anchor (heat=0; from t50 CSV when available).
    - gox_abs_activity_at_0_ref: GOx absolute-activity reference at heat=0 from per-run/round fallback resolver.
    - native_activity_rel_at_0: abs_activity_at_0 / gox_abs_activity_at_0.
    - native_activity_feasible: 1 when native_activity_rel_at_0 >= native_activity_min_rel, else 0.
    - fog_native_constrained: fog masked by native_activity_feasible (NaN when infeasible).
    - log_fog_native_constrained: log(fog_native_constrained), NaN when infeasible.
    - native_activity_soft_penalty: clip(native_activity_rel_at_0, 0, 1)^2 (default exponent).
    - fog_native_soft: fog * native_activity_soft_penalty.
    - log_fog_native_soft: log(fog_native_soft), NaN when missing/non-positive.
    - FoG-activity bonus/penalty objective:
      abs0_vs_solvent_control = abs0 / abs0_solvent_control
      fog_vs_solvent_control = t50 / t50_solvent_control
      fog_activity_bonus_penalty = fog_vs_solvent_control * abs_activity_balance(abs0_vs_solvent_control)
      where abs_activity_balance(U0*) = clip(U0*, 0, 1)^2 * [1 + 0.35 * clip(U0* - 1.05, 0, 0.30)].
      Legacy aliases are preserved: fog_solvent_balanced / log_fog_solvent_balanced.
    - Lineage: run_id, input_t50_file, input_tidy (from manifest or passed).
    """
    t50_path = Path(t50_path)
    run_id = str(run_id).strip()
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    df = pd.read_csv(t50_path)

    if "run_id" not in df.columns:
        df["run_id"] = run_id
    if "polymer_id" not in df.columns:
        raise ValueError(f"t50 CSV must have polymer_id, got: {list(df.columns)}")
    if "reference_polymer_id" in df.columns:
        df["reference_polymer_id"] = df["reference_polymer_id"].astype(str).str.strip()
        df["reference_polymer_id"] = np.where(
            df["reference_polymer_id"] != "",
            df["reference_polymer_id"],
            reference_polymer_id,
        )
    else:
        df["reference_polymer_id"] = reference_polymer_id

    native_activity_min_rel = float(native_activity_min_rel)
    if not np.isfinite(native_activity_min_rel) or native_activity_min_rel <= 0.0:
        raise ValueError(
            f"native_activity_min_rel must be a positive finite number, got {native_activity_min_rel!r}"
        )

    # Prefer t50_exp_min, fallback to t50_linear_min (unit: min)
    t50_exp = pd.to_numeric(df.get("t50_exp_min", np.nan), errors="coerce")
    t50_lin = pd.to_numeric(df.get("t50_linear_min", np.nan), errors="coerce")
    df["t50_min"] = np.where(np.isfinite(t50_exp), t50_exp, t50_lin)
    df["abs_activity_at_0"] = pd.to_numeric(df.get("abs_activity_at_0", np.nan), errors="coerce")
    df["abs_activity_at_20"] = pd.to_numeric(df.get("abs_activity_at_20", np.nan), errors="coerce")
    df["gox_abs_activity_at_0_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_0_ref", np.nan), errors="coerce")
    df["gox_abs_activity_at_20_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce")
    df["functional_activity_at_20_rel"] = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    df["functional_reference_source"] = df.get("functional_reference_source", pd.Series([""] * len(df))).astype(str)
    df["functional_reference_round_id"] = df.get("functional_reference_round_id", pd.Series([""] * len(df))).astype(str)
    df["functional_reference_run_id"] = df.get("functional_reference_run_id", pd.Series([""] * len(df))).astype(str)

    # Censored: no t50 or plateau >= target (REA did not reach t50 target)
    fit_plateau = pd.to_numeric(df.get("fit_plateau", np.nan), errors="coerce")
    fit_model = df.get("fit_model", "").astype(str)
    no_t50 = ~np.isfinite(df["t50_min"])
    target = pd.to_numeric(df.get("t50_target_rea_percent", np.nan), errors="coerce")
    target = np.where(np.isfinite(target), target, 50.0)
    plateau_not_reached = (
        fit_model.str.contains("exp_plateau", na=False)
        & np.isfinite(fit_plateau)
        & np.isfinite(target)
        & (fit_plateau >= target)
    )
    df["t50_censored"] = (no_t50 | plateau_not_reached).astype(int)

    # Bare reference-polymer t50 in same run (this CSV is single-run, so one value)
    ref_rows = df[
        df["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
    ]
    if ref_rows.empty:
        gox_t50_same_run = np.nan
        fog_missing_reason_default = _missing_reference_reason(reference_polymer_id)
    else:
        gox_t50_same_run = float(ref_rows["t50_min"].iloc[0]) if np.isfinite(ref_rows["t50_min"].iloc[0]) else np.nan
        fog_missing_reason_default = "" if np.isfinite(gox_t50_same_run) else _missing_reference_reason(reference_polymer_id)

    df["gox_t50_same_run_min"] = gox_t50_same_run
    df["fog_missing_reason"] = fog_missing_reason_default

    # Native-activity guard baseline: same-run reference-polymer absolute activity at heat=0.
    gox_abs0_same_run = np.nan
    if not ref_rows.empty:
        gox_abs0_vals = pd.to_numeric(ref_rows.get("abs_activity_at_0", np.nan), errors="coerce")
        gox_abs0_vals = gox_abs0_vals[np.isfinite(gox_abs0_vals) & (gox_abs0_vals > 0)]
        if not gox_abs0_vals.empty:
            gox_abs0_same_run = float(gox_abs0_vals.iloc[0])
    ref_abs0 = pd.to_numeric(df.get("gox_abs_activity_at_0_ref", np.nan), errors="coerce")
    if np.isfinite(gox_abs0_same_run):
        ref_abs0 = np.where(np.isfinite(ref_abs0), ref_abs0, float(gox_abs0_same_run))
    df["gox_abs_activity_at_0"] = pd.to_numeric(ref_abs0, errors="coerce")

    native_rel = np.where(
        np.isfinite(df["abs_activity_at_0"])
        & np.isfinite(df["gox_abs_activity_at_0"])
        & (df["gox_abs_activity_at_0"] > 0.0),
        df["abs_activity_at_0"] / df["gox_abs_activity_at_0"],
        np.nan,
    )
    df["native_activity_rel_at_0"] = native_rel
    df["native_activity_min_rel_threshold"] = float(native_activity_min_rel)
    native_feasible = np.isfinite(native_rel) & (native_rel >= float(native_activity_min_rel))
    df["native_activity_feasible"] = native_feasible.astype(int)

    native_reason = np.full(len(df), "", dtype=object)
    missing_abs0 = ~np.isfinite(df["abs_activity_at_0"])
    missing_gox0 = ~np.isfinite(df["gox_abs_activity_at_0"]) | (df["gox_abs_activity_at_0"] <= 0.0)
    below_native = np.isfinite(native_rel) & (native_rel < float(native_activity_min_rel))
    native_reason = np.where(missing_abs0, "missing_abs_activity_at_0", native_reason)
    native_reason = np.where(missing_gox0 & ~missing_abs0, "missing_gox_abs_activity_at_0", native_reason)
    native_reason = np.where(below_native, "native_activity_below_threshold", native_reason)
    df["fog_constraint_reason"] = native_reason

    # Functional activity at 20 min relative to reference polymer.
    # Prefer value pre-computed in t50 CSV; otherwise derive from abs_activity_at_20 / gox_abs_activity_at_20_ref.
    gox_abs20_same_run = np.nan
    if not ref_rows.empty:
        gox_abs20_vals = pd.to_numeric(ref_rows.get("abs_activity_at_20", np.nan), errors="coerce")
        gox_abs20_vals = gox_abs20_vals[np.isfinite(gox_abs20_vals) & (gox_abs20_vals > 0)]
        if not gox_abs20_vals.empty:
            gox_abs20_same_run = float(gox_abs20_vals.iloc[0])

    ref_abs20 = pd.to_numeric(df.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce")
    if not (np.isfinite(ref_abs20).any()):
        ref_abs20 = np.full(len(df), gox_abs20_same_run, dtype=float)
    ref_abs20 = pd.to_numeric(ref_abs20, errors="coerce")
    df["gox_abs_activity_at_20_ref"] = ref_abs20

    fa20 = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    fallback_fa20 = np.where(
        np.isfinite(df["abs_activity_at_20"]) & np.isfinite(df["gox_abs_activity_at_20_ref"]) & (df["gox_abs_activity_at_20_ref"] > 0),
        df["abs_activity_at_20"] / df["gox_abs_activity_at_20_ref"],
        np.nan,
    )
    df["functional_activity_at_20_rel"] = np.where(np.isfinite(fa20), fa20, fallback_fa20)
    if "functional_reference_source" in df.columns:
        src = df["functional_reference_source"].astype(str).str.strip()
        default_src_present = (
            "same_run_gox" if _is_reference_polymer_id(reference_polymer_id, DEFAULT_REFERENCE_POLYMER_ID) else "same_run_reference"
        )
        default_src_missing = (
            "missing_gox_reference"
            if _is_reference_polymer_id(reference_polymer_id, DEFAULT_REFERENCE_POLYMER_ID)
            else "missing_reference_polymer"
        )
        src = src.where(src != "", np.where(np.isfinite(df["gox_abs_activity_at_20_ref"]), default_src_present, default_src_missing))
        df["functional_reference_source"] = src

    fa20_rel = pd.to_numeric(df["functional_activity_at_20_rel"], errors="coerce")
    log_fa20 = np.full(len(df), np.nan, dtype=float)
    ok_fa = np.isfinite(fa20_rel) & (fa20_rel > 0)
    log_fa20[ok_fa] = np.log(fa20_rel[ok_fa])
    df["log_functional_activity_at_20_rel"] = log_fa20

    # FoG and log FoG
    with np.errstate(divide="ignore", invalid="ignore"):
        fog = np.where(
            np.isfinite(df["t50_min"]) & np.isfinite(df["gox_t50_same_run_min"]) & (df["gox_t50_same_run_min"] > 0),
            df["t50_min"] / df["gox_t50_same_run_min"],
            np.nan,
        )
    df["fog"] = fog
    # log_fog: only when fog > 0 and finite; else NaN (consistent for BO)
    log_fog = np.full_like(fog, np.nan, dtype=float)
    ok = np.isfinite(fog) & (fog > 0)
    log_fog[ok] = np.log(fog[ok])
    df["log_fog"] = log_fog

    # Constrained objective for BO/ranking:
    # optimize FoG only among polymers that keep enough native activity at heat=0.
    fog_native = np.where(native_feasible, fog, np.nan)
    df["fog_native_constrained"] = fog_native
    log_fog_native = np.full_like(fog_native, np.nan, dtype=float)
    ok_native = np.isfinite(fog_native) & (fog_native > 0)
    log_fog_native[ok_native] = np.log(fog_native[ok_native])
    df["log_fog_native_constrained"] = log_fog_native
    soft_penalty, fog_soft, log_fog_soft = _compute_penalized_fog_objective(fog, native_rel)
    df["native_activity_soft_penalty"] = soft_penalty
    df["fog_native_soft"] = fog_soft
    df["log_fog_native_soft"] = log_fog_soft
    stock_solvent_map, control_solvent_map = _load_polymer_solvent_maps(polymer_solvent_path)
    df = _apply_polymer_solvent_maps(
        df,
        stock_map=stock_solvent_map,
        control_map=control_solvent_map,
    )
    df = _add_solvent_balanced_objective_columns(
        df,
        reference_polymer_id=reference_polymer_id,
    )

    # Load use_for_bo from row_map if provided
    if row_map_path is not None and Path(row_map_path).is_file():
        from gox_plate_pipeline.loader import read_row_map_tsv
        row_map = read_row_map_tsv(Path(row_map_path))
        # Merge by polymer_id (t50 CSV aggregates by polymer_id, so we match by polymer_id)
        df["polymer_id_norm"] = df["polymer_id"].astype(str).str.strip()
        row_map["polymer_id_norm"] = row_map["polymer_id"].astype(str).str.strip()
        # Take the first match for each polymer_id (in case of duplicates, use first)
        use_for_bo_map = row_map.groupby("polymer_id_norm")["use_for_bo"].first().to_dict()
        df["use_for_bo"] = df["polymer_id_norm"].map(use_for_bo_map).fillna(True)  # Default True if not found
        df = df.drop(columns=["polymer_id_norm"])
    else:
        df["use_for_bo"] = True  # Default: include in BO

    # Lineage columns
    df["input_t50_file"] = t50_path.name
    if input_tidy_path is not None:
        df["input_tidy"] = str(Path(input_tidy_path).resolve())
    elif manifest_path is not None and manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            inputs = payload.get("manifest", {}).get("input_files", [])
            first_path = inputs[0].get("path", "") if inputs else ""
            df["input_tidy"] = first_path
        except Exception:
            df["input_tidy"] = ""
    else:
        df["input_tidy"] = ""

    # Output columns (order and names for BO and lineage)
    out_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "t50_min",
        "t50_definition",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "native_activity_rel_at_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "abs_activity_at_20",
        "gox_abs_activity_at_20_ref",
        "functional_activity_at_20_rel",
        "log_functional_activity_at_20_rel",
        "functional_reference_source",
        "functional_reference_round_id",
        "functional_reference_run_id",
        "t50_censored",
        "gox_t50_same_run_min",
        "fog",
        "log_fog",
        "fog_native_constrained",
        "log_fog_native_constrained",
        "native_activity_soft_penalty",
        "fog_native_soft",
        "log_fog_native_soft",
        "stock_solvent_group",
        "solvent_group",
        "solvent_control_polymer_id",
        "solvent_control_abs_activity_at_0",
        "solvent_control_t50_min",
        "solvent_control_fog",
        "abs0_vs_solvent_control",
        "fog_vs_solvent_control",
        "solvent_activity_down_penalty",
        "solvent_activity_up_bonus",
        "solvent_activity_balance_factor",
        "abs_activity_down_penalty",
        "abs_activity_up_bonus",
        "abs_activity_balance_factor",
        ABS_ACTIVITY_OBJECTIVE_COL,
        ABS_ACTIVITY_OBJECTIVE_LOG_COL,
        ABS_ACTIVITY_OBJECTIVE_SEM_COL,
        OBJECTIVE_LOGLINEAR_MAIN_COL,
        OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
        "fog_solvent_balanced",
        "log_fog_solvent_balanced",
        "fog_constraint_reason",
        "fog_missing_reason",
        "use_for_bo",
        "n_points",
        "input_t50_file",
        "input_tidy",
    ]
    # Keep only columns that exist
    available = [c for c in out_cols if c in df.columns]
    return _sync_abs_objective_alias_columns(df[available].copy())


def write_fog_summary_csv(
    fog_df: pd.DataFrame,
    out_path: Path,
    *,
    t50_unit: str = T50_UNIT,
) -> None:
    """Write FoG summary CSV. t50 and gox_t50 are in minutes (t50_unit)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fog_df.to_csv(out_path, index=False)
    # Document unit in a comment line would require custom write; unit is in T50_UNIT and README/doc.


def _load_polymer_color_map(color_map_path: Optional[Path]) -> Dict[str, str]:
    """Load polymer_id -> color map from YAML."""
    if color_map_path is None:
        return {}
    path = Path(color_map_path)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if isinstance(payload, dict) and "polymer_id" in payload and isinstance(payload["polymer_id"], dict):
        payload = payload["polymer_id"]
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in payload.items():
        if isinstance(v, str) and v.strip():
            out[str(k).strip()] = v.strip()
    return out


def _add_scatter_polymer_mapping_legend(
    ax: Any,
    scatter_df: pd.DataFrame,
    *,
    color_map: Optional[Dict[str, str]] = None,
    polymer_col: str = "polymer_id",
    feasible_col: str = "_feasible",
) -> None:
    """
    Add a compact lower-left mapping legend (marker/color -> polymer_id) for scatter plots.
    """
    if scatter_df.empty:
        return
    from matplotlib.lines import Line2D

    df = scatter_df.copy()
    if polymer_col not in df.columns:
        return
    df[polymer_col] = df[polymer_col].astype(str).str.strip()
    df = df[df[polymer_col] != ""].copy()
    if df.empty:
        return
    if "fog" in df.columns:
        df = df.sort_values("fog", ascending=False, kind="mergesort")

    def _polymer_map_order_key(polymer_id: str) -> tuple[int, int, str]:
        """Order legend as: GOx/GOx with*, PMPC, PMTAC, PMBTA-1..N, others."""
        pid = str(polymer_id).strip()
        norm = _normalize_polymer_id_token(pid)
        if norm == "GOX":
            return (0, 0, "")
        if norm.startswith("GOX WITH"):
            suffix = norm[len("GOX WITH") :].strip()
            return (0, 1, suffix)
        if norm.startswith("GOX"):
            return (0, 2, norm)
        if norm == "PMPC":
            return (1, 0, "")
        if norm == "PMTAC":
            return (2, 0, "")
        m = re.match(r"^PMBTA-(\d+)$", norm)
        if m is not None:
            return (3, int(m.group(1)), "")
        return (4, 0, norm)

    cmap = color_map or {}
    default_color = "#4C78A8"
    entries: List[tuple[str, Any]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        pid = str(row.get(polymer_col, "")).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        color = cmap.get(pid, default_color)
        feasible = bool(row.get(feasible_col, True))
        if feasible:
            handle = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4.2,
                markerfacecolor=color,
                markeredgecolor="0.2",
                markeredgewidth=0.5,
                alpha=0.9,
                label=pid,
            )
        else:
            handle = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4.2,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=0.9,
                alpha=0.9,
                label=pid,
            )
        entries.append((pid, handle))
    if not entries:
        return

    entries = sorted(entries, key=lambda x: _polymer_map_order_key(x[0]))
    handles: List[Any] = [h for _, h in entries]
    if not handles:
        return

    ncol = 1 if len(handles) <= 8 else 2 if len(handles) <= 18 else 3
    legend = ax.legend(
        handles=handles,
        title="Polymer map",
        loc="lower left",
        bbox_to_anchor=(0.012, 0.012),
        ncol=ncol,
        fontsize=4.6,
        title_fontsize=5.0,
        frameon=True,
        framealpha=0.90,
        borderpad=0.26,
        labelspacing=0.22,
        handletextpad=0.35,
        columnspacing=0.78,
        borderaxespad=0.14,
    )
    frame = legend.get_frame()
    frame.set_edgecolor("0.65")
    frame.set_linewidth(0.5)


def _plot_run_ranking_bar(
    rank_df: pd.DataFrame,
    *,
    value_col: str,
    error_col: Optional[str] = None,
    rank_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    Plot one run-level ranking bar chart (paper-grade, English).

    Returns True when a figure was written.
    """
    if rank_df.empty:
        return False
    data = rank_df.copy()
    data = data[np.isfinite(pd.to_numeric(data.get(value_col, np.nan), errors="coerce"))].copy()
    data = data[np.isfinite(pd.to_numeric(data.get(rank_col, np.nan), errors="coerce"))].copy()
    if data.empty:
        return False
    data = data.sort_values(rank_col, ascending=True).reset_index(drop=True)
    if error_col is not None and str(error_col).strip():
        if error_col in data.columns:
            data[error_col] = pd.to_numeric(data.get(error_col, np.nan), errors="coerce")
        else:
            error_col = None

    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    labels = []
    values = []
    colors = []
    cmap = color_map or {}
    default_color = "#4C78A8"
    for _, row in data.iterrows():
        pid = str(row.get("polymer_id", "")).strip()
        labels.append(pid)
        values.append(float(row[value_col]))
        if _is_reference_polymer_id(pid, reference_polymer_id=reference_polymer_id):
            colors.append("#808080")
        else:
            colors.append(cmap.get(pid, default_color))
    xerr_values: Optional[np.ndarray] = None
    if error_col is not None:
        err = pd.to_numeric(data.get(error_col, np.nan), errors="coerce").to_numpy(dtype=float)
        err = np.where(np.isfinite(err) & (err > 0.0), err, 0.0)
        if np.any(err > 0.0):
            xerr_values = err

    fig_h = max(2.2, 0.28 * len(labels) + 0.8)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(5.0, fig_h))
        y = np.arange(len(labels), dtype=float)
        bar_kwargs = dict(
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
        )
        if xerr_values is not None:
            bar_kwargs["xerr"] = xerr_values
            bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.9,
            }
        ax.barh(y, values, **bar_kwargs)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_fog_native_constrained_tradeoff(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    Plot trade-off scatter: x = native_activity_rel_at_0, y = FoG, with vertical line at theta.
    Paper-grade, English. Excludes the reference polymer row. Returns True when a figure was written.
    """
    df = run_df.copy()
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    x = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    x_sem = pd.to_numeric(df.get("native_activity_rel_at_0_sem", np.nan), errors="coerce")
    y = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    y_sem = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    has_any = np.any(np.isfinite(x)) and np.any(np.isfinite(y))
    if not has_any:
        return False
    fe_col = pd.to_numeric(df.get("native_activity_feasible", np.nan), errors="coerce")
    if np.any(np.isfinite(fe_col)):
        df["_feasible"] = fe_col.fillna(0).astype(int) > 0
    else:
        df["_feasible"] = np.isfinite(x) & (x >= float(theta))
    cmap = color_map or {}
    default_color = "#4C78A8"

    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        plotted: List[tuple[str, float, float, bool]] = []
        for _, row in df.iterrows():
            xi = float(row.get("native_activity_rel_at_0", np.nan))
            yi = float(row.get("fog", np.nan))
            if not (np.isfinite(xi) and np.isfinite(yi) and yi > 0):
                continue
            pid = str(row.get("polymer_id", "")).strip()
            fe = bool(row.get("_feasible", False))
            color = cmap.get(pid, default_color)
            xi_sem = float(row.get("native_activity_rel_at_0_sem", np.nan))
            yi_sem = float(row.get("fog_sem", np.nan))
            if (np.isfinite(xi_sem) and xi_sem > 0.0) or (np.isfinite(yi_sem) and yi_sem > 0.0):
                ax.errorbar(
                    xi,
                    yi,
                    xerr=(xi_sem if np.isfinite(xi_sem) and xi_sem > 0.0 else None),
                    yerr=(yi_sem if np.isfinite(yi_sem) and yi_sem > 0.0 else None),
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.8,
                    zorder=16,
                )
            ax.scatter(
                xi,
                yi,
                s=28,
                color=color,
                edgecolors="0.2",
                linewidths=0.5,
                alpha=0.9,
                zorder=20,
                marker="o" if fe else "s",
            )
            plotted.append((pid, xi, yi, fe))
        ax.axvline(
            x=theta,
            color="0.4",
            linestyle="--",
            linewidth=0.8,
            zorder=10,
            label=rf"$\theta$ = {theta:.2f}",
        )
        ax.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
        ax.set_ylabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
        ax.set_title("FoG vs native activity (constrained objective)")
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlim(0.0, None)
        ax.set_ylim(0.0, None)
        _add_scatter_polymer_mapping_legend(ax, df, color_map=cmap, feasible_col="_feasible")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_fog_native_soft_tradeoff(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    penalty_exponent: float = NATIVE_ACTIVITY_SOFT_PENALTY_EXPONENT,
) -> bool:
    """
    Plot FoG-vs-native map with iso-curves of the soft objective:
      score = FoG * clip(U0, 0, 1)^exponent
    """
    df = run_df.copy()
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False

    native_rel = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    if not np.isfinite(native_rel).any():
        native_rel = pd.to_numeric(df.get("abs0_vs_gox", np.nan), errors="coerce")
    fog = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    soft_penalty, fog_soft, _ = _compute_penalized_fog_objective(
        fog.to_numpy(dtype=float),
        native_rel.to_numpy(dtype=float),
        exponent=penalty_exponent,
    )
    df["_native_rel"] = native_rel
    df["_fog"] = fog
    df["_soft_penalty"] = soft_penalty
    df["_fog_native_soft"] = fog_soft
    df["_x_sem"] = pd.to_numeric(df.get("native_activity_rel_at_0_sem", np.nan), errors="coerce")
    df["_y_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df = df[
        np.isfinite(df["_native_rel"])
        & np.isfinite(df["_fog"])
        & (df["_fog"] > 0.0)
        & np.isfinite(df["_fog_native_soft"])
        & (df["_fog_native_soft"] > 0.0)
    ].copy()
    if df.empty:
        return False

    exp_lbl = _format_penalty_exponent_label(penalty_exponent)
    theta_vals = pd.to_numeric(df.get("native_activity_min_rel_threshold", np.nan), errors="coerce")
    theta = float(np.nanmedian(theta_vals)) if np.any(np.isfinite(theta_vals)) else np.nan
    feasible_vals = pd.to_numeric(df.get("native_activity_feasible", np.nan), errors="coerce")
    if np.isfinite(feasible_vals).any():
        df["_feasible"] = feasible_vals.fillna(0).astype(int) > 0
    elif np.isfinite(theta):
        df["_feasible"] = np.isfinite(df["_native_rel"]) & (df["_native_rel"] >= theta)
    else:
        df["_feasible"] = True
    x_max = float(np.nanmax(df["_native_rel"]))
    y_max = float(np.nanmax(df["_fog"]))
    x_right = max(1.05, x_max * 1.14)
    y_top = max(1.05, y_max * 1.16)
    score_vals = pd.to_numeric(df["_fog_native_soft"], errors="coerce").to_numpy(dtype=float)
    score_vals = score_vals[np.isfinite(score_vals) & (score_vals > 0.0)]
    if score_vals.size == 0:
        return False
    score_q = np.nanquantile(score_vals, [0.50, 0.75, 0.90])
    levels = sorted({float(v) for v in score_q if np.isfinite(v) and v > 0.0})

    cmap = color_map or {}
    default_color = "#4C78A8"

    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        x_curve = np.linspace(0.05, x_right, 500)
        clip_term = np.power(np.clip(x_curve, 0.0, 1.0), float(penalty_exponent))
        for lv in levels:
            y_curve = np.where(clip_term > 0.0, lv / clip_term, np.nan)
            y_curve = np.where(np.isfinite(y_curve) & (y_curve <= y_top * 1.35), y_curve, np.nan)
            if np.any(np.isfinite(y_curve)):
                ax.plot(
                    x_curve,
                    y_curve,
                    color="0.55",
                    linestyle=(0, (3, 2)),
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=8,
                )
                yi = y_curve[np.isfinite(y_curve)][-1]
                if np.isfinite(yi) and yi <= y_top * 1.30:
                    ax.text(
                        x_right * 0.985,
                        yi,
                        f"S={lv:.2f}",
                        ha="right",
                        va="bottom",
                        fontsize=5.2,
                        color="0.38",
                    )

        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_native_rel"])
            yi = float(row["_fog"])
            xi_sem = float(row.get("_x_sem", np.nan))
            yi_sem = float(row.get("_y_sem", np.nan))
            feasible = bool(row.get("_feasible", True))
            if (np.isfinite(xi_sem) and xi_sem > 0.0) or (np.isfinite(yi_sem) and yi_sem > 0.0):
                ax.errorbar(
                    xi,
                    yi,
                    xerr=(xi_sem if np.isfinite(xi_sem) and xi_sem > 0.0 else None),
                    yerr=(yi_sem if np.isfinite(yi_sem) and yi_sem > 0.0 else None),
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.72,
                    capsize=1.8,
                    alpha=0.78,
                    zorder=16,
                )
            if feasible:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=18,
                )

        if np.isfinite(theta):
            ax.axvline(
                x=theta,
                color="0.35",
                linestyle=(0, (3, 2)),
                linewidth=0.8,
                zorder=10,
            )
        ax.axvline(
            x=1.0,
            color="0.45",
            linestyle=(0, (1.5, 2.0)),
            linewidth=0.75,
            zorder=10,
        )
        ax.axhline(
            y=1.0,
            color="0.45",
            linestyle=(0, (1.5, 2.0)),
            linewidth=0.75,
            zorder=10,
        )
        ax.set_xlim(0.0, x_right)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
        ax.set_ylabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
        ax.set_title(
            rf"Soft objective map ($S=\mathrm{{FoG}}\times\mathrm{{clip}}(U_0,0,1)^{{{exp_lbl}}}$)"
        )
        ax.grid(True, linestyle=":", alpha=0.30)
        _add_scatter_polymer_mapping_legend(ax, df, color_map=cmap, feasible_col="_feasible")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_fog_solvent_balanced_tradeoff(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    Plot absolute-activity bonus/penalty trade-off map with iso-curves:
      S = FoG* x abs_activity_balance(U0*)
    """
    df = _add_solvent_balanced_objective_columns(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    # Focus on polymer candidates; keep solvent controls only in dedicated control checks.
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    df["_x"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["_y"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["_score"] = pd.to_numeric(df.get(ABS_ACTIVITY_OBJECTIVE_COL, np.nan), errors="coerce")
    if not np.isfinite(df["_score"]).any():
        _, _, _, score, _ = _compute_solvent_balanced_objective(df["_y"], df["_x"])
        df["_score"] = score
    df["_x_sem"] = np.nan
    df["_y_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df = df[
        np.isfinite(df["_x"])
        & np.isfinite(df["_y"])
        & (df["_y"] > 0.0)
        & np.isfinite(df["_score"])
        & (df["_score"] > 0.0)
    ].copy()
    if df.empty:
        return False
    df["_marker_filled"] = df["_x"] >= float(MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD)

    x_max = float(np.nanmax(df["_x"]))
    y_max = float(np.nanmax(df["_y"]))
    x_right = max(1.10, x_max * 1.14)
    y_top = max(1.10, y_max * 1.16)
    score_vals = df["_score"].to_numpy(dtype=float)
    score_vals = score_vals[np.isfinite(score_vals) & (score_vals > 0.0)]
    if score_vals.size == 0:
        return False
    score_q = np.nanquantile(score_vals, [0.50, 0.75, 0.90])
    levels = sorted({float(v) for v in score_q if np.isfinite(v) and v > 0.0})

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        x_curve = np.linspace(0.05, x_right, 500)
        _, _, factor_curve = _compute_solvent_balanced_activity_factor(
            x_curve,
            exponent=SOLVENT_BALANCED_PENALTY_EXPONENT,
            up_bonus_coef=SOLVENT_BALANCED_UP_BONUS_COEF,
            up_bonus_max_delta=SOLVENT_BALANCED_UP_BONUS_MAX_DELTA,
            up_bonus_deadband=SOLVENT_BALANCED_UP_BONUS_DEADBAND,
        )
        for lv in levels:
            y_curve = np.where(factor_curve > 0.0, lv / factor_curve, np.nan)
            y_curve = np.where(np.isfinite(y_curve) & (y_curve <= y_top * 1.35), y_curve, np.nan)
            if np.any(np.isfinite(y_curve)):
                ax.plot(
                    x_curve,
                    y_curve,
                    color="0.55",
                    linestyle=(0, (3, 2)),
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=8,
                )
                yi = y_curve[np.isfinite(y_curve)][-1]
                if np.isfinite(yi) and yi <= y_top * 1.30:
                    ax.text(
                        x_right * 0.985,
                        yi,
                        f"S={lv:.2f}",
                        ha="right",
                        va="bottom",
                        fontsize=5.2,
                        color="0.38",
                    )
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_x"])
            yi = float(row["_y"])
            yi_sem = float(row.get("_y_sem", np.nan))
            feasible = bool(row.get("_feasible", True))
            if np.isfinite(yi_sem) and yi_sem > 0.0:
                ax.errorbar(
                    xi,
                    yi,
                    yerr=yi_sem,
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.72,
                    capsize=1.8,
                    alpha=0.78,
                    zorder=16,
                )
            if feasible:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=18,
                )
        ax.axvline(
            x=1.0,
            color="0.45",
            linestyle=(0, (1.5, 2.0)),
            linewidth=0.75,
            zorder=10,
        )
        ax.axhline(
            y=1.0,
            color="0.45",
            linestyle=(0, (1.5, 2.0)),
            linewidth=0.75,
            zorder=10,
        )
        ax.set_xlim(0.0, x_right)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel(r"$U_{0}^{\mathrm{solv}}$ (vs solvent-matched control at 0 min)")
        ax.set_ylabel("FoG_solv = t50 / t50_solvent_control")
        ax.set_title("FoG-activity bonus/penalty objective map")
        ax.grid(True, linestyle=":", alpha=0.30)
        _add_scatter_polymer_mapping_legend(ax, df, color_map=cmap, feasible_col="_feasible")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _prepare_loglinear_objective_df(
    run_df: pd.DataFrame,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """Prepare solvent-matched U0*/FoG* table with primary log-linear objective and rank."""
    df = _add_solvent_balanced_objective_columns(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty or ("polymer_id" not in df.columns):
        return pd.DataFrame()
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return df
    df["abs0_vs_solvent_control"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["fog_vs_solvent_control"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["fog_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    if OBJECTIVE_LOGLINEAR_MAIN_COL in df.columns:
        df[OBJECTIVE_LOGLINEAR_MAIN_COL] = pd.to_numeric(df.get(OBJECTIVE_LOGLINEAR_MAIN_COL, np.nan), errors="coerce")
    else:
        score, _ = _compute_loglinear_main_objective(
            df["fog_vs_solvent_control"],
            df["abs0_vs_solvent_control"],
            weight_lambda=OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT,
        )
        df[OBJECTIVE_LOGLINEAR_MAIN_COL] = score
    if OBJECTIVE_LOGLINEAR_MAIN_EXP_COL in df.columns:
        df[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = pd.to_numeric(
            df.get(OBJECTIVE_LOGLINEAR_MAIN_EXP_COL, np.nan), errors="coerce"
        )
    else:
        df[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = np.where(
            np.isfinite(df[OBJECTIVE_LOGLINEAR_MAIN_COL]),
            np.exp(df[OBJECTIVE_LOGLINEAR_MAIN_COL]),
            np.nan,
        )
    df = df[
        np.isfinite(df["abs0_vs_solvent_control"])
        & (df["abs0_vs_solvent_control"] > 0.0)
        & np.isfinite(df["fog_vs_solvent_control"])
        & (df["fog_vs_solvent_control"] > 0.0)
    ].copy()
    if df.empty:
        return df
    df = _rank_by_loglinear_objective(
        df,
        score_col=OBJECTIVE_LOGLINEAR_MAIN_COL,
        rank_col=OBJECTIVE_LOGLINEAR_MAIN_RANK_COL,
    )
    df["_pareto"] = _compute_pareto_front_mask(
        df["abs0_vs_solvent_control"],
        df["fog_vs_solvent_control"],
    )
    df["_pmtac_like"] = (df["abs0_vs_solvent_control"] < 1.0) & (df["fog_vs_solvent_control"] > 1.0)
    return df


def _plot_maina_log_u0_vs_log_fog_iso_score(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainA: log(U0*) vs log(FoG*) with iso-score lines."""
    df = _prepare_loglinear_objective_df(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    df["_log_u0"] = np.log(df["abs0_vs_solvent_control"].to_numpy(dtype=float))
    df["_log_fog"] = np.log(df["fog_vs_solvent_control"].to_numpy(dtype=float))
    df = df[np.isfinite(df["_log_u0"]) & np.isfinite(df["_log_fog"])].copy()
    if df.empty:
        return False

    score_vals = pd.to_numeric(df.get(OBJECTIVE_LOGLINEAR_MAIN_COL, np.nan), errors="coerce").to_numpy(dtype=float)
    score_vals = score_vals[np.isfinite(score_vals)]
    if score_vals.size == 0:
        return False
    levels = sorted(
        {
            float(v)
            for v in np.nanquantile(score_vals, [0.40, 0.65, 0.85])
            if np.isfinite(v)
        }
    )
    if not levels:
        levels = [float(np.nanmedian(score_vals))]

    x_min = float(np.nanmin(df["_log_u0"]))
    x_max = float(np.nanmax(df["_log_u0"]))
    y_min = float(np.nanmin(df["_log_fog"]))
    y_max = float(np.nanmax(df["_log_fog"]))
    x_left = min(-0.25, x_min - 0.18)
    x_right = max(0.25, x_max + 0.18)
    y_bottom = min(-0.25, y_min - 0.18)
    y_top = max(0.25, y_max + 0.18)

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        x_curve = np.linspace(x_left, x_right, 420)
        for lv in levels:
            y_curve = lv - float(OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT) * x_curve
            ax.plot(
                x_curve,
                y_curve,
                color="0.55",
                linestyle=(0, (3, 2)),
                linewidth=0.7,
                alpha=0.85,
                zorder=8,
            )
        df["_marker_filled"] = pd.to_numeric(
            df.get("abs0_vs_solvent_control", np.nan), errors="coerce"
        ) >= float(MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD)
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_log_u0"])
            yi = float(row["_log_fog"])
            fog_sem = float(row.get("fog_sem", np.nan))
            fog_rel = float(row.get("fog_vs_solvent_control", np.nan))
            yerr = np.nan
            if np.isfinite(fog_sem) and np.isfinite(fog_rel) and fog_sem > 0.0 and fog_rel > 0.0:
                yerr = fog_sem / fog_rel
            if np.isfinite(yerr) and yerr > 0.0:
                ax.errorbar(
                    xi,
                    yi,
                    yerr=yerr,
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.70,
                    capsize=1.8,
                    alpha=0.76,
                    zorder=14,
                )
            if bool(row["_marker_filled"]):
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=20,
                )
        ax.axvline(x=0.0, color="0.40", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.axhline(y=0.0, color="0.40", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        ax.set_xlabel(r"$\log(U_{0}^{*})$")
        ax.set_ylabel(r"$\log(\mathrm{FoG}^{*})$")
        ax.set_title(
            rf"MainA: log-linear objective map ($S=\log(\mathrm{{FoG}}^*)+{OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT:.2f}\log(U_0^*)$)"
        )
        ax.grid(True, linestyle=":", alpha=0.30)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_mainb_u0_vs_fog_tradeoff_with_pareto(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    top_k: int = 5,
) -> bool:
    """MainB: U0* vs FoG* tradeoff scatter."""
    df = _prepare_loglinear_objective_df(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    x_max = float(np.nanmax(df["abs0_vs_solvent_control"]))
    y_max = float(np.nanmax(df["fog_vs_solvent_control"]))
    x_right = max(1.20, x_max * 1.14)
    y_top = max(1.20, y_max * 1.16)

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        df["_marker_filled"] = pd.to_numeric(
            df.get("abs0_vs_solvent_control", np.nan), errors="coerce"
        ) >= float(MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD)
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["abs0_vs_solvent_control"])
            yi = float(row["fog_vs_solvent_control"])
            yi_sem = float(row.get("fog_sem", np.nan))
            if np.isfinite(yi_sem) and yi_sem > 0.0:
                ax.errorbar(
                    xi,
                    yi,
                    yerr=yi_sem,
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.70,
                    capsize=1.8,
                    alpha=0.76,
                    zorder=14,
                )
            if bool(row["_marker_filled"]):
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.90,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=20,
                )
        ax.axvline(x=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.axhline(y=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.set_xlim(0.0, x_right)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel(r"$U_{0}^{*}$ (vs solvent-matched control)")
        ax.set_ylabel(r"$\mathrm{FoG}^{*}$")
        ax.set_title("MainB: U0*-FoG* tradeoff")
        ax.grid(True, linestyle=":", alpha=0.30)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _compute_loglog_regression_curve(
    x_values: Any,
    y_values: Any,
    *,
    n_curve_points: int = 220,
    line_span_factor: float = 1.35,
) -> Dict[str, Any]:
    """Fit log(y)=a+b*log(x) and return curve/statistics for plotting."""
    x = np.asarray(pd.to_numeric(x_values, errors="coerce"), dtype=float)
    y = np.asarray(pd.to_numeric(y_values, errors="coerce"), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    n_valid = int(np.count_nonzero(valid))
    out: Dict[str, Any] = {
        "ok": False,
        "n": n_valid,
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "x_curve": np.array([], dtype=float),
        "y_curve": np.array([], dtype=float),
    }
    if n_valid < 2:
        return out
    x_fit = x[valid]
    y_fit = y[valid]
    lx = np.log(x_fit)
    ly = np.log(y_fit)
    lx_span = float(np.nanmax(lx) - np.nanmin(lx))
    if not np.isfinite(lx_span) or lx_span <= 1e-12:
        return out
    coef = np.polyfit(lx, ly, 1)
    slope = float(coef[0])
    intercept = float(coef[1])
    ly_hat = slope * lx + intercept
    ss_res = float(np.nansum((ly - ly_hat) ** 2))
    ss_tot = float(np.nansum((ly - np.nanmean(ly)) ** 2))
    r2 = np.nan
    if np.isfinite(ss_tot) and ss_tot > 0.0:
        r2 = 1.0 - (ss_res / ss_tot)
    x_min = float(np.nanmin(x_fit))
    x_max = float(np.nanmax(x_fit))
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_min > 0.0 and x_max > 0.0):
        return out
    span = max(1.0, float(line_span_factor))
    x_lo = x_min / span
    x_hi = x_max * span
    if not (np.isfinite(x_lo) and np.isfinite(x_hi) and x_lo > 0.0 and x_hi > 0.0):
        return out
    if x_hi <= x_lo:
        x_curve = np.array([x_min, x_max], dtype=float)
    else:
        # Use log-spacing so this appears as a straight line on log-log axes.
        x_curve = np.geomspace(x_lo, x_hi, max(2, int(n_curve_points)))
    y_curve = np.exp(intercept) * np.power(x_curve, slope)
    curve_valid = np.isfinite(x_curve) & np.isfinite(y_curve) & (x_curve > 0.0) & (y_curve > 0.0)
    if not np.any(curve_valid):
        return out
    out["ok"] = True
    out["slope"] = slope
    out["intercept"] = intercept
    out["r2"] = r2
    out["x_curve"] = x_curve[curve_valid]
    out["y_curve"] = y_curve[curve_valid]
    return out


def _plot_maine_u0_vs_fog_loglog_regression(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainE: U0* vs FoG* tradeoff with extrapolated log-log regression line."""
    df = _prepare_loglinear_objective_df(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    df["_x"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["_y"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["_y_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df = df[np.isfinite(df["_x"]) & np.isfinite(df["_y"]) & (df["_x"] > 0.0) & (df["_y"] > 0.0)].copy()
    if df.empty:
        return False
    df["_marker_filled"] = df["_x"] >= float(MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD)
    reg = _compute_loglog_regression_curve(df["_x"], df["_y"])

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    x_min = float(np.nanmin(df["_x"]))
    x_max = float(np.nanmax(df["_x"]))
    y_min = float(np.nanmin(df["_y"]))
    y_max = float(np.nanmax(df["_y"]))
    x_left = max(1e-4, x_min * 0.76)
    x_right = max(x_max * 1.22, x_left * 1.8)
    y_bottom = max(1e-4, y_min * 0.76)
    y_top = max(y_max * 1.22, y_bottom * 1.8)

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_x"])
            yi = float(row["_y"])
            yi_sem = float(row.get("_y_sem", np.nan))
            if np.isfinite(yi_sem) and yi_sem > 0.0:
                yerr_low = min(yi_sem, yi * 0.95)
                ax.errorbar(
                    xi,
                    yi,
                    yerr=np.array([[yerr_low], [yi_sem]], dtype=float),
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.70,
                    capsize=1.8,
                    alpha=0.76,
                    zorder=14,
                )
            if bool(row["_marker_filled"]):
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.90,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=20,
                )
        if bool(reg.get("ok")):
            trend = "negative" if float(reg["slope"]) < 0.0 else "positive"
            ax.plot(
                reg["x_curve"],
                reg["y_curve"],
                color="0.10",
                linewidth=0.95,
                linestyle="-",
                zorder=24,
            )
            info_lines = [
                "log-log linear fit",
                f"slope = {float(reg['slope']):.3f}",
                f"R2 = {float(reg['r2']):.3f}" if np.isfinite(float(reg["r2"])) else "R2 = NA",
                f"trend = {trend}",
                f"n = {int(reg['n'])}",
            ]
        else:
            info_lines = [
                "log-log linear fit",
                "fit unavailable",
                f"n = {int(reg['n'])}",
            ]
        ax.text(
            0.03,
            0.97,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="0.12",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", linewidth=0.6, alpha=0.95),
            zorder=30,
        )
        ax.axvline(x=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.axhline(y=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        ax.set_xlabel(r"$U_{0}^{*}$ (vs solvent-matched control)")
        ax.set_ylabel(r"$\mathrm{FoG}^{*}$")
        ax.set_title("MainE: U0*-FoG* with extrapolated linear fit (log-log)")
        ax.grid(True, linestyle=":", alpha=0.30)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_mainf_u0_vs_t50_loglog_regression(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainF: U0* vs t50 with extrapolated log-log regression line."""
    df = _add_solvent_balanced_objective_columns(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    df["_x"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["_y"] = pd.to_numeric(df.get("t50_min", np.nan), errors="coerce")
    df["_y_sem"] = pd.to_numeric(df.get("t50_sem_min", np.nan), errors="coerce")
    df = df[np.isfinite(df["_x"]) & np.isfinite(df["_y"]) & (df["_x"] > 0.0) & (df["_y"] > 0.0)].copy()
    if df.empty:
        return False
    df["_marker_filled"] = df["_x"] >= float(MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD)
    reg = _compute_loglog_regression_curve(df["_x"], df["_y"])

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    x_min = float(np.nanmin(df["_x"]))
    x_max = float(np.nanmax(df["_x"]))
    y_min = float(np.nanmin(df["_y"]))
    y_max = float(np.nanmax(df["_y"]))
    x_left = max(1e-4, x_min * 0.76)
    x_right = max(x_max * 1.22, x_left * 1.8)
    y_bottom = max(1e-4, y_min * 0.76)
    y_top = max(y_max * 1.22, y_bottom * 1.8)

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_x"])
            yi = float(row["_y"])
            yi_sem = float(row.get("_y_sem", np.nan))
            if np.isfinite(yi_sem) and yi_sem > 0.0:
                yerr_low = min(yi_sem, yi * 0.95)
                ax.errorbar(
                    xi,
                    yi,
                    yerr=np.array([[yerr_low], [yi_sem]], dtype=float),
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.70,
                    capsize=1.8,
                    alpha=0.76,
                    zorder=14,
                )
            if bool(row["_marker_filled"]):
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.90,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=28,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=20,
                )
        if bool(reg.get("ok")):
            trend = "negative" if float(reg["slope"]) < 0.0 else "positive"
            ax.plot(
                reg["x_curve"],
                reg["y_curve"],
                color="0.10",
                linewidth=0.95,
                linestyle="-",
                zorder=24,
            )
            info_lines = [
                "log-log linear fit",
                f"slope = {float(reg['slope']):.3f}",
                f"R2 = {float(reg['r2']):.3f}" if np.isfinite(float(reg["r2"])) else "R2 = NA",
                f"trend = {trend}",
                f"n = {int(reg['n'])}",
            ]
        else:
            info_lines = [
                "log-log linear fit",
                "fit unavailable",
                f"n = {int(reg['n'])}",
            ]
        ax.text(
            0.03,
            0.97,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="0.12",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", linewidth=0.6, alpha=0.95),
            zorder=30,
        )
        ax.axvline(x=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        ax.set_xlabel(r"$U_{0}^{*}$ (vs solvent-matched control)")
        ax.set_ylabel(r"$t_{50}$ (min)")
        ax.set_title(r"MainF: $U_{0}^{*}$-$t_{50}$ with extrapolated linear fit (log-log)")
        ax.grid(True, linestyle=":", alpha=0.30)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_mainc_u0_vs_fog_objective_contour(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainC: full-area U0*-FoG* map with objective gradient and contour lines."""
    df = _prepare_loglinear_objective_df(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False

    df["_x"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["_y"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["_score"] = pd.to_numeric(df.get(OBJECTIVE_LOGLINEAR_MAIN_COL, np.nan), errors="coerce")
    df["_y_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df = df[np.isfinite(df["_x"]) & (df["_x"] > 0.0) & np.isfinite(df["_y"]) & (df["_y"] > 0.0)].copy()
    if df.empty:
        return False

    missing_score = ~np.isfinite(df["_score"])
    if np.any(missing_score):
        score_fill, _ = _compute_loglinear_main_objective(
            df.loc[missing_score, "_y"],
            df.loc[missing_score, "_x"],
            weight_lambda=OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT,
        )
        df.loc[missing_score, "_score"] = score_fill
    df = df[np.isfinite(df["_score"])].copy()
    if df.empty:
        return False

    x_max = float(np.nanmax(df["_x"]))
    y_max = float(np.nanmax(df["_y"]))
    x_right = max(1.20, x_max * 1.16)
    y_top = max(1.20, y_max * 1.18)
    # Objective contains log(U0*) and log(FoG*), so we evaluate from a small epsilon.
    x_floor = max(1e-3, 0.005 * x_right)
    y_floor = max(1e-3, 0.005 * y_top)

    x_grid = np.linspace(x_floor, x_right, 280)
    y_grid = np.linspace(y_floor, y_top, 280)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = np.log(yy) + float(OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT) * np.log(xx)
    finite_zz = zz[np.isfinite(zz)]
    if finite_zz.size == 0:
        return False
    z_lo = float(np.nanpercentile(finite_zz, 5.0))
    z_hi = float(np.nanpercentile(finite_zz, 95.0))
    if (not np.isfinite(z_lo)) or (not np.isfinite(z_hi)) or (z_hi <= z_lo + 1e-9):
        z_lo = float(np.nanmin(finite_zz))
        z_hi = float(np.nanmax(finite_zz))
    if z_hi <= z_lo + 1e-9:
        z_hi = z_lo + 1.0
    fill_levels = np.linspace(z_lo, z_hi, 40)
    line_levels = np.linspace(z_lo, z_hi, 10)

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        contourf = ax.contourf(
            xx,
            yy,
            zz,
            levels=fill_levels,
            cmap="viridis",
            alpha=0.50,
            antialiased=True,
            zorder=2,
        )
        ax.contour(
            xx,
            yy,
            zz,
            levels=line_levels,
            colors="0.45",
            linewidths=0.70,
            linestyles="dashed",
            alpha=0.85,
            zorder=4,
        )

        df["_marker_filled"] = df["_x"] >= float(MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD)
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_x"])
            yi = float(row["_y"])
            yi_sem = float(row.get("_y_sem", np.nan))
            if np.isfinite(yi_sem) and yi_sem > 0.0:
                ax.errorbar(
                    xi,
                    yi,
                    yerr=yi_sem,
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.70,
                    capsize=1.8,
                    alpha=0.78,
                    zorder=14,
                )
            if bool(row["_marker_filled"]):
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.92,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.95,
                    zorder=20,
                )
        ax.axvline(x=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.axhline(y=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
        ax.set_xlim(0.0, x_right)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel(r"$U_{0}^{*}$ (vs solvent-matched control)")
        ax.set_ylabel(r"$\mathrm{FoG}^{*}$")
        ax.set_title("MainC: objective topographic map (U0*-FoG*)")
        ax.grid(True, linestyle=":", alpha=0.28)
        _add_scatter_polymer_mapping_legend(ax, df, color_map=cmap, feasible_col="_marker_filled")
        cbar = fig.colorbar(contourf, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label(r"$S=\log(\mathrm{FoG}^{*})+\lambda\log(U_{0}^{*})$")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_maind_u0_vs_fog_objective_hill3d(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainD: 3D hill map of objective landscape with polymer points."""
    df = _prepare_loglinear_objective_df(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    df["_x"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["_y"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["_z"] = pd.to_numeric(df.get(OBJECTIVE_LOGLINEAR_MAIN_COL, np.nan), errors="coerce")
    df = df[np.isfinite(df["_x"]) & (df["_x"] > 0.0) & np.isfinite(df["_y"]) & (df["_y"] > 0.0)].copy()
    if df.empty:
        return False
    missing_z = ~np.isfinite(df["_z"])
    if np.any(missing_z):
        z_fill, _ = _compute_loglinear_main_objective(
            df.loc[missing_z, "_y"],
            df.loc[missing_z, "_x"],
            weight_lambda=OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT,
        )
        df.loc[missing_z, "_z"] = z_fill
    df = df[np.isfinite(df["_z"])].copy()
    if df.empty:
        return False

    x_right = max(1.20, float(np.nanmax(df["_x"])) * 1.16)
    y_top = max(1.20, float(np.nanmax(df["_y"])) * 1.18)
    x_floor = max(1e-3, 0.005 * x_right)
    y_floor = max(1e-3, 0.005 * y_top)
    x_grid = np.linspace(x_floor, x_right, 180)
    y_grid = np.linspace(y_floor, y_top, 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = np.log(yy) + float(OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT) * np.log(xx)
    finite_zz = zz[np.isfinite(zz)]
    if finite_zz.size == 0:
        return False
    z_lo = float(np.nanpercentile(finite_zz, 5.0))
    z_hi = float(np.nanpercentile(finite_zz, 95.0))
    if z_hi <= z_lo + 1e-9:
        z_lo = float(np.nanmin(finite_zz))
        z_hi = float(np.nanmax(finite_zz))
    if z_hi <= z_lo + 1e-9:
        z_hi = z_lo + 1.0

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig = plt.figure(figsize=(5.4, 4.1))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            alpha=0.78,
            zorder=2,
        )
        ax.contour(
            xx,
            yy,
            zz,
            zdir="z",
            offset=z_lo,
            levels=np.linspace(z_lo, z_hi, 10),
            colors="0.35",
            linewidths=0.50,
            linestyles="dashed",
            alpha=0.75,
            zorder=1,
        )
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_x"])
            yi = float(row["_y"])
            zi = float(row["_z"])
            ax.plot([xi, xi], [yi, yi], [z_lo, zi], color="0.40", linewidth=0.45, alpha=0.45, zorder=3)
            ax.scatter(
                [xi],
                [yi],
                [zi],
                s=22,
                color=color,
                edgecolors="0.15",
                linewidths=0.45,
                alpha=0.95,
                zorder=4,
            )

        ax.set_xlim(0.0, x_right)
        ax.set_ylim(0.0, y_top)
        ax.set_zlim(z_lo, z_hi)
        ax.view_init(elev=30, azim=-58)
        ax.set_xlabel(r"$U_{0}^{*}$")
        ax.set_ylabel(r"$\mathrm{FoG}^{*}$")
        ax.set_zlabel(r"$S=\log(\mathrm{FoG}^{*})+\lambda\log(U_{0}^{*})$")
        ax.set_title("MainD: 3D objective hill map")
        cbar = fig.colorbar(surf, ax=ax, fraction=0.05, pad=0.06, shrink=0.72)
        cbar.set_label("Objective level")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_loglinear_weight_sensitivity(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    weight_values: Tuple[float, ...] = OBJECTIVE_LOGLINEAR_WEIGHT_SWEEP,
) -> Tuple[bool, pd.DataFrame]:
    """Supplementary: lambda sensitivity for log-linear objective ranking."""
    base_df = _prepare_loglinear_objective_df(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if base_df.empty:
        return False, pd.DataFrame()
    valid = base_df[
        np.isfinite(base_df["abs0_vs_solvent_control"])
        & (base_df["abs0_vs_solvent_control"] > 0.0)
        & np.isfinite(base_df["fog_vs_solvent_control"])
        & (base_df["fog_vs_solvent_control"] > 0.0)
    ].copy()
    if valid.empty:
        return False, pd.DataFrame()
    lam_grid = sorted({float(v) for v in weight_values if np.isfinite(v)})
    if not lam_grid:
        lam_grid = [float(OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT)]
    rank_tables: Dict[float, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []
    for lam in lam_grid:
        score_lam, _ = _compute_loglinear_main_objective(
            valid["fog_vs_solvent_control"],
            valid["abs0_vs_solvent_control"],
            weight_lambda=float(lam),
        )
        tmp = valid.copy()
        tmp["_score_lam"] = score_lam
        tmp = _rank_by_loglinear_objective(
            tmp,
            score_col="_score_lam",
            rank_col="_rank_tmp",
        )
        ranked = tmp[np.isfinite(pd.to_numeric(tmp.get("_rank_tmp", np.nan), errors="coerce"))].copy()
        if ranked.empty:
            rank_tbl = pd.DataFrame(columns=["polymer_id", "rank"])
        else:
            rank_tbl = ranked[["polymer_id", "_rank_tmp"]].rename(columns={"_rank_tmp": "rank"}).copy()
            rank_tbl["rank"] = pd.to_numeric(rank_tbl["rank"], errors="coerce").astype(int)
            rank_tbl = rank_tbl.sort_values("rank", ascending=True, kind="mergesort").reset_index(drop=True)
        rank_tables[float(lam)] = rank_tbl
        top1 = str(rank_tbl["polymer_id"].iloc[0]) if len(rank_tbl) else ""
        top5 = ",".join(rank_tbl["polymer_id"].head(5).astype(str).tolist()) if len(rank_tbl) else ""
        summary_rows.append(
            {
                "analysis_type": "weight_sensitivity",
                "scenario": f"lambda={lam:.2f}",
                "weight_lambda": float(lam),
                "n_ranked": int(len(rank_tbl)),
                "top1_polymer_id": top1,
                "top5_polymer_ids": top5,
            }
        )
    summary = pd.DataFrame(summary_rows)
    base_lam = min(lam_grid, key=lambda x: abs(float(x) - float(OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT)))
    base_rank = rank_tables.get(float(base_lam), pd.DataFrame(columns=["polymer_id", "rank"]))
    corr_rows: List[Dict[str, Any]] = []
    for lam in lam_grid:
        rank_tbl = rank_tables.get(float(lam), pd.DataFrame(columns=["polymer_id", "rank"]))
        metrics = _rank_corr_metrics(base_rank, rank_tbl, top_k=5)
        corr_rows.append(
            {
                "analysis_type": "weight_sensitivity",
                "scenario": f"lambda={lam:.2f}",
                "reference_scenario": f"lambda={base_lam:.2f}",
                "weight_lambda": float(lam),
                "n_common": int(metrics["n_common"]),
                "kendall_tau": metrics["kendall_tau"],
                "spearman_rho": metrics["spearman_rho"],
                "top5_overlap": metrics["topk_overlap"],
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt
    with plt.rc_context(apply_paper_style()):
        fig, (ax_n, ax_corr) = plt.subplots(1, 2, figsize=(7.0, 2.9))
        ax_n.plot(
            summary["weight_lambda"].to_numpy(dtype=float),
            summary["n_ranked"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.0,
            color="#1f77b4",
        )
        ax_n.set_xlabel(r"$\lambda$")
        ax_n.set_ylabel("Ranked polymers (n)")
        ax_n.set_title("Supp: lambda sensitivity")
        ax_n.grid(True, linestyle=":", alpha=0.30)
        ax_corr.plot(
            corr_df["weight_lambda"].to_numpy(dtype=float),
            corr_df["top5_overlap"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.0,
            color="#2ca02c",
            label="Top5 overlap",
        )
        ax_corr.plot(
            corr_df["weight_lambda"].to_numpy(dtype=float),
            corr_df["kendall_tau"].to_numpy(dtype=float),
            marker="s",
            linewidth=1.0,
            color="#d62728",
            label="Kendall tau",
        )
        ax_corr.plot(
            corr_df["weight_lambda"].to_numpy(dtype=float),
            corr_df["spearman_rho"].to_numpy(dtype=float),
            marker="^",
            linewidth=1.0,
            color="#9467bd",
            label="Spearman rho",
        )
        ax_corr.set_xlabel(r"$\lambda$")
        ax_corr.set_ylabel("Stability metric")
        ax_corr.set_ylim(-1.05, 1.05)
        ax_corr.set_title("Supp: ranking stability")
        ax_corr.grid(True, linestyle=":", alpha=0.30)
        ax_corr.legend(loc="lower left", fontsize=5.2, frameon=True, framealpha=0.9)
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True, corr_df


def _plot_loglinear_threshold_sensitivity(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    thresholds: Tuple[float, ...] = OBJECTIVE_LOGLINEAR_THRESHOLD_SWEEP,
) -> Tuple[bool, pd.DataFrame]:
    """Supplementary: threshold sensitivity on U0* with rank correlation vs unconstrained ranking."""
    base_df = _prepare_loglinear_objective_df(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if base_df.empty:
        return False, pd.DataFrame()
    base_ranked = base_df[np.isfinite(pd.to_numeric(base_df.get(OBJECTIVE_LOGLINEAR_MAIN_RANK_COL, np.nan), errors="coerce"))].copy()
    if base_ranked.empty:
        return False, pd.DataFrame()
    base_rank = base_ranked[["polymer_id", OBJECTIVE_LOGLINEAR_MAIN_RANK_COL]].rename(
        columns={OBJECTIVE_LOGLINEAR_MAIN_RANK_COL: "rank"}
    )
    base_rank["rank"] = pd.to_numeric(base_rank["rank"], errors="coerce").astype(int)
    base_rank = base_rank.sort_values("rank", ascending=True, kind="mergesort").reset_index(drop=True)

    th_grid = sorted({float(v) for v in thresholds if np.isfinite(v) and float(v) > 0.0})
    if not th_grid:
        th_grid = [0.7, 0.8, 0.9]
    rank_tables: Dict[float, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []
    corr_rows: List[Dict[str, Any]] = []
    for th in th_grid:
        sub = base_df[pd.to_numeric(base_df["abs0_vs_solvent_control"], errors="coerce") >= float(th)].copy()
        if sub.empty:
            rank_tbl = pd.DataFrame(columns=["polymer_id", "rank"])
        else:
            sub = _rank_by_loglinear_objective(
                sub,
                score_col=OBJECTIVE_LOGLINEAR_MAIN_COL,
                rank_col="_rank_tmp",
            )
            rank_tbl = sub[np.isfinite(pd.to_numeric(sub.get("_rank_tmp", np.nan), errors="coerce"))][
                ["polymer_id", "_rank_tmp"]
            ].rename(columns={"_rank_tmp": "rank"})
            if not rank_tbl.empty:
                rank_tbl["rank"] = pd.to_numeric(rank_tbl["rank"], errors="coerce").astype(int)
                rank_tbl = rank_tbl.sort_values("rank", ascending=True, kind="mergesort").reset_index(drop=True)
        rank_tables[float(th)] = rank_tbl
        top1 = str(rank_tbl["polymer_id"].iloc[0]) if len(rank_tbl) else ""
        top5 = ",".join(rank_tbl["polymer_id"].head(5).astype(str).tolist()) if len(rank_tbl) else ""
        summary_rows.append(
            {
                "analysis_type": "threshold_sensitivity",
                "scenario": f"u0_threshold={th:.2f}",
                "u0_threshold": float(th),
                "n_ranked": int(len(rank_tbl)),
                "top1_polymer_id": top1,
                "top5_polymer_ids": top5,
            }
        )
        metrics = _rank_corr_metrics(base_rank, rank_tbl, top_k=5)
        corr_rows.append(
            {
                "analysis_type": "threshold_sensitivity",
                "scenario": f"u0_threshold={th:.2f}",
                "reference_scenario": "u0_threshold=none",
                "u0_threshold": float(th),
                "n_common": int(metrics["n_common"]),
                "kendall_tau": metrics["kendall_tau"],
                "spearman_rho": metrics["spearman_rho"],
                "top5_overlap": metrics["topk_overlap"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    corr_df = pd.DataFrame(corr_rows)

    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt
    with plt.rc_context(apply_paper_style()):
        fig, (ax_n, ax_corr) = plt.subplots(1, 2, figsize=(7.0, 2.9))
        ax_n.plot(
            summary["u0_threshold"].to_numpy(dtype=float),
            summary["n_ranked"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.0,
            color="#1f77b4",
        )
        ax_n.set_xlabel(r"$U_{0}^{*}$ threshold")
        ax_n.set_ylabel("Ranked polymers (n)")
        ax_n.set_title("Supp: threshold sensitivity")
        ax_n.grid(True, linestyle=":", alpha=0.30)
        ax_corr.plot(
            corr_df["u0_threshold"].to_numpy(dtype=float),
            corr_df["top5_overlap"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.0,
            color="#2ca02c",
            label="Top5 overlap",
        )
        ax_corr.plot(
            corr_df["u0_threshold"].to_numpy(dtype=float),
            corr_df["kendall_tau"].to_numpy(dtype=float),
            marker="s",
            linewidth=1.0,
            color="#d62728",
            label="Kendall tau",
        )
        ax_corr.plot(
            corr_df["u0_threshold"].to_numpy(dtype=float),
            corr_df["spearman_rho"].to_numpy(dtype=float),
            marker="^",
            linewidth=1.0,
            color="#9467bd",
            label="Spearman rho",
        )
        ax_corr.set_xlabel(r"$U_{0}^{*}$ threshold")
        ax_corr.set_ylabel("Stability metric")
        ax_corr.set_ylim(-1.05, 1.05)
        ax_corr.set_title("Supp: ranking stability")
        ax_corr.grid(True, linestyle=":", alpha=0.30)
        ax_corr.legend(loc="lower left", fontsize=5.2, frameon=True, framealpha=0.9)
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True, corr_df


def _plot_maina_abs_vs_fog_solvent(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainA variant: solvent-matched abs0 vs solvent-matched FoG scatter."""
    df = _add_solvent_balanced_objective_columns(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    df["_x"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["_y"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["_y_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df = df[np.isfinite(df["_x"]) & np.isfinite(df["_y"]) & (df["_y"] > 0)].copy()
    if df.empty:
        return False
    df["_marker_filled"] = df["_x"] >= float(MAIN_PLOT_LOW_U0_HOLLOW_THRESHOLD)

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row["_x"])
            yi = float(row["_y"])
            yi_sem = float(row.get("_y_sem", np.nan))
            if np.isfinite(yi_sem) and yi_sem > 0.0:
                ax.errorbar(
                    xi,
                    yi,
                    yerr=yi_sem,
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.8,
                    zorder=16,
                )
            if bool(row["_marker_filled"]):
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    marker="o",
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    marker="o",
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=18,
                )
        ax.axhline(y=1.0, color="0.45", linestyle=(0, (3, 2)), linewidth=0.7, zorder=10)
        ax.set_xlabel(r"$U_{0}^{\mathrm{solv}}$ (vs solvent-matched control)")
        ax.set_ylabel(r"$\mathrm{FoG}^{\mathrm{solv}}$")
        ax.set_title("MainA: solvent-matched abs0 and FoG")
        ax.grid(True, linestyle=":", alpha=0.30)
        ax.set_xlim(0.0, max(1.10, float(np.nanmax(df["_x"])) * 1.12))
        ax.set_ylim(0.0, max(1.10, float(np.nanmax(df["_y"])) * 1.12))
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_fog_native_constrained_decision(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    Plot decision chart for constrained FoG objective (paper main figure style):
      left  = native_0 (vs reference at 0 min) with gate line (theta)
      right = FoG with infeasible rows visually marked.
    Returns True when a figure was written.
    """
    df = run_df.copy()
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False

    df["native_activity_rel_at_0"] = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df = df[np.isfinite(df["native_activity_rel_at_0"]) & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False

    if "native_activity_feasible" in df.columns and np.any(pd.notna(df["native_activity_feasible"])):
        feasible = pd.to_numeric(df["native_activity_feasible"], errors="coerce").fillna(0).astype(int) > 0
    else:
        feasible = df["native_activity_rel_at_0"] >= float(theta)
    df["native_activity_feasible"] = feasible.astype(int)

    df = df.sort_values(["native_activity_feasible", "fog"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    if df.empty:
        return False

    cmap = color_map or {}
    default_color = "#4C78A8"
    colors: List[str] = []
    for _, row in df.iterrows():
        pid = str(row.get("polymer_id", "")).strip()
        colors.append(cmap.get(pid, default_color))
    labels = [str(pid) for pid in df["polymer_id"].tolist()]
    y = np.arange(len(df), dtype=float)
    h = max(2.5, 0.30 * len(df) + 0.9)

    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, (ax_native, ax_fog) = plt.subplots(
            1,
            2,
            figsize=(7.2, h),
            sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.2]},
        )

        native_vals = df["native_activity_rel_at_0"].to_numpy(dtype=float)
        fog_vals = df["fog"].to_numpy(dtype=float)
        feasible_vals = df["native_activity_feasible"].to_numpy(dtype=int)
        if "native_activity_rel_at_0_sem" in df.columns:
            native_sem_vals = pd.to_numeric(df["native_activity_rel_at_0_sem"], errors="coerce").to_numpy(dtype=float)
        else:
            native_sem_vals = np.full(len(df), np.nan, dtype=float)
        if "fog_sem" in df.columns:
            fog_sem_vals = pd.to_numeric(df["fog_sem"], errors="coerce").to_numpy(dtype=float)
        else:
            fog_sem_vals = np.full(len(df), np.nan, dtype=float)
        if "fog_native_constrained_sem" in df.columns:
            fog_native_sem_vals = pd.to_numeric(df["fog_native_constrained_sem"], errors="coerce").to_numpy(dtype=float)
        else:
            fog_native_sem_vals = np.full(len(df), np.nan, dtype=float)
        fog_err_vals = np.where(np.isfinite(fog_native_sem_vals), fog_native_sem_vals, fog_sem_vals)

        # Left panel: native gate.
        native_xerr = np.where(np.isfinite(native_sem_vals) & (native_sem_vals > 0.0), native_sem_vals, 0.0)
        native_bar_kwargs = {}
        if np.any(native_xerr > 0.0):
            native_bar_kwargs["xerr"] = native_xerr
            native_bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.85,
            }
        native_bars = ax_native.barh(
            y,
            native_vals,
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
            alpha=0.92,
            zorder=15,
            **native_bar_kwargs,
        )
        for bar, fe in zip(native_bars, feasible_vals):
            if int(fe) <= 0:
                bar.set_alpha(0.40)
                bar.set_hatch("//")
                bar.set_edgecolor("0.45")
        ax_native.axvline(
            x=float(theta),
            color="0.25",
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            alpha=0.95,
            zorder=20,
        )
        ax_native.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
        ax_native.set_yticks(y)
        ax_native.set_yticklabels(labels)
        ax_native.invert_yaxis()
        ax_native.grid(axis="x", linestyle=":", alpha=0.28)

        # Right panel: FoG with feasible/infeasible marking.
        fog_xerr = np.where(np.isfinite(fog_err_vals) & (fog_err_vals > 0.0), fog_err_vals, 0.0)
        fog_bar_kwargs = {}
        if np.any(fog_xerr > 0.0):
            fog_bar_kwargs["xerr"] = fog_xerr
            fog_bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.85,
            }
        fog_bars = ax_fog.barh(
            y,
            fog_vals,
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
            alpha=0.92,
            zorder=15,
            **fog_bar_kwargs,
        )
        for bar, fe in zip(fog_bars, feasible_vals):
            if int(fe) <= 0:
                bar.set_alpha(0.40)
                bar.set_hatch("//")
                bar.set_edgecolor("0.45")
        ax_fog.axvline(
            x=1.0,
            color="0.35",
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            alpha=0.85,
            zorder=20,
        )
        ax_fog.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
        ax_fog.grid(axis="x", linestyle=":", alpha=0.28)

        # Keep y labels only on the left.
        ax_fog.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Include horizontal error bars in axis limits so caps stay inside the plot area.
        fog_upper_with_err = float(np.nanmax(fog_vals + fog_xerr))
        x_right = max(1.4, float(np.nanmax(fog_vals)) * 1.16, fog_upper_with_err * 1.08)
        ax_fog.set_xlim(0.0, x_right)
        native_upper_with_err = float(np.nanmax(native_vals + native_xerr))
        native_right = max(
            float(theta) * 1.25,
            float(np.nanmax(native_vals)) * 1.15,
            native_upper_with_err * 1.08,
            1.05,
        )
        ax_native.set_xlim(0.0, native_right)

        feasible_n = int(np.sum(feasible_vals > 0))
        fig.suptitle(
            rf"Feasibility-constrained FoG ranking ({feasible_n}/{len(df)} feasible, $U_{{0}}\geq\theta$, $\theta={float(theta):.2f}$)",
            y=0.995,
            fontsize=7.0,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985], pad=0.35)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_maina_native_vs_fog(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    MainA: native_0 vs FoG scatter with feasibility gate.
    Feasible points: filled circles. Infeasible: open circles.
    """
    df = run_df.copy()
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    x_col = "native_0" if "native_0" in df.columns else "native_activity_rel_at_0"
    x_sem_col = "native_activity_rel_at_0_sem" if "native_activity_rel_at_0_sem" in df.columns else ""
    df[x_col] = pd.to_numeric(df.get(x_col, np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df["_fog_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    if x_sem_col:
        df["_x_sem"] = pd.to_numeric(df.get(x_sem_col, np.nan), errors="coerce")
    else:
        df["_x_sem"] = np.nan
    fe = pd.to_numeric(df.get("native_activity_feasible", np.nan), errors="coerce")
    if np.isfinite(fe).any():
        df["_feasible"] = fe.fillna(0).astype(int) > 0
    else:
        df["_feasible"] = np.isfinite(df[x_col]) & (df[x_col] >= float(theta))
    df = df[np.isfinite(df[x_col]) & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row[x_col])
            yi = float(row["fog"])
            xi_sem = float(row.get("_x_sem", np.nan))
            yi_sem = float(row.get("_fog_sem", np.nan))
            if (np.isfinite(xi_sem) and xi_sem > 0.0) or (np.isfinite(yi_sem) and yi_sem > 0.0):
                ax.errorbar(
                    xi,
                    yi,
                    xerr=(xi_sem if np.isfinite(xi_sem) and xi_sem > 0.0 else None),
                    yerr=(yi_sem if np.isfinite(yi_sem) and yi_sem > 0.0 else None),
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.8,
                    zorder=16,
                )
            if bool(row["_feasible"]):
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    marker="o",
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=18,
                )
        ax.axvline(x=float(theta), color="0.30", linestyle=(0, (3, 2)), linewidth=0.8, zorder=10)
        ax.axhline(y=1.0, color="0.45", linestyle=(0, (3, 2)), linewidth=0.7, zorder=10)
        ax.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
        ax.set_ylabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
        ax.set_title("MainA: native gate and FoG")
        ax.grid(True, linestyle=":", alpha=0.30)
        ax.set_xlim(0.0, max(float(theta) * 1.35, float(np.nanmax(df[x_col])) * 1.12, 1.05))
        ax.set_ylim(0.0, max(1.5, float(np.nanmax(df["fog"])) * 1.12))
        _add_scatter_polymer_mapping_legend(ax, df, color_map=cmap, feasible_col="_feasible")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_activity_bonus_penalty_proxy_curves(
    run_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    top_n: int = 12,
) -> bool:
    """
    Plot proxy activity-decay curves that combine FoG-side stability and U0-side bonus/penalty.

    Curves are REA50-equivalent exponential proxies derived from t50:
      control(t) = exp(-ln(2) * t / t50_control)
      abs-normalized(t) = U0* * exp(-ln(2) * t / t50_polymer)
      score-normalized(t) = B(U0*) * exp(-ln(2) * t / t50_polymer)
    """
    df = _add_solvent_balanced_objective_columns(
        run_df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty or ("polymer_id" not in df.columns):
        return False
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False

    df["_t50"] = pd.to_numeric(df.get("t50_min", np.nan), errors="coerce")
    df["_t50_ctrl"] = pd.to_numeric(df.get("solvent_control_t50_min", np.nan), errors="coerce")
    df["_u0"] = pd.to_numeric(df.get("abs0_vs_solvent_control", np.nan), errors="coerce")
    df["_fog_rel"] = pd.to_numeric(df.get("fog_vs_solvent_control", np.nan), errors="coerce")
    df["_bal"] = pd.to_numeric(df.get("abs_activity_balance_factor", np.nan), errors="coerce")
    df["_score"] = pd.to_numeric(df.get(ABS_ACTIVITY_OBJECTIVE_COL, np.nan), errors="coerce")
    df = df[
        np.isfinite(df["_t50"])
        & (df["_t50"] > 0.0)
        & np.isfinite(df["_t50_ctrl"])
        & (df["_t50_ctrl"] > 0.0)
        & np.isfinite(df["_u0"])
        & np.isfinite(df["_bal"])
        & np.isfinite(df["_score"])
        & (df["_score"] > 0.0)
    ].copy()
    if df.empty:
        return False

    df = df.sort_values(["_score", "_fog_rel", "_u0"], ascending=[False, False, False], kind="mergesort")
    n_take = max(1, int(top_n))
    df = df.head(n_take).copy()
    if df.empty:
        return False

    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt

    cmap = color_map or {}
    default_color = "#4C78A8"
    n = len(df)
    ncols = 3 if n >= 7 else 2 if n >= 3 else 1
    nrows = int(np.ceil(n / float(ncols)))
    fig_w = 5.2 if ncols == 2 else 7.2 if ncols == 3 else 3.6
    fig_h = max(2.8, 2.25 * nrows)

    with plt.rc_context(apply_paper_style()):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
        axes_flat = list(axes.flat)
        ln2 = float(np.log(2.0))
        for idx, (_, row) in enumerate(df.iterrows()):
            ax = axes_flat[idx]
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            t50 = float(row["_t50"])
            t50_ctrl = float(row["_t50_ctrl"])
            u0 = float(row["_u0"])
            bal = float(row["_bal"])
            fog_rel = float(row.get("_fog_rel", np.nan))
            score = float(row["_score"])
            t_end = max(20.0, 1.45 * max(t50, t50_ctrl))
            tt = np.linspace(0.0, t_end, 240)
            ctrl = np.exp(-ln2 * tt / t50_ctrl)
            abs_curve = u0 * np.exp(-ln2 * tt / t50)
            score_curve = bal * np.exp(-ln2 * tt / t50)

            ax.plot(tt, ctrl, color="0.45", linewidth=0.95, linestyle="-", label="Solvent control proxy")
            ax.plot(tt, abs_curve, color=color, linewidth=1.05, linestyle="-", label="Abs-normalized proxy")
            ax.plot(tt, score_curve, color=color, linewidth=1.0, linestyle=(0, (3, 2)), label="Objective-normalized proxy")
            ax.axvline(t50_ctrl, color="0.55", linewidth=0.75, linestyle=(0, (2, 2)))
            ax.axvline(t50, color=color, linewidth=0.8, linestyle=(0, (2, 2)))
            ymax = float(np.nanmax(np.r_[ctrl, abs_curve, score_curve]))
            ax.set_ylim(0.0, max(1.05, 1.12 * ymax))
            ax.set_xlim(0.0, t_end)
            ax.set_title(pid, fontsize=6.0)
            ax.text(
                0.03,
                0.97,
                f"FoG*={fog_rel:.2f}  U0*={u0:.2f}\nB(U0*)={bal:.2f}  S={score:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.0,
                color="0.20",
            )
            ax.grid(True, linestyle=":", alpha=0.28)
            if (idx // ncols) == (nrows - 1):
                ax.set_xlabel("Heat time (min)")
            else:
                ax.set_xticklabels([])
            if (idx % ncols) == 0:
                ax.set_ylabel("Relative activity (proxy)")
            else:
                ax.set_yticklabels([])
        for ax in axes_flat[n:]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                fontsize=5.5,
                frameon=True,
                framealpha=0.9,
                bbox_to_anchor=(0.5, 1.01),
            )
        fig.suptitle(
            "FoG-activity bonus/penalty proxy curves (REA50 exponential proxy)",
            y=1.03,
            fontsize=6.6,
        )
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_activity_bonus_penalty_profile_tradeoff_grid(
    profile_rank_df: pd.DataFrame,
    *,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
) -> bool:
    """Plot a 2x2 profile grid to audit objective arbitrariness around defaults."""
    if profile_rank_df.empty:
        return False
    req = {"polymer_id", "profile_id", "profile_label", "abs0_vs_solvent_control", "fog_vs_solvent_control", "score"}
    if not req.issubset(set(profile_rank_df.columns)):
        return False
    df = profile_rank_df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    df["abs0_vs_solvent_control"] = pd.to_numeric(df["abs0_vs_solvent_control"], errors="coerce")
    df["fog_vs_solvent_control"] = pd.to_numeric(df["fog_vs_solvent_control"], errors="coerce")
    df["fog_vs_solvent_control_sem"] = pd.to_numeric(df.get("fog_vs_solvent_control_sem", np.nan), errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[
        np.isfinite(df["abs0_vs_solvent_control"])
        & np.isfinite(df["fog_vs_solvent_control"])
        & (df["fog_vs_solvent_control"] > 0.0)
        & np.isfinite(df["score"])
        & (df["score"] > 0.0)
    ].copy()
    if df.empty:
        return False
    profile_order = [pid for pid, _, _, _ in OBJECTIVE_PROFILE_SPECS if pid in set(df["profile_id"].tolist())]
    if not profile_order:
        return False

    n_panels = len(profile_order)
    ncols = 2
    nrows = int(np.ceil(n_panels / float(ncols)))
    x_right = max(1.10, float(np.nanmax(df["abs0_vs_solvent_control"])) * 1.12)
    y_top = max(1.10, float(np.nanmax(df["fog_vs_solvent_control"])) * 1.14)
    x_curve = np.linspace(0.05, x_right, 420)

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.8, max(3.4, 2.95 * nrows)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = list(axes.flat)

        for idx, profile_id in enumerate(profile_order):
            ax = axes_flat[idx]
            sub = df[df["profile_id"] == profile_id].copy()
            if sub.empty:
                ax.axis("off")
                continue
            label = str(sub["profile_label"].iloc[0])
            exp = float(pd.to_numeric(sub["penalty_exponent"], errors="coerce").iloc[0])
            bonus_coef = float(pd.to_numeric(sub["bonus_coef"], errors="coerce").iloc[0])
            _, _, factor_curve = _compute_solvent_balanced_activity_factor(
                x_curve,
                exponent=exp,
                up_bonus_coef=bonus_coef,
                up_bonus_max_delta=SOLVENT_BALANCED_UP_BONUS_MAX_DELTA,
                up_bonus_deadband=SOLVENT_BALANCED_UP_BONUS_DEADBAND,
            )
            score_vals = sub["score"].to_numpy(dtype=float)
            score_vals = score_vals[np.isfinite(score_vals) & (score_vals > 0.0)]
            if score_vals.size > 0:
                qvals = np.nanquantile(score_vals, [0.50, 0.75, 0.90])
                levels = sorted({float(v) for v in qvals if np.isfinite(v) and v > 0.0})
            else:
                levels = []
            for lv in levels:
                y_curve = np.where(factor_curve > 0.0, lv / factor_curve, np.nan)
                y_curve = np.where(np.isfinite(y_curve) & (y_curve <= y_top * 1.35), y_curve, np.nan)
                if np.any(np.isfinite(y_curve)):
                    ax.plot(
                        x_curve,
                        y_curve,
                        color="0.58",
                        linestyle=(0, (3, 2)),
                        linewidth=0.65,
                        alpha=0.80,
                        zorder=8,
                    )
            sub["_feasible"] = sub["abs0_vs_solvent_control"] >= 1.0
            for _, row in sub.iterrows():
                pid = str(row["polymer_id"])
                xi = float(row["abs0_vs_solvent_control"])
                yi = float(row["fog_vs_solvent_control"])
                yi_sem = float(row.get("fog_vs_solvent_control_sem", np.nan))
                color = cmap.get(pid, default_color)
                if np.isfinite(yi_sem) and yi_sem > 0.0:
                    ax.errorbar(
                        xi,
                        yi,
                        yerr=yi_sem,
                        fmt="none",
                        ecolor=PAPER_ERRORBAR_COLOR,
                        elinewidth=0.70,
                        capsize=1.8,
                        alpha=0.78,
                        zorder=14,
                    )
                if bool(row["_feasible"]):
                    ax.scatter(
                        xi,
                        yi,
                        s=24,
                        color=color,
                        edgecolors="0.2",
                        linewidths=0.5,
                        alpha=0.9,
                        zorder=20,
                    )
                else:
                    ax.scatter(
                        xi,
                        yi,
                        s=24,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.9,
                        alpha=0.82,
                        zorder=19,
                    )
            # Annotate top-3 to make ranking shifts readable without a large legend.
            top3 = sub.sort_values("rank", ascending=True, kind="mergesort").head(3).copy()
            for _, row in top3.iterrows():
                pid = str(row["polymer_id"])
                xi = float(row["abs0_vs_solvent_control"])
                yi = float(row["fog_vs_solvent_control"])
                rank_i = int(pd.to_numeric(row["rank"], errors="coerce"))
                ax.text(
                    xi + 0.012 * x_right,
                    yi + 0.012 * y_top,
                    f"#{rank_i} {pid}",
                    fontsize=5.0,
                    color="0.20",
                    ha="left",
                    va="bottom",
                    zorder=30,
                )
            ax.axvline(x=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
            ax.axhline(y=1.0, color="0.45", linestyle=(0, (1.5, 2.0)), linewidth=0.75, zorder=10)
            ax.set_title(label, fontsize=6.3)
            ax.grid(True, linestyle=":", alpha=0.30)

        for ax in axes_flat[n_panels:]:
            ax.axis("off")
        for ax in axes_flat[:n_panels]:
            ax.set_xlim(0.0, x_right)
            ax.set_ylim(0.0, y_top)
        for i, ax in enumerate(axes_flat[:n_panels]):
            if (i % ncols) == 0:
                ax.set_ylabel("FoG_solv = t50 / t50_solvent_control")
            if (i // ncols) == (nrows - 1):
                ax.set_xlabel(r"$U_{0}^{\mathrm{solv}}$ (vs solvent-matched control at 0 min)")
        fig.suptitle("Objective profile sensitivity map (all polymers)", y=1.01, fontsize=6.8)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_activity_bonus_penalty_profile_rank_heatmap(
    profile_rank_df: pd.DataFrame,
    *,
    out_path: Path,
) -> bool:
    """Plot per-polymer rank stability across objective profile settings."""
    if profile_rank_df.empty:
        return False
    req = {"polymer_id", "profile_id", "profile_label", "rank"}
    if not req.issubset(set(profile_rank_df.columns)):
        return False
    df = profile_rank_df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df[np.isfinite(df["rank"])].copy()
    if df.empty:
        return False
    profile_order = [pid for pid, _, _, _ in OBJECTIVE_PROFILE_SPECS if pid in set(df["profile_id"].tolist())]
    if not profile_order:
        return False
    label_map = (
        df[["profile_id", "profile_label"]]
        .drop_duplicates("profile_id")
        .set_index("profile_id")["profile_label"]
        .to_dict()
    )
    pivot = df.pivot_table(index="polymer_id", columns="profile_id", values="rank", aggfunc="first")
    use_cols = [c for c in profile_order if c in pivot.columns]
    if not use_cols:
        return False
    pivot = pivot[use_cols].copy()
    if "default" in pivot.columns:
        pivot = pivot.sort_values("default", ascending=True, kind="mergesort")
    else:
        pivot = pivot.sort_index()
    mat = pivot.to_numpy(dtype=float)
    if mat.size == 0:
        return False
    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt

    nrows, ncols = mat.shape
    fig_w = max(4.6, 1.2 + 1.6 * ncols)
    fig_h = max(3.2, 1.2 + 0.34 * nrows)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(mat, cmap="viridis_r", aspect="auto", vmin=1.0, vmax=max(1.0, vmax))
        xticklabels = [label_map.get(pid, str(pid)) for pid in use_cols]
        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels(xticklabels, rotation=16, ha="right")
        ax.set_yticks(np.arange(nrows))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xlabel("Objective profile")
        ax.set_ylabel("Polymer")
        ax.set_title("Rank stability across objective profiles (1 = best)")
        for i in range(nrows):
            for j in range(ncols):
                val = mat[i, j]
                if np.isfinite(val):
                    txt_color = "black" if val >= (0.55 * max(1.0, vmax)) else "white"
                    ax.text(
                        j,
                        i,
                        f"{int(round(val))}",
                        ha="center",
                        va="center",
                        fontsize=5.2,
                        color=txt_color,
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
        cbar.set_label("Rank (1=best)")
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.45, alpha=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _write_ranking_figure_guide(run_id: str, out_dir: Path, outputs: Dict[str, Path]) -> Path:
    """Write a compact figure-to-interpretation guide for one run."""
    guide_path = out_dir / f"figure_guide__{run_id}.md"
    rows: List[Tuple[str, str]] = [
        ("t50_ranking_png", "Activity-adjusted t50 ranking. Best single-value stability summary."),
        ("mainA_log_u0_vs_log_fog_iso_score_png", "Primary log-linear objective map on log(U0*) vs log(FoG*) with iso-score lines."),
        ("mainB_u0_vs_fog_tradeoff_with_pareto_png", "U0* vs FoG* tradeoff scatter (clean marker view)."),
        ("mainE_u0_vs_fog_loglog_regression_png", "U0* vs FoG* tradeoff with log-log regression line (slope/R2 shown)."),
        ("mainF_u0_vs_t50_loglog_regression_png", "U0* vs t50 tradeoff with log-log regression line (slope/R2 shown)."),
        ("mainC_u0_vs_fog_objective_contour_png", "U0* vs FoG* full-area topographic map (gradient + contour)."),
        ("mainD_u0_vs_fog_objective_hill3d_png", "3D hill view of the same objective landscape with polymer points."),
        ("mainA_native0_vs_fog_png", "Native U0 vs FoG map (GOx-referenced). Gate and tradeoff overview."),
        ("mainA_abs0_vs_fog_solvent_png", "Solvent-matched U0* vs FoG* map. Main control-balanced view."),
        ("objective_loglinear_main_ranking_png", "Primary objective ranking based on log(FoG*) + lambda*log(U0*)."),
        ("objective_activity_bonus_penalty_ranking_png", "Current objective S ranking (FoG* with activity bonus/penalty)."),
        ("objective_activity_bonus_penalty_tradeoff_png", "Objective map with iso-S curves."),
        ("objective_activity_bonus_penalty_proxy_curves_png", "Top-ranked proxy curves (curve-style intuition)."),
        ("objective_activity_bonus_penalty_proxy_curves_grid_png", "All-polymer proxy-curve grid."),
        ("objective_activity_bonus_penalty_profile_tradeoff_grid_png", "Objective profile sensitivity map (default vs nearby variants)."),
        ("objective_activity_bonus_penalty_profile_rank_heatmap_png", "Per-polymer rank stability across objective profiles."),
        ("supp_weight_sensitivity_png", "Weight (lambda) sensitivity for primary objective ranking."),
        ("supp_threshold_sensitivity_png", "U0 threshold sensitivity (secondary analysis) for ranking stability."),
        ("functional_ranking_png", "20 min functional activity ranking."),
    ]
    lines = [
        f"# Figure Guide ({run_id})",
        "",
        "This file maps each key figure to its interpretation role.",
        "",
        "| Figure | Role |",
        "|---|---|",
    ]
    for key, role in rows:
        path = outputs.get(key)
        if path is None:
            continue
        rel = str(path)
        try:
            rel = str(Path(path).resolve().relative_to(Path(out_dir).resolve()))
        except Exception:
            rel = str(path)
        lines.append(f"| `{rel}` | {role} |")
    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return guide_path


def _plot_mainb_feasible_ranking(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainB: feasible-only FoG ranking bar chart."""
    df = run_df.copy()
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    x_col = "native_0" if "native_0" in df.columns else "native_activity_rel_at_0"
    df[x_col] = pd.to_numeric(df.get(x_col, np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df["fog_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    fe = pd.to_numeric(df.get("native_activity_feasible", np.nan), errors="coerce")
    if np.isfinite(fe).any():
        df["_feasible"] = fe.fillna(0).astype(int) > 0
    else:
        df["_feasible"] = np.isfinite(df[x_col]) & (df[x_col] >= float(theta))
    df = df[df["_feasible"] & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False
    rank = (
        df.groupby("polymer_id", as_index=False)
        .agg(
            fog_median=("fog", "median"),
            fog_sem_median=("fog_sem", "median"),
            native0_median=(x_col, "median"),
            n_rows=("polymer_id", "size"),
        )
        .sort_values("fog_median", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    rank["rank"] = np.arange(1, len(rank) + 1, dtype=int)
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt
    cmap = color_map or {}
    default_color = "#4C78A8"
    labels = [str(pid) for pid in rank["polymer_id"]]
    colors = [cmap.get(str(pid), default_color) for pid in rank["polymer_id"]]
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(5.2, max(2.2, 0.30 * len(rank) + 0.9)))
        y = np.arange(len(rank), dtype=float)
        if "fog_sem_median" in rank.columns:
            fog_err = pd.to_numeric(rank["fog_sem_median"], errors="coerce").to_numpy(dtype=float)
        else:
            fog_err = np.full(len(rank), np.nan, dtype=float)
        fog_err = np.where(np.isfinite(fog_err) & (fog_err > 0.0), fog_err, 0.0)
        bar_kwargs = {}
        if np.any(fog_err > 0.0):
            bar_kwargs["xerr"] = fog_err
            bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.85,
            }
        ax.barh(
            y,
            rank["fog_median"].to_numpy(dtype=float),
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
            **bar_kwargs,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(x=1.0, color="0.45", linestyle=(0, (3, 2)), linewidth=0.75)
        ax.set_xlabel(r"$\mathrm{FoG}$ median among feasible")
        ax.set_title("MainB: feasible-only ranking")
        ax.grid(axis="x", linestyle=":", alpha=0.30)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_theta_sensitivity_summary(
    run_df: pd.DataFrame,
    *,
    theta_values: List[float],
    out_path: Path,
    csv_out_path: Optional[Path] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """Supplementary: theta sensitivity (feasible n and top-k overlap)."""
    df = run_df.copy()
    df = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if df.empty:
        return False
    x_col = "native_0" if "native_0" in df.columns else "native_activity_rel_at_0"
    df[x_col] = pd.to_numeric(df.get(x_col, np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df = df[np.isfinite(df[x_col]) & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False
    values = sorted(set(float(v) for v in theta_values if np.isfinite(v) and float(v) > 0.0))
    if not values:
        values = [0.60, 0.70, 0.75]
    rows: List[Dict[str, Any]] = []
    top_sets: Dict[float, set[str]] = {}
    for th in values:
        sub = df[df[x_col] >= float(th)].copy()
        if sub.empty:
            rows.append({"theta": float(th), "feasible_polymers": 0})
            top_sets[float(th)] = set()
            continue
        agg = (
            sub.groupby("polymer_id", as_index=False)
            .agg(fog_median=("fog", "median"))
            .sort_values("fog_median", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        top_sets[float(th)] = set(agg["polymer_id"].head(5).astype(str).tolist())
        rows.append({"theta": float(th), "feasible_polymers": int(len(agg))})
    sens = pd.DataFrame(rows)
    base_theta = min(values, key=lambda x: abs(float(x) - 0.70))
    base_set = top_sets.get(float(base_theta), set())
    sens["top5_overlap_vs_theta70"] = [
        float(len(top_sets.get(float(th), set()) & base_set)) / max(1, len(base_set))
        for th in sens["theta"].tolist()
    ]
    # write companion csv
    sens_out = Path(csv_out_path) if csv_out_path is not None else out_path.with_suffix(".csv")
    sens_out.parent.mkdir(parents=True, exist_ok=True)
    sens.to_csv(sens_out, index=False)
    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt
    with plt.rc_context(apply_paper_style()):
        fig, (ax_n, ax_o) = plt.subplots(1, 2, figsize=(7.0, 2.8))
        ax_n.plot(sens["theta"], sens["feasible_polymers"], marker="o", linewidth=1.0, color="#1f77b4")
        ax_n.set_xlabel("theta")
        ax_n.set_ylabel("Feasible polymers (n)")
        ax_n.set_title("Supp: feasible count")
        ax_n.grid(True, linestyle=":", alpha=0.30)
        ax_o.plot(sens["theta"], sens["top5_overlap_vs_theta70"], marker="o", linewidth=1.0, color="#2ca02c")
        ax_o.set_xlabel("theta")
        ax_o.set_ylabel(r"Top5 overlap vs $\theta=0.70$")
        ax_o.set_ylim(0.0, 1.05)
        ax_o.set_title("Supp: top-k stability")
        ax_o.grid(True, linestyle=":", alpha=0.30)
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def write_run_ranking_outputs(
    fog_df: pd.DataFrame,
    run_id: str,
    out_dir: Path,
    *,
    color_map_path: Optional[Path] = None,
    polymer_solvent_path: Optional[Path] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> Dict[str, Path]:
    """
    Write per-run ranking CSVs and bar charts for t50/FoG/functional.

    Outputs:
      - CSV: out_dir/csv/
        - t50_ranking__{run_id}.csv
        - fog_ranking__{run_id}.csv
      - objective_activity_bonus_penalty_ranking__{run_id}.csv
      - objective_loglinear_main_ranking__{run_id}.csv
        - functional_ranking__{run_id}.csv
        - objective_activity_bonus_penalty_profile_ranks__{run_id}.csv
        - supp_rank_correlation__{run_id}.csv
      - CSV: out_dir/old/csv/ (legacy objective outputs kept for traceability)
        - fog_ranking__{run_id}.csv
        - fog_native_constrained_ranking__{run_id}.csv
        - objective_native_soft_ranking__{run_id}.csv
      - t50_ranking__{run_id}.png (if plottable rows exist)
      - fog_ranking__{run_id}.png (if plottable rows exist)
      - objective_activity_bonus_penalty_ranking__{run_id}.png (if plottable rows exist)
      - objective_activity_bonus_penalty_tradeoff__{run_id}.png (supplementary objective-map figure)
      - objective_activity_bonus_penalty_proxy_curves__{run_id}.png (curve-style objective proxy)
      - new/objective_activity_bonus_penalty_proxy_curves_grid__{run_id}.png (all-polymer curve grid)
      - new/objective_activity_bonus_penalty_profile_tradeoff_grid__{run_id}.png (profile sensitivity grid)
      - new/objective_activity_bonus_penalty_profile_rank_heatmap__{run_id}.png (profile rank stability heatmap)
      - objective_loglinear_main_ranking__{run_id}.png (primary objective ranking bar)
      - mainA_log_u0_vs_log_fog_iso_score__{run_id}.png (log-space iso-score map)
      - mainB_u0_vs_fog_tradeoff_with_pareto__{run_id}.png (tradeoff scatter, clean marker view)
      - mainE_u0_vs_fog_loglog_regression__{run_id}.png (tradeoff with log-log regression)
      - mainF_u0_vs_t50_loglog_regression__{run_id}.png (t50 tradeoff with log-log regression)
      - mainC_u0_vs_fog_objective_contour__{run_id}.png (U0*-FoG* objective topographic map)
      - mainD_u0_vs_fog_objective_hill3d__{run_id}.png (3D objective hill map)
      - supp_weight_sensitivity__{run_id}.png (lambda sweep)
      - supp_threshold_sensitivity__{run_id}.png (U0 threshold sweep)
      - mainA_native0_vs_fog__{run_id}.png (paper main scatter)
      - mainA_abs0_vs_fog_solvent__{run_id}.png (paper main scatter, solvent-matched axes)
      - out_dir/csv/primary_objective_table__{run_id}.csv (native_0, FoG, t50, t_theta, QC flags)
      - figure_guide__{run_id}.md (what each figure means)
      - legacy figures are written under out_dir/old/

    Ranking score applies an absolute-activity guard when available:
      activity_weight = clip(abs_activity_at_0 / GOx_abs_activity_at_0, 0, 1)
      t50_activity_adjusted_min = t50_min * activity_weight
      fog_activity_adjusted = fog * activity_weight
      fog_native_soft = fog * clip(native_activity_rel_at_0, 0, 1)^2
      fog_activity_bonus_penalty = (t50 / t50_control) * abs_activity_balance(abs0 / abs0_control)
      abs_activity_balance(U0*) = clip(U0*, 0, 1)^2 * [1 + 0.35 * clip(U0* - 1.05, 0, 0.30)]
    """
    run_id = str(run_id).strip()
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = _ensure_csv_subdir(out_dir)
    old_dir = out_dir / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    old_csv_dir = _ensure_csv_subdir(old_dir)
    new_dir = out_dir / "new"
    new_dir.mkdir(parents=True, exist_ok=True)
    legacy_csv_names = [
        f"fog_native_constrained_ranking__{run_id}.csv",
        f"objective_native_soft_ranking__{run_id}.csv",
        f"objective_abs_bonus_penalty_ranking__{run_id}.csv",
        f"objective_abs_modulated_ranking__{run_id}.csv",
        f"objective_solvent_balanced_ranking__{run_id}.csv",
        f"supp_theta_sensitivity__{run_id}.csv",
    ]
    for name in legacy_csv_names:
        _archive_legacy_file(csv_dir / name, old_csv_dir / name)
        _archive_legacy_file(out_dir / name, old_csv_dir / name)
    current_csv_names = [
        f"t50_ranking__{run_id}.csv",
        f"fog_ranking__{run_id}.csv",
        f"objective_activity_bonus_penalty_ranking__{run_id}.csv",
        f"objective_loglinear_main_ranking__{run_id}.csv",
        f"functional_ranking__{run_id}.csv",
        f"primary_objective_table__{run_id}.csv",
        f"objective_activity_bonus_penalty_profile_ranks__{run_id}.csv",
        f"supp_rank_correlation__{run_id}.csv",
    ]
    for name in current_csv_names:
        _archive_legacy_file(csv_dir / name, old_csv_dir / name)
    legacy_png_names = [
        f"fog_native_constrained_ranking__{run_id}.png",
        f"objective_native_soft_ranking__{run_id}.png",
        f"objective_abs_bonus_penalty_ranking__{run_id}.png",
        f"objective_abs_modulated_ranking__{run_id}.png",
        f"objective_solvent_balanced_ranking__{run_id}.png",
        f"fog_native_constrained_decision__{run_id}.png",
        f"fog_native_constrained_tradeoff__{run_id}.png",
        f"objective_native_soft_tradeoff__{run_id}.png",
        f"objective_abs_bonus_penalty_tradeoff__{run_id}.png",
        f"objective_abs_modulated_tradeoff__{run_id}.png",
        f"objective_solvent_balanced_tradeoff__{run_id}.png",
        f"objective_abs_bonus_penalty_proxy_curves__{run_id}.png",
        f"objective_abs_modulated_proxy_curves__{run_id}.png",
        f"objective_solvent_balanced_proxy_curves__{run_id}.png",
        f"mainB_feasible_fog_ranking__{run_id}.png",
        f"supp_theta_sensitivity__{run_id}.png",
    ]
    for name in legacy_png_names:
        _archive_legacy_file(out_dir / name, old_dir / name)
    current_png_names = [
        f"t50_ranking__{run_id}.png",
        f"fog_ranking__{run_id}.png",
        f"objective_activity_bonus_penalty_ranking__{run_id}.png",
        f"objective_loglinear_main_ranking__{run_id}.png",
        f"objective_activity_bonus_penalty_tradeoff__{run_id}.png",
        f"objective_activity_bonus_penalty_proxy_curves__{run_id}.png",
        f"mainA_log_u0_vs_log_fog_iso_score__{run_id}.png",
        f"mainB_u0_vs_fog_tradeoff_with_pareto__{run_id}.png",
        f"mainE_u0_vs_fog_loglog_regression__{run_id}.png",
        f"mainF_u0_vs_t50_loglog_regression__{run_id}.png",
        f"mainC_u0_vs_fog_objective_contour__{run_id}.png",
        f"mainD_u0_vs_fog_objective_hill3d__{run_id}.png",
        f"supp_weight_sensitivity__{run_id}.png",
        f"supp_threshold_sensitivity__{run_id}.png",
        f"mainA_native0_vs_fog__{run_id}.png",
        f"mainA_abs0_vs_fog_solvent__{run_id}.png",
        f"functional_ranking__{run_id}.png",
    ]
    for name in current_png_names:
        _archive_legacy_file(out_dir / name, old_dir / name)
    current_new_png_names = [
        f"objective_activity_bonus_penalty_proxy_curves_grid__{run_id}.png",
        f"objective_activity_bonus_penalty_profile_tradeoff_grid__{run_id}.png",
        f"objective_activity_bonus_penalty_profile_rank_heatmap__{run_id}.png",
    ]
    for name in current_new_png_names:
        _archive_legacy_file(new_dir / name, old_dir / "new" / name)
    _archive_legacy_file(
        out_dir / f"figure_guide__{run_id}.md",
        old_dir / f"figure_guide__{run_id}.md",
    )

    df = fog_df.copy()
    if "run_id" not in df.columns:
        df["run_id"] = run_id
    df["run_id"] = df["run_id"].astype(str)
    df = df[df["run_id"] == run_id].copy()
    if df.empty:
        # Keep reproducible empty outputs when no rows are available.
        df = pd.DataFrame(columns=["run_id", "polymer_id", "t50_min", "t50_censored", "fog", "fog_missing_reason"])
    if "reference_polymer_id" in df.columns:
        ref_col = df["reference_polymer_id"].astype(str).str.strip()
        ref_col = ref_col.where(ref_col != "", reference_polymer_id)
        df["reference_polymer_id"] = ref_col
    else:
        df["reference_polymer_id"] = reference_polymer_id
    stock_solvent_map, control_solvent_map = _load_polymer_solvent_maps(polymer_solvent_path)
    df = _apply_polymer_solvent_maps(
        df,
        stock_map=stock_solvent_map,
        control_map=control_solvent_map,
    )

    df["t50_min"] = pd.to_numeric(df.get("t50_min", np.nan), errors="coerce")
    df["t50_sem_min"] = pd.to_numeric(df.get("t50_sem_min", np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df["fog_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df["fog_native_constrained"] = pd.to_numeric(df.get("fog_native_constrained", np.nan), errors="coerce")
    df["fog_native_constrained_sem"] = pd.to_numeric(
        df.get("fog_native_constrained_sem", np.nan), errors="coerce"
    )
    df["abs_activity_at_0"] = pd.to_numeric(df.get("abs_activity_at_0", np.nan), errors="coerce")
    df["gox_abs_activity_at_0_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_0_ref", np.nan), errors="coerce")
    df["gox_abs_activity_at_0"] = pd.to_numeric(df.get("gox_abs_activity_at_0", np.nan), errors="coerce")
    df["native_activity_rel_at_0"] = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    df["native_activity_min_rel_threshold"] = pd.to_numeric(
        df.get("native_activity_min_rel_threshold", np.nan), errors="coerce"
    )
    if "fog_constraint_reason" in df.columns:
        df["fog_constraint_reason"] = df["fog_constraint_reason"].astype(str)
    else:
        df["fog_constraint_reason"] = ""
    df["abs_activity_at_20"] = pd.to_numeric(df.get("abs_activity_at_20", np.nan), errors="coerce")
    df["gox_abs_activity_at_20_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce")
    df["functional_activity_at_20_rel"] = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    df["functional_activity_at_20_rel_sem"] = pd.to_numeric(
        df.get("functional_activity_at_20_rel_sem", np.nan), errors="coerce"
    )
    if "functional_reference_source" in df.columns:
        df["functional_reference_source"] = df["functional_reference_source"].astype(str)
    if "t50_censored" in df.columns:
        df["t50_censored"] = pd.to_numeric(df["t50_censored"], errors="coerce").fillna(1).astype(int)
    gox_abs_activity_at_0 = np.nan
    ref_mask = df["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
    if np.any(ref_mask):
        ref_vals = df.loc[ref_mask, "abs_activity_at_0"]
        ref_vals = ref_vals[np.isfinite(ref_vals) & (ref_vals > 0)]
        if not ref_vals.empty:
            gox_abs_activity_at_0 = float(ref_vals.iloc[0])
    if np.isfinite(gox_abs_activity_at_0) and gox_abs_activity_at_0 > 0:
        df["gox_abs_activity_at_0"] = np.where(
            np.isfinite(df["gox_abs_activity_at_0"]),
            df["gox_abs_activity_at_0"],
            float(gox_abs_activity_at_0),
        )
    else:
        df["gox_abs_activity_at_0"] = np.where(
            np.isfinite(df["gox_abs_activity_at_0"]),
            df["gox_abs_activity_at_0"],
            df["gox_abs_activity_at_0_ref"],
        )
    df = _add_solvent_balanced_objective_columns(
        df,
        reference_polymer_id=reference_polymer_id,
    )

    abs0_vs_gox = np.where(
        np.isfinite(df["abs_activity_at_0"])
        & np.isfinite(df["gox_abs_activity_at_0"])
        & (df["gox_abs_activity_at_0"] > 0.0),
        df["abs_activity_at_0"] / df["gox_abs_activity_at_0"],
        np.nan,
    )
    abs0_vs_gox = pd.to_numeric(abs0_vs_gox, errors="coerce")
    activity_weight = np.clip(abs0_vs_gox, 0.0, 1.0)
    activity_weight = np.where(np.isfinite(activity_weight), activity_weight, 1.0)
    df["abs0_vs_gox"] = abs0_vs_gox
    df["activity_weight"] = activity_weight
    df["t50_activity_adjusted_min"] = df["t50_min"] * df["activity_weight"]
    df["t50_activity_adjusted_sem_min"] = np.where(
        np.isfinite(df["t50_sem_min"]),
        df["t50_sem_min"] * df["activity_weight"],
        np.nan,
    )
    df["fog_activity_adjusted"] = df["fog"] * df["activity_weight"]
    df["fog_activity_adjusted_sem"] = np.where(
        np.isfinite(df["fog_sem"]),
        df["fog_sem"] * df["activity_weight"],
        np.nan,
    )
    native_rel_for_soft = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    native_rel_for_soft = np.where(np.isfinite(native_rel_for_soft), native_rel_for_soft, df["abs0_vs_gox"])
    soft_penalty, fog_soft, log_fog_soft = _compute_penalized_fog_objective(df["fog"], native_rel_for_soft)
    df["native_activity_soft_penalty"] = soft_penalty
    df["fog_native_soft"] = fog_soft
    df["log_fog_native_soft"] = log_fog_soft
    df["fog_native_soft_sem"] = np.where(
        np.isfinite(df["fog_sem"]) & np.isfinite(df["native_activity_soft_penalty"]),
        df["fog_sem"] * df["native_activity_soft_penalty"],
        np.nan,
    )

    # Native-activity feasibility and constrained FoG objective.
    native_thr = pd.to_numeric(df.get("native_activity_min_rel_threshold", np.nan), errors="coerce")
    if not np.isfinite(native_thr).any():
        native_thr = np.full(len(df), float(NATIVE_ACTIVITY_MIN_REL_DEFAULT), dtype=float)
    else:
        fallback_thr = float(np.nanmedian(native_thr[np.isfinite(native_thr)]))
        native_thr = np.where(np.isfinite(native_thr), native_thr, fallback_thr)
    df["native_activity_min_rel_threshold"] = native_thr
    rel = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    if not np.isfinite(rel).any():
        rel = pd.to_numeric(df.get("abs0_vs_gox", np.nan), errors="coerce")
    df["native_activity_rel_at_0"] = rel
    native_feasible = np.isfinite(rel) & np.isfinite(native_thr) & (rel >= native_thr)
    df["native_activity_feasible"] = native_feasible.astype(int)
    fog_native = pd.to_numeric(df.get("fog_native_constrained", np.nan), errors="coerce")
    fog_native_fallback = np.where(native_feasible, df["fog"], np.nan)
    df["fog_native_constrained"] = np.where(np.isfinite(fog_native), fog_native, fog_native_fallback)

    # Fill functional metric from abs20/reference if missing.
    fa20 = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    fa20_fallback = np.where(
        np.isfinite(df["abs_activity_at_20"]) & np.isfinite(df["gox_abs_activity_at_20_ref"]) & (df["gox_abs_activity_at_20_ref"] > 0),
        df["abs_activity_at_20"] / df["gox_abs_activity_at_20_ref"],
        np.nan,
    )
    df["functional_activity_at_20_rel"] = np.where(np.isfinite(fa20), fa20, fa20_fallback)

    # t50 ranking table
    t50_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "t50_min",
        "t50_sem_min",
        "t50_activity_adjusted_min",
        "t50_activity_adjusted_sem_min",
        "t50_censored",
        "t50_definition",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "activity_weight",
        "use_for_bo",
    ]
    t50_available = [c for c in t50_cols if c in df.columns]
    t50_tbl = df[t50_available].copy()
    t50_tbl["rank_t50"] = np.nan
    t50_valid = t50_tbl[
        np.isfinite(t50_tbl["t50_min"])
        & (t50_tbl["t50_min"] > 0)
        & np.isfinite(t50_tbl["t50_activity_adjusted_min"])
        & (t50_tbl["t50_activity_adjusted_min"] > 0)
    ].copy()
    if not t50_valid.empty:
        t50_valid = t50_valid.sort_values(
            ["t50_activity_adjusted_min", "t50_min"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        t50_valid["rank_t50"] = np.arange(1, len(t50_valid) + 1, dtype=int)
        t50_tbl = t50_valid
    t50_out = csv_dir / f"t50_ranking__{run_id}.csv"
    t50_tbl.to_csv(t50_out, index=False)
    _archive_legacy_file(out_dir / f"t50_ranking__{run_id}.csv", old_csv_dir / f"t50_ranking__{run_id}.csv")

    # FoG ranking table
    fog_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "fog",
        "fog_sem",
        "fog_activity_adjusted",
        "fog_activity_adjusted_sem",
        "log_fog",
        "fog_missing_reason",
        "gox_t50_same_run_min",
        "t50_min",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "activity_weight",
        "t50_censored",
        "use_for_bo",
    ]
    fog_available = [c for c in fog_cols if c in df.columns]
    fog_tbl = df[fog_available].copy()
    fog_tbl["rank_fog"] = np.nan
    fog_valid = fog_tbl[
        np.isfinite(fog_tbl["fog"])
        & (fog_tbl["fog"] > 0)
        & np.isfinite(fog_tbl["fog_activity_adjusted"])
        & (fog_tbl["fog_activity_adjusted"] > 0)
    ].copy()
    if not fog_valid.empty:
        fog_valid = fog_valid.sort_values(
            ["fog_activity_adjusted", "fog"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        fog_valid["rank_fog"] = np.arange(1, len(fog_valid) + 1, dtype=int)
        fog_tbl = fog_valid
    fog_out = csv_dir / f"fog_ranking__{run_id}.csv"
    fog_out_legacy = old_csv_dir / f"fog_ranking__{run_id}.csv"
    fog_tbl.to_csv(fog_out, index=False)
    fog_tbl.to_csv(fog_out_legacy, index=False)

    # Native-constrained FoG ranking table
    fog_constrained_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "fog_native_constrained",
        "fog_native_constrained_sem",
        "fog",
        "native_activity_rel_at_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "fog_constraint_reason",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "t50_min",
        "use_for_bo",
    ]
    fog_constrained_available = [c for c in fog_constrained_cols if c in df.columns]
    fog_constrained_tbl = df[fog_constrained_available].copy()
    fog_constrained_tbl["rank_fog_native_constrained"] = np.nan
    fog_constrained_valid = fog_constrained_tbl[
        np.isfinite(fog_constrained_tbl["fog_native_constrained"])
        & (fog_constrained_tbl["fog_native_constrained"] > 0)
    ].copy()
    if not fog_constrained_valid.empty:
        fog_constrained_valid = fog_constrained_valid.sort_values(
            ["fog_native_constrained", "fog"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        fog_constrained_valid["rank_fog_native_constrained"] = np.arange(
            1, len(fog_constrained_valid) + 1, dtype=int
        )
        fog_constrained_tbl = fog_constrained_valid
    fog_constrained_out = old_csv_dir / f"fog_native_constrained_ranking__{run_id}.csv"
    fog_constrained_tbl.to_csv(fog_constrained_out, index=False)

    # Soft-penalized objective ranking table.
    objective_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "fog_native_soft",
        "fog_native_soft_sem",
        "log_fog_native_soft",
        "native_activity_soft_penalty",
        "native_activity_rel_at_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "fog",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "t50_min",
        "use_for_bo",
    ]
    objective_available = [c for c in objective_cols if c in df.columns]
    objective_tbl = df[objective_available].copy()
    objective_tbl["rank_objective_native_soft"] = np.nan
    objective_valid = objective_tbl[
        np.isfinite(objective_tbl["fog_native_soft"])
        & (objective_tbl["fog_native_soft"] > 0)
    ].copy()
    if not objective_valid.empty:
        objective_valid = objective_valid.sort_values(
            ["fog_native_soft", "fog"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        objective_valid["rank_objective_native_soft"] = np.arange(
            1, len(objective_valid) + 1, dtype=int
        )
        objective_tbl = objective_valid
    objective_out = old_csv_dir / f"objective_native_soft_ranking__{run_id}.csv"
    objective_tbl.to_csv(objective_out, index=False)

    # FoG-activity bonus/penalty objective ranking table (current BO target).
    objective_balanced_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        ABS_ACTIVITY_OBJECTIVE_COL,
        ABS_ACTIVITY_OBJECTIVE_OLD_COL,
        ABS_ACTIVITY_OBJECTIVE_LEGACY_COL,
        "fog_solvent_balanced",
        ABS_ACTIVITY_OBJECTIVE_SEM_COL,
        ABS_ACTIVITY_OBJECTIVE_SEM_OLD_COL,
        ABS_ACTIVITY_OBJECTIVE_SEM_LEGACY_COL,
        "fog_solvent_balanced_sem",
        ABS_ACTIVITY_OBJECTIVE_LOG_COL,
        ABS_ACTIVITY_OBJECTIVE_LOG_OLD_COL,
        ABS_ACTIVITY_OBJECTIVE_LOG_LEGACY_COL,
        "log_fog_solvent_balanced",
        "stock_solvent_group",
        "solvent_group",
        "solvent_control_polymer_id",
        "solvent_control_abs_activity_at_0",
        "solvent_control_t50_min",
        "abs0_vs_solvent_control",
        "fog_vs_solvent_control",
        "abs_activity_down_penalty",
        "abs_activity_up_bonus",
        "abs_activity_balance_factor",
        "solvent_activity_down_penalty",
        "solvent_activity_up_bonus",
        "solvent_activity_balance_factor",
        "fog",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "t50_min",
        "use_for_bo",
    ]
    objective_balanced_available = [c for c in objective_balanced_cols if c in df.columns]
    objective_balanced_tbl = _sync_abs_objective_alias_columns(df[objective_balanced_available].copy())
    objective_balanced_tbl[ABS_ACTIVITY_OBJECTIVE_RANK_COL] = np.nan
    objective_balanced_valid = objective_balanced_tbl[
        np.isfinite(objective_balanced_tbl[ABS_ACTIVITY_OBJECTIVE_COL])
        & (objective_balanced_tbl[ABS_ACTIVITY_OBJECTIVE_COL] > 0)
    ].copy()
    if not objective_balanced_valid.empty:
        objective_balanced_valid = objective_balanced_valid.sort_values(
            [ABS_ACTIVITY_OBJECTIVE_COL, "fog_vs_solvent_control", "abs0_vs_solvent_control"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        objective_balanced_valid[ABS_ACTIVITY_OBJECTIVE_RANK_COL] = np.arange(
            1, len(objective_balanced_valid) + 1, dtype=int
        )
        objective_balanced_tbl = objective_balanced_valid
    objective_balanced_tbl["rank_objective_solvent_balanced"] = objective_balanced_tbl[ABS_ACTIVITY_OBJECTIVE_RANK_COL]
    objective_balanced_tbl[ABS_ACTIVITY_OBJECTIVE_RANK_OLD_COL] = objective_balanced_tbl[
        ABS_ACTIVITY_OBJECTIVE_RANK_COL
    ]
    objective_balanced_tbl[ABS_ACTIVITY_OBJECTIVE_RANK_LEGACY_COL] = objective_balanced_tbl[
        ABS_ACTIVITY_OBJECTIVE_RANK_COL
    ]
    objective_balanced_out = csv_dir / f"objective_activity_bonus_penalty_ranking__{run_id}.csv"
    objective_balanced_tbl.to_csv(objective_balanced_out, index=False)
    _archive_legacy_file(
        out_dir / f"objective_activity_bonus_penalty_ranking__{run_id}.csv",
        old_csv_dir / f"objective_activity_bonus_penalty_ranking__{run_id}.csv",
    )
    _archive_legacy_file(
        out_dir / f"objective_solvent_balanced_ranking__{run_id}.csv",
        old_csv_dir / f"objective_solvent_balanced_ranking__{run_id}.csv",
    )

    # Primary objective ranking table: S = log(FoG*) + lambda*log(U0*), lambda fixed a priori.
    objective_loglinear_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        OBJECTIVE_LOGLINEAR_MAIN_COL,
        OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
        "abs0_vs_solvent_control",
        "fog_vs_solvent_control",
        "fog_sem",
        "fog",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "t50_min",
        "solvent_group",
        "solvent_control_polymer_id",
        "use_for_bo",
    ]
    objective_loglinear_available = [c for c in objective_loglinear_cols if c in df.columns]
    objective_loglinear_tbl = df[objective_loglinear_available].copy()
    if OBJECTIVE_LOGLINEAR_MAIN_COL not in objective_loglinear_tbl.columns:
        score_main, score_main_exp = _compute_loglinear_main_objective(
            objective_loglinear_tbl.get("fog_vs_solvent_control", np.nan),
            objective_loglinear_tbl.get("abs0_vs_solvent_control", np.nan),
            weight_lambda=OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT,
        )
        objective_loglinear_tbl[OBJECTIVE_LOGLINEAR_MAIN_COL] = score_main
        objective_loglinear_tbl[OBJECTIVE_LOGLINEAR_MAIN_EXP_COL] = score_main_exp
    objective_loglinear_tbl = _rank_by_loglinear_objective(
        objective_loglinear_tbl,
        score_col=OBJECTIVE_LOGLINEAR_MAIN_COL,
        rank_col=OBJECTIVE_LOGLINEAR_MAIN_RANK_COL,
    )
    objective_loglinear_tbl = objective_loglinear_tbl[
        np.isfinite(pd.to_numeric(objective_loglinear_tbl[OBJECTIVE_LOGLINEAR_MAIN_RANK_COL], errors="coerce"))
    ].copy()
    objective_loglinear_tbl[OBJECTIVE_LOGLINEAR_MAIN_RANK_COL] = pd.to_numeric(
        objective_loglinear_tbl[OBJECTIVE_LOGLINEAR_MAIN_RANK_COL], errors="coerce"
    ).astype(int)
    objective_loglinear_tbl = objective_loglinear_tbl.sort_values(
        OBJECTIVE_LOGLINEAR_MAIN_RANK_COL,
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)
    objective_loglinear_out = csv_dir / f"objective_loglinear_main_ranking__{run_id}.csv"
    objective_loglinear_tbl.to_csv(objective_loglinear_out, index=False)
    _archive_legacy_file(
        out_dir / f"objective_loglinear_main_ranking__{run_id}.csv",
        old_csv_dir / f"objective_loglinear_main_ranking__{run_id}.csv",
    )

    objective_profile_rank_tbl = _compute_activity_objective_profile_ranks(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    objective_profile_rank_out = csv_dir / f"objective_activity_bonus_penalty_profile_ranks__{run_id}.csv"
    if objective_profile_rank_tbl.empty:
        objective_profile_rank_tbl = pd.DataFrame(
            columns=[
                "polymer_id",
                "abs0_vs_solvent_control",
                "fog_vs_solvent_control",
                "fog_vs_solvent_control_sem",
                "profile_id",
                "profile_label",
                "penalty_exponent",
                "bonus_coef",
                "bonus_deadband",
                "bonus_max_delta",
                "score",
                "rank",
                "rank_default",
                "rank_delta_vs_default",
            ]
        )
    objective_profile_rank_tbl.to_csv(objective_profile_rank_out, index=False)

    # Functional (20 min) ranking table
    func_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "functional_activity_at_20_rel",
        "functional_activity_at_20_rel_sem",
        "abs_activity_at_20",
        "gox_abs_activity_at_20_ref",
        "functional_reference_source",
        "functional_reference_round_id",
        "functional_reference_run_id",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "use_for_bo",
    ]
    func_available = [c for c in func_cols if c in df.columns]
    func_tbl = df[func_available].copy()
    func_tbl["rank_functional"] = np.nan
    func_valid = func_tbl[
        np.isfinite(func_tbl["functional_activity_at_20_rel"])
        & (func_tbl["functional_activity_at_20_rel"] > 0)
    ].copy()
    if not func_valid.empty:
        func_valid = func_valid.sort_values(
            ["functional_activity_at_20_rel", "abs_activity_at_20"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        func_valid["rank_functional"] = np.arange(1, len(func_valid) + 1, dtype=int)
        func_tbl = func_valid
    func_out = csv_dir / f"functional_ranking__{run_id}.csv"
    func_tbl.to_csv(func_out, index=False)
    _archive_legacy_file(
        out_dir / f"functional_ranking__{run_id}.csv",
        old_csv_dir / f"functional_ranking__{run_id}.csv",
    )

    cmap = _load_polymer_color_map(color_map_path)
    t50_png = out_dir / f"t50_ranking__{run_id}.png"
    fog_png = out_dir / f"fog_ranking__{run_id}.png"
    fog_png_legacy = old_dir / f"fog_ranking__{run_id}.png"
    fog_constrained_png = old_dir / f"fog_native_constrained_ranking__{run_id}.png"
    objective_png = old_dir / f"objective_native_soft_ranking__{run_id}.png"
    objective_balanced_png = out_dir / f"objective_activity_bonus_penalty_ranking__{run_id}.png"
    objective_loglinear_png = out_dir / f"objective_loglinear_main_ranking__{run_id}.png"
    fog_decision_png = old_dir / f"fog_native_constrained_decision__{run_id}.png"
    func_png = out_dir / f"functional_ranking__{run_id}.png"
    wrote_t50_png = _plot_run_ranking_bar(
        t50_tbl,
        value_col="t50_activity_adjusted_min",
        error_col="t50_activity_adjusted_sem_min",
        rank_col="rank_t50",
        title=r"$t_{50}$ ranking (activity-adjusted)",
        xlabel=r"Activity-adjusted $t_{50}$ (min)",
        out_path=t50_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_fog_png = _plot_run_ranking_bar(
        fog_tbl,
        value_col="fog_activity_adjusted",
        error_col="fog_activity_adjusted_sem",
        rank_col="rank_fog",
        title="FoG ranking (activity-adjusted)",
        xlabel="activity-adjusted FoG",
        out_path=fog_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_fog_png_legacy = False
    if wrote_fog_png:
        wrote_fog_png_legacy = _plot_run_ranking_bar(
            fog_tbl,
            value_col="fog_activity_adjusted",
            error_col="fog_activity_adjusted_sem",
            rank_col="rank_fog",
            title="FoG ranking (activity-adjusted)",
            xlabel="activity-adjusted FoG",
            out_path=fog_png_legacy,
            color_map=cmap,
            reference_polymer_id=reference_polymer_id,
        )
    wrote_fog_constrained_png = _plot_run_ranking_bar(
        fog_constrained_tbl,
        value_col="fog_native_constrained",
        error_col="fog_native_constrained_sem",
        rank_col="rank_fog_native_constrained",
        title=r"FoG ranking ($U_{0}\geq\theta$ constrained)",
        xlabel=r"$\mathrm{FoG}$ ($U_{0}\geq\theta$ constrained)",
        out_path=fog_constrained_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    soft_exp_lbl = _format_penalty_exponent_label(NATIVE_ACTIVITY_SOFT_PENALTY_EXPONENT)
    wrote_objective_png = _plot_run_ranking_bar(
        objective_tbl,
        value_col="fog_native_soft",
        error_col="fog_native_soft_sem",
        rank_col="rank_objective_native_soft",
        title="FoG ranking (native-penalized soft objective)",
        xlabel=rf"$\mathrm{{FoG}}\times\mathrm{{clip}}(U_{{0}},0,1)^{{{soft_exp_lbl}}}$",
        out_path=objective_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_objective_balanced_png = _plot_run_ranking_bar(
        objective_balanced_tbl,
        value_col=ABS_ACTIVITY_OBJECTIVE_COL,
        error_col=ABS_ACTIVITY_OBJECTIVE_SEM_COL,
        rank_col=ABS_ACTIVITY_OBJECTIVE_RANK_COL,
        title="FoG-activity bonus/penalty objective ranking",
        xlabel="S = FoG* x activity bonus/penalty(U0*)",
        out_path=objective_balanced_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_objective_loglinear_png = _plot_run_ranking_bar(
        objective_loglinear_tbl,
        value_col=OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
        error_col="fog_sem",
        rank_col=OBJECTIVE_LOGLINEAR_MAIN_RANK_COL,
        title=(
            "Primary objective ranking "
            rf"(exp(log(FoG*)+{OBJECTIVE_LOGLINEAR_MAIN_LAMBDA_DEFAULT:.2f}log(U0*)))"
        ),
        xlabel=r"exp(log(FoG*) + \lambda log(U0*))",
        out_path=objective_loglinear_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_func_png = _plot_run_ranking_bar(
        func_tbl,
        value_col="functional_activity_at_20_rel",
        error_col="functional_activity_at_20_rel_sem",
        rank_col="rank_functional",
        title="Functional ranking (20 min)",
        xlabel=f"functional activity ratio at 20 min (vs {reference_polymer_id})",
        out_path=func_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_t50_png and t50_png.exists():
        t50_png.unlink(missing_ok=True)
    if not wrote_fog_png and fog_png.exists():
        fog_png.unlink(missing_ok=True)
    if not wrote_fog_png_legacy and fog_png_legacy.exists():
        fog_png_legacy.unlink(missing_ok=True)
    if not wrote_fog_constrained_png and fog_constrained_png.exists():
        fog_constrained_png.unlink(missing_ok=True)
    if not wrote_objective_png and objective_png.exists():
        objective_png.unlink(missing_ok=True)
    if not wrote_objective_balanced_png and objective_balanced_png.exists():
        objective_balanced_png.unlink(missing_ok=True)
    if not wrote_objective_loglinear_png and objective_loglinear_png.exists():
        objective_loglinear_png.unlink(missing_ok=True)
    if not wrote_func_png and func_png.exists():
        func_png.unlink(missing_ok=True)

    # Native-gated decision figure (main) + trade-off scatter (supplementary).
    theta = float(np.nanmedian(df["native_activity_min_rel_threshold"])) if "native_activity_min_rel_threshold" in df.columns and np.any(np.isfinite(df["native_activity_min_rel_threshold"])) else NATIVE_ACTIVITY_MIN_REL_DEFAULT
    wrote_decision = _plot_fog_native_constrained_decision(
        df,
        theta=theta,
        out_path=fog_decision_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_decision and fog_decision_png.exists():
        fog_decision_png.unlink(missing_ok=True)

    tradeoff_png = old_dir / f"fog_native_constrained_tradeoff__{run_id}.png"
    wrote_tradeoff = _plot_fog_native_constrained_tradeoff(
        df,
        theta=theta,
        out_path=tradeoff_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_tradeoff and tradeoff_png.exists():
        tradeoff_png.unlink(missing_ok=True)
    objective_tradeoff_png = old_dir / f"objective_native_soft_tradeoff__{run_id}.png"
    wrote_objective_tradeoff = _plot_fog_native_soft_tradeoff(
        df,
        out_path=objective_tradeoff_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
        penalty_exponent=NATIVE_ACTIVITY_SOFT_PENALTY_EXPONENT,
    )
    if not wrote_objective_tradeoff and objective_tradeoff_png.exists():
        objective_tradeoff_png.unlink(missing_ok=True)
    objective_balanced_tradeoff_png = out_dir / f"objective_activity_bonus_penalty_tradeoff__{run_id}.png"
    wrote_objective_balanced_tradeoff = _plot_fog_solvent_balanced_tradeoff(
        df,
        out_path=objective_balanced_tradeoff_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_objective_balanced_tradeoff and objective_balanced_tradeoff_png.exists():
        objective_balanced_tradeoff_png.unlink(missing_ok=True)
    objective_balanced_proxy_png = out_dir / f"objective_activity_bonus_penalty_proxy_curves__{run_id}.png"
    wrote_objective_balanced_proxy = _plot_activity_bonus_penalty_proxy_curves(
        df,
        out_path=objective_balanced_proxy_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_objective_balanced_proxy and objective_balanced_proxy_png.exists():
        objective_balanced_proxy_png.unlink(missing_ok=True)
    objective_balanced_proxy_grid_png = new_dir / f"objective_activity_bonus_penalty_proxy_curves_grid__{run_id}.png"
    n_proxy_all = int(
        np.sum(
            ~df["polymer_id"].astype(str).map(
                lambda x: _is_reference_like_polymer_id(
                    x,
                    reference_polymer_id=reference_polymer_id,
                )
            )
        )
    )
    wrote_objective_balanced_proxy_grid = _plot_activity_bonus_penalty_proxy_curves(
        df,
        out_path=objective_balanced_proxy_grid_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
        top_n=max(1, n_proxy_all),
    )
    if not wrote_objective_balanced_proxy_grid and objective_balanced_proxy_grid_png.exists():
        objective_balanced_proxy_grid_png.unlink(missing_ok=True)
    objective_profile_tradeoff_grid_png = new_dir / f"objective_activity_bonus_penalty_profile_tradeoff_grid__{run_id}.png"
    wrote_objective_profile_tradeoff_grid = _plot_activity_bonus_penalty_profile_tradeoff_grid(
        objective_profile_rank_tbl,
        out_path=objective_profile_tradeoff_grid_png,
        color_map=cmap,
    )
    if not wrote_objective_profile_tradeoff_grid and objective_profile_tradeoff_grid_png.exists():
        objective_profile_tradeoff_grid_png.unlink(missing_ok=True)
    objective_profile_rank_heatmap_png = new_dir / f"objective_activity_bonus_penalty_profile_rank_heatmap__{run_id}.png"
    wrote_objective_profile_rank_heatmap = _plot_activity_bonus_penalty_profile_rank_heatmap(
        objective_profile_rank_tbl,
        out_path=objective_profile_rank_heatmap_png,
    )
    if not wrote_objective_profile_rank_heatmap and objective_profile_rank_heatmap_png.exists():
        objective_profile_rank_heatmap_png.unlink(missing_ok=True)

    # Additional paper-oriented main figure set.
    maina_log_png = out_dir / f"mainA_log_u0_vs_log_fog_iso_score__{run_id}.png"
    mainb_tradeoff_png = out_dir / f"mainB_u0_vs_fog_tradeoff_with_pareto__{run_id}.png"
    maine_reg_png = out_dir / f"mainE_u0_vs_fog_loglog_regression__{run_id}.png"
    mainf_reg_png = out_dir / f"mainF_u0_vs_t50_loglog_regression__{run_id}.png"
    mainc_contour_png = out_dir / f"mainC_u0_vs_fog_objective_contour__{run_id}.png"
    maind_hill3d_png = out_dir / f"mainD_u0_vs_fog_objective_hill3d__{run_id}.png"
    maina_png = out_dir / f"mainA_native0_vs_fog__{run_id}.png"
    maina_solvent_png = out_dir / f"mainA_abs0_vs_fog_solvent__{run_id}.png"
    mainb_png = old_dir / f"mainB_feasible_fog_ranking__{run_id}.png"
    supp_theta_png = old_dir / f"supp_theta_sensitivity__{run_id}.png"
    supp_theta_csv = old_csv_dir / f"supp_theta_sensitivity__{run_id}.csv"
    supp_weight_png = out_dir / f"supp_weight_sensitivity__{run_id}.png"
    supp_threshold_png = out_dir / f"supp_threshold_sensitivity__{run_id}.png"
    supp_rank_corr_csv = csv_dir / f"supp_rank_correlation__{run_id}.csv"

    wrote_maina_log = _plot_maina_log_u0_vs_log_fog_iso_score(
        df,
        out_path=maina_log_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_maina_log and maina_log_png.exists():
        maina_log_png.unlink(missing_ok=True)
    wrote_mainb_tradeoff = _plot_mainb_u0_vs_fog_tradeoff_with_pareto(
        df,
        out_path=mainb_tradeoff_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_mainb_tradeoff and mainb_tradeoff_png.exists():
        mainb_tradeoff_png.unlink(missing_ok=True)
    wrote_maine_reg = _plot_maine_u0_vs_fog_loglog_regression(
        df,
        out_path=maine_reg_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_maine_reg and maine_reg_png.exists():
        maine_reg_png.unlink(missing_ok=True)
    wrote_mainf_reg = _plot_mainf_u0_vs_t50_loglog_regression(
        df,
        out_path=mainf_reg_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_mainf_reg and mainf_reg_png.exists():
        mainf_reg_png.unlink(missing_ok=True)
    wrote_mainc_contour = _plot_mainc_u0_vs_fog_objective_contour(
        df,
        out_path=mainc_contour_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_mainc_contour and mainc_contour_png.exists():
        mainc_contour_png.unlink(missing_ok=True)
    wrote_maind_hill3d = _plot_maind_u0_vs_fog_objective_hill3d(
        df,
        out_path=maind_hill3d_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_maind_hill3d and maind_hill3d_png.exists():
        maind_hill3d_png.unlink(missing_ok=True)
    wrote_maina = _plot_maina_native_vs_fog(
        df,
        theta=theta,
        out_path=maina_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_maina and maina_png.exists():
        maina_png.unlink(missing_ok=True)
    wrote_maina_solvent = _plot_maina_abs_vs_fog_solvent(
        df,
        out_path=maina_solvent_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_maina_solvent and maina_solvent_png.exists():
        maina_solvent_png.unlink(missing_ok=True)
    wrote_mainb = _plot_mainb_feasible_ranking(
        df,
        theta=theta,
        out_path=mainb_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_mainb and mainb_png.exists():
        mainb_png.unlink(missing_ok=True)
    wrote_supp_theta = _plot_theta_sensitivity_summary(
        df,
        theta_values=[0.60, 0.70, 0.75],
        out_path=supp_theta_png,
        csv_out_path=supp_theta_csv,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_supp_theta and supp_theta_png.exists():
        supp_theta_png.unlink(missing_ok=True)
    if not wrote_supp_theta and supp_theta_csv.exists():
        supp_theta_csv.unlink(missing_ok=True)
    wrote_supp_weight, corr_weight_df = _plot_loglinear_weight_sensitivity(
        df,
        out_path=supp_weight_png,
        reference_polymer_id=reference_polymer_id,
        weight_values=OBJECTIVE_LOGLINEAR_WEIGHT_SWEEP,
    )
    if not wrote_supp_weight and supp_weight_png.exists():
        supp_weight_png.unlink(missing_ok=True)
    wrote_supp_threshold, corr_threshold_df = _plot_loglinear_threshold_sensitivity(
        df,
        out_path=supp_threshold_png,
        reference_polymer_id=reference_polymer_id,
        thresholds=OBJECTIVE_LOGLINEAR_THRESHOLD_SWEEP,
    )
    if not wrote_supp_threshold and supp_threshold_png.exists():
        supp_threshold_png.unlink(missing_ok=True)
    corr_cols = [
        "analysis_type",
        "scenario",
        "reference_scenario",
        "weight_lambda",
        "u0_threshold",
        "n_common",
        "kendall_tau",
        "spearman_rho",
        "top5_overlap",
    ]
    corr_frames = [cdf for cdf in [corr_weight_df, corr_threshold_df] if isinstance(cdf, pd.DataFrame) and not cdf.empty]
    if corr_frames:
        supp_rank_corr_tbl = pd.concat(corr_frames, axis=0, ignore_index=True)
    else:
        supp_rank_corr_tbl = pd.DataFrame(columns=corr_cols)
    for col in corr_cols:
        if col not in supp_rank_corr_tbl.columns:
            supp_rank_corr_tbl[col] = np.nan
    supp_rank_corr_tbl = supp_rank_corr_tbl[corr_cols].copy()
    supp_rank_corr_tbl.to_csv(supp_rank_corr_csv, index=False)

    # Primary objective table for quick PI/reviewer reading.
    primary_tbl = _exclude_reference_like_polymers(
        df,
        reference_polymer_id=reference_polymer_id,
    )
    if "native_0" not in primary_tbl.columns and "native_activity_rel_at_0" in primary_tbl.columns:
        primary_tbl["native_0"] = pd.to_numeric(primary_tbl.get("native_activity_rel_at_0", np.nan), errors="coerce")
    keep_cols = [
        "run_id",
        "polymer_id",
        "native_0",
        "native_activity_rel_at_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "native_activity_soft_penalty",
        "fog",
        "fog_native_constrained",
        "fog_native_soft",
        "log_fog_native_soft",
        "solvent_group",
        "solvent_control_polymer_id",
        "abs0_vs_solvent_control",
        "fog_vs_solvent_control",
        "solvent_activity_down_penalty",
        "solvent_activity_up_bonus",
        "solvent_activity_balance_factor",
        "abs_activity_down_penalty",
        "abs_activity_up_bonus",
        "abs_activity_balance_factor",
        OBJECTIVE_LOGLINEAR_MAIN_COL,
        OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
        OBJECTIVE_LOGLINEAR_MAIN_RANK_COL,
        ABS_ACTIVITY_OBJECTIVE_COL,
        ABS_ACTIVITY_OBJECTIVE_LOG_COL,
        "fog_solvent_balanced",
        "log_fog_solvent_balanced",
        "t50_min",
        "t_theta",
        "t_theta_censor_flag",
        "reference_qc_fail",
        "reference_qc_reason",
    ]
    keep = [c for c in keep_cols if c in primary_tbl.columns]
    primary_tbl = primary_tbl[keep].copy()
    if OBJECTIVE_LOGLINEAR_MAIN_COL in primary_tbl.columns:
        primary_tbl = _rank_by_loglinear_objective(
            primary_tbl,
            score_col=OBJECTIVE_LOGLINEAR_MAIN_COL,
            rank_col=OBJECTIVE_LOGLINEAR_MAIN_RANK_COL,
        )
        sort_cols = [
            c
            for c in [
                OBJECTIVE_LOGLINEAR_MAIN_RANK_COL,
                OBJECTIVE_LOGLINEAR_MAIN_COL,
                "abs0_vs_solvent_control",
                "fog_vs_solvent_control",
            ]
            if c in primary_tbl.columns
        ]
        sort_asc = [True, False, False, False][: len(sort_cols)]
        if sort_cols:
            primary_tbl = primary_tbl.sort_values(
                sort_cols,
                ascending=sort_asc,
                kind="mergesort",
            )
    elif ABS_ACTIVITY_OBJECTIVE_COL in primary_tbl.columns:
        primary_tbl = primary_tbl.sort_values(
            [ABS_ACTIVITY_OBJECTIVE_COL, "fog_vs_solvent_control", "abs0_vs_solvent_control", "fog"],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
    elif "fog_native_soft" in primary_tbl.columns:
        primary_tbl = primary_tbl.sort_values(
            ["fog_native_soft", "fog_native_constrained", "fog"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    elif "fog_native_constrained" in primary_tbl.columns:
        primary_tbl = primary_tbl.sort_values(
            ["native_activity_feasible", "fog_native_constrained", "fog"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    primary_tbl_out = csv_dir / f"primary_objective_table__{run_id}.csv"
    primary_tbl.to_csv(primary_tbl_out, index=False)
    _archive_legacy_file(
        out_dir / f"primary_objective_table__{run_id}.csv",
        old_csv_dir / f"primary_objective_table__{run_id}.csv",
    )

    outputs: Dict[str, Path] = {
        "t50_ranking_csv": t50_out,
        "fog_ranking_csv": fog_out,
        "fog_ranking_legacy_csv": fog_out_legacy,
        "fog_native_constrained_ranking_csv": fog_constrained_out,
        "objective_native_soft_ranking_csv": objective_out,
        "objective_activity_bonus_penalty_ranking_csv": objective_balanced_out,
        "objective_loglinear_main_ranking_csv": objective_loglinear_out,
        "objective_activity_bonus_penalty_profile_ranks_csv": objective_profile_rank_out,
        "objective_abs_bonus_penalty_ranking_csv": objective_balanced_out,
        "objective_abs_modulated_ranking_csv": objective_balanced_out,
        "objective_solvent_balanced_ranking_csv": objective_balanced_out,
        "functional_ranking_csv": func_out,
        "primary_objective_table_csv": primary_tbl_out,
        "supp_rank_correlation_csv": supp_rank_corr_csv,
    }
    if wrote_t50_png:
        outputs["t50_ranking_png"] = t50_png
    if wrote_fog_png:
        outputs["fog_ranking_png"] = fog_png
    if wrote_fog_png_legacy:
        outputs["fog_ranking_legacy_png"] = fog_png_legacy
    if wrote_fog_constrained_png:
        outputs["fog_native_constrained_ranking_png"] = fog_constrained_png
    if wrote_objective_png:
        outputs["objective_native_soft_ranking_png"] = objective_png
    if wrote_objective_balanced_png:
        outputs["objective_activity_bonus_penalty_ranking_png"] = objective_balanced_png
        outputs["objective_abs_bonus_penalty_ranking_png"] = objective_balanced_png
        outputs["objective_abs_modulated_ranking_png"] = objective_balanced_png
        outputs["objective_solvent_balanced_ranking_png"] = objective_balanced_png
    if wrote_objective_loglinear_png:
        outputs["objective_loglinear_main_ranking_png"] = objective_loglinear_png
    if wrote_decision:
        outputs["fog_native_constrained_decision_png"] = fog_decision_png
    if wrote_tradeoff:
        outputs["fog_native_constrained_tradeoff_png"] = tradeoff_png
    if wrote_objective_tradeoff:
        outputs["objective_native_soft_tradeoff_png"] = objective_tradeoff_png
    if wrote_objective_balanced_tradeoff:
        outputs["objective_activity_bonus_penalty_tradeoff_png"] = objective_balanced_tradeoff_png
        outputs["objective_abs_bonus_penalty_tradeoff_png"] = objective_balanced_tradeoff_png
        outputs["objective_abs_modulated_tradeoff_png"] = objective_balanced_tradeoff_png
        outputs["objective_solvent_balanced_tradeoff_png"] = objective_balanced_tradeoff_png
    if wrote_objective_balanced_proxy:
        outputs["objective_activity_bonus_penalty_proxy_curves_png"] = objective_balanced_proxy_png
        outputs["objective_abs_bonus_penalty_proxy_curves_png"] = objective_balanced_proxy_png
        outputs["objective_abs_modulated_proxy_curves_png"] = objective_balanced_proxy_png
        outputs["objective_solvent_balanced_proxy_curves_png"] = objective_balanced_proxy_png
    if wrote_objective_balanced_proxy_grid:
        outputs["objective_activity_bonus_penalty_proxy_curves_grid_png"] = objective_balanced_proxy_grid_png
    if wrote_objective_profile_tradeoff_grid:
        outputs["objective_activity_bonus_penalty_profile_tradeoff_grid_png"] = objective_profile_tradeoff_grid_png
    if wrote_objective_profile_rank_heatmap:
        outputs["objective_activity_bonus_penalty_profile_rank_heatmap_png"] = objective_profile_rank_heatmap_png
    if wrote_maina:
        outputs["mainA_native0_vs_fog_png"] = maina_png
    if wrote_maina_log:
        outputs["mainA_log_u0_vs_log_fog_iso_score_png"] = maina_log_png
    if wrote_maina_solvent:
        outputs["mainA_abs0_vs_fog_solvent_png"] = maina_solvent_png
    if wrote_mainb_tradeoff:
        outputs["mainB_u0_vs_fog_tradeoff_with_pareto_png"] = mainb_tradeoff_png
    if wrote_maine_reg:
        outputs["mainE_u0_vs_fog_loglog_regression_png"] = maine_reg_png
    if wrote_mainf_reg:
        outputs["mainF_u0_vs_t50_loglog_regression_png"] = mainf_reg_png
    if wrote_mainc_contour:
        outputs["mainC_u0_vs_fog_objective_contour_png"] = mainc_contour_png
    if wrote_maind_hill3d:
        outputs["mainD_u0_vs_fog_objective_hill3d_png"] = maind_hill3d_png
    if wrote_mainb:
        outputs["mainB_feasible_fog_ranking_png"] = mainb_png
    if wrote_supp_theta:
        outputs["supp_theta_sensitivity_png"] = supp_theta_png
        outputs["supp_theta_sensitivity_csv"] = supp_theta_csv
    if wrote_supp_weight:
        outputs["supp_weight_sensitivity_png"] = supp_weight_png
    if wrote_supp_threshold:
        outputs["supp_threshold_sensitivity_png"] = supp_threshold_png
    if wrote_func_png:
        outputs["functional_ranking_png"] = func_png
    guide_path = _write_ranking_figure_guide(run_id, out_dir, outputs)
    outputs["figure_guide_md"] = guide_path
    return outputs


def build_round_averaged_fog(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Build round-averaged FoG from per-run fog_summary CSVs.

    - FoG is currently computed per run (same-run GOx as denominator). This function
      averages FoG by (round_id, polymer_id) across all runs in that round.
    - GOx row is excluded from the output. Only rows with finite fog and fog > 0 are averaged.
    - If a round has no run with GOx (all runs have gox_t50_same_run_min NaN), raises ValueError.
    - Output columns:
      round_id, polymer_id, mean_fog, mean_log_fog, robust_fog, robust_log_fog,
      log_fog_mad, mean_fog_native_constrained, mean_log_fog_native_constrained,
      robust_fog_native_constrained, robust_log_fog_native_constrained,
      log_fog_native_constrained_mad, mean_fog_native_soft, mean_log_fog_native_soft,
      robust_fog_native_soft, robust_log_fog_native_soft, log_fog_native_soft_mad,
      mean_fog_solvent_balanced, mean_log_fog_solvent_balanced,
      robust_fog_solvent_balanced, robust_log_fog_solvent_balanced, log_fog_solvent_balanced_mad,
      mean/robust objective_loglinear_main (+ MAD) and exp(objective_loglinear_main),
      native_feasible_fraction, n_observations, run_ids.
    """
    processed_dir = Path(processed_dir)
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    # round_id -> list of run_ids
    round_to_runs: Dict[str, List[str]] = {}
    for rid, oid in run_round_map.items():
        rid, oid = str(rid).strip(), str(oid).strip()
        if not oid:
            continue
        round_to_runs.setdefault(oid, []).append(rid)

    if not round_to_runs:
        return pd.DataFrame(
            columns=[
                "round_id",
                "polymer_id",
                "reference_polymer_id",
                "mean_fog",
                "mean_log_fog",
                "robust_fog",
                "robust_log_fog",
                "log_fog_mad",
                "mean_fog_native_constrained",
                "mean_log_fog_native_constrained",
                "robust_fog_native_constrained",
                "robust_log_fog_native_constrained",
                "log_fog_native_constrained_mad",
                "mean_fog_native_soft",
                "mean_log_fog_native_soft",
                "robust_fog_native_soft",
                "robust_log_fog_native_soft",
                "log_fog_native_soft_mad",
                "mean_fog_solvent_balanced",
                "mean_log_fog_solvent_balanced",
                "robust_fog_solvent_balanced",
                "robust_log_fog_solvent_balanced",
                "log_fog_solvent_balanced_mad",
                "mean_fog_activity_bonus_penalty",
                "mean_log_fog_activity_bonus_penalty",
                "robust_fog_activity_bonus_penalty",
                "robust_log_fog_activity_bonus_penalty",
                "log_fog_activity_bonus_penalty_mad",
                "mean_objective_loglinear_main",
                "robust_objective_loglinear_main",
                "objective_loglinear_main_mad",
                "mean_objective_loglinear_main_exp",
                "robust_objective_loglinear_main_exp",
                "native_feasible_fraction",
                "n_observations",
                "run_ids",
            ]
        )

    rows: List[dict] = []
    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        # Load fog_summary for each run in this round
        dfs: List[pd.DataFrame] = []
        round_has_reference = False
        for rid in run_ids:
            path = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            df["run_id"] = df.get("run_id", rid)
            df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
            gox_t50 = df.get("gox_t50_same_run_min", pd.Series(dtype=float))
            if gox_t50.notna().any() and np.isfinite(gox_t50).any():
                round_has_reference = True
            dfs.append(df)

        if not dfs:
            raise ValueError(f"Round {round_id!r} has no fog_summary CSV under processed_dir for runs {run_ids}.")
        if not round_has_reference:
            raise ValueError(
                f"Round {round_id!r} has no reference polymer {reference_polymer_id!r} in any run "
                "(all runs have no bare reference polymer). "
                "FoG cannot be computed for this round."
            )

        combined = pd.concat(dfs, ignore_index=True)
        # Only valid fog
        fog = pd.to_numeric(combined.get("fog", np.nan), errors="coerce")
        log_fog = pd.to_numeric(combined.get("log_fog", np.nan), errors="coerce")
        valid = np.isfinite(fog) & (fog > 0)
        combined = combined.loc[valid].copy()
        combined["_fog"] = fog[valid].values
        combined["_log_fog"] = log_fog[valid].values
        fog_native = pd.to_numeric(combined.get("fog_native_constrained", np.nan), errors="coerce")
        log_fog_native = pd.to_numeric(combined.get("log_fog_native_constrained", np.nan), errors="coerce")
        combined["_fog_native"] = fog_native
        combined["_log_fog_native"] = log_fog_native
        fog_native_soft = pd.to_numeric(combined.get("fog_native_soft", np.nan), errors="coerce")
        log_fog_native_soft = pd.to_numeric(combined.get("log_fog_native_soft", np.nan), errors="coerce")
        if not np.isfinite(fog_native_soft).any():
            native_rel = pd.to_numeric(combined.get("native_activity_rel_at_0", np.nan), errors="coerce")
            _, fog_native_soft_calc, log_fog_native_soft_calc = _compute_penalized_fog_objective(
                combined["_fog"],
                native_rel,
            )
            fog_native_soft = fog_native_soft_calc
            log_fog_native_soft = log_fog_native_soft_calc
        else:
            miss_log_soft = ~np.isfinite(log_fog_native_soft) & np.isfinite(fog_native_soft) & (fog_native_soft > 0.0)
            if np.any(miss_log_soft):
                log_fog_native_soft = np.where(
                    miss_log_soft,
                    np.log(fog_native_soft),
                    log_fog_native_soft,
                )
        combined["_fog_native_soft"] = fog_native_soft
        combined["_log_fog_native_soft"] = log_fog_native_soft
        combined = _add_solvent_balanced_objective_columns(
            combined,
            reference_polymer_id=reference_polymer_id,
        )
        combined["_fog_solvent_balanced"] = pd.to_numeric(
            combined.get("fog_solvent_balanced", np.nan),
            errors="coerce",
        )
        combined["_log_fog_solvent_balanced"] = pd.to_numeric(
            combined.get("log_fog_solvent_balanced", np.nan),
            errors="coerce",
        )
        combined["_objective_loglinear_main"] = pd.to_numeric(
            combined.get(OBJECTIVE_LOGLINEAR_MAIN_COL, np.nan),
            errors="coerce",
        )
        combined["_objective_loglinear_main_exp"] = pd.to_numeric(
            combined.get(OBJECTIVE_LOGLINEAR_MAIN_EXP_COL, np.nan),
            errors="coerce",
        )
        # Exclude reference-polymer row from round-level ranking output after control-matched metrics are computed.
        combined = combined[
            combined["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) != reference_polymer_norm)
        ].copy()
        native_feasible = pd.to_numeric(combined.get("native_activity_feasible", np.nan), errors="coerce")
        if not np.isfinite(native_feasible).any():
            native_feasible = np.isfinite(fog_native).astype(float)
        combined["_native_feasible"] = np.where(np.isfinite(native_feasible), native_feasible, 0.0)

        for polymer_id, g in combined.groupby("polymer_id", sort=False):
            pid = str(polymer_id).strip()
            mean_fog = float(g["_fog"].mean())
            mean_log_fog = float(g["_log_fog"].mean())
            robust_fog = float(np.nanmedian(g["_fog"]))
            robust_log_fog = float(np.nanmedian(g["_log_fog"]))
            log_fog_mad = float(np.nanmedian(np.abs(g["_log_fog"] - robust_log_fog)))
            g_native = g[np.isfinite(g["_fog_native"]) & (g["_fog_native"] > 0)].copy()
            if not g_native.empty:
                mean_fog_native = float(g_native["_fog_native"].mean())
                mean_log_fog_native = float(g_native["_log_fog_native"].mean())
                robust_fog_native = float(np.nanmedian(g_native["_fog_native"]))
                robust_log_fog_native = float(np.nanmedian(g_native["_log_fog_native"]))
                log_fog_native_mad = float(
                    np.nanmedian(np.abs(g_native["_log_fog_native"] - robust_log_fog_native))
                )
            else:
                mean_fog_native = np.nan
                mean_log_fog_native = np.nan
                robust_fog_native = np.nan
                robust_log_fog_native = np.nan
                log_fog_native_mad = np.nan
            g_soft = g[np.isfinite(g["_fog_native_soft"]) & (g["_fog_native_soft"] > 0)].copy()
            if not g_soft.empty:
                mean_fog_native_soft = float(g_soft["_fog_native_soft"].mean())
                mean_log_fog_native_soft = float(g_soft["_log_fog_native_soft"].mean())
                robust_fog_native_soft = float(np.nanmedian(g_soft["_fog_native_soft"]))
                robust_log_fog_native_soft = float(np.nanmedian(g_soft["_log_fog_native_soft"]))
                log_fog_native_soft_mad = float(
                    np.nanmedian(np.abs(g_soft["_log_fog_native_soft"] - robust_log_fog_native_soft))
                )
            else:
                mean_fog_native_soft = np.nan
                mean_log_fog_native_soft = np.nan
                robust_fog_native_soft = np.nan
                robust_log_fog_native_soft = np.nan
                log_fog_native_soft_mad = np.nan
            g_solvent = g[np.isfinite(g["_fog_solvent_balanced"]) & (g["_fog_solvent_balanced"] > 0)].copy()
            if not g_solvent.empty:
                mean_fog_solvent_balanced = float(g_solvent["_fog_solvent_balanced"].mean())
                mean_log_fog_solvent_balanced = float(g_solvent["_log_fog_solvent_balanced"].mean())
                robust_fog_solvent_balanced = float(np.nanmedian(g_solvent["_fog_solvent_balanced"]))
                robust_log_fog_solvent_balanced = float(np.nanmedian(g_solvent["_log_fog_solvent_balanced"]))
                log_fog_solvent_balanced_mad = float(
                    np.nanmedian(
                        np.abs(g_solvent["_log_fog_solvent_balanced"] - robust_log_fog_solvent_balanced)
                    )
                )
            else:
                mean_fog_solvent_balanced = np.nan
                mean_log_fog_solvent_balanced = np.nan
                robust_fog_solvent_balanced = np.nan
                robust_log_fog_solvent_balanced = np.nan
                log_fog_solvent_balanced_mad = np.nan
            g_loglinear = g[np.isfinite(g["_objective_loglinear_main"])].copy()
            if not g_loglinear.empty:
                mean_objective_loglinear_main = float(g_loglinear["_objective_loglinear_main"].mean())
                robust_objective_loglinear_main = float(np.nanmedian(g_loglinear["_objective_loglinear_main"]))
                objective_loglinear_main_mad = float(
                    np.nanmedian(
                        np.abs(g_loglinear["_objective_loglinear_main"] - robust_objective_loglinear_main)
                    )
                )
            else:
                mean_objective_loglinear_main = np.nan
                robust_objective_loglinear_main = np.nan
                objective_loglinear_main_mad = np.nan
            g_loglinear_exp = g[np.isfinite(g["_objective_loglinear_main_exp"]) & (g["_objective_loglinear_main_exp"] > 0.0)].copy()
            if not g_loglinear_exp.empty:
                mean_objective_loglinear_main_exp = float(g_loglinear_exp["_objective_loglinear_main_exp"].mean())
                robust_objective_loglinear_main_exp = float(np.nanmedian(g_loglinear_exp["_objective_loglinear_main_exp"]))
            else:
                mean_objective_loglinear_main_exp = np.nan
                robust_objective_loglinear_main_exp = np.nan
            native_feasible_fraction = float(np.nanmean(np.clip(g["_native_feasible"], 0.0, 1.0)))
            n_obs = int(len(g))
            run_list = sorted(g["run_id"].astype(str).unique().tolist())
            rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "reference_polymer_id": reference_polymer_id,
                "mean_fog": mean_fog,
                "mean_log_fog": mean_log_fog,
                "robust_fog": robust_fog,
                "robust_log_fog": robust_log_fog,
                "log_fog_mad": log_fog_mad,
                "mean_fog_native_constrained": mean_fog_native,
                "mean_log_fog_native_constrained": mean_log_fog_native,
                "robust_fog_native_constrained": robust_fog_native,
                "robust_log_fog_native_constrained": robust_log_fog_native,
                "log_fog_native_constrained_mad": log_fog_native_mad,
                "mean_fog_native_soft": mean_fog_native_soft,
                "mean_log_fog_native_soft": mean_log_fog_native_soft,
                "robust_fog_native_soft": robust_fog_native_soft,
                "robust_log_fog_native_soft": robust_log_fog_native_soft,
                "log_fog_native_soft_mad": log_fog_native_soft_mad,
                "mean_fog_solvent_balanced": mean_fog_solvent_balanced,
                "mean_log_fog_solvent_balanced": mean_log_fog_solvent_balanced,
                "robust_fog_solvent_balanced": robust_fog_solvent_balanced,
                "robust_log_fog_solvent_balanced": robust_log_fog_solvent_balanced,
                "log_fog_solvent_balanced_mad": log_fog_solvent_balanced_mad,
                "mean_objective_loglinear_main": mean_objective_loglinear_main,
                "robust_objective_loglinear_main": robust_objective_loglinear_main,
                "objective_loglinear_main_mad": objective_loglinear_main_mad,
                "mean_objective_loglinear_main_exp": mean_objective_loglinear_main_exp,
                "robust_objective_loglinear_main_exp": robust_objective_loglinear_main_exp,
                "native_feasible_fraction": native_feasible_fraction,
                "n_observations": n_obs,
                "run_ids": ",".join(run_list),
            })

    return _sync_abs_objective_alias_columns(pd.DataFrame(rows))


def build_round_gox_traceability(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Build per-round GOx traceability: abs_activity (and REA_percent) by heat_min for each run,
    listing all pre-averaged well-level values so which GOx was used for that round is auditable.

    - Reads rates_with_rea.csv for each run in each round; keeps only rows with polymer_id == GOx.
    - Output columns: round_id, run_id, heat_min, plate_id, well, abs_activity, REA_percent.
    - One row per (round_id, run_id, heat_min, well); no averaging (all raw values for that run).
    """
    processed_dir = Path(processed_dir)
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    round_to_runs: Dict[str, List[str]] = {}
    for rid, oid in run_round_map.items():
        rid, oid = str(rid).strip(), str(oid).strip()
        if not oid:
            continue
        round_to_runs.setdefault(oid, []).append(rid)

    rows: List[dict] = []
    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        for run_id in run_ids:
            path = processed_dir / run_id / "fit" / "rates_with_rea.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            df["polymer_id"] = df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
            gox = df[
                df["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
            ].copy()
            if gox.empty:
                continue
            for _, row in gox.iterrows():
                rows.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "reference_polymer_id": reference_polymer_id,
                    "heat_min": pd.to_numeric(row.get("heat_min", np.nan), errors="coerce"),
                    "plate_id": str(row.get("plate_id", "")),
                    "well": str(row.get("well", "")),
                    "abs_activity": pd.to_numeric(row.get("abs_activity", np.nan), errors="coerce"),
                    "REA_percent": pd.to_numeric(row.get("REA_percent", np.nan), errors="coerce"),
                })
    out_cols = [
        "round_id",
        "run_id",
        "reference_polymer_id",
        "heat_min",
        "plate_id",
        "well",
        "abs_activity",
        "REA_percent",
    ]
    return pd.DataFrame(rows, columns=out_cols)


def compute_per_plate_t50_from_rates(
    rates_df: pd.DataFrame,
    run_id: str,
    *,
    t50_definition: str = "y0_half",
) -> pd.DataFrame:
    """
    Compute t50 per (run_id, plate_id, polymer_id) from rates_with_rea-style DataFrame.

    - Uses only rows with status == 'ok' and finite REA_percent.
    - Aggregates by (plate_id, polymer_id, heat_min) -> mean REA_percent, then fits t50 per (plate_id, polymer_id).
    - Prefers exp/exp_plateau t50; fallback t50_linear.
    - t50_definition: "y0_half" or "rea50".
    - Output: run_id, plate_id, polymer_id, t50_min, n_points, fit_model, t50_definition.
    """
    from gox_plate_pipeline.polymer_timeseries import (
        T50_DEFINITION_REA50,
        fit_exponential_decay,
        normalize_t50_definition,
        t50_linear,
        value_at_time_linear,
    )  # avoid circular import

    t50_definition = normalize_t50_definition(t50_definition)

    df = rates_df.copy()
    df["polymer_id"] = df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
    df["plate_id"] = df.get("plate_id", pd.Series(dtype=str)).astype(str)
    df["heat_min"] = pd.to_numeric(df.get("heat_min", np.nan), errors="coerce")
    df["REA_percent"] = pd.to_numeric(df.get("REA_percent", np.nan), errors="coerce")
    df["abs_activity"] = pd.to_numeric(df.get("abs_activity", np.nan), errors="coerce")
    df["status"] = df.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    ok = df[(df["status"] == "ok") & np.isfinite(df["REA_percent"]) & (df["REA_percent"] > 0)].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "plate_id",
                "polymer_id",
                "t50_min",
                "n_points",
                "fit_model",
                "t50_definition",
                "abs_activity_at_0",
            ]
        )

    # Aggregate by (plate_id, polymer_id, heat_min) -> mean REA/absolute activity
    agg = ok.groupby(["plate_id", "polymer_id", "heat_min"], dropna=False).agg(
        REA_percent=("REA_percent", "mean"),
        abs_activity=("abs_activity", "mean"),
    ).reset_index()

    rows: List[dict] = []
    for (plate_id, polymer_id), g in agg.groupby(["plate_id", "polymer_id"], dropna=False):
        g = g.sort_values("heat_min")
        t = g["heat_min"].to_numpy(dtype=float)
        y = g["REA_percent"].to_numpy(dtype=float)
        aa = g["abs_activity"].to_numpy(dtype=float)
        n_pts = int(len(g))
        if t.size < 3 or np.unique(t).size < 2:
            continue
        abs_activity_at_0 = value_at_time_linear(t, aa, at_time_min=0.0)
        abs_activity_at_0 = (
            float(abs_activity_at_0)
            if abs_activity_at_0 is not None and np.isfinite(float(abs_activity_at_0))
            else np.nan
        )
        y0_init = float(np.nanmax(y)) if np.any(np.isfinite(y)) else None
        fit = fit_exponential_decay(
            t,
            y,
            y0=y0_init,
            fixed_y0=100.0,
            min_points=3,
            t50_definition=t50_definition,
        )
        t50_exp = float(fit.t50) if (fit is not None and fit.t50 is not None and np.isfinite(fit.t50)) else np.nan
        y0_lin = 100.0
        t50_lin_val = t50_linear(
            t,
            y,
            y0=y0_lin,
            target_frac=0.5,
            target_value=(50.0 if t50_definition == T50_DEFINITION_REA50 else None),
        )
        t50_lin_f = float(t50_lin_val) if t50_lin_val is not None and np.isfinite(t50_lin_val) else np.nan
        t50_min = t50_exp if np.isfinite(t50_exp) and t50_exp > 0 else t50_lin_f
        if not np.isfinite(t50_min) or t50_min <= 0:
            continue
        fit_model = "exp" if (fit is not None and fit.model == "exp") else ("exp_plateau" if (fit is not None and fit.model == "exp_plateau") else "linear_only")
        rows.append({
            "run_id": run_id,
            "plate_id": str(plate_id),
            "polymer_id": str(polymer_id).strip(),
            "t50_min": t50_min,
            "n_points": n_pts,
            "fit_model": fit_model,
            "t50_definition": t50_definition,
            "abs_activity_at_0": abs_activity_at_0,
        })
    return pd.DataFrame(rows)


def build_fog_plate_aware(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    t50_definition: str = "y0_half",
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    polymer_solvent_path: Optional[Path] = None,
    native_activity_min_rel: float = NATIVE_ACTIVITY_MIN_REL_DEFAULT,
    exclude_outlier_gox: bool = False,
    gox_outlier_low_threshold: float = 0.33,
    gox_outlier_high_threshold: float = 3.0,
    gox_guard_same_plate: bool = True,
    gox_guard_low_threshold: Optional[float] = None,
    gox_guard_high_threshold: Optional[float] = None,
    gox_round_fallback_stat: str = "median",
    gox_round_trimmed_mean_proportion: float = 0.1,
    native_reference_mode: str = "same_plate_then_round",
    ref_agg_method: str = "median",
    ref_trimmed_mean_proportion: float = 0.1,
    reference_qc_mad_rel_threshold: float = 0.25,
    reference_qc_min_abs0: float = 0.0,
    reference_qc_exclude: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FogWarningInfo]:
    """
    Build FoG with denominator rule: same plate GOx → same round GOx.

    - Per-plate t50 is computed from rates_with_rea per run (REA vs heat_min per plate).
    - t50_definition controls how per-plate t50 is computed ("y0_half" or "rea50").
    - For each (run_id, plate_id, polymer_id): gox_t50 = GOx t50 from same (run_id, plate_id) if present,
      else mean of GOx t50 across all (run_id, plate_id) in that round. FoG = t50_polymer / gox_t50.
    - Round average GOx t50 calculation: All (run, plate) GOx t50 values are averaged with equal weight
      (run-level weighting is not applied; each plate contributes equally).
    - Outlier detection: GOx t50 values that are < median * gox_outlier_low_threshold or
      > median * gox_outlier_high_threshold are detected and warned. If exclude_outlier_gox=True,
      outliers are excluded from round average GOx t50 calculation.
    - Denominator guard: when gox_guard_same_plate=True, even if same-plate GOx exists, values outside
      [round_median * gox_guard_low_threshold, round_median * gox_guard_high_threshold] are treated as
      unstable and fall back to round representative GOx t50.
    - Round fallback denominator: median (default), mean, or trimmed_mean of round GOx t50 after optional outlier exclusion.
    - If a round has no GOx in any (run, plate), raises ValueError.
    - Returns (per_row_df, round_averaged_df, gox_traceability_df, warning_info).
              per_row_df columns: run_id, plate_id, polymer_id, t50_min, gox_t50_used_min, denominator_source, fog, log_fog,
                          abs_activity_at_0, gox_abs_activity_at_0_ref, native_activity_rel_at_0,
                          native_activity_feasible, fog_native_constrained, log_fog_native_constrained,
                          native_activity_soft_penalty, fog_native_soft, log_fog_native_soft,
                          fog_solvent_balanced, log_fog_solvent_balanced,
                          native_0, U_*, t_theta, t_theta_censor_flag, reference_qc_*.
      warning_info: FogWarningInfo object containing warning details.

    Args:
        exclude_outlier_gox: If True, exclude outlier GOx t50 values from round average calculation.
        gox_outlier_low_threshold: Lower threshold multiplier for outlier detection (default: 0.33).
        gox_outlier_high_threshold: Upper threshold multiplier for outlier detection (default: 3.0).
        gox_guard_same_plate: If True, guard same_plate denominator and fallback to same_round when extreme.
        gox_guard_low_threshold: Lower multiplier for same_plate guard. If None, uses gox_outlier_low_threshold.
        gox_guard_high_threshold: Upper multiplier for same_plate guard. If None, uses gox_outlier_high_threshold.
        gox_round_fallback_stat: Round representative denominator ("median", "mean", or "trimmed_mean").
        gox_round_trimmed_mean_proportion: Proportion to trim from each tail for "trimmed_mean" (default 0.1).
        native_reference_mode: Native/U(t) reference mode:
            "same_plate_then_round" (legacy) or "same_run_then_round" (policy v2 primary).
        ref_agg_method: Aggregation for same-round or same-run reference abs0: "median" (default),
            "trimmed_mean", or "mean".
        ref_trimmed_mean_proportion: Trim proportion for ref_agg_method="trimmed_mean".
        reference_qc_mad_rel_threshold: Relative MAD threshold for run-level reference abs0 QC fail.
        reference_qc_min_abs0: Absolute minimum for run-level reference abs0 median QC.
        reference_qc_exclude: If True, rows from QC-failed runs become infeasible for constrained objective.
    """
    from gox_plate_pipeline.polymer_timeseries import normalize_t50_definition  # avoid circular import

    t50_definition = normalize_t50_definition(t50_definition)
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    native_activity_min_rel = float(native_activity_min_rel)
    if not np.isfinite(native_activity_min_rel) or native_activity_min_rel <= 0.0:
        raise ValueError(
            f"native_activity_min_rel must be a positive finite number, got {native_activity_min_rel!r}"
        )
    processed_dir = Path(processed_dir)
    warning_info = FogWarningInfo()
    round_to_runs: Dict[str, List[str]] = {}
    for rid, oid in run_round_map.items():
        rid, oid = str(rid).strip(), str(oid).strip()
        if not oid:
            continue
        round_to_runs.setdefault(oid, []).append(rid)

    if not round_to_runs:
        empty = pd.DataFrame(columns=[
            "round_id",
            "run_id",
            "plate_id",
            "polymer_id",
            "reference_polymer_id",
            "t50_min",
            "t50_definition",
            "gox_t50_used_min",
            "denominator_source",
            "fog",
            "log_fog",
            "abs_activity_at_0",
            "gox_abs_activity_at_0_ref",
            "native_activity_rel_at_0",
            "native_activity_min_rel_threshold",
            "native_activity_feasible",
            "fog_native_constrained",
            "log_fog_native_constrained",
            "native_activity_soft_penalty",
            "fog_native_soft",
            "log_fog_native_soft",
            "stock_solvent_group",
            "solvent_group",
            "solvent_control_polymer_id",
            "solvent_control_abs_activity_at_0",
            "solvent_control_t50_min",
            "solvent_control_fog",
            "abs0_vs_solvent_control",
            "fog_vs_solvent_control",
            "abs_activity_down_penalty",
            "abs_activity_up_bonus",
            "abs_activity_balance_factor",
            "solvent_activity_down_penalty",
            "solvent_activity_up_bonus",
            "solvent_activity_balance_factor",
            ABS_ACTIVITY_OBJECTIVE_COL,
            ABS_ACTIVITY_OBJECTIVE_LOG_COL,
            OBJECTIVE_LOGLINEAR_MAIN_COL,
            OBJECTIVE_LOGLINEAR_MAIN_EXP_COL,
            "fog_solvent_balanced",
            "log_fog_solvent_balanced",
            "fog_constraint_reason",
            "native_0",
            "native_0_reference_source",
            "t_theta",
            "t_theta_censor_flag",
            "censor_flag",
            "reference_qc_fail",
            "reference_qc_reason",
            "reference_qc_ref_abs0_median",
            "reference_qc_ref_abs0_mad",
            "reference_qc_ref_abs0_rel_mad",
        ])
        return empty, pd.DataFrame(columns=[
            "round_id",
            "polymer_id",
            "reference_polymer_id",
            "mean_fog",
            "mean_log_fog",
            "robust_fog",
            "robust_log_fog",
            "log_fog_mad",
            "mean_fog_native_constrained",
            "mean_log_fog_native_constrained",
            "robust_fog_native_constrained",
            "robust_log_fog_native_constrained",
            "log_fog_native_constrained_mad",
            "mean_fog_native_soft",
            "mean_log_fog_native_soft",
            "robust_fog_native_soft",
            "robust_log_fog_native_soft",
            "log_fog_native_soft_mad",
            "mean_fog_activity_bonus_penalty",
            "mean_log_fog_activity_bonus_penalty",
            "robust_fog_activity_bonus_penalty",
            "robust_log_fog_activity_bonus_penalty",
            "log_fog_activity_bonus_penalty_mad",
            "mean_objective_loglinear_main",
            "robust_objective_loglinear_main",
            "objective_loglinear_main_mad",
            "mean_objective_loglinear_main_exp",
            "robust_objective_loglinear_main_exp",
            "mean_fog_solvent_balanced",
            "mean_log_fog_solvent_balanced",
            "robust_fog_solvent_balanced",
            "robust_log_fog_solvent_balanced",
            "log_fog_solvent_balanced_mad",
            "native_feasible_fraction",
            "n_observations",
            "run_ids",
        ]), empty, FogWarningInfo()

    guard_low_mult = float(gox_guard_low_threshold if gox_guard_low_threshold is not None else gox_outlier_low_threshold)
    guard_high_mult = float(gox_guard_high_threshold if gox_guard_high_threshold is not None else gox_outlier_high_threshold)
    if guard_low_mult <= 0 or guard_high_mult <= 0:
        raise ValueError("gox_guard thresholds must be positive multipliers.")
    fallback_stat = str(gox_round_fallback_stat).strip().lower()
    if fallback_stat not in {"median", "mean", "trimmed_mean"}:
        raise ValueError(
            f"gox_round_fallback_stat must be 'median', 'mean', or 'trimmed_mean', got {gox_round_fallback_stat!r}"
        )
    trimmed_proportion = float(gox_round_trimmed_mean_proportion)
    if fallback_stat == "trimmed_mean" and not (0.0 <= trimmed_proportion < 0.5):
        raise ValueError("gox_round_trimmed_mean_proportion must be in [0, 0.5).")
    native_ref_mode = str(native_reference_mode).strip().lower()
    if native_ref_mode not in {"same_plate_then_round", "same_run_then_round"}:
        raise ValueError(
            "native_reference_mode must be 'same_plate_then_round' or 'same_run_then_round', "
            f"got {native_reference_mode!r}"
        )
    ref_method = str(ref_agg_method).strip().lower()
    if ref_method not in {"median", "mean", "trimmed_mean"}:
        raise ValueError(
            "ref_agg_method must be 'median', 'mean', or 'trimmed_mean', "
            f"got {ref_agg_method!r}"
        )
    ref_trim_prop = float(ref_trimmed_mean_proportion)
    if ref_method == "trimmed_mean" and not (0.0 <= ref_trim_prop < 0.5):
        raise ValueError("ref_trimmed_mean_proportion must be in [0, 0.5).")
    qc_rel_mad_thr = float(reference_qc_mad_rel_threshold)
    qc_min_abs0 = float(reference_qc_min_abs0)
    if not np.isfinite(qc_rel_mad_thr) or qc_rel_mad_thr < 0.0:
        raise ValueError("reference_qc_mad_rel_threshold must be a finite non-negative number.")
    if not np.isfinite(qc_min_abs0) or qc_min_abs0 < 0.0:
        raise ValueError("reference_qc_min_abs0 must be a finite non-negative number.")

    # Collect per-plate t50 for all runs in all rounds
    run_plate_t50: Dict[str, pd.DataFrame] = {}  # run_id -> DataFrame run_id, plate_id, polymer_id, t50_min
    run_abs_profiles: Dict[str, pd.DataFrame] = {}  # run_id -> DataFrame plate_id, polymer_id, heat_min, abs_activity
    missing_rates_files: List[tuple[str, str]] = []  # (round_id, run_id) pairs
    heat_grid_vals: set[float] = set()
    for round_id, run_ids in round_to_runs.items():
        for run_id in run_ids:
            if run_id in run_plate_t50:
                continue
            path = processed_dir / run_id / "fit" / "rates_with_rea.csv"
            if not path.is_file():
                missing_rates_files.append((round_id, run_id))
                warning_info.missing_rates_files.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "expected_path": str(path),
                })
                import warnings
                import sys
                # Print a more visible warning message
                print("\n" + "=" * 80, file=sys.stderr)
                print("⚠️  警告: rates_with_rea.csv が見つかりません", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print(f"Round: {round_id}", file=sys.stderr)
                print(f"Run ID: {run_id}", file=sys.stderr)
                print(f"期待されるパス: {path}", file=sys.stderr)
                print("→ このrunはFoG計算からスキップされます。", file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)
                # Also issue standard warning for programmatic access
                warnings.warn(
                    f"Round {round_id!r}, run {run_id!r}: rates_with_rea.csv not found at {path}. "
                    "This run will be skipped in FoG calculation.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            df = pd.read_csv(path)
            run_plate_t50[run_id] = compute_per_plate_t50_from_rates(
                df,
                run_id,
                t50_definition=t50_definition,
            )
            abs_df = df.copy()
            abs_df["plate_id"] = abs_df.get("plate_id", pd.Series(dtype=str)).astype(str)
            abs_df["polymer_id"] = abs_df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
            abs_df["heat_min"] = pd.to_numeric(abs_df.get("heat_min", np.nan), errors="coerce")
            abs_df["abs_activity"] = pd.to_numeric(abs_df.get("abs_activity", np.nan), errors="coerce")
            abs_df["status"] = abs_df.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
            abs_df = abs_df[
                (abs_df["status"] == "ok")
                & np.isfinite(abs_df["heat_min"])
                & np.isfinite(abs_df["abs_activity"])
                & (abs_df["abs_activity"] > 0.0)
            ].copy()
            if abs_df.empty:
                run_abs_profiles[run_id] = pd.DataFrame(columns=["plate_id", "polymer_id", "heat_min", "abs_activity"])
            else:
                abs_agg = (
                    abs_df.groupby(["plate_id", "polymer_id", "heat_min"], dropna=False)
                    .agg(abs_activity=("abs_activity", "mean"))
                    .reset_index()
                )
                run_abs_profiles[run_id] = abs_agg
                for hh in abs_agg["heat_min"].tolist():
                    hv = pd.to_numeric(hh, errors="coerce")
                    if np.isfinite(hv):
                        heat_grid_vals.add(float(hv))

    heat_grid = sorted([float(h) for h in heat_grid_vals if np.isfinite(float(h))])
    per_row_rows: List[dict] = []
    round_gox_t50: Dict[str, List[float]] = {}  # round_id -> list of GOx t50 values (all run,plate in that round)

    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        gox_in_round: List[float] = []
        gox_plate_info: List[tuple[str, str, float, float]] = []  # (run_id, plate_id, t50, abs0)
        for run_id in run_ids:
            pt50 = run_plate_t50.get(run_id)
            if pt50 is None or pt50.empty:
                continue
            gox_plate = pt50[
                pt50["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
            ]
            for _, r in gox_plate.iterrows():
                plate_id = str(r.get("plate_id", ""))
                t = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                abs0 = pd.to_numeric(r.get("abs_activity_at_0", np.nan), errors="coerce")
                if np.isfinite(t) and t > 0:
                    gox_in_round.append(float(t))
                    gox_plate_info.append((run_id, plate_id, float(t), float(abs0) if np.isfinite(abs0) else np.nan))

        if not gox_in_round:
            raise ValueError(
                f"Round {round_id!r} has no reference polymer {reference_polymer_id!r} t50 in any (run, plate). "
                "FoG with same_plate/same_round denominator cannot be computed."
            )

        # Outlier detection: median-based
        import warnings
        median_gox = float(np.median(gox_in_round))
        low_threshold = median_gox * gox_outlier_low_threshold
        high_threshold = median_gox * gox_outlier_high_threshold
        outliers: List[tuple[str, str, float]] = []
        for run_id, plate_id, t50_val, _abs0 in gox_plate_info:
            if t50_val < low_threshold or t50_val > high_threshold:
                outliers.append((run_id, plate_id, t50_val))

        if outliers:
            outlier_details = ", ".join([f"{run_id}/{plate_id}={t50_val:.3f}min" for run_id, plate_id, t50_val in outliers])
            # Collect warning info
            warning_info.outlier_gox.append({
                "round_id": round_id,
                "median_gox_t50": float(median_gox),
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
                "outliers": [{"run_id": run_id, "plate_id": plate_id, "gox_t50": float(t50_val)} for run_id, plate_id, t50_val in outliers],
                "excluded": exclude_outlier_gox,
            })
            import sys
            # Print a more visible warning message
            print("\n" + "=" * 80, file=sys.stderr)
            print(f"⚠️  警告: 異常{reference_polymer_id} t50値が検出されました", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"Round: {round_id}", file=sys.stderr)
            print(f"{reference_polymer_id} t50 中央値: {median_gox:.3f} min", file=sys.stderr)
            print(f"異常値の閾値: [{low_threshold:.3f}, {high_threshold:.3f}] min", file=sys.stderr)
            print(f"検出された異常値: {outlier_details}", file=sys.stderr)
            if exclude_outlier_gox:
                print(f"→ 異常値はround平均{reference_polymer_id} t50の計算から除外されます。", file=sys.stderr)
            else:
                print(f"→ 異常値はround平均{reference_polymer_id} t50の計算に含まれます。", file=sys.stderr)
                print("→ 除外する場合は、--exclude_outlier_gox フラグを使用してください。", file=sys.stderr)
            print("=" * 80 + "\n", file=sys.stderr)
            # Also issue standard warning for programmatic access
            warnings.warn(
                f"Round {round_id!r} has outlier {reference_polymer_id} t50 values (median={median_gox:.3f}min, "
                f"thresholds=[{low_threshold:.3f}, {high_threshold:.3f}]min): {outlier_details}. "
                f"{'Excluding outliers from round average.' if exclude_outlier_gox else 'Outliers are included in round average. Use --exclude_outlier_gox to exclude.'}",
                UserWarning,
                stacklevel=2,
            )
            if exclude_outlier_gox:
                outlier_t50s = {t50_val for _, _, t50_val in outliers}
                gox_in_round = [t for t in gox_in_round if t not in outlier_t50s]
                if not gox_in_round:
                    raise ValueError(
                        f"Round {round_id!r} has no reference-polymer t50 remaining after excluding outliers. "
                        "Cannot compute FoG."
                    )

        round_gox_t50[round_id] = gox_in_round
        outlier_keys = (
            {(rid, pid, float(t50_val)) for rid, pid, t50_val in outliers}
            if exclude_outlier_gox
            else set()
        )
        valid_gox_plate_info = [
            (rid, pid, t50_val, abs0)
            for rid, pid, t50_val, abs0 in gox_plate_info
            if (not outlier_keys) or ((rid, pid, float(t50_val)) not in outlier_keys)
        ]
        gox_abs0_round_vals = [
            float(abs0)
            for _rid, _pid, _t50_val, abs0 in valid_gox_plate_info
            if np.isfinite(abs0) and float(abs0) > 0.0
        ]
        round_gox_abs0_fallback = _aggregate_reference_values(
            gox_abs0_round_vals,
            method=ref_method,
            trimmed_mean_proportion=ref_trim_prop,
        )

        # same-run reference abs0 (policy-v2 primary reference)
        run_gox_abs0_by_run: Dict[str, float] = {}
        run_ref_abs0_vals_map: Dict[str, List[float]] = {}
        for rid, _pid, _t50_val, abs0 in valid_gox_plate_info:
            if np.isfinite(abs0) and float(abs0) > 0.0:
                run_ref_abs0_vals_map.setdefault(str(rid), []).append(float(abs0))
        for rid, vals in run_ref_abs0_vals_map.items():
            run_gox_abs0_by_run[rid] = _aggregate_reference_values(
                vals,
                method=ref_method,
                trimmed_mean_proportion=ref_trim_prop,
            )

        # Run-level QC on same-run reference abs0 spread.
        run_ref_qc: Dict[str, Dict[str, Any]] = {}
        for rid in run_ids:
            vals = run_ref_abs0_vals_map.get(rid, [])
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0.0)]
            n_ref = int(arr.size)
            if n_ref == 0:
                run_ref_qc[rid] = {
                    "fail": True,
                    "reason": "missing_same_run_reference_abs0",
                    "median": np.nan,
                    "mad": np.nan,
                    "rel_mad": np.nan,
                    "n_ref": 0,
                }
                continue
            med = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - med)))
            rel_mad = float(mad / med) if np.isfinite(med) and med > 0 else np.nan
            reason_parts: List[str] = []
            fail = False
            if np.isfinite(med) and med < qc_min_abs0:
                fail = True
                reason_parts.append("ref_abs0_median_below_min")
            if n_ref >= 2 and np.isfinite(rel_mad) and rel_mad > qc_rel_mad_thr:
                fail = True
                reason_parts.append("ref_abs0_rel_mad_above_threshold")
            run_ref_qc[rid] = {
                "fail": bool(fail),
                "reason": ";".join(reason_parts),
                "median": med,
                "mad": mad,
                "rel_mad": rel_mad,
                "n_ref": n_ref,
            }
            if fail:
                warning_info.reference_qc_fail_runs.append(
                    {
                        "round_id": round_id,
                        "run_id": rid,
                        "n_reference_wells": n_ref,
                        "ref_abs0_median": med,
                        "ref_abs0_mad": mad,
                        "ref_abs0_rel_mad": rel_mad,
                        "reason": ";".join(reason_parts),
                    }
                )

        # Round representative GOx t50 (used for same_round fallback and same_plate guard fallback).
        # Values are taken after optional outlier exclusion above.
        round_gox_arr = np.asarray(gox_in_round, dtype=float)
        round_gox_median = float(np.nanmedian(round_gox_arr))
        round_gox_mean = float(np.nanmean(round_gox_arr))
        if fallback_stat == "median":
            round_gox_fallback = round_gox_median
        elif fallback_stat == "trimmed_mean":
            n_gox = len(round_gox_arr)
            if n_gox >= 3:
                round_gox_fallback = float(
                    trim_mean(round_gox_arr, proportiontocut=min(trimmed_proportion, (n_gox - 1) / (2 * n_gox)))
                )
            else:
                round_gox_fallback = round_gox_median
        else:
            round_gox_fallback = round_gox_mean
        guard_low_threshold = round_gox_median * guard_low_mult
        guard_high_threshold = round_gox_median * guard_high_mult
        guard_logged_plates: set[tuple[str, str, str]] = set()

        for run_id in run_ids:
            pt50 = run_plate_t50.get(run_id)
            if pt50 is None or pt50.empty:
                continue
            run_abs_df = run_abs_profiles.get(run_id)
            if run_abs_df is None:
                run_abs_df = pd.DataFrame(columns=["plate_id", "polymer_id", "heat_min", "abs_activity"])
            gox_by_plate: Dict[str, float] = {}
            gox_abs0_by_plate: Dict[str, float] = {}
            gox_plate = pt50[
                pt50["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
            ]
            for _, r in gox_plate.iterrows():
                plate_id = str(r.get("plate_id", ""))
                t = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                abs0 = pd.to_numeric(r.get("abs_activity_at_0", np.nan), errors="coerce")
                if np.isfinite(t) and t > 0:
                    # Check for duplicate GOx t50 for the same plate (should not happen)
                    if plate_id in gox_by_plate:
                        existing_t50 = gox_by_plate[plate_id]
                        raise ValueError(
                            f"Run {run_id!r}, plate {plate_id!r} has multiple reference-polymer t50 values: "
                            f"{existing_t50:.3f}min and {t:.3f}min. "
                            "Each plate should have exactly one reference-polymer t50 value."
                        )
                    gox_by_plate[plate_id] = float(t)
                    if np.isfinite(abs0) and abs0 > 0:
                        gox_abs0_by_plate[plate_id] = float(abs0)
            run_abs0_ref = run_gox_abs0_by_run.get(run_id, np.nan)
            run_qc = run_ref_qc.get(
                run_id,
                {
                    "fail": True,
                    "reason": "missing_same_run_reference_abs0",
                    "median": np.nan,
                    "mad": np.nan,
                    "rel_mad": np.nan,
                    "n_ref": 0,
                },
            )

            polymers = pt50[
                pt50["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) != reference_polymer_norm)
            ]
            for _, r in polymers.iterrows():
                plate_id = str(r.get("plate_id", ""))
                polymer_id = str(r.get("polymer_id", "")).strip()
                t50_min = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                abs_activity_at_0 = pd.to_numeric(r.get("abs_activity_at_0", np.nan), errors="coerce")
                if not np.isfinite(t50_min) or t50_min <= 0:
                    continue
                gox_t50 = gox_by_plate.get(plate_id)
                if gox_t50 is not None and gox_t50 > 0:
                    use_same_plate = True
                    if gox_guard_same_plate:
                        if (gox_t50 < guard_low_threshold) or (gox_t50 > guard_high_threshold):
                            use_same_plate = False
                            key = (round_id, run_id, plate_id)
                            if key not in guard_logged_plates:
                                warning_info.guarded_same_plate.append({
                                    "round_id": round_id,
                                    "run_id": run_id,
                                    "plate_id": plate_id,
                                    "same_plate_gox_t50_min": float(gox_t50),
                                    "fallback_gox_t50_min": float(round_gox_fallback),
                                    "guard_low_threshold_min": float(guard_low_threshold),
                                    "guard_high_threshold_min": float(guard_high_threshold),
                                })
                                guard_logged_plates.add(key)
                    if use_same_plate:
                        denominator_source = "same_plate"
                    else:
                        gox_t50 = round_gox_fallback
                        denominator_source = "same_round"
                else:
                    gox_t50 = round_gox_fallback
                    denominator_source = "same_round"
                fog = t50_min / gox_t50
                log_fog = np.log(fog) if fog > 0 else np.nan
                same_plate_abs0 = gox_abs0_by_plate.get(plate_id, np.nan)
                if native_ref_mode == "same_run_then_round":
                    if np.isfinite(run_abs0_ref) and float(run_abs0_ref) > 0.0:
                        gox_abs0_ref = float(run_abs0_ref)
                        native_ref_source = "same_run"
                    else:
                        gox_abs0_ref = float(round_gox_abs0_fallback) if np.isfinite(round_gox_abs0_fallback) and float(round_gox_abs0_fallback) > 0.0 else np.nan
                        native_ref_source = "same_round"
                else:
                    if np.isfinite(same_plate_abs0) and float(same_plate_abs0) > 0.0:
                        gox_abs0_ref = float(same_plate_abs0)
                        native_ref_source = "same_plate"
                    else:
                        gox_abs0_ref = float(round_gox_abs0_fallback) if np.isfinite(round_gox_abs0_fallback) and float(round_gox_abs0_fallback) > 0.0 else np.nan
                        native_ref_source = "same_round"
                native_rel = (
                    float(abs_activity_at_0) / float(gox_abs0_ref)
                    if np.isfinite(abs_activity_at_0) and np.isfinite(gox_abs0_ref) and gox_abs0_ref > 0
                    else np.nan
                )
                native_feasible = bool(np.isfinite(native_rel) and native_rel >= float(native_activity_min_rel))
                qc_fail = bool(run_qc.get("fail", False))
                fog_native = float(fog) if native_feasible else np.nan
                log_fog_native = np.log(fog_native) if np.isfinite(fog_native) and fog_native > 0 else np.nan
                soft_penalty_arr, fog_native_soft_arr, log_fog_native_soft_arr = _compute_penalized_fog_objective(
                    [fog],
                    [native_rel],
                )
                native_soft_penalty = float(soft_penalty_arr[0]) if np.isfinite(soft_penalty_arr[0]) else np.nan
                fog_native_soft = float(fog_native_soft_arr[0]) if np.isfinite(fog_native_soft_arr[0]) else np.nan
                log_fog_native_soft = (
                    float(log_fog_native_soft_arr[0]) if np.isfinite(log_fog_native_soft_arr[0]) else np.nan
                )
                if not np.isfinite(abs_activity_at_0):
                    constraint_reason = "missing_abs_activity_at_0"
                elif not np.isfinite(gox_abs0_ref):
                    constraint_reason = "missing_gox_abs_activity_at_0"
                elif np.isfinite(native_rel) and native_rel < float(native_activity_min_rel):
                    constraint_reason = "native_activity_below_threshold"
                else:
                    constraint_reason = ""
                if reference_qc_exclude and qc_fail:
                    native_feasible = False
                    fog_native = np.nan
                    log_fog_native = np.nan
                    fog_native_soft = np.nan
                    log_fog_native_soft = np.nan
                    constraint_reason = (
                        f"{constraint_reason};reference_qc_fail"
                        if constraint_reason
                        else "reference_qc_fail"
                    )

                # U(t) series uses reference abs activity at t=0 (same-run first in policy v2 mode).
                u_cols: Dict[str, float] = {}
                t_theta = np.nan
                t_theta_flag = "missing_profile"
                if np.isfinite(gox_abs0_ref) and float(gox_abs0_ref) > 0.0 and (run_abs_df is not None) and (not run_abs_df.empty):
                    prof = run_abs_df[
                        (run_abs_df["plate_id"].astype(str) == str(plate_id))
                        & (run_abs_df["polymer_id"].astype(str).str.strip() == polymer_id)
                    ].copy()
                    if not prof.empty:
                        prof["heat_min"] = pd.to_numeric(prof["heat_min"], errors="coerce")
                        prof["abs_activity"] = pd.to_numeric(prof["abs_activity"], errors="coerce")
                        prof = prof[np.isfinite(prof["heat_min"]) & np.isfinite(prof["abs_activity"])].copy()
                        if not prof.empty:
                            prof = prof.sort_values("heat_min")
                            u_vals = prof["abs_activity"].to_numpy(dtype=float) / float(gox_abs0_ref)
                            t_vals = prof["heat_min"].to_numpy(dtype=float)
                            t_theta, t_theta_flag = _compute_t_theta_from_series(
                                t_vals,
                                u_vals,
                                theta=float(native_activity_min_rel),
                            )
                            for hh, uv in zip(t_vals, u_vals):
                                u_cols[_u_col_name(float(hh))] = float(uv) if np.isfinite(uv) else np.nan
                elif not (np.isfinite(gox_abs0_ref) and float(gox_abs0_ref) > 0.0):
                    t_theta_flag = "missing_reference_abs0"
                for hh in heat_grid:
                    col = _u_col_name(hh)
                    if col not in u_cols:
                        u_cols[col] = np.nan
                per_row_rows.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "plate_id": plate_id,
                    "polymer_id": polymer_id,
                    "reference_polymer_id": reference_polymer_id,
                    "t50_min": t50_min,
                    "t50_definition": str(r.get("t50_definition", t50_definition)),
                    "gox_t50_used_min": gox_t50,
                    "denominator_source": denominator_source,
                    "fog": fog,
                    "log_fog": log_fog,
                    "abs_activity_at_0": float(abs_activity_at_0) if np.isfinite(abs_activity_at_0) else np.nan,
                    "gox_abs_activity_at_0_ref": gox_abs0_ref,
                    "native_activity_rel_at_0": native_rel,
                    "native_activity_min_rel_threshold": float(native_activity_min_rel),
                    "native_activity_feasible": int(native_feasible),
                    "fog_native_constrained": fog_native,
                    "log_fog_native_constrained": log_fog_native,
                    "native_activity_soft_penalty": native_soft_penalty,
                    "fog_native_soft": fog_native_soft,
                    "log_fog_native_soft": log_fog_native_soft,
                    "fog_constraint_reason": constraint_reason,
                    "native_0": native_rel,
                    "native_0_reference_source": native_ref_source,
                    "t_theta": float(t_theta) if np.isfinite(t_theta) else np.nan,
                    "t_theta_censor_flag": str(t_theta_flag),
                    "censor_flag": str(t_theta_flag),
                    "reference_qc_fail": int(qc_fail),
                    "reference_qc_reason": str(run_qc.get("reason", "")),
                    "reference_qc_ref_abs0_median": float(run_qc.get("median", np.nan)) if np.isfinite(pd.to_numeric(run_qc.get("median", np.nan), errors="coerce")) else np.nan,
                    "reference_qc_ref_abs0_mad": float(run_qc.get("mad", np.nan)) if np.isfinite(pd.to_numeric(run_qc.get("mad", np.nan), errors="coerce")) else np.nan,
                    "reference_qc_ref_abs0_rel_mad": float(run_qc.get("rel_mad", np.nan)) if np.isfinite(pd.to_numeric(run_qc.get("rel_mad", np.nan), errors="coerce")) else np.nan,
                    **u_cols,
                })

    per_row_df = pd.DataFrame(per_row_rows)
    if not per_row_df.empty:
        stock_solvent_map, control_solvent_map = _load_polymer_solvent_maps(polymer_solvent_path)
        per_row_df = _apply_polymer_solvent_maps(
            per_row_df,
            stock_map=stock_solvent_map,
            control_map=control_solvent_map,
        )
        per_row_df = _add_solvent_balanced_objective_columns(
            per_row_df,
            reference_polymer_id=reference_polymer_id,
        )

    # Round-averaged: by (round_id, polymer_id)
    round_av_rows: List[dict] = []
    for round_id in sorted(round_to_runs.keys()):
        sub = per_row_df[per_row_df["round_id"] == round_id].copy() if not per_row_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        fog = pd.to_numeric(sub["fog"], errors="coerce")
        log_fog = pd.to_numeric(sub["log_fog"], errors="coerce")
        valid = np.isfinite(fog) & (fog > 0)
        sub = sub.loc[valid].copy()
        for polymer_id, g in sub.groupby("polymer_id", sort=False):
            pid = str(polymer_id).strip()
            robust_fog = float(np.nanmedian(g["fog"]))
            robust_log_fog = float(np.nanmedian(g["log_fog"]))
            log_fog_mad = float(np.nanmedian(np.abs(g["log_fog"] - robust_log_fog)))
            fog_native = pd.to_numeric(g.get("fog_native_constrained", np.nan), errors="coerce")
            log_fog_native = pd.to_numeric(g.get("log_fog_native_constrained", np.nan), errors="coerce")
            fog_native_soft = pd.to_numeric(g.get("fog_native_soft", np.nan), errors="coerce")
            log_fog_native_soft = pd.to_numeric(g.get("log_fog_native_soft", np.nan), errors="coerce")
            fog_solvent_balanced = pd.to_numeric(g.get("fog_solvent_balanced", np.nan), errors="coerce")
            log_fog_solvent_balanced = pd.to_numeric(g.get("log_fog_solvent_balanced", np.nan), errors="coerce")
            if not np.isfinite(fog_native_soft).any():
                native_rel = pd.to_numeric(g.get("native_activity_rel_at_0", np.nan), errors="coerce")
                _, fog_native_soft_calc, log_fog_native_soft_calc = _compute_penalized_fog_objective(
                    g["fog"],
                    native_rel,
                )
                fog_native_soft = fog_native_soft_calc
                log_fog_native_soft = log_fog_native_soft_calc
            else:
                miss_log_soft = (
                    ~np.isfinite(log_fog_native_soft)
                    & np.isfinite(fog_native_soft)
                    & (fog_native_soft > 0.0)
                )
                if np.any(miss_log_soft):
                    log_fog_native_soft = np.where(
                        miss_log_soft,
                        np.log(fog_native_soft),
                        log_fog_native_soft,
                    )
            if not np.isfinite(fog_solvent_balanced).any():
                fog_rel = pd.to_numeric(g.get("fog_vs_solvent_control", np.nan), errors="coerce")
                abs_rel = pd.to_numeric(g.get("abs0_vs_solvent_control", np.nan), errors="coerce")
                _, _, _, fog_solvent_calc, log_fog_solvent_calc = _compute_solvent_balanced_objective(
                    fog_rel,
                    abs_rel,
                )
                fog_solvent_balanced = fog_solvent_calc
                log_fog_solvent_balanced = log_fog_solvent_calc
            else:
                miss_log_solvent = (
                    ~np.isfinite(log_fog_solvent_balanced)
                    & np.isfinite(fog_solvent_balanced)
                    & (fog_solvent_balanced > 0.0)
                )
                if np.any(miss_log_solvent):
                    log_fog_solvent_balanced = np.where(
                        miss_log_solvent,
                        np.log(fog_solvent_balanced),
                        log_fog_solvent_balanced,
                    )
            native_feasible = pd.to_numeric(g.get("native_activity_feasible", np.nan), errors="coerce")
            native_valid = np.isfinite(fog_native) & (fog_native > 0)
            if np.any(native_valid):
                fog_native_vals = fog_native[native_valid]
                log_fog_native_vals = log_fog_native[native_valid]
                robust_fog_native = float(np.nanmedian(fog_native_vals))
                robust_log_fog_native = float(np.nanmedian(log_fog_native_vals))
                log_fog_native_mad = float(
                    np.nanmedian(np.abs(log_fog_native_vals - robust_log_fog_native))
                )
                mean_fog_native = float(np.nanmean(fog_native_vals))
                mean_log_fog_native = float(np.nanmean(log_fog_native_vals))
            else:
                robust_fog_native = np.nan
                robust_log_fog_native = np.nan
                log_fog_native_mad = np.nan
                mean_fog_native = np.nan
                mean_log_fog_native = np.nan
            native_soft_valid = np.isfinite(fog_native_soft) & (fog_native_soft > 0)
            if np.any(native_soft_valid):
                fog_native_soft_vals = fog_native_soft[native_soft_valid]
                log_fog_native_soft_vals = log_fog_native_soft[native_soft_valid]
                robust_fog_native_soft = float(np.nanmedian(fog_native_soft_vals))
                robust_log_fog_native_soft = float(np.nanmedian(log_fog_native_soft_vals))
                log_fog_native_soft_mad = float(
                    np.nanmedian(np.abs(log_fog_native_soft_vals - robust_log_fog_native_soft))
                )
                mean_fog_native_soft = float(np.nanmean(fog_native_soft_vals))
                mean_log_fog_native_soft = float(np.nanmean(log_fog_native_soft_vals))
            else:
                robust_fog_native_soft = np.nan
                robust_log_fog_native_soft = np.nan
                log_fog_native_soft_mad = np.nan
                mean_fog_native_soft = np.nan
                mean_log_fog_native_soft = np.nan
            solvent_valid = np.isfinite(fog_solvent_balanced) & (fog_solvent_balanced > 0)
            if np.any(solvent_valid):
                fog_solvent_vals = fog_solvent_balanced[solvent_valid]
                log_fog_solvent_vals = log_fog_solvent_balanced[solvent_valid]
                robust_fog_solvent_balanced = float(np.nanmedian(fog_solvent_vals))
                robust_log_fog_solvent_balanced = float(np.nanmedian(log_fog_solvent_vals))
                log_fog_solvent_balanced_mad = float(
                    np.nanmedian(np.abs(log_fog_solvent_vals - robust_log_fog_solvent_balanced))
                )
                mean_fog_solvent_balanced = float(np.nanmean(fog_solvent_vals))
                mean_log_fog_solvent_balanced = float(np.nanmean(log_fog_solvent_vals))
            else:
                robust_fog_solvent_balanced = np.nan
                robust_log_fog_solvent_balanced = np.nan
                log_fog_solvent_balanced_mad = np.nan
                mean_fog_solvent_balanced = np.nan
                mean_log_fog_solvent_balanced = np.nan
            if not np.isfinite(native_feasible).any():
                native_feasible_fraction = float(np.nanmean(native_valid.astype(float)))
            else:
                native_feasible_fraction = float(np.nanmean(np.clip(native_feasible, 0.0, 1.0)))
            round_av_rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "reference_polymer_id": reference_polymer_id,
                "mean_fog": float(g["fog"].mean()),
                "mean_log_fog": float(g["log_fog"].mean()),
                "robust_fog": robust_fog,
                "robust_log_fog": robust_log_fog,
                "log_fog_mad": log_fog_mad,
                "mean_fog_native_constrained": mean_fog_native,
                "mean_log_fog_native_constrained": mean_log_fog_native,
                "robust_fog_native_constrained": robust_fog_native,
                "robust_log_fog_native_constrained": robust_log_fog_native,
                "log_fog_native_constrained_mad": log_fog_native_mad,
                "mean_fog_native_soft": mean_fog_native_soft,
                "mean_log_fog_native_soft": mean_log_fog_native_soft,
                "robust_fog_native_soft": robust_fog_native_soft,
                "robust_log_fog_native_soft": robust_log_fog_native_soft,
                "log_fog_native_soft_mad": log_fog_native_soft_mad,
                "mean_fog_solvent_balanced": mean_fog_solvent_balanced,
                "mean_log_fog_solvent_balanced": mean_log_fog_solvent_balanced,
                "robust_fog_solvent_balanced": robust_fog_solvent_balanced,
                "robust_log_fog_solvent_balanced": robust_log_fog_solvent_balanced,
                "log_fog_solvent_balanced_mad": log_fog_solvent_balanced_mad,
                "native_feasible_fraction": native_feasible_fraction,
                "n_observations": int(len(g)),
                "run_ids": ",".join(sorted(g["run_id"].astype(str).unique().tolist())),
            })
    round_averaged_df = pd.DataFrame(round_av_rows)

    gox_trace_df = build_round_gox_traceability(
        run_round_map,
        processed_dir,
        reference_polymer_id=reference_polymer_id,
    )
    return (
        _sync_abs_objective_alias_columns(per_row_df),
        _sync_abs_objective_alias_columns(round_averaged_df),
        gox_trace_df,
        warning_info,
    )
