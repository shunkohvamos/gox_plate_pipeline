# src/gox_plate_pipeline/polymer_timeseries.py
from __future__ import annotations

import hashlib
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import colorsys
import numpy as np
import pandas as pd
import yaml


_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _is_ascii_printable(s: str) -> bool:
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    # printable ASCII (space..~)
    return all(32 <= ord(ch) <= 126 for ch in s)


def _short_hash(s: str, n: int = 8) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:n]


def safe_stem(text: str, *, max_len: int = 80) -> str:
    """
    Make a filesystem-safe ASCII stem from an arbitrary string.
    Keeps traceability by appending a short hash only when needed (non-ASCII or truncation).
    """
    raw = "" if text is None else str(text)
    raw_strip = raw.strip()
    base = _SAFE_STEM_RE.sub("_", raw_strip).strip("_")
    if not base:
        base = "polymer"
    needs_hash = (not _is_ascii_printable(raw_strip))
    if len(base) > max_len:
        base = base[:max_len].rstrip("_")
        needs_hash = True
    if needs_hash:
        base = f"{base}__{_short_hash(raw_strip)}"
    return base


def safe_label(text: str) -> str:
    """
    Figure text must be English-only and should not contain full-width Japanese.
    If the polymer_id is non-ASCII, replace it with a stable hash label.
    """
    s = "" if text is None else str(text).strip()
    if s and _is_ascii_printable(s):
        return s
    return f"polymer_{_short_hash(s)}"


def _format_exp_rhs_simple(y0: float, k: float) -> str:
    """Mathtext RHS: y = y0 * exp(-k x). According to CHAT_HANDOVER.md: e^{-k x} format."""
    return f"{y0:.4g} e^{{-{k:.4g} x}}"


def _format_exp_rhs_plateau(c: float, y0: float, k: float) -> str:
    """Mathtext RHS: y = c + (y0-c) * exp(-k x). According to CHAT_HANDOVER.md: e^{-k x} format."""
    return f"{c:.4g} + ({y0:.4g}-{c:.4g}) e^{{-{k:.4g} x}}"


# Heat time axis: 0–60 min, 7 ticks (used for per-polymer plots)
HEAT_TICKS_0_60 = [0, 10, 20, 30, 40, 50, 60]

# t50 definition modes
T50_DEFINITION_Y0_HALF = "y0_half"
T50_DEFINITION_REA50 = "rea50"


def normalize_t50_definition(t50_definition: str) -> str:
    """
    Normalize t50 definition aliases to canonical values.

    Canonical:
      - y0_half: threshold is 0.5 * fitted y0
      - rea50:   threshold is fixed REA = 50%
    """
    raw = "" if t50_definition is None else str(t50_definition).strip().lower()
    if raw in {"y0_half", "y0half", "y0/2", "half_y0", "half"}:
        return T50_DEFINITION_Y0_HALF
    if raw in {"rea50", "rea_50", "rea=50", "rea50_percent", "rea50%"}:
        return T50_DEFINITION_REA50
    raise ValueError(
        "t50_definition must be one of {'y0_half', 'rea50'} "
        f"(accepted aliases include 'y0/2' and 'rea=50'), got: {t50_definition!r}"
    )


def t50_target_rea_percent(y0: float, *, t50_definition: str) -> float:
    """
    Return target REA (%) used for t50 crossing.
    """
    mode = normalize_t50_definition(t50_definition)
    if mode == T50_DEFINITION_Y0_HALF:
        return 0.5 * float(y0)
    return 50.0


def _compute_t50_from_exp_params(
    *,
    y0: float,
    k: float,
    c: Optional[float],
    t50_definition: str,
) -> Optional[float]:
    """
    Compute t50 from exponential parameters and definition mode.
    """
    if not np.isfinite(float(y0)) or not np.isfinite(float(k)) or float(y0) <= 0.0 or float(k) <= 0.0:
        return None

    target = t50_target_rea_percent(float(y0), t50_definition=t50_definition)

    # Already below/equal target at the start.
    if float(y0) <= float(target):
        return 0.0

    # Simple exponential (no plateau): y = y0 * exp(-k t)
    if c is None or not np.isfinite(float(c)):
        ratio = float(y0) / max(float(target), 1e-12)
        if ratio <= 1.0:
            return 0.0
        return float(np.log(ratio) / float(k))

    c_val = float(c)
    if c_val >= float(target):
        # Never reaches target because plateau stays above/equal target.
        return None

    denom = max(float(y0) - c_val, 1e-12)
    frac = (float(target) - c_val) / denom
    if frac <= 0.0:
        return None
    if frac >= 1.0:
        return 0.0
    return float(-np.log(frac) / float(k))


def value_at_time_linear(
    t_min: np.ndarray,
    y: np.ndarray,
    *,
    at_time_min: float,
) -> Optional[float]:
    """
    Linear interpolation for y(at_time_min) on observed curve.
    Returns None when at_time_min is outside observed time range.
    """
    t = np.asarray(t_min, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(yv)
    t = t[mask]
    yv = yv[mask]
    if t.size == 0 or not np.isfinite(float(at_time_min)):
        return None

    order = np.argsort(t)
    t = t[order]
    yv = yv[order]
    q = float(at_time_min)

    if q < float(t[0]) or q > float(t[-1]):
        return None

    exact = np.isclose(t, q, atol=1e-12)
    if np.any(exact):
        return float(np.nanmean(yv[exact]))

    for i in range(int(t.size) - 1):
        t0 = float(t[i])
        t1 = float(t[i + 1])
        if not (t0 <= q <= t1):
            continue
        if t1 == t0:
            continue
        y0 = float(yv[i])
        y1 = float(yv[i + 1])
        frac = (q - t0) / (t1 - t0)
        return float(y0 + frac * (y1 - y0))

    return None


def _normalize_summary_simple_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["polymer_id"] = out.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
    out["heat_min"] = pd.to_numeric(out.get("heat_min", np.nan), errors="coerce")
    out["abs_activity"] = pd.to_numeric(out.get("abs_activity", np.nan), errors="coerce")
    out = out[np.isfinite(out["heat_min"])].copy()
    return out


def _gox_profile_from_summary_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    gox = df[df["polymer_id"].str.upper() == "GOX"].copy()
    if gox.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    agg = (
        gox.groupby("heat_min", as_index=False)
        .agg(abs_activity=("abs_activity", "mean"))
        .sort_values("heat_min")
    )
    t = agg["heat_min"].to_numpy(dtype=float)
    y = agg["abs_activity"].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0.0)
    return t[mask], y[mask]


def _polymer_abs_at_time_map(
    df: pd.DataFrame,
    *,
    at_time_min: float,
    include_gox: bool = False,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for pid, g in df.groupby("polymer_id", sort=False):
        pid_s = str(pid).strip()
        if not include_gox and pid_s.upper() == "GOX":
            continue
        t = g["heat_min"].to_numpy(dtype=float)
        y = g["abs_activity"].to_numpy(dtype=float)
        val = value_at_time_linear(t, y, at_time_min=float(at_time_min))
        if val is None or (not np.isfinite(float(val))) or float(val) <= 0.0:
            continue
        out[pid_s] = float(val)
    return out


def _median_abs_log_ratio(
    lhs: dict[str, float],
    rhs: dict[str, float],
    *,
    min_shared: int,
) -> tuple[float, int]:
    shared = [k for k in lhs.keys() if k in rhs and np.isfinite(lhs[k]) and np.isfinite(rhs[k]) and lhs[k] > 0.0 and rhs[k] > 0.0]
    if len(shared) < int(min_shared):
        return float("nan"), len(shared)
    vals = [abs(math.log(float(lhs[k]) / float(rhs[k]))) for k in shared]
    return float(np.nanmedian(vals)), len(shared)


def _load_summary_simple_from_processed(processed_dir: Path, run_id: str) -> Optional[pd.DataFrame]:
    p = Path(processed_dir) / str(run_id) / "fit" / "summary_simple.csv"
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    return _normalize_summary_simple_df(df)


def resolve_gox_reference_profile(
    *,
    run_id: str,
    summary_df: pd.DataFrame,
    at_time_min: float = 20.0,
    processed_dir: Optional[Path] = None,
    run_round_map_path: Optional[Path] = None,
    drift_log_threshold: float = math.log(1.5),
    nearest_log_threshold: float = math.log(1.25),
    min_shared_polymers: int = 2,
) -> dict[str, Any]:
    """
    Resolve GOx absolute-activity reference profile for a run.

    Priority:
      1) same_run_gox
      2) same_round_mean_gox (mean over runs that have GOx)
      3) nearest_round_run_gox when target run is clearly shifted from round center
         and one round run has matching non-GOx profile at `at_time_min`.
    """
    rid = str(run_id).strip()
    df = _normalize_summary_simple_df(summary_df)
    result: dict[str, Any] = {
        "source": "missing_gox_reference",
        "round_id": "",
        "reference_run_id": "",
        "gox_t_min": np.array([], dtype=float),
        "gox_abs_activity": np.array([], dtype=float),
        "gox_abs_activity_at_time": np.nan,
        "round_mean_gox_abs_activity_at_time": np.nan,
        "target_round_drift_log_median": np.nan,
        "nearest_run_log_median": np.nan,
        "nearest_run_shared_polymers": 0,
        "reference_note": "no_reference",
    }

    # 1) same-run GOx
    t_same, y_same = _gox_profile_from_summary_df(df)
    if t_same.size > 0:
        gox_at = value_at_time_linear(t_same, y_same, at_time_min=float(at_time_min))
        if gox_at is not None and np.isfinite(float(gox_at)) and float(gox_at) > 0.0:
            result.update({
                "source": "same_run_gox",
                "reference_run_id": rid,
                "gox_t_min": t_same,
                "gox_abs_activity": y_same,
                "gox_abs_activity_at_time": float(gox_at),
                "round_mean_gox_abs_activity_at_time": float(gox_at),
                "reference_note": "same_run_gox_available",
            })
            return result

    if processed_dir is None or run_round_map_path is None:
        result["reference_note"] = "same_run_gox_missing_and_no_round_fallback"
        return result

    processed_dir = Path(processed_dir)
    run_round_map_path = Path(run_round_map_path)
    if (not processed_dir.is_dir()) or (not run_round_map_path.is_file()):
        result["reference_note"] = "round_fallback_inputs_missing"
        return result

    try:
        from gox_plate_pipeline.bo_data import load_run_round_map
        run_round_map = load_run_round_map(run_round_map_path)
    except Exception:
        result["reference_note"] = "failed_to_load_run_round_map"
        return result

    round_id = str(run_round_map.get(rid, "")).strip()
    if not round_id:
        result["reference_note"] = "run_not_in_round_map"
        return result
    result["round_id"] = round_id

    candidate_runs = sorted([r for r, oid in run_round_map.items() if str(oid).strip() == round_id and str(r).strip() != rid])
    if not candidate_runs:
        result["reference_note"] = "no_other_runs_in_round"
        return result

    target_map = _polymer_abs_at_time_map(df, at_time_min=float(at_time_min), include_gox=False)
    candidates: list[dict[str, Any]] = []
    for cand_run in candidate_runs:
        cand_df = _load_summary_simple_from_processed(processed_dir, cand_run)
        if cand_df is None or cand_df.empty:
            continue
        t_c, y_c = _gox_profile_from_summary_df(cand_df)
        if t_c.size == 0:
            continue
        gox20 = value_at_time_linear(t_c, y_c, at_time_min=float(at_time_min))
        if gox20 is None or (not np.isfinite(float(gox20))) or float(gox20) <= 0.0:
            continue
        candidates.append({
            "run_id": str(cand_run).strip(),
            "gox_t": t_c,
            "gox_y": y_c,
            "gox_at_time": float(gox20),
            "polymer_abs_at_time": _polymer_abs_at_time_map(cand_df, at_time_min=float(at_time_min), include_gox=False),
        })

    if not candidates:
        result["reference_note"] = "no_round_runs_with_valid_gox"
        return result

    # same-round mean GOx profile
    heat_to_vals: dict[float, list[float]] = {}
    for cand in candidates:
        for hh, vv in zip(cand["gox_t"], cand["gox_y"]):
            if not np.isfinite(float(hh)) or not np.isfinite(float(vv)) or float(vv) <= 0.0:
                continue
            heat_to_vals.setdefault(float(hh), []).append(float(vv))
    mean_t = np.array(sorted(heat_to_vals.keys()), dtype=float)
    mean_y = np.array([float(np.mean(heat_to_vals[h])) for h in mean_t], dtype=float) if mean_t.size else np.array([], dtype=float)
    mean_gox_at = value_at_time_linear(mean_t, mean_y, at_time_min=float(at_time_min)) if mean_t.size else None
    if mean_gox_at is None or (not np.isfinite(float(mean_gox_at))) or float(mean_gox_at) <= 0.0:
        mean_gox_at = float(np.mean([c["gox_at_time"] for c in candidates]))
    result["round_mean_gox_abs_activity_at_time"] = float(mean_gox_at)

    # Decide nearest-run override when target run is clearly shifted.
    round_poly_vals: dict[str, list[float]] = {}
    for cand in candidates:
        for pid, val in cand["polymer_abs_at_time"].items():
            if not np.isfinite(float(val)) or float(val) <= 0.0:
                continue
            round_poly_vals.setdefault(str(pid), []).append(float(val))
    round_poly_center = {k: float(np.mean(v)) for k, v in round_poly_vals.items() if len(v) > 0}
    drift_log, _n_shared_round = _median_abs_log_ratio(
        target_map,
        round_poly_center,
        min_shared=max(1, int(min_shared_polymers)),
    )

    best_cand: Optional[dict[str, Any]] = None
    best_dist = float("inf")
    best_shared = 0
    for cand in candidates:
        dist, n_shared = _median_abs_log_ratio(
            target_map,
            cand["polymer_abs_at_time"],
            min_shared=max(1, int(min_shared_polymers)),
        )
        if not np.isfinite(dist):
            continue
        if dist < best_dist:
            best_dist = float(dist)
            best_cand = cand
            best_shared = int(n_shared)

    use_nearest = (
        best_cand is not None
        and np.isfinite(drift_log)
        and float(drift_log) >= float(drift_log_threshold)
        and np.isfinite(best_dist)
        and float(best_dist) <= float(nearest_log_threshold)
    )

    if use_nearest and best_cand is not None:
        result.update({
            "source": "nearest_round_run_gox",
            "reference_run_id": str(best_cand["run_id"]),
            "gox_t_min": np.asarray(best_cand["gox_t"], dtype=float),
            "gox_abs_activity": np.asarray(best_cand["gox_y"], dtype=float),
            "gox_abs_activity_at_time": float(best_cand["gox_at_time"]),
            "target_round_drift_log_median": float(drift_log),
            "nearest_run_log_median": float(best_dist),
            "nearest_run_shared_polymers": int(best_shared),
            "reference_note": "nearest_round_run_selected_for_shift",
        })
        return result

    result.update({
        "source": "same_round_mean_gox",
        "reference_run_id": "",
        "gox_t_min": mean_t,
        "gox_abs_activity": mean_y,
        "gox_abs_activity_at_time": float(mean_gox_at),
        "target_round_drift_log_median": float(drift_log) if np.isfinite(drift_log) else np.nan,
        "nearest_run_log_median": float(best_dist) if np.isfinite(best_dist) else np.nan,
        "nearest_run_shared_polymers": int(best_shared),
        "reference_note": "same_round_mean_gox_selected",
    })
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=True, allow_unicode=True)


def _default_palette_hex() -> list[str]:
    # Neon-ish, high-chroma palette on white.
    # Note: keep the list reasonably long so new polymer_ids get distinct colors.
    return [
        "#40C4FF",  # neon sky blue
        "#18FFFF",  # cyan
        "#00E676",  # neon green
        "#C6FF00",  # lime
        "#FFEA00",  # neon yellow
        "#FFC400",  # amber
        "#FF9100",  # neon orange
        "#FF5252",  # red
        "#FF4081",  # pink
        "#E040FB",  # magenta/purple
        "#7C4DFF",  # violet
        "#64FFDA",  # mint
    ]


def load_or_create_polymer_color_map(path: Path) -> dict[str, str]:
    obj = _read_yaml(path)
    cmap = obj.get("polymer_id", obj) if isinstance(obj, dict) else {}
    if not isinstance(cmap, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in cmap.items():
        if k is None:
            continue
        kk = str(k)
        vv = "" if v is None else str(v).strip()
        if vv:
            out[kk] = vv
    return out


def _color_distance(hex1: str, hex2: str) -> float:
    """
    Calculate perceptual color distance between two hex colors using Euclidean distance in RGB space.
    Returns a value between 0 (same color) and ~441 (max distance in RGB cube).
    """
    hex1 = hex1.lstrip('#')
    hex2 = hex2.lstrip('#')
    r1 = int(hex1[0:2], 16)
    g1 = int(hex1[2:4], 16)
    b1 = int(hex1[4:6], 16)
    r2 = int(hex2[0:2], 16)
    g2 = int(hex2[2:4], 16)
    b2 = int(hex2[4:6], 16)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def ensure_polymer_colors(
    polymer_ids: list[str],
    *,
    color_map_path: Path,
) -> dict[str, str]:
    """
    Ensure a persistent polymer_id -> color map exists and includes all polymer_ids.
    Existing IDs keep the same color; new IDs are appended and saved.
    
    For new IDs, colors are chosen to:
    1. Maximize distance from all existing colors
    2. Maximize distance from other new IDs in the same experiment (polymer_ids list)
    """
    color_map_path = Path(color_map_path)
    cmap = load_or_create_polymer_color_map(color_map_path)
    changed = False

    # GOx always uses gray
    GOX_COLOR = "#808080"  # Medium gray
    
    used = set(v.lower() for v in cmap.values())
    palette = _default_palette_hex()
    
    # Separate existing and new polymer IDs
    existing_pids = [pid for pid in polymer_ids if str(pid) in cmap]
    new_pids = [pid for pid in polymer_ids if str(pid) not in cmap]
    
    # Handle GOx first (create a new list without GOx for color assignment)
    new_pids_for_color = []
    for pid in new_pids:
        pid_str = str(pid)
        if pid_str.upper() == "GOX":
            cmap[pid_str] = GOX_COLOR
            used.add(GOX_COLOR.lower())
            changed = True
        else:
            new_pids_for_color.append(pid)
    
    # Assign colors to new non-GOx polymer IDs
    for pid in new_pids_for_color:
        pid_str = "" if pid is None else str(pid)
        if pid_str in cmap:
            continue
        
        # Get all existing colors (from saved map and already assigned new IDs)
        all_existing_colors = [v.lower() for v in cmap.values()]
        # Also consider colors assigned to other new IDs in this batch
        other_new_colors = [cmap[str(other_pid)].lower() for other_pid in new_pids_for_color if str(other_pid) in cmap and str(other_pid) != pid_str]
        
        # Find the best color: maximize minimum distance to all existing and other new colors
        best_color = None
        best_min_distance = -1.0
        
        # First try palette colors
        for c in palette:
            c_lower = c.lower()
            if c_lower in used:
                continue
            
            # Calculate minimum distance to all existing colors
            min_dist_to_existing = min(
                [_color_distance(c, existing) for existing in all_existing_colors] + [float('inf')]
            )
            # Calculate minimum distance to other new colors in this batch
            min_dist_to_new = min(
                [_color_distance(c, other_new) for other_new in other_new_colors] + [float('inf')]
            )
            # Use the smaller of the two (we want to maximize the minimum distance)
            min_dist = min(min_dist_to_existing, min_dist_to_new)
            
            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_color = c
        
        # If no palette color is suitable (all too close), generate a new color
        if best_color is None or best_min_distance < 50.0:  # Threshold: if min distance < 50, generate new
            # Find the hue that maximizes distance from existing colors
            best_hue = None
            best_hue_min_dist = -1.0
            
            for hue_candidate in np.linspace(0.0, 1.0, 360):  # Try 360 different hues
                r, g, b = colorsys.hsv_to_rgb(hue_candidate, 0.65, 0.85)
                candidate_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
                
                min_dist_to_existing = min(
                    [_color_distance(candidate_color, existing) for existing in all_existing_colors] + [float('inf')]
                )
                min_dist_to_new = min(
                    [_color_distance(candidate_color, other_new) for other_new in other_new_colors] + [float('inf')]
                )
                min_dist = min(min_dist_to_existing, min_dist_to_new)
                
                if min_dist > best_hue_min_dist:
                    best_hue_min_dist = min_dist
                    best_hue = hue_candidate
            
            if best_hue is not None:
                r, g, b = colorsys.hsv_to_rgb(best_hue, 0.65, 0.85)
                best_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            else:
                # Fallback: golden ratio hue stepping
                i = int(len(used))
                hue = (0.61803398875 * (i + 1)) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
                best_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        
        cmap[pid_str] = best_color
        used.add(best_color.lower())
        changed = True

    if changed:
        # Wrap as a mapping so we can extend later without breaking existing files.
        _write_yaml(color_map_path, {"polymer_id": cmap})

    return cmap


@dataclass(frozen=True)
class ExpDecayFit:
    model: str  # "exp" or "exp_plateau"
    y0: float
    k: float  # 1 / min
    c: Optional[float]  # plateau (same unit as y), only for exp_plateau
    r2: float
    aic: float
    t50: Optional[float]  # minutes; None if undefined


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _aic(rss: float, n: int, p: int) -> float:
    rss = float(max(rss, 1e-24))
    n = int(max(n, 1))
    return float(n * np.log(rss / n) + 2.0 * p)


def fit_exponential_decay(
    t_min: np.ndarray,
    y: np.ndarray,
    *,
    y0: Optional[float] = None,
    fixed_y0: Optional[float] = None,
    min_points: int = 4,
    t50_definition: str = T50_DEFINITION_Y0_HALF,
) -> Optional[ExpDecayFit]:
    """
    Exponential decay fit.

    Models:
      - exp:         y = y0 * exp(-k t), k>=0, y0>0
      - exp_plateau: y = c + (y0-c)*exp(-k t), 0<=c<=y0, k>=0, y0>0

    Picks the model with lower AIC. Returns None when fitting is unsafe.
    y0 is an optional initial guess for free-y0 fitting.
    fixed_y0:
      - None: fit y0 freely from all points
      - finite positive value: constrain y0 to that value
    t50_definition:
      - y0_half: t50 is the time to reach 0.5 * fitted y0
      - rea50:   t50 is the time to reach REA=50
    """
    t50_definition = normalize_t50_definition(t50_definition)
    t = np.asarray(t_min, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if t.size < int(min_points):
        return None
    if np.unique(t).size < 3:
        return None
    if np.any(y <= 0.0):
        # log-space / exp fit becomes unstable if any point is non-positive
        return None

    # Sort by time.
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    fixed_y0_val: Optional[float] = None
    if fixed_y0 is not None:
        try:
            fixed_y0_val = float(fixed_y0)
        except Exception:
            return None
        if not np.isfinite(fixed_y0_val) or fixed_y0_val <= 0.0:
            return None

    # Estimate y0 from data if not provided (free-y0 mode only).
    # Use the maximum value as initial guess (typically the first or early point).
    if fixed_y0_val is None:
        if y0 is None or not np.isfinite(float(y0)) or float(y0) <= 0.0:
            y0_init = float(np.max(y))
        else:
            y0_init = float(y0)
    else:
        y0_init = float(fixed_y0_val)

    # Only fit decay models when the overall trend is decreasing.
    # This avoids producing meaningless "t50" for non-decaying / drifting traces.
    if float(y[-1]) >= float(y0_init):
        return None

    rt = pd.Series(t).rank(method="average")
    ry = pd.Series(y).rank(method="average")
    spearman_rho = float(rt.corr(ry))
    if not np.isfinite(spearman_rho) or spearman_rho >= -0.20:
        return None

    from scipy.optimize import curve_fit

    t_end = float(t[-1]) if t.size else 0.0
    y_end = float(y[-1]) if y.size else float(y0_init)
    if t_end <= 0.0:
        return None

    k_init = 1e-6
    if y_end > 0.0 and y0_init > 0.0 and y_end < y0_init:
        k_init = max(1e-6, float(np.log(y0_init / y_end) / max(t_end, 1e-12)))

    best: Optional[ExpDecayFit] = None
    if fixed_y0_val is None:
        # --- Model 1 (free y0): y = y0 * exp(-k t)
        def _m1(tt: np.ndarray, y0_param: float, k_param: float) -> np.ndarray:
            return y0_param * np.exp(-k_param * tt)

        try:
            (y0_1, k1), _ = curve_fit(
                _m1,
                t,
                y,
                p0=[y0_init, k_init],
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=20_000,
            )
            y0_1 = float(y0_1)
            k1 = float(k1)
            if y0_1 <= 0.0 or k1 < 0.0:
                raise ValueError("Invalid fitted parameters")
            yhat1 = _m1(t, y0_1, k1)
            rss1 = float(np.sum((y - yhat1) ** 2))
            aic1 = _aic(rss1, int(t.size), 2)  # 2 parameters: y0 and k
            r21 = _r2(y, yhat1)
            t50_1 = _compute_t50_from_exp_params(
                y0=y0_1,
                k=k1,
                c=None,
                t50_definition=t50_definition,
            )
            best = ExpDecayFit(model="exp", y0=y0_1, k=k1, c=None, r2=float(r21), aic=float(aic1), t50=t50_1)
        except Exception:
            best = None
    else:
        # --- Model 1 (fixed y0): y = fixed_y0 * exp(-k t)
        def _m1_fixed(tt: np.ndarray, k_param: float) -> np.ndarray:
            return fixed_y0_val * np.exp(-k_param * tt)

        try:
            (k1,), _ = curve_fit(
                _m1_fixed,
                t,
                y,
                p0=[k_init],
                bounds=([0.0], [np.inf]),
                maxfev=20_000,
            )
            y0_1 = float(fixed_y0_val)
            k1 = float(k1)
            if y0_1 <= 0.0 or k1 < 0.0:
                raise ValueError("Invalid fitted parameters")
            yhat1 = _m1_fixed(t, k1)
            rss1 = float(np.sum((y - yhat1) ** 2))
            aic1 = _aic(rss1, int(t.size), 1)  # 1 parameter: k
            r21 = _r2(y, yhat1)
            t50_1 = _compute_t50_from_exp_params(
                y0=y0_1,
                k=k1,
                c=None,
                t50_definition=t50_definition,
            )
            best = ExpDecayFit(model="exp", y0=y0_1, k=k1, c=None, r2=float(r21), aic=float(aic1), t50=t50_1)
        except Exception:
            best = None

    if t.size >= 5:
        if fixed_y0_val is None:
            # --- Model 2 (free y0): y = c + (y0-c)*exp(-k t)
            def _m2(tt: np.ndarray, c_param: float, y0_param: float, k_param: float) -> np.ndarray:
                return c_param + (y0_param - c_param) * np.exp(-k_param * tt)

            c_init = float(max(0.0, min(float(np.min(y)), 0.2 * float(y0_init))))
            try:
                (c2, y0_2, k2), _ = curve_fit(
                    _m2,
                    t,
                    y,
                    p0=[c_init, y0_init, k_init],
                    bounds=([0.0, 0.0, 0.0], [float(y0_init), np.inf, np.inf]),
                    maxfev=30_000,
                )
                c2 = float(c2)
                y0_2 = float(y0_2)
                k2 = float(k2)
                if y0_2 <= 0.0 or k2 < 0.0 or c2 < 0.0 or c2 >= y0_2:
                    raise ValueError("Invalid fitted parameters")
                yhat2 = _m2(t, c2, y0_2, k2)
                rss2 = float(np.sum((y - yhat2) ** 2))
                aic2 = _aic(rss2, int(t.size), 3)  # 3 parameters: c, y0, and k
                r22 = _r2(y, yhat2)

                t50_2 = _compute_t50_from_exp_params(
                    y0=y0_2,
                    k=k2,
                    c=c2,
                    t50_definition=t50_definition,
                )

                cand = ExpDecayFit(
                    model="exp_plateau",
                    y0=y0_2,
                    k=k2,
                    c=c2,
                    r2=float(r22),
                    aic=float(aic2),
                    t50=t50_2,
                )
                if best is None or (np.isfinite(cand.aic) and cand.aic < best.aic):
                    best = cand
            except Exception:
                pass
        else:
            # --- Model 2 (fixed y0): y = c + (fixed_y0-c)*exp(-k t)
            def _m2_fixed(tt: np.ndarray, c_param: float, k_param: float) -> np.ndarray:
                return c_param + (fixed_y0_val - c_param) * np.exp(-k_param * tt)

            c_init = float(max(0.0, min(float(np.min(y)), 0.2 * float(fixed_y0_val))))
            try:
                (c2, k2), _ = curve_fit(
                    _m2_fixed,
                    t,
                    y,
                    p0=[c_init, k_init],
                    bounds=([0.0, 0.0], [float(fixed_y0_val), np.inf]),
                    maxfev=30_000,
                )
                c2 = float(c2)
                y0_2 = float(fixed_y0_val)
                k2 = float(k2)
                if y0_2 <= 0.0 or k2 < 0.0 or c2 < 0.0 or c2 >= y0_2:
                    raise ValueError("Invalid fitted parameters")
                yhat2 = _m2_fixed(t, c2, k2)
                rss2 = float(np.sum((y - yhat2) ** 2))
                aic2 = _aic(rss2, int(t.size), 2)  # 2 parameters: c and k
                r22 = _r2(y, yhat2)

                t50_2 = _compute_t50_from_exp_params(
                    y0=y0_2,
                    k=k2,
                    c=c2,
                    t50_definition=t50_definition,
                )

                cand = ExpDecayFit(
                    model="exp_plateau",
                    y0=y0_2,
                    k=k2,
                    c=c2,
                    r2=float(r22),
                    aic=float(aic2),
                    t50=t50_2,
                )
                if best is None or (np.isfinite(cand.aic) and cand.aic < best.aic):
                    best = cand
            except Exception:
                pass

    return best


def t50_linear(
    t_min: np.ndarray,
    y: np.ndarray,
    *,
    y0: float,
    target_frac: float = 0.5,
    target_value: Optional[float] = None,
) -> Optional[float]:
    """
    Estimate t50 by linear interpolation on the observed curve.
    Returns None if the curve never crosses target in the observed domain.
    If target_value is provided, it is used as absolute target (REA %).
    Otherwise target = y0 * target_frac.
    """
    t = np.asarray(t_min, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 2:
        return None

    if target_value is not None and np.isfinite(float(target_value)):
        target = float(target_value)
    else:
        if not np.isfinite(float(y0)) or float(y0) <= 0.0:
            return None
        target = float(y0) * float(target_frac)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # If already below/equal target at earliest observed time, t50 is reached at the first point.
    if float(y[0]) <= target:
        return float(t[0])

    for i in range(int(t.size) - 1):
        y0i = float(y[i])
        y1i = float(y[i + 1])
        if (y0i >= target and y1i <= target) or (y0i <= target and y1i >= target):
            if y1i == y0i:
                return float(t[i])
            frac = (target - y0i) / (y1i - y0i)
            return float(t[i] + frac * (t[i + 1] - t[i]))
    return None


def plot_per_polymer_timeseries(
    *,
    summary_simple_path: Path,
    run_id: str,
    out_fit_dir: Path,
    color_map_path: Path,
    dpi: int = 600,
    row_map_path: Optional[Path] = None,
    t50_definition: str = T50_DEFINITION_Y0_HALF,
    processed_dir: Optional[Path] = None,
    run_round_map_path: Optional[Path] = None,
) -> Path:
    """
    Create per-polymer time series plots: one figure per polymer with
    Absolute activity (left), REA (center), and Functional activity (right).

    Outputs (PNG only):
      - out_fit_dir/per_polymer__{run_id}/{polymer_stem}__{run_id}.png  (combined 3-panel figure)
      - out_fit_dir/t50/t50__{run_id}.csv
      - out_fit_dir/all_polymers__{run_id}.png (include_in_all_polymers=True only, default)
      - out_fit_dir/all_polymers_all__{run_id}.png (all polymers, for debugging, only if different from filtered)
      - out_fit_dir/all_polymers_pair__{run_id}.png (custom pair from TSV, if specified)

    If row_map_path is provided and contains all_polymers_pair column with two polymer IDs
    (comma-separated), generates an additional plot with only those two polymers.

    Functional activity reference for panel/metric:
      - same-run GOx first
      - if missing and round info is provided: same-round mean GOx
      - when the run is clearly shifted from the round center, nearest round run GOx.

    Returns path to the written t50 CSV.
    """
    summary_simple_path = Path(summary_simple_path)
    out_fit_dir = Path(out_fit_dir)
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")
    t50_definition = normalize_t50_definition(t50_definition)

    df = pd.read_csv(summary_simple_path)
    required = {"polymer_id", "heat_min", "abs_activity", "REA_percent"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"summary_simple is missing required columns: {missing}")

    df = df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str)
    df["heat_min"] = pd.to_numeric(df["heat_min"], errors="coerce")
    df["abs_activity"] = pd.to_numeric(df["abs_activity"], errors="coerce")
    df["REA_percent"] = pd.to_numeric(df["REA_percent"], errors="coerce")
    df = df.dropna(subset=["polymer_id", "heat_min"])
    
    # Handle include_in_all_polymers flag (default True if missing)
    if "include_in_all_polymers" in df.columns:
        # Convert string "True"/"False" or boolean to bool
        def _parse_bool_flag(v):
            if pd.isna(v):
                return True  # Default True if missing
            if isinstance(v, bool):
                return v
            s = str(v).strip().upper()
            if s in ("TRUE", "1", "YES"):
                return True
            if s in ("FALSE", "0", "NO"):
                return False
            return True  # Default True for unrecognized values
        df["include_in_all_polymers"] = df["include_in_all_polymers"].apply(_parse_bool_flag)
    else:
        df["include_in_all_polymers"] = True
    
    # Handle all_polymers_pair flag (default False if missing)
    if "all_polymers_pair" in df.columns:
        # Convert string "True"/"False" or boolean to bool
        def _parse_bool_flag_pair(v):
            if pd.isna(v):
                return False  # Default False if missing
            if isinstance(v, bool):
                return v
            s = str(v).strip().upper()
            if s in ("TRUE", "1", "YES"):
                return True
            if s in ("FALSE", "0", "NO", ""):
                return False
            return False  # Default False for unrecognized values
        df["all_polymers_pair"] = df["all_polymers_pair"].apply(_parse_bool_flag_pair)
    else:
        df["all_polymers_pair"] = False

    # GOx reference for functional activity panel/metric (same run -> same round fallback).
    gox_ref = resolve_gox_reference_profile(
        run_id=run_id,
        summary_df=df,
        at_time_min=20.0,
        processed_dir=(Path(processed_dir) if processed_dir is not None else None),
        run_round_map_path=(Path(run_round_map_path) if run_round_map_path is not None else None),
    )
    gox_ref_t = np.asarray(gox_ref.get("gox_t_min", np.array([], dtype=float)), dtype=float)
    gox_ref_y = np.asarray(gox_ref.get("gox_abs_activity", np.array([], dtype=float)), dtype=float)
    gox_ref_at_20 = pd.to_numeric(gox_ref.get("gox_abs_activity_at_time", np.nan), errors="coerce")
    gox_ref_source = str(gox_ref.get("source", "missing_gox_reference"))
    gox_ref_round_id = str(gox_ref.get("round_id", "")).strip()
    gox_ref_run_id = str(gox_ref.get("reference_run_id", "")).strip()

    polymer_ids = sorted(df["polymer_id"].astype(str).unique().tolist())
    cmap = ensure_polymer_colors(polymer_ids, color_map_path=Path(color_map_path))

    # Resolve per-polymer file stems (avoid collisions after sanitization).
    stems: dict[str, str] = {pid: safe_stem(pid) for pid in polymer_ids}
    stem_counts: dict[str, int] = {}
    for st in stems.values():
        stem_counts[st] = stem_counts.get(st, 0) + 1
    for pid, st in list(stems.items()):
        if stem_counts.get(st, 0) > 1:
            stems[pid] = f"{st}__{_short_hash(pid)}"

    # Create t50 directory first, then per_polymer inside it
    out_t50_dir = out_fit_dir / "t50"
    out_t50_dir.mkdir(parents=True, exist_ok=True)
    out_per_polymer = out_t50_dir / f"per_polymer__{run_id}"
    out_per_polymer.mkdir(parents=True, exist_ok=True)

    # Remove legacy per-polymer folders (abs/rea split) so only the combined output remains.
    for legacy in (out_fit_dir / f"per_polymer_abs__{run_id}", out_fit_dir / f"per_polymer_rea__{run_id}"):
        if legacy.is_dir():
            try:
                shutil.rmtree(legacy)
            except Exception:
                pass

    # Clean stale outputs so the folder contains exactly one plot per polymer_id.
    expected_pngs = {f"{stems[pid]}__{run_id}.png" for pid in polymer_ids}

    def _clean_stale(dir_path: Path, expected: set[str]) -> None:
        for p in dir_path.glob(f"*__{run_id}.png"):
            if p.name not in expected:
                try:
                    p.unlink()
                except Exception:
                    pass

    _clean_stale(out_per_polymer, expected_pngs)

    from gox_plate_pipeline.fitting.core import (
        apply_paper_style,
        INFO_BOX_MARGIN_PT,
        INFO_BOX_FACE_COLOR,
        INFO_BOX_PAD_PER_POLYMER,
        get_info_box_gradient_shadow,
    )
    import matplotlib.pyplot as plt
    import gc

    t50_columns = [
        "run_id",
        "polymer_id",
        "polymer_label",
        "n_points",
        "abs_activity_at_0",
        "abs_activity_at_20",
        "functional_activity_at_20_rel",
        "gox_abs_activity_at_20_ref",
        "functional_reference_source",
        "functional_reference_round_id",
        "functional_reference_run_id",
        "y0_REA_percent",
        "t50_definition",
        "t50_target_rea_percent",
        "t50_linear_min",
        "t50_exp_min",
        "rea_at_20_percent",
        "fit_model",
        "fit_k_per_min",
        "fit_tau_min",
        "fit_plateau",
        "fit_r2",
        "rea_connector",
    ]
    t50_rows: list[dict[str, Any]] = []
    # Calculate padding for info box (larger for per_polymer to reduce cramped appearance)
    if isinstance(INFO_BOX_PAD_PER_POLYMER, (tuple, list)):
        # Use average of all sides + extra margin for more spacious feel
        info_pad = sum(float(v) for v in INFO_BOX_PAD_PER_POLYMER) / len(INFO_BOX_PAD_PER_POLYMER) + 0.2
    else:
        info_pad = float(INFO_BOX_PAD_PER_POLYMER) + 0.2

    def _eval_fit_curve(fit_obj: ExpDecayFit, tt: np.ndarray) -> np.ndarray:
        if fit_obj.model == "exp":
            return fit_obj.y0 * np.exp(-fit_obj.k * tt)
        c = float(fit_obj.c) if fit_obj.c is not None else 0.0
        return c + (fit_obj.y0 - c) * np.exp(-fit_obj.k * tt)

    def _to_fluorescent_color(color_hex: str) -> str:
        """
        Convert a color to a fluorescent (bright, high-saturation) version.
        Increases saturation and brightness while maintaining hue.
        """
        # Remove '#' if present
        hex_str = color_hex.lstrip('#')
        # Convert hex to RGB (0-255)
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Increase saturation (toward 1.0) and brightness (toward 1.0) for fluorescent effect
        # Blend original saturation with 0.9 (high saturation)
        s_fluorescent = min(1.0, s * 0.3 + 0.9 * 0.7)
        # Blend original brightness with 1.0 (maximum brightness)
        v_fluorescent = min(1.0, v * 0.4 + 1.0 * 0.6)
        
        # Convert back to RGB
        r_new, g_new, b_new = colorsys.hsv_to_rgb(h, s_fluorescent, v_fluorescent)
        
        # Convert to hex
        return "#{:02x}{:02x}{:02x}".format(
            int(r_new * 255),
            int(g_new * 255),
            int(b_new * 255)
        )

    def _draw_fit_with_extension(ax: Any, fit_obj: ExpDecayFit, t_obs: np.ndarray, color_hex: str, *, use_dashed_main: bool = False, preserve_gray: bool = False) -> None:
        """
        Draw fitted curve with extension.
        
        Args:
            ax: Matplotlib axes
            fit_obj: Exponential decay fit object
            t_obs: Observed time points
            color_hex: Base color in hex format (same as scatter/project color; curve uses corresponding fluorescent)
            use_dashed_main: If True, curves have higher transparency (for all_polymers plots).
                            If False, curves have normal transparency (for per_polymer plots).
                            Both use solid line for main curve and dashed line for extensions.
            preserve_gray: If True, do not convert gray colors (#808080) to fluorescent (for GOx).
        """
        t_obs = np.asarray(t_obs, dtype=float)
        t_obs = t_obs[np.isfinite(t_obs)]
        if t_obs.size == 0:
            return
        t_min_obs = float(np.min(t_obs))
        t_max_obs = float(np.max(t_obs))

        # Corresponding curve color: fluorescent from plot color (all_polymers / all_polymers_pair same logic)
        if preserve_gray and color_hex.upper() == "#808080":
            color_fluorescent = color_hex  # Keep gray as-is
        else:
            color_fluorescent = _to_fluorescent_color(color_hex)
        
        # Main curve on the observed domain used for fitting.
        # Both per_polymer and all_polymers use solid line for main curve
        tt_main = np.linspace(t_min_obs, t_max_obs, 220)
        yy_main = _eval_fit_curve(fit_obj, tt_main)
        if use_dashed_main:
            # all_polymers: main curve is solid with higher transparency
            ax.plot(tt_main, yy_main, color=color_fluorescent, linewidth=1.7, alpha=0.40, zorder=8)
        else:
            # per_polymer: main curve is solid with normal transparency
            ax.plot(tt_main, yy_main, color=color_fluorescent, linewidth=1.7, alpha=0.50, zorder=8)

        # Dashed extension where points are missing (0-60 min design range).
        if t_min_obs > 0.0:
            tt_pre = np.linspace(0.0, t_min_obs, 120)
            yy_pre = _eval_fit_curve(fit_obj, tt_pre)
            if use_dashed_main:
                # all_polymers: extension with higher transparency
                ax.plot(tt_pre, yy_pre, color=color_fluorescent, linewidth=1.5, alpha=0.30, linestyle=(0, (2.4, 2.4)), zorder=7)
            else:
                # per_polymer: extension with normal transparency
                ax.plot(tt_pre, yy_pre, color=color_fluorescent, linewidth=1.5, alpha=0.40, linestyle=(0, (2.4, 2.4)), zorder=7)
        if t_max_obs < 60.0:
            tt_post = np.linspace(t_max_obs, 60.0, 140)
            yy_post = _eval_fit_curve(fit_obj, tt_post)
            if use_dashed_main:
                # all_polymers: extension with higher transparency
                ax.plot(tt_post, yy_post, color=color_fluorescent, linewidth=1.5, alpha=0.30, linestyle=(0, (2.4, 2.4)), zorder=7)
            else:
                # per_polymer: extension with normal transparency
                ax.plot(tt_post, yy_post, color=color_fluorescent, linewidth=1.5, alpha=0.40, linestyle=(0, (2.4, 2.4)), zorder=7)

    def _info_box_text(rhs: str, r2: Optional[float], t50: Optional[float] = None) -> str:
        # Format according to CHAT_HANDOVER.md: mathtext with R² and t₅₀
        # Align = signs vertically by using spaces (phantom not supported in matplotlib mathtext)
        r2_txt = f"{float(r2):.3f}" if (r2 is not None and np.isfinite(float(r2))) else r"\mathrm{NA}"
        if t50 is None or not np.isfinite(float(t50)):
            t50_txt = r"\mathrm{NA}"
        else:
            t50_txt = f"{float(t50):.3g}\\,\\mathrm{{min}}"
        if t50 is None:
            # Align = by matching width: y with space and R^2
            return rf"$y\  = {rhs}$" + "\n" + rf"$R^2 = {r2_txt}$"
        # Align = by matching width: y with space, R^2, and t_{50}
        return rf"$y\  = {rhs}$" + "\n" + rf"$R^2 = {r2_txt}$" + "\n" + rf"$t_{{50}} = {t50_txt}$"

    for pid, g in df.groupby("polymer_id", sort=False):
        g = g.sort_values("heat_min").reset_index(drop=True)
        t = g["heat_min"].to_numpy(dtype=float)
        aa = g["abs_activity"].to_numpy(dtype=float)
        rea = g["REA_percent"].to_numpy(dtype=float)
        # Absolute activity anchor for ranking: prefer heat=0, fallback to first finite point.
        abs_activity_at_0 = np.nan
        heat0_mask = np.isfinite(t) & np.isfinite(aa) & np.isclose(t, 0.0, atol=1e-9)
        if np.any(heat0_mask):
            abs_activity_at_0 = float(aa[np.where(heat0_mask)[0][0]])
        else:
            finite_idx = np.where(np.isfinite(aa))[0]
            if finite_idx.size > 0:
                abs_activity_at_0 = float(aa[finite_idx[0]])
        abs_activity_at_20 = value_at_time_linear(t, aa, at_time_min=20.0)
        abs_activity_at_20 = float(abs_activity_at_20) if abs_activity_at_20 is not None and np.isfinite(abs_activity_at_20) else np.nan

        # Functional activity: absolute activity relative to GOx reference at the same heat time.
        func = np.full_like(aa, np.nan, dtype=float)
        if gox_ref_t.size > 0 and gox_ref_y.size > 0:
            denom = np.array(
                [
                    value_at_time_linear(gox_ref_t, gox_ref_y, at_time_min=float(tt))
                    if np.isfinite(float(tt)) else np.nan
                    for tt in t
                ],
                dtype=float,
            )
            ok_func = np.isfinite(aa) & np.isfinite(denom) & (denom > 0.0)
            func[ok_func] = aa[ok_func] / denom[ok_func]
        func_at_20 = value_at_time_linear(t, func, at_time_min=20.0)
        func_at_20 = float(func_at_20) if func_at_20 is not None and np.isfinite(func_at_20) else np.nan

        # Debug output removed for memory optimization

        # GOx always uses gray color
        pid_str = str(pid)
        is_gox_per_polymer = pid_str.upper() == "GOX" or pid_str == "GOx"
        if is_gox_per_polymer:
            color = "#808080"  # Medium gray
        else:
            color = cmap.get(pid_str, "#0072B2")
        pid_label = safe_label(str(pid))
        stem = stems.get(pid_str, safe_stem(pid_str))

        # --- Absolute activity (left panel)
        # y0 is optional: if provided, used as initial guess; otherwise estimated from all data points
        y0_abs_init = None
        if aa.size > 0 and np.isfinite(float(aa[0])):
            y0_abs_init = float(aa[0])  # Use first point as initial guess if available

        fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
        use_exp_abs = bool(fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70)
        r2_abs = float(fit_abs.r2) if (fit_abs is not None and np.isfinite(float(fit_abs.r2))) else None

        # --- Functional activity (new left panel)
        y0_func_init = None
        if func.size > 0 and np.any(np.isfinite(func)):
            first_func_idx = np.where(np.isfinite(func))[0]
            if first_func_idx.size > 0:
                y0_func_init = float(func[first_func_idx[0]])
        fit_func = fit_exponential_decay(t, func, y0=y0_func_init, min_points=4)
        use_exp_func = bool(fit_func is not None and np.isfinite(float(fit_func.r2)) and float(fit_func.r2) >= 0.70)

        # --- REA (right panel)
        # REA is anchored at 100% for heat=0 by definition, so y0 is constrained to 100.
        y0_rea_init = None
        if rea.size > 0 and np.isfinite(float(rea[0])):
            y0_rea_init = float(rea[0])  # Use first point as initial guess if available
        else:
            y0_rea_init = 100.0  # Default initial guess for REA

        fit_rea = fit_exponential_decay(
            t,
            rea,
            y0=y0_rea_init,
            fixed_y0=100.0,
            min_points=4,
            t50_definition=t50_definition,
        )
        use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
        r2_rea = float(fit_rea.r2) if (fit_rea is not None and np.isfinite(float(fit_rea.r2))) else None
        
        # Debug output removed for memory optimization
        
        # REA is defined as % of heat=0 baseline; keep y0 anchored at 100 for t50 thresholding.
        y0_rea_for_t50 = 100.0
        t50_target_rea = t50_target_rea_percent(y0_rea_for_t50, t50_definition=t50_definition)
        t50_lin = t50_linear(
            t,
            rea,
            y0=y0_rea_for_t50,
            target_frac=0.5,
            target_value=(50.0 if t50_definition == T50_DEFINITION_REA50 else None),
        )
        t50_model = fit_rea.t50 if (fit_rea is not None and use_exp_rea) else None
        rea_at_20 = value_at_time_linear(t, rea, at_time_min=20.0)
        # Keep 'fit' variable for backward compatibility in t50_rows.append
        fit = fit_rea

        # --- One figure: Absolute (left), REA (center), Functional (right)
        # STIX fontset for mathtext (R², t_{50}, e^{-kt}) so sub/superscripts render correctly
        _style = {**apply_paper_style(), "mathtext.fontset": "stix"}
        with plt.rc_context(_style):
            fig, (ax_left, ax_right, ax_func) = plt.subplots(1, 3, figsize=(10.4, 2.8))

            # Right: Functional activity ratio (abs / GOx reference)
            ax_func.scatter(
                t,
                func,
                s=12,
                color=color,
                edgecolors="0.2",
                linewidths=0.4,
                alpha=1.0,
                zorder=30,
                clip_on=False,
            )
            if gox_ref_t.size == 0 or gox_ref_y.size == 0:
                info_text_func = _info_box_text(r"\mathrm{ref\ unavailable}", None)
            elif use_exp_func and fit_func is not None:
                if fit_func.model == "exp":
                    func_rhs = _format_exp_rhs_simple(float(fit_func.y0), float(fit_func.k))
                else:
                    c = float(fit_func.c) if fit_func.c is not None else 0.0
                    func_rhs = _format_exp_rhs_plateau(c, float(fit_func.y0), float(fit_func.k))
                _draw_fit_with_extension(ax_func, fit_func, t, color, use_dashed_main=False, preserve_gray=is_gox_per_polymer)
                info_text_func = _info_box_text(func_rhs, float(fit_func.r2))
            else:
                ax_func.plot(t, func, color=color, linewidth=0.8, alpha=0.85, zorder=8, clip_on=False)
                info_text_func = _info_box_text(r"\mathrm{polyline}", None)
            txt_func = ax_func.annotate(
                info_text_func,
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                multialignment="left",
                fontsize=8.0,
                bbox=dict(
                    boxstyle=f"round,pad={info_pad}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.95,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_func.get_bbox_patch() is not None:
                txt_func.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())
            ax_func.set_title(f"{pid_label} | Functional (vs GOx)")
            ax_func.set_xlabel("Heat time (min)")
            ax_func.set_ylabel("Functional ratio (-)")
            ax_func.tick_params(axis="x", which="both", length=0, labelsize=6)
            ax_func.tick_params(axis="y", which="both", length=0, labelsize=6)
            ax_func.set_xlim(0.0, 62.5)
            if func.size > 0 and np.any(np.isfinite(func)):
                func_finite = func[np.isfinite(func)]
                fmax = float(np.max(func_finite))
                fmin = float(np.min(func_finite))
                fmargin = (fmax - fmin) * 0.05 if fmax > fmin else (fmax * 0.05 if fmax > 0 else 0.1)
                y_top_func = fmax + fmargin
                if not np.isfinite(y_top_func) or y_top_func <= 0.0:
                    y_top_func = 1.0
                ax_func.set_ylim(0.0, y_top_func)
            else:
                ax_func.set_ylim(0.0, 1.0)
            ax_func.spines["top"].set_visible(False)
            ax_func.spines["right"].set_visible(False)
            ax_func.spines["left"].set_visible(True)
            ax_func.spines["left"].set_color("0.7")
            ax_func.spines["left"].set_zorder(-10)
            ax_func.spines["bottom"].set_visible(True)
            ax_func.spines["bottom"].set_color("0.7")
            ax_func.spines["bottom"].set_zorder(-10)

            # Left: Absolute activity
            # Set zorder high so points appear in front of axes (especially heat time 0 points)
            # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
            # alpha=1.0 for fully opaque plots
            ax_left.scatter(t, aa, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=30, clip_on=False)
            if use_exp_abs:
                if fit_abs.model == "exp":
                    abs_rhs = _format_exp_rhs_simple(float(fit_abs.y0), float(fit_abs.k))
                else:
                    c = float(fit_abs.c) if fit_abs.c is not None else 0.0
                    abs_rhs = _format_exp_rhs_plateau(c, float(fit_abs.y0), float(fit_abs.k))
                _draw_fit_with_extension(ax_left, fit_abs, t, color, use_dashed_main=False, preserve_gray=is_gox_per_polymer)
                info_text_left = _info_box_text(abs_rhs, float(fit_abs.r2))
            else:
                ax_left.plot(t, aa, color=color, linewidth=0.8, alpha=0.85, zorder=8, clip_on=False)
                info_text_left = _info_box_text(r"\mathrm{polyline}", None)
            txt_left = ax_left.annotate(
                info_text_left,
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                multialignment="left",
                fontsize=8.0,
                bbox=dict(
                    boxstyle=f"round,pad={info_pad}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.95,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_left.get_bbox_patch() is not None:
                txt_left.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())
            ax_left.set_title(f"{pid_label} | Absolute activity")
            ax_left.set_xlabel("Heat time (min)")
            ax_left.set_ylabel("Absolute activity (a.u./s)")
            # Keep tick labels but hide tick lines
            ax_left.tick_params(axis="x", which="both", length=0, labelsize=6)
            ax_left.tick_params(axis="y", which="both", length=0, labelsize=6)
            # Set limits with careful margin to prevent points from touching frame
            # Marker size s=12 (area in points^2) ≈ radius ~1.95pt, edge linewidth=0.4pt
            # Total marker radius ≈ 2.35pt. At figsize=7.0in, 60min range ≈ 504pt
            # So 1min ≈ 8.4pt. Need margin ≥ 2.35/8.4 ≈ 0.28min, use 2.5min for safety
            # Also check if data points are exactly at 0 or 60
            if t.size > 0:
                t_min = float(np.min(t[np.isfinite(t)])) if np.any(np.isfinite(t)) else 0.0
                t_max = float(np.max(t[np.isfinite(t)])) if np.any(np.isfinite(t)) else 60.0
                # Add extra margin if points are at boundaries
                x_margin_left = 2.5 if np.any(np.isclose(t, 0.0, atol=0.1)) else 2.5
                x_margin_right = 2.5 if np.any(np.isclose(t, 60.0, atol=0.1)) else 2.5
            else:
                x_margin_left = 2.5
                x_margin_right = 2.5
            # Calculate y_margin_abs using only finite values
            if aa.size > 0 and np.any(np.isfinite(aa)):
                aa_finite = aa[np.isfinite(aa)]
                aa_max = np.max(aa_finite)
                aa_min = np.min(aa_finite)
                if aa_max > aa_min:
                    y_margin_abs = (aa_max - aa_min) * 0.05
                else:
                    y_margin_abs = aa_max * 0.05 if aa_max > 0 else 1.0
            else:
                y_margin_abs = 1.0
            # Set x-axis to start at 0 so x and y axes intersect at (0, 0)
            ax_left.set_xlim(0.0, 60 + x_margin_right)
            if aa.size > 0:
                y_min_abs = float(np.min(aa[np.isfinite(aa)])) if np.any(np.isfinite(aa)) else 0.0
                y_max_abs = float(np.max(aa[np.isfinite(aa)])) if np.any(np.isfinite(aa)) else 1.0
                y_top_abs = y_max_abs + y_margin_abs
                # Ensure y_top_abs is finite and positive
                if not np.isfinite(y_top_abs) or y_top_abs <= 0:
                    y_top_abs = 1.0
                ax_left.set_ylim(0.0, y_top_abs)  # Start y-axis at 0
            # Hide top and right spines (keep only x-axis and y-axis)
            # Set after limits to ensure it takes effect
            ax_left.spines["top"].set_visible(False)
            ax_left.spines["right"].set_visible(False)
            # Ensure left and bottom spines remain visible, set color to light gray, zorder low so axes are behind data
            ax_left.spines["left"].set_visible(True)
            ax_left.spines["left"].set_color("0.7")  # Light gray
            ax_left.spines["left"].set_zorder(-10)  # Behind data points
            ax_left.spines["bottom"].set_visible(True)
            ax_left.spines["bottom"].set_color("0.7")  # Light gray
            ax_left.spines["bottom"].set_zorder(-10)  # Behind data points

            # Center: REA (%)
            # Set zorder high so points appear in front of axes (especially heat time 0 points)
            # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
            # alpha=1.0 for fully opaque plots
            ax_right.scatter(t, rea, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=30, clip_on=False)
            if use_exp_rea and fit_rea is not None:
                if fit_rea.model == "exp":
                    rea_rhs = _format_exp_rhs_simple(float(fit_rea.y0), float(fit_rea.k))
                else:
                    c = float(fit_rea.c) if fit_rea.c is not None else 0.0
                    rea_rhs = _format_exp_rhs_plateau(c, float(fit_rea.y0), float(fit_rea.k))
                # Draw fitted curve for REA in both exp and plateau cases
                _draw_fit_with_extension(ax_right, fit_rea, t, color, use_dashed_main=False, preserve_gray=is_gox_per_polymer)
                t50_show = t50_model if t50_model is not None else t50_lin
                info_text_right = _info_box_text(rea_rhs, float(fit_rea.r2), t50=t50_show)
            else:
                ax_right.plot(t, rea, color=color, linewidth=0.8, alpha=0.85, zorder=8, clip_on=False)
                t50_show = t50_lin
                info_text_right = _info_box_text(r"\mathrm{polyline}", None, t50=t50_show)
            txt_right = ax_right.annotate(
                info_text_right,
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                multialignment="left",
                fontsize=8.0,
                bbox=dict(
                    boxstyle=f"round,pad={info_pad}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.95,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_right.get_bbox_patch() is not None:
                txt_right.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())
            ax_right.set_title(f"{pid_label} | REA (%)")
            ax_right.set_xlabel("Heat time (min)")
            ax_right.set_ylabel("REA (%)")
            # Keep tick labels but hide tick lines
            ax_right.tick_params(axis="x", which="both", length=0, labelsize=6)
            ax_right.tick_params(axis="y", which="both", length=0, labelsize=6)
            # Set limits with careful margin to prevent points from touching frame
            # Marker size s=12 (area in points^2) ≈ radius ~1.95pt, edge linewidth=0.4pt
            # Total marker radius ≈ 2.35pt. At figsize=7.0in, 60min range ≈ 504pt
            # So 1min ≈ 8.4pt. Need margin ≥ 2.35/8.4 ≈ 0.28min, use 2.5min for safety
            # Also check if data points are exactly at 0 or 60
            if t.size > 0:
                t_min = float(np.min(t[np.isfinite(t)])) if np.any(np.isfinite(t)) else 0.0
                t_max = float(np.max(t[np.isfinite(t)])) if np.any(np.isfinite(t)) else 60.0
                # Add extra margin if points are at boundaries
                x_margin_left = 2.5 if np.any(np.isclose(t, 0.0, atol=0.1)) else 2.5
                x_margin_right = 2.5 if np.any(np.isclose(t, 60.0, atol=0.1)) else 2.5
            else:
                x_margin_left = 2.5
                x_margin_right = 2.5
            # Calculate y_margin_rea using only finite values
            if rea.size > 0 and np.any(np.isfinite(rea)):
                rea_finite = rea[np.isfinite(rea)]
                rea_max = np.max(rea_finite)
                rea_min = np.min(rea_finite)
                if rea_max > rea_min:
                    y_margin_rea = (rea_max - rea_min) * 0.05
                else:
                    y_margin_rea = rea_max * 0.05 if rea_max > 0 else 2.0
            else:
                y_margin_rea = 2.0
            # Set x-axis to start at 0 so x and y axes intersect at (0, 0)
            ax_right.set_xlim(0.0, 60 + x_margin_right)
            if rea.size > 0:
                y_min_rea = float(np.min(rea[np.isfinite(rea)])) if np.any(np.isfinite(rea)) else 0.0
                y_max_rea = float(np.max(rea[np.isfinite(rea)])) if np.any(np.isfinite(rea)) else 100.0
                y_top_rea = y_max_rea + y_margin_rea
                # Ensure y_top_rea is finite and positive
                if not np.isfinite(y_top_rea) or y_top_rea <= 0:
                    y_top_rea = 100.0
                ax_right.set_ylim(0.0, y_top_rea)  # Start y-axis at 0
            # Hide top and right spines (keep only x-axis and y-axis)
            # Set after limits to ensure it takes effect
            ax_right.spines["top"].set_visible(False)
            ax_right.spines["right"].set_visible(False)
            # Ensure left and bottom spines remain visible, set color to light gray, zorder low so axes are behind data
            ax_right.spines["left"].set_visible(True)
            ax_right.spines["left"].set_color("0.7")  # Light gray
            ax_right.spines["left"].set_zorder(-10)  # Behind data points
            ax_right.spines["bottom"].set_visible(True)
            ax_right.spines["bottom"].set_color("0.7")  # Light gray
            ax_right.spines["bottom"].set_zorder(-10)  # Behind data points
            
            # Draw t50 intersection lines: left and bottom only (not right and top)
            # Intersect at the center of the fitted curve (not at the bottom edge)
            # Must be drawn AFTER set_ylim to get correct y-axis limits
            if t50_show is not None and np.isfinite(float(t50_show)) and float(t50_show) > 0.0:
                t50_val = float(t50_show)
                # Get y-axis limits to draw line from bottom (after set_ylim)
                ylim = ax_right.get_ylim()
                y_bottom = ylim[0]
                # Keep the reference line at REA=50 for rea50 mode.
                if t50_definition == T50_DEFINITION_REA50:
                    y_at_t50 = 50.0
                elif use_exp_rea and fit_rea is not None:
                    y_at_t50 = float(_eval_fit_curve(fit_rea, np.array([t50_val]))[0])
                else:
                    y_at_t50 = float(t50_target_rea)
                # Horizontal line: from left edge (x=0) to t50 intersection (left side only)
                # Use zorder=5 to be behind fitted curve (zorder=8) but still visible
                ax_right.plot([0.0, t50_val], [y_at_t50, y_at_t50], linestyle=(0, (3, 2)), color="0.5", linewidth=0.6, alpha=0.8, zorder=5)
                # Vertical line: from bottom to t50 intersection (bottom side only)
                ax_right.plot([t50_val, t50_val], [y_bottom, y_at_t50], linestyle=(0, (3, 2)), color="0.4", linewidth=0.6, alpha=0.8, zorder=5)

            fig.tight_layout(pad=0.3)
            # Increase left margin to accommodate clip_on=False markers (especially heat_time=0 points)
            fig.subplots_adjust(left=0.08, wspace=0.28)
            # Ensure spines visibility and color after tight_layout (per_polymer: only x-axis and y-axis, light gray)
            ax_func.spines["top"].set_visible(False)
            ax_func.spines["right"].set_visible(False)
            ax_func.spines["left"].set_visible(True)
            ax_func.spines["left"].set_color("0.7")
            ax_func.spines["left"].set_zorder(-10)
            ax_func.spines["bottom"].set_visible(True)
            ax_func.spines["bottom"].set_color("0.7")
            ax_func.spines["bottom"].set_zorder(-10)
            ax_left.spines["top"].set_visible(False)
            ax_left.spines["right"].set_visible(False)
            ax_left.spines["left"].set_visible(True)
            ax_left.spines["left"].set_color("0.7")  # Light gray
            ax_left.spines["left"].set_zorder(-10)  # Behind data points
            ax_left.spines["bottom"].set_visible(True)
            ax_left.spines["bottom"].set_color("0.7")  # Light gray
            ax_left.spines["bottom"].set_zorder(-10)  # Behind data points
            ax_right.spines["top"].set_visible(False)
            ax_right.spines["right"].set_visible(False)
            ax_right.spines["left"].set_visible(True)
            ax_right.spines["left"].set_color("0.7")  # Light gray
            ax_right.spines["left"].set_zorder(-10)  # Behind data points
            ax_right.spines["bottom"].set_visible(True)
            ax_right.spines["bottom"].set_color("0.7")  # Light gray
            ax_right.spines["bottom"].set_zorder(-10)  # Behind data points
            
            # Ensure spines zorder is set correctly before saving (savefig may reset it)
            # Set zorder very low so axes are definitely behind data points
            ax_func.spines["left"].set_zorder(-10)
            ax_func.spines["bottom"].set_zorder(-10)
            ax_left.spines["left"].set_zorder(-10)
            ax_left.spines["bottom"].set_zorder(-10)
            ax_right.spines["left"].set_zorder(-10)
            ax_right.spines["bottom"].set_zorder(-10)
            
            out_path = out_per_polymer / f"{stem}__{run_id}.png"
            fig.savefig(
                out_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig)
            # Memory optimization: periodically force garbage collection after plot generation
            if len(t50_rows) % 5 == 0:  # Every 5 polymers
                gc.collect()

        t50_rows.append(
            {
                "run_id": run_id,
                "polymer_id": str(pid),
                "polymer_label": pid_label,
                "n_points": int(len(g)),
                "abs_activity_at_0": abs_activity_at_0,
                "abs_activity_at_20": abs_activity_at_20,
                "functional_activity_at_20_rel": func_at_20,
                "gox_abs_activity_at_20_ref": float(gox_ref_at_20) if np.isfinite(gox_ref_at_20) else np.nan,
                "functional_reference_source": gox_ref_source,
                "functional_reference_round_id": gox_ref_round_id,
                "functional_reference_run_id": gox_ref_run_id,
                "y0_REA_percent": float(y0_rea_for_t50),
                "t50_definition": str(t50_definition),
                "t50_target_rea_percent": float(t50_target_rea),
                "t50_linear_min": float(t50_lin) if t50_lin is not None else np.nan,
                "t50_exp_min": float(t50_model) if t50_model is not None else np.nan,
                "rea_at_20_percent": float(rea_at_20) if rea_at_20 is not None else np.nan,
                "fit_model": fit_rea.model if (fit_rea is not None and use_exp_rea) else "",
                "fit_k_per_min": float(fit_rea.k) if (fit_rea is not None and use_exp_rea) else np.nan,
                "fit_tau_min": float(1.0 / fit_rea.k) if (fit_rea is not None and use_exp_rea and fit_rea.k > 0) else np.nan,
                "fit_plateau": float(fit_rea.c) if (fit_rea is not None and use_exp_rea and fit_rea.c is not None) else np.nan,
                "fit_r2": float(fit_rea.r2) if (fit_rea is not None and use_exp_rea) else np.nan,
                "rea_connector": "exp" if use_exp_rea else "polyline",
            }
        )

    t50_df = pd.DataFrame(t50_rows, columns=t50_columns)
    # out_t50_dir is already created earlier (before out_per_polymer)
    t50_path = out_t50_dir / f"t50__{run_id}.csv"
    t50_df.to_csv(t50_path, index=False)

    # --- All polymers comparison plots (overlay all polymers in one figure: Absolute, REA, Functional)
    # Generate two versions: one with include_in_all_polymers=True only (default), one with all polymers (for debugging)
    import gc
    _style = {**apply_paper_style(), "mathtext.fontset": "stix"}
    
    # Version 1: Filtered (only include_in_all_polymers=True) - this is the default/main output
    # Filter out polymers with include_in_all_polymers=False
    # After _parse_bool_flag, the column should be boolean, but ensure it's bool type for comparison
    mask = df["include_in_all_polymers"].astype(bool) == True
    df_filtered = df[mask].copy()
    polymer_ids_filtered = sorted(df_filtered["polymer_id"].astype(str).unique().tolist()) if not df_filtered.empty else []
    
    # Debug: print excluded polymers (can be removed later)
    excluded_polymers = sorted(df[~mask]["polymer_id"].astype(str).unique().tolist()) if not df[~mask].empty else []
    if excluded_polymers:
        print(f"Info: Excluding polymers from all_polymers plot (include_in_all_polymers=False): {excluded_polymers}")
    
    # Version 2: All polymers (for comparison/debugging) - only generated if different from filtered
    df_all = df.copy()
    polymer_ids_all = sorted(df_all["polymer_id"].astype(str).unique().tolist())
    
    def _plot_all_polymers_subplot(df_plot: pd.DataFrame, polymer_ids_plot: list[str], suffix: str = "") -> None:
        """Helper function to plot all polymers comparison."""
        if not polymer_ids_plot:
            return
        with plt.rc_context(_style):
            fig_all, (ax_abs, ax_rea, ax_func) = plt.subplots(1, 3, figsize=(13.2, 3.5))
            n_polymers = len(polymer_ids_plot)

            # Left: Absolute activity
            for pid, g in df_plot.groupby("polymer_id", sort=False):
                g = g.sort_values("heat_min").reset_index(drop=True)
                t = g["heat_min"].to_numpy(dtype=float)
                aa = g["abs_activity"].to_numpy(dtype=float)
                # GOx always uses gray color
                pid_str = str(pid)
                if pid_str.upper() == "GOX":
                    color = "#808080"  # Medium gray
                else:
                    color = cmap.get(pid_str, "#0072B2")
                pid_label = safe_label(pid_str)

                # Plot data points (high zorder so points appear in front of axes, especially heat time 0)
                # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
                # alpha=1.0 for fully opaque plots
                ax_abs.scatter(t, aa, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)

                # Plot fit curve: same color correspondence as per_polymer (polymer_id → plot color, curve = fluorescent or gray for GOx)
                # REA fit keeps y0 constrained at 100.
                y0_abs_init = float(aa[0]) if aa.size > 0 and np.isfinite(float(aa[0])) else None
                fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
                is_gox = pid_str.upper() == "GOX"
                if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                    _draw_fit_with_extension(ax_abs, fit_abs, t, color, use_dashed_main=True, preserve_gray=is_gox)
                else:
                    ax_abs.plot(t, aa, color=color, linewidth=0.7, alpha=0.6, zorder=8, clip_on=False)

            ax_abs.set_xlabel("Heat time (min)")
            ax_abs.set_ylabel("Absolute activity (a.u./s)")
            ax_abs.set_xticks(HEAT_TICKS_0_60)
            # Set x-axis to start at 0 so x and y axes intersect at (0, 0)
            # Calculate margin for x-axis (same as per_polymer)
            x_margin_right = 2.5
            ax_abs.set_xlim(0.0, 60 + x_margin_right)
            # Calculate y-axis limits based on data
            all_aa = []
            for pid, g in df_plot.groupby("polymer_id", sort=False):
                aa = g["abs_activity"].to_numpy(dtype=float)
                all_aa.extend(aa[np.isfinite(aa)].tolist())
            if all_aa:
                y_min_abs = float(np.min(all_aa))
                y_max_abs = float(np.max(all_aa))
                y_margin_abs = (y_max_abs - y_min_abs) * 0.05 if y_max_abs > y_min_abs else y_max_abs * 0.05 if y_max_abs > 0 else 1.0
                y_top_abs = y_max_abs + y_margin_abs
                ax_abs.set_ylim(0.0, y_top_abs)  # Start y-axis at 0
            else:
                ax_abs.set_ylim(0.0, 1.0)
            ax_abs.spines["top"].set_visible(False)
            ax_abs.spines["right"].set_visible(False)
            ax_abs.spines["left"].set_visible(True)
            ax_abs.spines["left"].set_color("0.7")  # Light gray
            ax_abs.spines["left"].set_zorder(-10)  # Behind data points
            ax_abs.spines["bottom"].set_visible(True)
            ax_abs.spines["bottom"].set_color("0.7")  # Light gray
            ax_abs.spines["bottom"].set_zorder(-10)  # Behind data points

            # Center: REA (%)
            for pid, g in df_plot.groupby("polymer_id", sort=False):
                g = g.sort_values("heat_min").reset_index(drop=True)
                t = g["heat_min"].to_numpy(dtype=float)
                rea = g["REA_percent"].to_numpy(dtype=float)
                # GOx always uses gray color
                pid_str = str(pid)
                if pid_str.upper() == "GOX":
                    color = "#808080"  # Medium gray
                else:
                    color = cmap.get(pid_str, "#0072B2")
                pid_label = safe_label(pid_str)

                # Plot data points (high zorder so points appear in front of axes, especially heat time 0)
                # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
                # alpha=1.0 for fully opaque plots
                ax_rea.scatter(t, rea, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)

                # Plot fit curve: same color correspondence as per_polymer (polymer_id → plot color, curve = fluorescent or gray for GOx)
                # REA fit keeps y0 constrained at 100.
                y0_rea_init = float(rea[0]) if rea.size > 0 and np.isfinite(float(rea[0])) else 100.0
                fit_rea = fit_exponential_decay(
                    t,
                    rea,
                    y0=y0_rea_init,
                    fixed_y0=100.0,
                    min_points=4,
                    t50_definition=t50_definition,
                )
                use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
                is_gox = pid_str.upper() == "GOX"
                if use_exp_rea:
                    _draw_fit_with_extension(ax_rea, fit_rea, t, color, use_dashed_main=True, preserve_gray=is_gox)
                else:
                    ax_rea.plot(t, rea, color=color, linewidth=0.7, alpha=0.6, zorder=8, clip_on=False)

            ax_rea.set_xlabel("Heat time (min)")
            ax_rea.set_ylabel("REA (%)")
            ax_rea.set_xticks(HEAT_TICKS_0_60)
            # Set x-axis to start at 0 so x and y axes intersect at (0, 0)
            # Calculate margin for x-axis (same as per_polymer)
            x_margin_right = 2.5
            ax_rea.set_xlim(0.0, 60 + x_margin_right)
            # Calculate y-axis limits based on data
            all_rea = []
            for pid, g in df_plot.groupby("polymer_id", sort=False):
                rea = g["REA_percent"].to_numpy(dtype=float)
                all_rea.extend(rea[np.isfinite(rea)].tolist())
            if all_rea:
                y_min_rea = float(np.min(all_rea))
                y_max_rea = float(np.max(all_rea))
                y_margin_rea = (y_max_rea - y_min_rea) * 0.05 if y_max_rea > y_min_rea else y_max_rea * 0.05 if y_max_rea > 0 else 2.0
                y_top_rea = y_max_rea + y_margin_rea
                ax_rea.set_ylim(0.0, y_top_rea)  # Start y-axis at 0
            else:
                ax_rea.set_ylim(0.0, 100.0)
            ax_rea.spines["top"].set_visible(False)
            ax_rea.spines["right"].set_visible(False)
            ax_rea.spines["left"].set_visible(True)
            ax_rea.spines["left"].set_color("0.7")  # Light gray
            ax_rea.spines["left"].set_zorder(-10)  # Behind data points
            ax_rea.spines["bottom"].set_visible(True)
            ax_rea.spines["bottom"].set_color("0.7")  # Light gray
            ax_rea.spines["bottom"].set_zorder(-10)  # Behind data points

            # Right: Functional activity ratio (abs / GOx reference)
            for pid, g in df_plot.groupby("polymer_id", sort=False):
                g = g.sort_values("heat_min").reset_index(drop=True)
                t = g["heat_min"].to_numpy(dtype=float)
                aa = g["abs_activity"].to_numpy(dtype=float)
                pid_str = str(pid)
                if pid_str.upper() == "GOX":
                    color = "#808080"  # Medium gray
                else:
                    color = cmap.get(pid_str, "#0072B2")
                pid_label = safe_label(pid_str)

                func = np.full_like(aa, np.nan, dtype=float)
                if gox_ref_t.size > 0 and gox_ref_y.size > 0:
                    denom = np.array(
                        [
                            value_at_time_linear(gox_ref_t, gox_ref_y, at_time_min=float(tt))
                            if np.isfinite(float(tt)) else np.nan
                            for tt in t
                        ],
                        dtype=float,
                    )
                    ok_func = np.isfinite(aa) & np.isfinite(denom) & (denom > 0.0)
                    func[ok_func] = aa[ok_func] / denom[ok_func]

                ax_func.scatter(
                    t,
                    func,
                    s=10,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.3,
                    alpha=1.0,
                    zorder=30,
                    label=pid_label,
                    clip_on=False,
                )

                y0_func_init = None
                if func.size > 0 and np.any(np.isfinite(func)):
                    first_func_idx = np.where(np.isfinite(func))[0]
                    if first_func_idx.size > 0:
                        y0_func_init = float(func[first_func_idx[0]])
                fit_func = fit_exponential_decay(t, func, y0=y0_func_init, min_points=4)
                use_exp_func = bool(fit_func is not None and np.isfinite(float(fit_func.r2)) and float(fit_func.r2) >= 0.70)
                is_gox = pid_str.upper() == "GOX"
                if use_exp_func:
                    _draw_fit_with_extension(ax_func, fit_func, t, color, use_dashed_main=True, preserve_gray=is_gox)
                else:
                    ax_func.plot(t, func, color=color, linewidth=0.7, alpha=0.6, zorder=8, clip_on=False)

            ax_func.set_xlabel("Heat time (min)")
            ax_func.set_ylabel("Functional ratio (-)")
            ax_func.set_xticks(HEAT_TICKS_0_60)
            ax_func.set_xlim(0.0, 62.5)
            all_func = []
            if gox_ref_t.size > 0 and gox_ref_y.size > 0:
                for _, g in df_plot.groupby("polymer_id", sort=False):
                    t = g["heat_min"].to_numpy(dtype=float)
                    aa = g["abs_activity"].to_numpy(dtype=float)
                    denom = np.array(
                        [
                            value_at_time_linear(gox_ref_t, gox_ref_y, at_time_min=float(tt))
                            if np.isfinite(float(tt)) else np.nan
                            for tt in t
                        ],
                        dtype=float,
                    )
                    func = np.full_like(aa, np.nan, dtype=float)
                    ok_func = np.isfinite(aa) & np.isfinite(denom) & (denom > 0.0)
                    func[ok_func] = aa[ok_func] / denom[ok_func]
                    all_func.extend(func[np.isfinite(func)].tolist())
            if all_func:
                y_min_func = float(np.min(all_func))
                y_max_func = float(np.max(all_func))
                y_margin_func = (y_max_func - y_min_func) * 0.05 if y_max_func > y_min_func else y_max_func * 0.05 if y_max_func > 0 else 0.1
                y_top_func = y_max_func + y_margin_func
                ax_func.set_ylim(0.0, y_top_func if np.isfinite(y_top_func) and y_top_func > 0 else 1.0)
            else:
                ax_func.set_ylim(0.0, 1.0)
            ax_func.spines["top"].set_visible(False)
            ax_func.spines["right"].set_visible(False)
            ax_func.spines["left"].set_visible(True)
            ax_func.spines["left"].set_color("0.7")  # Light gray
            ax_func.spines["left"].set_zorder(-10)  # Behind data points
            ax_func.spines["bottom"].set_visible(True)
            ax_func.spines["bottom"].set_color("0.7")  # Light gray
            ax_func.spines["bottom"].set_zorder(-10)  # Behind data points

            # Legend: place on functional panel (rightmost)
            if n_polymers > 8:
                # Place legend outside on the right
                ax_func.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    fontsize=6,
                    ncol=1,
                    columnspacing=0.5,
                    handlelength=1.0,
                    handletextpad=0.3,
                )
                fig_all.tight_layout(rect=[0, 0, 0.88, 1])
            else:
                # Place legend inside (upper right of functional panel)
                ax_func.legend(
                    loc="upper right",
                    frameon=True,
                    fontsize=6,
                    ncol=1,
                    columnspacing=0.5,
                    handlelength=1.0,
                    handletextpad=0.3,
                )
                fig_all.tight_layout(pad=0.3)
                # Increase left margin to accommodate clip_on=False markers (especially heat_time=0 points)
                fig_all.subplots_adjust(left=0.08, wspace=0.28)
            
            # Ensure spines zorder and color are set correctly after tight_layout (savefig may reset it)
            # Set zorder very low so axes are definitely behind data points
            ax_abs.spines["left"].set_color("0.7")  # Light gray
            ax_abs.spines["left"].set_zorder(-10)
            ax_abs.spines["bottom"].set_color("0.7")  # Light gray
            ax_abs.spines["bottom"].set_zorder(-10)
            ax_rea.spines["left"].set_color("0.7")  # Light gray
            ax_rea.spines["left"].set_zorder(-10)
            ax_rea.spines["bottom"].set_color("0.7")  # Light gray
            ax_rea.spines["bottom"].set_zorder(-10)
            ax_func.spines["left"].set_color("0.7")  # Light gray
            ax_func.spines["left"].set_zorder(-10)
            ax_func.spines["bottom"].set_color("0.7")  # Light gray
            ax_func.spines["bottom"].set_zorder(-10)

            # Save combined figure
            out_all_path = out_fit_dir / f"all_polymers{suffix}__{run_id}.png"
            fig_all.savefig(
                out_all_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig_all)
            # Memory optimization: force garbage collection after large plot generation
            gc.collect()
    
    # Generate both versions: filtered (default) and all (for debugging if different)
    _plot_all_polymers_subplot(df_filtered, polymer_ids_filtered, suffix="")  # Default: only include_in_all_polymers=True
    if polymer_ids_filtered != polymer_ids_all:  # Only generate "all" version if different from filtered
        _plot_all_polymers_subplot(df_all, polymer_ids_all, suffix="_all")
    
    # Generate custom pair plot if all_polymers_pair=True polymers are specified
    # Only include polymers with all_polymers_pair=True (exclude all others)
    if "all_polymers_pair" in df.columns and df["all_polymers_pair"].any():
        # Ensure boolean comparison works correctly
        mask_pair = df["all_polymers_pair"].astype(bool) == True
        df_pair = df[mask_pair].copy()
        polymer_ids_pair = sorted(df_pair["polymer_id"].astype(str).unique().tolist())
        if polymer_ids_pair:
            _plot_all_polymers_subplot(df_pair, polymer_ids_pair, suffix="_pair")
            print(f"Info: Generated all_polymers_pair plot with {len(polymer_ids_pair)} polymer(s): {polymer_ids_pair}")

    # Remove old separate files if they exist
    for old_file in (out_fit_dir / f"all_polymers_abs__{run_id}.png", out_fit_dir / f"all_polymers_rea__{run_id}.png"):
        if old_file.exists():
            try:
                old_file.unlink()
            except Exception:
                pass

    # --- Representative 4 polymers comparison plot (GOx, PMPC, t50 top, t50 bottom)
    # Fixed polymers: GOx and PMPC (always included if present)
    # Additional: t50 highest and lowest polymers (excluding GOx and PMPC)
    import gc
    representative_pids = []
    
    # Fixed: Always include GOx if present
    if "GOx" in polymer_ids:
        representative_pids.append("GOx")
    
    # Fixed: Always include PMPC if present
    if "PMPC" in polymer_ids:
        representative_pids.append("PMPC")
    
    # Find t50 top and bottom from remaining polymers (exclude GOx, PMPC, and include_in_all_polymers=False)
    # Filter out GOx, PMPC, and polymers with include_in_all_polymers=False
    excluded_for_t50_selection = {"GOx", "PMPC"}
    if "include_in_all_polymers" in df.columns:
        # Ensure boolean comparison works correctly
        mask_excluded = df["include_in_all_polymers"].astype(bool) == False
        excluded_pids = set(df[mask_excluded]["polymer_id"].astype(str).unique())
        excluded_for_t50_selection.update(excluded_pids)
    
    # Get best t50 per polymer (use exp if available, otherwise linear)
    t50_per_polymer = []
    for pid in t50_df["polymer_id"].unique():
        # Skip GOx, PMPC, and excluded polymers
        if pid in excluded_for_t50_selection:
            continue
        
        pid_df = t50_df[t50_df["polymer_id"] == pid]
        t50_val = None
        if pid_df["t50_exp_min"].notna().any() and np.isfinite(pid_df["t50_exp_min"]).any():
            t50_val = float(pid_df["t50_exp_min"].iloc[0])
        elif pid_df["t50_linear_min"].notna().any() and np.isfinite(pid_df["t50_linear_min"]).any():
            t50_val = float(pid_df["t50_linear_min"].iloc[0])
        if t50_val is not None and np.isfinite(t50_val):
            t50_per_polymer.append((pid, t50_val))
    
    if t50_per_polymer:
        t50_per_polymer.sort(key=lambda x: x[1])
        # Top (highest t50) - add if not already in list
        top_pid = t50_per_polymer[-1][0]
        if top_pid not in representative_pids:
            representative_pids.append(top_pid)
        # Bottom (lowest t50) - add if not already in list
        bottom_pid = t50_per_polymer[0][0]
        if bottom_pid not in representative_pids:
            representative_pids.append(bottom_pid)
    
    # Create representative plot if we have at least one polymer
    if representative_pids:
        _style = {**apply_paper_style(), "mathtext.fontset": "stix"}
        with plt.rc_context(_style):
            fig_rep, (ax_abs_rep, ax_rea_rep) = plt.subplots(1, 2, figsize=(10.0, 3.5))
            
            # Left: Absolute activity
            for pid in representative_pids:
                if pid not in df["polymer_id"].values:
                    continue
                g = df[df["polymer_id"] == pid].sort_values("heat_min").reset_index(drop=True)
                t = g["heat_min"].to_numpy(dtype=float)
                aa = g["abs_activity"].to_numpy(dtype=float)
                # GOx always uses gray color (exact match, case-insensitive)
                pid_str = str(pid).strip()
                if pid_str.upper() == "GOX" or pid_str == "GOx":
                    color = "#808080"  # Medium gray
                else:
                    color = cmap.get(pid_str, "#0072B2")
                pid_label = safe_label(pid_str)
                
                # Plot data points (high zorder so points appear in front of axes, especially heat time 0)
                # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
                # alpha=1.0 for fully opaque plots
                ax_abs_rep.scatter(t, aa, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)
                
                # Plot fit curve if available
                y0_abs_init = float(aa[0]) if aa.size > 0 and np.isfinite(float(aa[0])) else None
                fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
                is_gox = pid_str.upper() == "GOX" or pid_str == "GOx"
                if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                    _draw_fit_with_extension(ax_abs_rep, fit_abs, t, color, preserve_gray=is_gox)
                else:
                    ax_abs_rep.plot(t, aa, color=color, linewidth=0.7, alpha=0.6, zorder=8, clip_on=False)
            
            ax_abs_rep.set_xlabel("Heat time (min)")
            ax_abs_rep.set_ylabel("Absolute activity (a.u./s)")
            ax_abs_rep.set_xticks(HEAT_TICKS_0_60)
            ax_abs_rep.set_xlim(-2.5, 62.5)
            ax_abs_rep.set_ylim(bottom=0.0)
            ax_abs_rep.spines["top"].set_visible(False)
            ax_abs_rep.spines["right"].set_visible(False)
            ax_abs_rep.spines["left"].set_visible(True)
            ax_abs_rep.spines["left"].set_color("0.7")  # Light gray
            ax_abs_rep.spines["left"].set_zorder(-1)
            ax_abs_rep.spines["bottom"].set_visible(True)
            ax_abs_rep.spines["bottom"].set_color("0.7")  # Light gray
            ax_abs_rep.spines["bottom"].set_zorder(-1)
            
            # Right: REA (%)
            # Plot data first, then set axis limits based on actual data range
            y_max_rea_rep = 0.0
            
            for pid in representative_pids:
                if pid not in df["polymer_id"].values:
                    continue
                g = df[df["polymer_id"] == pid].sort_values("heat_min").reset_index(drop=True)
                t = g["heat_min"].to_numpy(dtype=float)
                rea = g["REA_percent"].to_numpy(dtype=float)
                # GOx always uses gray color (exact match, case-insensitive)
                pid_str = str(pid).strip()
                if pid_str.upper() == "GOX" or pid_str == "GOx":
                    color = "#808080"  # Medium gray
                else:
                    color = cmap.get(pid_str, "#0072B2")
                pid_label = safe_label(pid_str)
                
                # Track maximum y value for setting y-axis limit
                if rea.size > 0:
                    rea_finite = rea[np.isfinite(rea)]
                    if rea_finite.size > 0:
                        y_max_rea_rep = max(y_max_rea_rep, float(np.max(rea_finite)))
                
                # Plot data points (high zorder so points appear in front of axes, especially heat time 0)
                # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
                # alpha=1.0 for fully opaque plots
                ax_rea_rep.scatter(t, rea, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)
                
                # Plot fit curve if available
                y0_rea_init = float(rea[0]) if rea.size > 0 and np.isfinite(float(rea[0])) else 100.0
                fit_rea = fit_exponential_decay(
                    t,
                    rea,
                    y0=y0_rea_init,
                    fixed_y0=100.0,
                    min_points=4,
                    t50_definition=t50_definition,
                )
                use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
                is_gox = pid_str.upper() == "GOX" or pid_str == "GOx"
                if use_exp_rea:
                    _draw_fit_with_extension(ax_rea_rep, fit_rea, t, color, preserve_gray=is_gox)
                    # Also track maximum from fitted curve
                    if fit_rea is not None:
                        t_eval = np.linspace(0.0, 60.0, 200)
                        y_eval = _eval_fit_curve(fit_rea, t_eval)
                        y_eval_finite = y_eval[np.isfinite(y_eval)]
                        if y_eval_finite.size > 0:
                            y_max_rea_rep = max(y_max_rea_rep, float(np.max(y_eval_finite)))
                else:
                    ax_rea_rep.plot(t, rea, color=color, linewidth=0.7, alpha=0.6, zorder=8, clip_on=False)
            
            # Set axis limits after plotting data
            x_margin_right = 2.5
            ax_rea_rep.set_xlim(0.0, 60.0 + x_margin_right)
            y_top_rea_rep = y_max_rea_rep * 1.1 if y_max_rea_rep > 0 else 100.0
            ax_rea_rep.set_ylim(0.0, y_top_rea_rep)
            
            # Get y_bottom for t50 lines (after setting ylim)
            ylim_rea_rep = ax_rea_rep.get_ylim()
            y_bottom_rea_rep = ylim_rea_rep[0]
            
            # Draw t50 lines after setting axis limits
            for pid in representative_pids:
                if pid not in df["polymer_id"].values:
                    continue
                g = df[df["polymer_id"] == pid].sort_values("heat_min").reset_index(drop=True)
                t = g["heat_min"].to_numpy(dtype=float)
                rea = g["REA_percent"].to_numpy(dtype=float)
                
                # Calculate t50 and draw intersection lines (left and bottom only)
                y0_rea_init = float(rea[0]) if rea.size > 0 and np.isfinite(float(rea[0])) else 100.0
                fit_rea = fit_exponential_decay(
                    t,
                    rea,
                    y0=y0_rea_init,
                    fixed_y0=100.0,
                    min_points=4,
                    t50_definition=t50_definition,
                )
                use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
                t50_model = fit_rea.t50 if (fit_rea is not None and use_exp_rea) else None
                y0_rea_for_t50 = 100.0
                t50_target_rea = t50_target_rea_percent(y0_rea_for_t50, t50_definition=t50_definition)
                t50_lin = t50_linear(
                    t,
                    rea,
                    y0=y0_rea_for_t50,
                    target_frac=0.5,
                    target_value=(50.0 if t50_definition == T50_DEFINITION_REA50 else None),
                )
                t50_show = t50_model if t50_model is not None else t50_lin
                
                if t50_show is not None and np.isfinite(float(t50_show)) and float(t50_show) > 0.0:
                    t50_val = float(t50_show)
                    # Keep the reference line at REA=50 for rea50 mode.
                    if t50_definition == T50_DEFINITION_REA50:
                        y_at_t50 = 50.0
                    elif use_exp_rea and fit_rea is not None:
                        y_at_t50 = float(_eval_fit_curve(fit_rea, np.array([t50_val]))[0])
                    else:
                        y_at_t50 = float(t50_target_rea)
                    # Horizontal line: from left edge (x=0) to t50 intersection (left side only)
                    # Use zorder=5 to be behind fitted curve (zorder=8) but still visible
                    ax_rea_rep.plot([0.0, t50_val], [y_at_t50, y_at_t50], linestyle=(0, (3, 2)), color="0.5", linewidth=0.6, alpha=0.8, zorder=5)
                    # Vertical line: from bottom to t50 intersection (bottom side only)
                    ax_rea_rep.plot([t50_val, t50_val], [y_bottom_rea_rep, y_at_t50], linestyle=(0, (3, 2)), color="0.4", linewidth=0.6, alpha=0.8, zorder=5)
            
            ax_rea_rep.set_xlabel("Heat time (min)")
            ax_rea_rep.set_ylabel("REA (%)")
            ax_rea_rep.set_xticks(HEAT_TICKS_0_60)
            ax_rea_rep.spines["top"].set_visible(False)
            ax_rea_rep.spines["right"].set_visible(False)
            ax_rea_rep.spines["left"].set_visible(True)
            ax_rea_rep.spines["left"].set_color("0.7")  # Light gray
            ax_rea_rep.spines["left"].set_zorder(-1)
            ax_rea_rep.spines["bottom"].set_visible(True)
            ax_rea_rep.spines["bottom"].set_color("0.7")  # Light gray
            ax_rea_rep.spines["bottom"].set_zorder(-1)
            
            # Legend
            n_rep = len(representative_pids)
            if n_rep > 8:
                ax_rea_rep.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    fontsize=6,
                    ncol=1,
                    columnspacing=0.5,
                    handlelength=1.0,
                    handletextpad=0.3,
                )
                fig_rep.tight_layout(rect=[0, 0, 0.88, 1])
                # Increase left margin to accommodate clip_on=False markers (especially heat_time=0 points)
                # Note: rect already sets right margin, so adjust left only
                fig_rep.subplots_adjust(left=0.12)
            else:
                ax_rea_rep.legend(
                    loc="upper right",
                    frameon=True,
                    fontsize=6,
                    ncol=1,
                    columnspacing=0.5,
                    handlelength=1.0,
                    handletextpad=0.3,
                )
                fig_rep.tight_layout(pad=0.3)
                # Increase left margin to accommodate clip_on=False markers (especially heat_time=0 points)
                fig_rep.subplots_adjust(left=0.12)
            
            # Ensure spines zorder and color are set correctly before saving (savefig may reset it)
            # Set zorder very low so axes are definitely behind data points
            ax_abs_rep.spines["left"].set_color("0.7")  # Light gray
            ax_abs_rep.spines["left"].set_zorder(-10)
            ax_abs_rep.spines["bottom"].set_color("0.7")  # Light gray
            ax_abs_rep.spines["bottom"].set_zorder(-10)
            ax_rea_rep.spines["left"].set_color("0.7")  # Light gray
            ax_rea_rep.spines["left"].set_zorder(-10)
            ax_rea_rep.spines["bottom"].set_color("0.7")  # Light gray
            ax_rea_rep.spines["bottom"].set_zorder(-10)
            
            # Save representative figure
            out_rep_path = out_fit_dir / f"representative_4__{run_id}.png"
            fig_rep.savefig(
                out_rep_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig_rep)
            # Memory optimization: force garbage collection after plot generation
            gc.collect()

    return t50_path


def plot_per_polymer_timeseries_with_error_band(
    *,
    summary_stats_path: Path,
    run_id: str,
    out_fit_dir: Path,
    color_map_path: Path,
    dpi: int = 600,
) -> Optional[Path]:
    """
    Additional per-polymer time-series plots with error bands (mean ± SEM).

    This is intended for runs where the same polymer_id has replicate wells
    at the same heat_min (n > 1). Output:
      - out_fit_dir/per_polymer_with_error__{run_id}/{polymer_stem}__{run_id}.png

    Returns output directory path when plots are written, otherwise None.
    """
    summary_stats_path = Path(summary_stats_path)
    out_fit_dir = Path(out_fit_dir)
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if not summary_stats_path.is_file():
        return None

    df = pd.read_csv(summary_stats_path)
    required = {
        "polymer_id",
        "heat_min",
        "n",
        "mean_abs_activity",
        "sem_abs_activity",
        "mean_REA_percent",
        "sem_REA_percent",
    }
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        return None

    df = df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str)
    for c in ["heat_min", "n", "mean_abs_activity", "sem_abs_activity", "mean_REA_percent", "sem_REA_percent"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["polymer_id", "heat_min"])

    # Only add this version when replicate information exists.
    if not (df["n"] > 1).any():
        return None

    polymer_ids = sorted(df["polymer_id"].astype(str).unique().tolist())
    cmap = ensure_polymer_colors(polymer_ids, color_map_path=Path(color_map_path))

    stems: dict[str, str] = {pid: safe_stem(pid) for pid in polymer_ids}
    stem_counts: dict[str, int] = {}
    for st in stems.values():
        stem_counts[st] = stem_counts.get(st, 0) + 1
    for pid, st in list(stems.items()):
        if stem_counts.get(st, 0) > 1:
            stems[pid] = f"{st}__{_short_hash(pid)}"

    out_dir = out_fit_dir / f"per_polymer_with_error__{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_pngs = {f"{stems[pid]}__{run_id}.png" for pid in polymer_ids}
    for p in out_dir.glob(f"*__{run_id}.png"):
        if p.name not in expected_pngs:
            try:
                p.unlink()
            except Exception:
                pass

    from gox_plate_pipeline.fitting.core import apply_paper_style
    import matplotlib.pyplot as plt
    import gc

    plot_count = 0
    for pid, g in df.groupby("polymer_id", sort=False):
        g = g.sort_values("heat_min").reset_index(drop=True)
        t = g["heat_min"].to_numpy(dtype=float)
        n = g["n"].to_numpy(dtype=float)
        aa = g["mean_abs_activity"].to_numpy(dtype=float)
        aa_sem = g["sem_abs_activity"].to_numpy(dtype=float)
        rea = g["mean_REA_percent"].to_numpy(dtype=float)
        rea_sem = g["sem_REA_percent"].to_numpy(dtype=float)

        color = cmap.get(str(pid), "#0072B2")
        pid_label = safe_label(str(pid))
        stem = stems.get(str(pid), safe_stem(str(pid)))

        with plt.rc_context(apply_paper_style()):
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 2.6))

            ax_left.plot(t, aa, color=color, linewidth=0.9, alpha=0.95)
            ax_left.scatter(t, aa, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=0.95, zorder=5)
            band_ok_abs = np.isfinite(aa_sem) & (aa_sem > 0) & (n > 1)
            if np.any(band_ok_abs):
                y_low = aa - aa_sem
                y_high = aa + aa_sem
                ax_left.fill_between(t, y_low, y_high, where=band_ok_abs, color=color, alpha=0.18, linewidth=0)
            ax_left.set_title(f"{pid_label} | Absolute activity (mean ± SEM)")
            ax_left.set_xlabel("Heat time (min)")
            ax_left.set_ylabel("Absolute activity (a.u./s)")
            ax_left.set_xlim(0.0, 62.5)
            if np.isfinite(aa).any():
                ymax = float(np.nanmax(aa + np.where(np.isfinite(aa_sem), aa_sem, 0.0)))
                ax_left.set_ylim(0.0, max(ymax * 1.05, 1e-9))

            ax_right.plot(t, rea, color=color, linewidth=0.9, alpha=0.95)
            ax_right.scatter(t, rea, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=0.95, zorder=5)
            band_ok_rea = np.isfinite(rea_sem) & (rea_sem > 0) & (n > 1)
            if np.any(band_ok_rea):
                y_low = rea - rea_sem
                y_high = rea + rea_sem
                ax_right.fill_between(t, y_low, y_high, where=band_ok_rea, color=color, alpha=0.18, linewidth=0)
            ax_right.set_title(f"{pid_label} | REA (mean ± SEM)")
            ax_right.set_xlabel("Heat time (min)")
            ax_right.set_ylabel("REA (%)")
            ax_right.set_xlim(0.0, 62.5)
            if np.isfinite(rea).any():
                ymax = float(np.nanmax(rea + np.where(np.isfinite(rea_sem), rea_sem, 0.0)))
                ax_right.set_ylim(0.0, max(ymax * 1.05, 1.0))

            fig.tight_layout(pad=0.3)
            out_path = out_dir / f"{stem}__{run_id}.png"
            fig.savefig(
                out_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig)
            # Memory optimization: periodically force garbage collection after plot generation
            plot_count += 1
            if plot_count % 5 == 0:  # Every 5 polymers
                gc.collect()

    return out_dir


def extract_measurement_date_from_run_id(run_id: str) -> str:
    """
    Extract measurement date from run_id.
    
    Examples:
        "260205-R1" -> "260205"
        "260205-R2" -> "260205"
        "260203-1" -> "260203"
        "260204-2" -> "260204"
    
    Returns the date part (YYMMDD format) before the first hyphen or dash.
    """
    run_id = str(run_id).strip()
    # Split by hyphen/dash and take the first part (date)
    parts = run_id.split("-")
    if parts:
        return parts[0]
    return run_id


def find_same_date_runs(run_id: str, processed_dir: Path) -> list[str]:
    """
    Find all run_ids with the same measurement date as the given run_id.
    
    Returns list of run_ids (including the input run_id) that have the same date prefix.
    """
    date_prefix = extract_measurement_date_from_run_id(run_id)
    same_date_runs: list[str] = []
    
    if not processed_dir.is_dir():
        return same_date_runs
    
    for run_dir in processed_dir.iterdir():
        if not run_dir.is_dir():
            continue
        candidate_run_id = run_dir.name
        candidate_date = extract_measurement_date_from_run_id(candidate_run_id)
        if candidate_date == date_prefix:
            # Check if summary_simple.csv exists
            summary_path = run_dir / "fit" / "summary_simple.csv"
            if summary_path.is_file():
                same_date_runs.append(candidate_run_id)
    
    return sorted(same_date_runs)


def plot_per_polymer_timeseries_across_runs_with_error_band(
    *,
    run_id: str,
    processed_dir: Path,
    out_fit_dir: Path,
    color_map_path: Path,
    same_date_runs: Optional[list[str]] = None,
    group_label: Optional[str] = None,
    dpi: int = 600,
) -> Optional[Path]:
    """
    Plot per-polymer time-series with error bars across a run group.

    Aggregates runs (auto-discovered by date when same_date_runs is None, or provided
    explicitly via same_date_runs), computes mean ± SEM per polymer_id × heat_min,
    and fits curves to the mean trace.

    Output:
      - out_fit_dir/per_polymer_across_runs__{group_label}__{run_id}/{polymer_stem}__{group_label}.png

    Returns output directory path when plots are written, otherwise None.
    """
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")
    
    processed_dir = Path(processed_dir)
    out_fit_dir = Path(out_fit_dir)
    
    # Find all runs with the same measurement date unless explicitly provided
    if same_date_runs is None:
        same_date_runs = find_same_date_runs(run_id, processed_dir)
    else:
        same_date_runs = sorted({str(r).strip() for r in same_date_runs if str(r).strip()})
    if len(same_date_runs) < 2:
        # Need at least 2 runs to calculate SEM
        return None
    
    label = "" if group_label is None else str(group_label).strip()
    if not label:
        label = extract_measurement_date_from_run_id(run_id)
    
    # Load summary_simple.csv from all same-date runs
    all_data: list[pd.DataFrame] = []
    for rid in same_date_runs:
        summary_path = processed_dir / rid / "fit" / "summary_simple.csv"
        if not summary_path.is_file():
            continue
        df = pd.read_csv(summary_path)
        df["run_id"] = rid  # Keep track of which run this came from
        all_data.append(df)
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Required columns
    required = {"polymer_id", "heat_min", "abs_activity", "REA_percent"}
    missing = [c for c in sorted(required) if c not in combined.columns]
    if missing:
        return None
    
    # Clean and prepare data
    combined = combined.copy()
    combined["polymer_id"] = combined["polymer_id"].astype(str)
    combined["heat_min"] = pd.to_numeric(combined["heat_min"], errors="coerce")
    combined["abs_activity"] = pd.to_numeric(combined["abs_activity"], errors="coerce")
    combined["REA_percent"] = pd.to_numeric(combined["REA_percent"], errors="coerce")
    combined = combined.dropna(subset=["polymer_id", "heat_min", "abs_activity", "REA_percent"])
    
    # Aggregate across runs: group by polymer_id × heat_min
    agg_data = []
    for (pid, heat_min), g in combined.groupby(["polymer_id", "heat_min"], dropna=False):
        n_runs = len(g["run_id"].unique())
        if n_runs < 2:
            continue  # Need at least 2 runs for error bands
        
        aa_values = g["abs_activity"].dropna()
        rea_values = g["REA_percent"].dropna()
        
        if len(aa_values) == 0 or len(rea_values) == 0:
            continue
        
        agg_data.append({
            "polymer_id": str(pid),
            "heat_min": float(heat_min),
            "n_runs": n_runs,
            "mean_abs_activity": float(aa_values.mean()),
            "std_abs_activity": float(aa_values.std()) if len(aa_values) > 1 else np.nan,
            "sem_abs_activity": float(aa_values.sem()) if len(aa_values) > 1 else np.nan,
            "mean_REA_percent": float(rea_values.mean()),
            "std_REA_percent": float(rea_values.std()) if len(rea_values) > 1 else np.nan,
            "sem_REA_percent": float(rea_values.sem()) if len(rea_values) > 1 else np.nan,
        })
    
    if not agg_data:
        return None
    
    agg_df = pd.DataFrame(agg_data)
    polymer_ids = sorted(agg_df["polymer_id"].astype(str).unique().tolist())
    cmap = ensure_polymer_colors(polymer_ids, color_map_path=Path(color_map_path))
    
    # Resolve file stems
    stems: dict[str, str] = {pid: safe_stem(pid) for pid in polymer_ids}
    stem_counts: dict[str, int] = {}
    for st in stems.values():
        stem_counts[st] = stem_counts.get(st, 0) + 1
    for pid, st in list(stems.items()):
        if stem_counts.get(st, 0) > 1:
            stems[pid] = f"{st}__{_short_hash(pid)}"
    
    out_dir = out_fit_dir / f"per_polymer_across_runs__{label}__{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from gox_plate_pipeline.fitting.core import apply_paper_style
    import matplotlib.pyplot as plt
    import gc
    
    # Helper functions for fit curve drawing (same as in plot_per_polymer_timeseries)
    def _eval_fit_curve(fit_obj: ExpDecayFit, tt: np.ndarray) -> np.ndarray:
        if fit_obj.model == "exp":
            return fit_obj.y0 * np.exp(-fit_obj.k * tt)
        c = float(fit_obj.c) if fit_obj.c is not None else 0.0
        return c + (fit_obj.y0 - c) * np.exp(-fit_obj.k * tt)
    
    def _to_fluorescent_color(color_hex: str) -> str:
        """Convert a color to a fluorescent (bright, high-saturation) version."""
        hex_str = color_hex.lstrip('#')
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, s * 1.4)  # Increase saturation
        v = min(1.0, v * 1.1)  # Increase brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    def _draw_fit_with_extension(ax: Any, fit_obj: ExpDecayFit, t_obs: np.ndarray, color_hex: str) -> None:
        """Draw fitted curve with extension."""
        t_obs = np.asarray(t_obs, dtype=float)
        t_obs = t_obs[np.isfinite(t_obs)]
        if t_obs.size == 0:
            return
        t_min_obs = float(np.min(t_obs))
        t_max_obs = float(np.max(t_obs))
        color_fluorescent = _to_fluorescent_color(color_hex)
        tt_main = np.linspace(t_min_obs, t_max_obs, 220)
        yy_main = _eval_fit_curve(fit_obj, tt_main)
        ax.plot(tt_main, yy_main, color=color_fluorescent, linewidth=1.7, alpha=0.50, zorder=8)
        if t_min_obs > 0.0:
            tt_pre = np.linspace(0.0, t_min_obs, 120)
            yy_pre = _eval_fit_curve(fit_obj, tt_pre)
            ax.plot(tt_pre, yy_pre, color=color_fluorescent, linewidth=1.5, alpha=0.40, linestyle=(0, (2.4, 2.4)), zorder=7)
        if t_max_obs < 60.0:
            tt_post = np.linspace(t_max_obs, 60.0, 140)
            yy_post = _eval_fit_curve(fit_obj, tt_post)
            ax.plot(tt_post, yy_post, color=color_fluorescent, linewidth=1.5, alpha=0.40, linestyle=(0, (2.4, 2.4)), zorder=7)
    
    plot_count = 0
    for pid, g in agg_df.groupby("polymer_id", sort=False):
        g = g.sort_values("heat_min").reset_index(drop=True)
        t = g["heat_min"].to_numpy(dtype=float)
        n_runs = g["n_runs"].to_numpy(dtype=int)
        aa_mean = g["mean_abs_activity"].to_numpy(dtype=float)
        aa_sem = g["sem_abs_activity"].to_numpy(dtype=float)
        rea_mean = g["mean_REA_percent"].to_numpy(dtype=float)
        rea_sem = g["sem_REA_percent"].to_numpy(dtype=float)
        
        color = cmap.get(str(pid), "#0072B2")
        pid_label = safe_label(str(pid))
        stem = stems.get(str(pid), safe_stem(str(pid)))
        
        with plt.rc_context(apply_paper_style()):
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
            
            # Left: Absolute activity with error bars
            ax_left.plot(t, aa_mean, color=color, linewidth=0.9, alpha=0.95, zorder=10)
            ax_left.scatter(t, aa_mean, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=0.95, zorder=15)

            # Error bars (SEM)
            band_ok_abs = np.isfinite(aa_sem) & (aa_sem > 0) & (n_runs >= 2)
            if np.any(band_ok_abs):
                ax_left.errorbar(
                    t[band_ok_abs],
                    aa_mean[band_ok_abs],
                    yerr=aa_sem[band_ok_abs],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=0.75,
                    zorder=9,
                )
            
            # Fit curve
            y0_abs_init = float(aa_mean[0]) if aa_mean.size > 0 and np.isfinite(float(aa_mean[0])) else None
            fit_abs = fit_exponential_decay(t, aa_mean, y0=y0_abs_init, min_points=4)
            if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                _draw_fit_with_extension(ax_left, fit_abs, t, color)
            
            ax_left.set_title(f"{pid_label} | Absolute activity (mean ± SEM, n={len(same_date_runs)} runs)")
            ax_left.set_xlabel("Heat time (min)")
            ax_left.set_ylabel("Absolute activity (a.u./s)")
            ax_left.set_xlim(0.0, 62.5)
            if np.isfinite(aa_mean).any():
                ymax = float(np.nanmax(aa_mean + np.where(np.isfinite(aa_sem), aa_sem, 0.0)))
                ax_left.set_ylim(0.0, max(ymax * 1.05, 1e-9))
            
            # Right: REA with error bars
            ax_right.plot(t, rea_mean, color=color, linewidth=0.9, alpha=0.95, zorder=10)
            ax_right.scatter(t, rea_mean, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=0.95, zorder=15)

            # Error bars (SEM)
            band_ok_rea = np.isfinite(rea_sem) & (rea_sem > 0) & (n_runs >= 2)
            if np.any(band_ok_rea):
                ax_right.errorbar(
                    t[band_ok_rea],
                    rea_mean[band_ok_rea],
                    yerr=rea_sem[band_ok_rea],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=0.75,
                    zorder=9,
                )
            
            # Fit curve
            y0_rea_init = float(rea_mean[0]) if rea_mean.size > 0 and np.isfinite(float(rea_mean[0])) else 100.0
            fit_rea = fit_exponential_decay(
                t,
                rea_mean,
                y0=y0_rea_init,
                fixed_y0=100.0,
                min_points=4,
            )
            if fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70:
                _draw_fit_with_extension(ax_right, fit_rea, t, color)
            
            ax_right.set_title(f"{pid_label} | REA (mean ± SEM, n={len(same_date_runs)} runs)")
            ax_right.set_xlabel("Heat time (min)")
            ax_right.set_ylabel("REA (%)")
            ax_right.set_xlim(0.0, 62.5)
            if np.isfinite(rea_mean).any():
                ymax = float(np.nanmax(rea_mean + np.where(np.isfinite(rea_sem), rea_sem, 0.0)))
                ax_right.set_ylim(0.0, max(ymax * 1.05, 1.0))
            
            fig.tight_layout(pad=0.3)
            out_path = out_dir / f"{stem}__{label}.png"
            fig.savefig(
                out_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig)
            plot_count += 1
            if plot_count % 5 == 0:
                gc.collect()
    
    return out_dir


def plot_per_polymer_timeseries_across_runs_with_error_bars(
    *,
    run_id: str,
    processed_dir: Path,
    out_fit_dir: Path,
    color_map_path: Path,
    same_date_runs: Optional[list[str]] = None,
    group_label: Optional[str] = None,
    dpi: int = 600,
) -> Optional[Path]:
    """
    Alias of plot_per_polymer_timeseries_across_runs_with_error_band().
    Kept for explicit naming when using SEM error bars.
    """
    return plot_per_polymer_timeseries_across_runs_with_error_band(
        run_id=run_id,
        processed_dir=processed_dir,
        out_fit_dir=out_fit_dir,
        color_map_path=color_map_path,
        same_date_runs=same_date_runs,
        group_label=group_label,
        dpi=dpi,
    )
