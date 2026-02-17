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


def _escape_mathtext_token(text: str) -> str:
    """Escape minimal characters for embedding a token inside mathtext \\mathrm{...}."""
    s = "" if text is None else str(text)
    s = s.replace("\\", r"\\")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    return s


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
DEFAULT_REFERENCE_POLYMER_ID = "GOX"


def _normalize_polymer_id_token(polymer_id: str) -> str:
    return str(polymer_id).strip().upper()


def _polymer_plot_order_key(polymer_id: str) -> tuple[int, int, str]:
    """
    Consistent polymer ordering for plot panels/legends:
      1) GOx / GOx with*
      2) PMPC
      3) PMTAC
      4) PMBTA-1..N (numeric)
      5) others (alphabetical by normalized token)
    """
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


def _is_reference_polymer_id(
    polymer_id: str,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    return _normalize_polymer_id_token(polymer_id) == _normalize_polymer_id_token(reference_polymer_id)


def _is_background_polymer_id(polymer_id: str) -> bool:
    """Return True for background/control-only IDs that should be excluded from downstream t50/FoG plotting."""
    norm = _normalize_polymer_id_token(polymer_id)
    return norm in {
        "BACKGROUND",
        "BACKGROUND CONTROL",
        "BLANK",
        "BLANK CONTROL",
    }


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


def _gox_profile_from_summary_df(
    df: pd.DataFrame,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> tuple[np.ndarray, np.ndarray]:
    reference_norm = _normalize_polymer_id_token(reference_polymer_id)
    ref_df = df[df["polymer_id"].astype(str).str.strip().str.upper() == reference_norm].copy()
    if ref_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    agg = (
        ref_df.groupby("heat_min", as_index=False)
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
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> dict[str, float]:
    out: dict[str, float] = {}
    reference_norm = _normalize_polymer_id_token(reference_polymer_id)
    for pid, g in df.groupby("polymer_id", sort=False):
        pid_s = str(pid).strip()
        if not include_gox and _normalize_polymer_id_token(pid_s) == reference_norm:
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
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
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
    is_default_ref = _is_reference_polymer_id(reference_polymer_id, DEFAULT_REFERENCE_POLYMER_ID)
    same_run_source = "same_run_gox" if is_default_ref else "same_run_reference"
    same_round_source = "same_round_mean_gox" if is_default_ref else "same_round_mean_reference"
    nearest_source = "nearest_round_run_gox" if is_default_ref else "nearest_round_run_reference"
    missing_source = "missing_gox_reference" if is_default_ref else "missing_reference_polymer"
    df = _normalize_summary_simple_df(summary_df)
    result: dict[str, Any] = {
        "source": missing_source,
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

    # 1) same-run reference polymer
    t_same, y_same = _gox_profile_from_summary_df(df, reference_polymer_id=reference_polymer_id)
    if t_same.size > 0:
        gox_at = value_at_time_linear(t_same, y_same, at_time_min=float(at_time_min))
        if gox_at is not None and np.isfinite(float(gox_at)) and float(gox_at) > 0.0:
            result.update({
                "source": same_run_source,
                "reference_run_id": rid,
                "gox_t_min": t_same,
                "gox_abs_activity": y_same,
                "gox_abs_activity_at_time": float(gox_at),
                "round_mean_gox_abs_activity_at_time": float(gox_at),
                "reference_note": "same_run_gox_available",
            })
            return result

    if processed_dir is None or run_round_map_path is None:
        result["reference_note"] = "same_run_reference_missing_and_no_round_fallback"
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

    target_map = _polymer_abs_at_time_map(
        df,
        at_time_min=float(at_time_min),
        include_gox=False,
        reference_polymer_id=reference_polymer_id,
    )
    candidates: list[dict[str, Any]] = []
    for cand_run in candidate_runs:
        cand_df = _load_summary_simple_from_processed(processed_dir, cand_run)
        if cand_df is None or cand_df.empty:
            continue
        t_c, y_c = _gox_profile_from_summary_df(cand_df, reference_polymer_id=reference_polymer_id)
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
            "polymer_abs_at_time": _polymer_abs_at_time_map(
                cand_df,
                at_time_min=float(at_time_min),
                include_gox=False,
                reference_polymer_id=reference_polymer_id,
            ),
        })

    if not candidates:
        result["reference_note"] = "no_round_runs_with_valid_reference"
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
            "source": nearest_source,
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
        "source": same_round_source,
        "reference_run_id": "",
        "gox_t_min": mean_t,
        "gox_abs_activity": mean_y,
        "gox_abs_activity_at_time": float(mean_gox_at),
        "target_round_drift_log_median": float(drift_log) if np.isfinite(drift_log) else np.nan,
        "nearest_run_log_median": float(best_dist) if np.isfinite(best_dist) else np.nan,
        "nearest_run_shared_polymers": int(best_shared),
        "reference_note": "same_round_mean_reference_selected",
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
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
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

    # Reference polymer always uses gray.
    REFERENCE_COLOR = "#808080"  # Medium gray
    reference_norm = _normalize_polymer_id_token(reference_polymer_id)

    # Enforce reference color for already-registered IDs as well.
    for pid, c in list(cmap.items()):
        if _normalize_polymer_id_token(pid) == reference_norm and str(c).strip().lower() != REFERENCE_COLOR.lower():
            cmap[pid] = REFERENCE_COLOR
            changed = True
    
    used = set(v.lower() for v in cmap.values())
    palette = _default_palette_hex()
    
    # Separate existing and new polymer IDs
    existing_pids = [pid for pid in polymer_ids if str(pid) in cmap]
    new_pids = [pid for pid in polymer_ids if str(pid) not in cmap]
    
    # Handle reference polymer first (create a new list without reference for color assignment)
    new_pids_for_color = []
    for pid in new_pids:
        pid_str = str(pid)
        if _normalize_polymer_id_token(pid_str) == reference_norm:
            cmap[pid_str] = REFERENCE_COLOR
            used.add(REFERENCE_COLOR.lower())
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


def first_crossing_below_linear(
    t_min: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
) -> tuple[Optional[float], str]:
    """
    First time when y crosses below threshold by linear interpolation.

    Returns:
      (t_cross, flag)
      - flag="" when interpolated crossing exists
      - flag="already_below" if first finite y <= threshold
      - flag="never_cross" when never below threshold (right-censored)
      - flag="missing_profile" when no finite profile
    """
    t = np.asarray(t_min, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(yy)
    t = t[mask]
    yy = yy[mask]
    if t.size == 0:
        return None, "missing_profile"
    order = np.argsort(t)
    t = t[order]
    yy = yy[order]
    th = float(threshold)
    if float(yy[0]) <= th:
        return float(t[0]), "already_below"
    for i in range(int(t.size) - 1):
        y0i = float(yy[i])
        y1i = float(yy[i + 1])
        if (y0i >= th and y1i <= th) or (y0i > th and y1i < th):
            if math.isclose(y1i, y0i, rel_tol=0.0, abs_tol=1e-12):
                return float(t[i]), ""
            frac = (th - y0i) / (y1i - y0i)
            return float(t[i] + frac * (t[i + 1] - t[i])), ""
    return float(t[-1]), "never_cross"


def _format_t_theta_flag_note(flag: str) -> str:
    """
    Return annotation text for t_theta censor status.
    `already_below` is intentionally hidden because U(t) plots make it obvious.
    """
    flag_s = str(flag).strip()
    if not flag_s or flag_s == "already_below":
        return ""
    return f"flag={flag_s}"


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
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    native_activity_min_rel: float = 0.70,
) -> Path:
    """
    Create per-polymer time series plots: one figure per polymer with
    Absolute activity (left), REA (center), and objective-understanding panel (right).

    Outputs (PNG only):
      - out_fit_dir/per_polymer__{run_id}/{polymer_stem}__{run_id}.png  (combined 3-panel figure)
      - out_fit_dir/t50/per_polymer_v2__{run_id}/{polymer_stem}__{run_id}.png (Abs + REA + U(t))
      - out_fit_dir/t50/per_polymer_refnorm__{run_id}/{polymer_stem}__{run_id}.png
        (reference-normalized activity panel: U_ref(t) + theta + t_theta + reference overlay)
      - out_fit_dir/t50/csv/t50__{run_id}.csv
      - out_fit_dir/all_polymers__{run_id}.png (include_in_all_polymers=True only, default)
      - out_fit_dir/all_polymers_all__{run_id}.png (all polymers, for debugging, only if different from filtered)
      - out_fit_dir/all_polymers_pair__{run_id}.png (custom pair from TSV, if specified)

    If row_map_path is provided and contains all_polymers_pair column with two polymer IDs
    (comma-separated), generates an additional plot with only those two polymers.

    Reference profile for objective panel/metrics:
      - same-run reference polymer first (default reference is GOx)
      - if missing and round info is provided: same-round mean reference profile
      - when the run is clearly shifted from the round center, nearest round run reference profile.

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

    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)

    # Reference profile for objective metrics (same run -> same round fallback).
    gox_ref = resolve_gox_reference_profile(
        run_id=run_id,
        summary_df=df,
        at_time_min=20.0,
        processed_dir=(Path(processed_dir) if processed_dir is not None else None),
        run_round_map_path=(Path(run_round_map_path) if run_round_map_path is not None else None),
        reference_polymer_id=reference_polymer_id,
    )
    gox_ref_t = np.asarray(gox_ref.get("gox_t_min", np.array([], dtype=float)), dtype=float)
    gox_ref_y = np.asarray(gox_ref.get("gox_abs_activity", np.array([], dtype=float)), dtype=float)
    gox_ref_at_20 = pd.to_numeric(gox_ref.get("gox_abs_activity_at_time", np.nan), errors="coerce")
    gox_ref_at_0 = value_at_time_linear(gox_ref_t, gox_ref_y, at_time_min=0.0) if gox_ref_t.size > 0 else None
    gox_ref_at_0 = (
        float(gox_ref_at_0)
        if gox_ref_at_0 is not None and np.isfinite(float(gox_ref_at_0)) and float(gox_ref_at_0) > 0.0
        else np.nan
    )
    gox_ref_source = str(gox_ref.get("source", "missing_gox_reference"))
    gox_ref_round_id = str(gox_ref.get("round_id", "")).strip()
    gox_ref_run_id = str(gox_ref.get("reference_run_id", "")).strip()
    ref_rea_t = np.array([], dtype=float)
    ref_rea = np.array([], dtype=float)
    fit_ref_rea: Optional[ExpDecayFit] = None
    ref_rea_t50_lin: Optional[float] = None
    ref_rea_t50_show: Optional[float] = None
    use_exp_ref_rea = False
    if (
        gox_ref_t.size > 0
        and gox_ref_y.size > 0
        and np.isfinite(gox_ref_at_0)
        and float(gox_ref_at_0) > 0.0
    ):
        ref_rea_t = np.asarray(gox_ref_t, dtype=float)
        ref_rea = np.asarray(gox_ref_y, dtype=float) / float(gox_ref_at_0) * 100.0
        fit_ref_rea = fit_exponential_decay(
            ref_rea_t,
            ref_rea,
            y0=100.0,
            fixed_y0=100.0,
            min_points=4,
            t50_definition=t50_definition,
        )
        use_exp_ref_rea = bool(
            fit_ref_rea is not None
            and np.isfinite(float(fit_ref_rea.r2))
            and float(fit_ref_rea.r2) >= 0.70
        )
        ref_rea_t50_lin = t50_linear(
            ref_rea_t,
            ref_rea,
            y0=100.0,
            target_frac=0.5,
            target_value=(50.0 if t50_definition == T50_DEFINITION_REA50 else None),
        )
        ref_rea_t50_show = (
            fit_ref_rea.t50 if (fit_ref_rea is not None and use_exp_ref_rea) else ref_rea_t50_lin
        )

    polymer_ids_all = sorted(
        df["polymer_id"].astype(str).unique().tolist(),
        key=_polymer_plot_order_key,
    )
    excluded_background_ids = [pid for pid in polymer_ids_all if _is_background_polymer_id(pid)]
    polymer_ids = [pid for pid in polymer_ids_all if not _is_background_polymer_id(pid)]
    if excluded_background_ids:
        print(
            "Info: Excluding background-like polymers from downstream t50/FoG plots: "
            f"{excluded_background_ids}"
        )
    if not polymer_ids:
        raise ValueError("No analyzable polymers after excluding background-like IDs.")
    cmap = ensure_polymer_colors(
        polymer_ids,
        color_map_path=Path(color_map_path),
        reference_polymer_id=reference_polymer_id,
    )

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
    out_t50_csv_dir = out_t50_dir / "csv"
    out_t50_csv_dir.mkdir(parents=True, exist_ok=True)
    out_per_polymer = out_t50_dir / f"per_polymer__{run_id}"
    out_per_polymer.mkdir(parents=True, exist_ok=True)
    out_per_polymer_v2 = out_t50_dir / f"per_polymer_v2__{run_id}"
    out_per_polymer_v2.mkdir(parents=True, exist_ok=True)
    out_per_polymer_refnorm = out_t50_dir / f"per_polymer_refnorm__{run_id}"
    out_per_polymer_refnorm.mkdir(parents=True, exist_ok=True)
    out_rea_fog_panel = out_t50_dir / f"rea_comparison_fog_panel__{run_id}"
    out_rea_fog_panel.mkdir(parents=True, exist_ok=True)

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
    _clean_stale(out_per_polymer_v2, expected_pngs)
    _clean_stale(out_per_polymer_refnorm, expected_pngs)
    _clean_stale(out_rea_fog_panel, expected_pngs)

    def _write_rea_fog_gallery_from_per_polymer(
        *,
        per_polymer_png_paths: list[Path],
        out_png: Path,
        n_cols: int = 5,
        crop_left_frac: float = 0.0,
        gap_px: int = 12,
        outer_pad_px: int = 12,
    ) -> Optional[Path]:
        """
        Build a gallery of the right panel ("REA comparison and FoG ratio").

        Quality policy:
          - no resizing (each tile keeps original pixel resolution)
          - PNG output (lossless)
        """
        try:
            from PIL import Image
        except Exception:
            return None

        tile_images: list[Any] = []
        for p in per_polymer_png_paths:
            if not p.is_file():
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    if w <= 1 or h <= 1:
                        continue
                    # When crop_left_frac <= 0, keep full image (used for pre-exported right panels).
                    x0 = 0 if float(crop_left_frac) <= 0.0 else int(round(float(w) * float(crop_left_frac)))
                    x0 = max(0, min(w - 1, x0))
                    tile = im.crop((x0, 0, w, h)).copy()
                    tile_images.append(tile)
            except Exception:
                continue

        if not tile_images:
            if out_png.exists():
                out_png.unlink(missing_ok=True)
            return None

        cols = max(1, int(n_cols))
        rows = int(math.ceil(len(tile_images) / float(cols)))
        tile_w = max(int(im.width) for im in tile_images)
        tile_h = max(int(im.height) for im in tile_images)
        gap = max(0, int(gap_px))
        outer = max(0, int(outer_pad_px))

        out_w = outer * 2 + cols * tile_w + (cols - 1) * gap
        out_h = outer * 2 + rows * tile_h + (rows - 1) * gap

        # Match background to plot canvas (light gray in current paper style).
        bg_rgb = (245, 245, 245)
        try:
            px = tile_images[0].convert("RGB").getpixel((0, 0))
            if isinstance(px, tuple) and len(px) >= 3:
                bg_rgb = (int(px[0]), int(px[1]), int(px[2]))
        except Exception:
            pass

        canvas = Image.new("RGB", (out_w, out_h), bg_rgb)
        for idx, tile in enumerate(tile_images):
            rr = idx // cols
            cc = idx % cols
            x = outer + cc * (tile_w + gap)
            y = outer + rr * (tile_h + gap)
            ox = (tile_w - int(tile.width)) // 2
            oy = (tile_h - int(tile.height)) // 2
            if tile.mode == "RGBA":
                canvas.paste(tile, (x + ox, y + oy), tile)
            else:
                canvas.paste(tile.convert("RGB"), (x + ox, y + oy))

        out_png.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_png, format="PNG", optimize=False, compress_level=1)
        return out_png

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
        "reference_polymer_id",
        "n_points",
        "abs_activity_at_0",
        "native_activity_rel_at_0",
        "native_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "abs_activity_at_20",
        "u_at_20",
        "t_theta_min",
        "t_theta_censor_flag",
        "functional_activity_at_20_rel",
        "gox_abs_activity_at_0_ref",
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

    def _draw_fit_with_extension(
        ax: Any,
        fit_obj: ExpDecayFit,
        t_obs: np.ndarray,
        color_hex: str,
        *,
        use_dashed_main: bool = False,
        preserve_gray: bool = False,
        low_confidence: bool = False,
        force_dashed_main: bool = False,
    ) -> None:
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
        main_alpha = 0.40 if use_dashed_main else 0.50
        ext_alpha = 0.30 if use_dashed_main else 0.40
        main_ls = (0, (2.0, 2.0)) if (low_confidence or force_dashed_main) else "-"
        ext_ls = (0, (1.6, 1.9)) if low_confidence else (0, (2.4, 2.4))
        if low_confidence:
            main_alpha = min(0.65, main_alpha + 0.12)
            ext_alpha = min(0.55, ext_alpha + 0.10)

        ax.plot(
            tt_main,
            yy_main,
            color=color_fluorescent,
            linewidth=1.7,
            alpha=main_alpha,
            linestyle=main_ls,
            zorder=8,
        )

        # Dashed extension where points are missing (0-60 min design range).
        if t_min_obs > 0.0:
            tt_pre = np.linspace(0.0, t_min_obs, 120)
            yy_pre = _eval_fit_curve(fit_obj, tt_pre)
            ax.plot(
                tt_pre,
                yy_pre,
                color=color_fluorescent,
                linewidth=1.5,
                alpha=ext_alpha,
                linestyle=ext_ls,
                zorder=7,
            )
        if t_max_obs < 60.0:
            tt_post = np.linspace(t_max_obs, 60.0, 140)
            yy_post = _eval_fit_curve(fit_obj, tt_post)
            ax.plot(
                tt_post,
                yy_post,
                color=color_fluorescent,
                linewidth=1.5,
                alpha=ext_alpha,
                linestyle=ext_ls,
                zorder=7,
            )

    def _info_box_text(rhs: str, r2: Optional[float], t50: Optional[float] = None) -> str:
        # Keep superscript compact and consistent across figures.
        r2_txt = f"{float(r2):.3f}" if (r2 is not None and np.isfinite(float(r2))) else "NA"
        if t50 is None or not np.isfinite(float(t50)):
            t50_math = r"\mathrm{NA}"
        else:
            t50_math = f"{float(t50):.3g}\\,\\mathrm{{min}}"
        if t50 is None:
            return rf"$y = {rhs}$" + "\n" + rf"R$^{{2}}$ = {r2_txt}"
        return rf"$y = {rhs}$" + "\n" + rf"R$^{{2}}$ = {r2_txt}" + "\n" + rf"$t_{{50}} = {t50_math}$"

    grouped_by_pid: dict[str, pd.DataFrame] = {str(pid): g for pid, g in df.groupby("polymer_id", sort=False)}
    for pid in polymer_ids:
        g = grouped_by_pid.get(str(pid))
        if g is None:
            continue
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

        # Reference-relative activity at each heat time (kept for CSV metric at 20 min).
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
        u_series = np.full_like(aa, np.nan, dtype=float)
        if np.isfinite(gox_ref_at_0) and float(gox_ref_at_0) > 0.0:
            ok_u = np.isfinite(aa)
            u_series[ok_u] = aa[ok_u] / float(gox_ref_at_0)
        u_at_20 = value_at_time_linear(t, u_series, at_time_min=20.0)
        u_at_20 = float(u_at_20) if u_at_20 is not None and np.isfinite(u_at_20) else np.nan
        t_theta_val, t_theta_flag = first_crossing_below_linear(
            t,
            u_series,
            threshold=float(native_activity_min_rel),
        )

        # Debug output removed for memory optimization

        # Reference polymer always uses gray color.
        pid_str = str(pid)
        is_ref_per_polymer = _normalize_polymer_id_token(pid_str) == reference_polymer_norm
        if is_ref_per_polymer:
            color = "#808080"  # Medium gray
        else:
            color = cmap.get(pid_str, "#0072B2")
        pid_label = safe_label(str(pid))
        stem = stems.get(pid_str, safe_stem(pid_str))
        native_rel_at_0 = (
            float(abs_activity_at_0) / float(gox_ref_at_0)
            if np.isfinite(abs_activity_at_0) and np.isfinite(gox_ref_at_0) and float(gox_ref_at_0) > 0.0
            else np.nan
        )
        native_feasible = bool(
            np.isfinite(native_rel_at_0) and native_rel_at_0 >= float(native_activity_min_rel)
        )

        # --- Absolute activity (left panel)
        # y0 is optional: if provided, used as initial guess; otherwise estimated from all data points
        y0_abs_init = None
        if aa.size > 0 and np.isfinite(float(aa[0])):
            y0_abs_init = float(aa[0])  # Use first point as initial guess if available

        fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
        use_exp_abs = bool(fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70)
        r2_abs = float(fit_abs.r2) if (fit_abs is not None and np.isfinite(float(fit_abs.r2))) else None

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

        # --- One figure: Absolute (left), REA (center), objective-understanding (right)
        _style = apply_paper_style()
        with plt.rc_context(_style):
            fig, (ax_left, ax_right, ax_func) = plt.subplots(1, 3, figsize=(10.4, 2.8))

            # Right: Objective components (polymer REA vs reference REA with native-activity gate).
            ax_func.scatter(
                t,
                rea,
                s=12,
                color=color,
                edgecolors="0.2",
                linewidths=0.4,
                alpha=1.0,
                zorder=30,
                clip_on=False,
            )
            should_draw_rea_fit_obj = bool(fit_rea is not None and use_exp_rea)
            if should_draw_rea_fit_obj:
                _draw_fit_with_extension(
                    ax_func,
                    fit_rea,
                    t,
                    color,
                    use_dashed_main=False,
                    preserve_gray=is_ref_per_polymer,
                    low_confidence=(not is_ref_per_polymer) and (not native_feasible),
                )

            if ref_rea_t.size > 0 and ref_rea.size > 0:
                ax_func.scatter(
                    ref_rea_t,
                    ref_rea,
                    s=10,
                    marker="s",
                    color="#808080",
                    edgecolors="0.2",
                    linewidths=0.3,
                    alpha=0.85,
                    zorder=24,
                    clip_on=False,
                )
                if fit_ref_rea is not None:
                    _draw_fit_with_extension(
                        ax_func,
                        fit_ref_rea,
                        ref_rea_t,
                        "#808080",
                        use_dashed_main=True,
                        preserve_gray=True,
                        low_confidence=not use_exp_ref_rea,
                        force_dashed_main=True,
                    )
                elif ref_rea_t.size > 1:
                    # Keep reference trace visible even when exponential fit is unavailable.
                    ax_func.plot(
                        ref_rea_t,
                        ref_rea,
                        color="#808080",
                        linewidth=1.3,
                        alpha=0.75,
                        linestyle=(0, (2.0, 2.0)),
                        zorder=7,
                    )

            t50_poly = t50_model if t50_model is not None else t50_lin
            fog_from_t50 = (
                float(t50_poly) / float(ref_rea_t50_show)
                if (
                    t50_poly is not None
                    and np.isfinite(float(t50_poly))
                    and float(t50_poly) > 0.0
                    and ref_rea_t50_show is not None
                    and np.isfinite(float(ref_rea_t50_show))
                    and float(ref_rea_t50_show) > 0.0
                )
                else np.nan
            )
            target_line = float(t50_target_rea)
            t50_poly_guide_x: Optional[float] = (
                float(t50_poly)
                if (t50_poly is not None and np.isfinite(float(t50_poly)) and float(t50_poly) > 0.0)
                else None
            )
            t50_ref_guide_x: Optional[float] = (
                float(ref_rea_t50_show)
                if (
                    ref_rea_t50_show is not None
                    and np.isfinite(float(ref_rea_t50_show))
                    and float(ref_rea_t50_show) > 0.0
                )
                else None
            )

            t50_poly_math = (
                f"{float(t50_poly):.3g}\\,\\mathrm{{min}}"
                if (t50_poly is not None and np.isfinite(float(t50_poly)) and float(t50_poly) > 0.0)
                else r"\mathrm{NA}"
            )
            t50_ref_math = (
                f"{float(ref_rea_t50_show):.3g}\\,\\mathrm{{min}}"
                if (
                    ref_rea_t50_show is not None
                    and np.isfinite(float(ref_rea_t50_show))
                    and float(ref_rea_t50_show) > 0.0
                )
                else r"\mathrm{NA}"
            )
            pid_norm = _normalize_polymer_id_token(pid_str)
            t50_poly_label = "sample" if pid_norm.startswith("GOX WITH") else "poly"
            t50_ref_label = "GOx" if reference_polymer_norm == "GOX" else safe_label(reference_polymer_id)
            t50_poly_label_math = _escape_mathtext_token(t50_poly_label)
            t50_ref_label_math = _escape_mathtext_token(t50_ref_label)
            objective_lines: list[str] = []
            if is_ref_per_polymer:
                objective_lines.append(rf"$\mathrm{{reference}}:\ {t50_ref_label_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
            elif np.isfinite(fog_from_t50):
                fog_math = f"{float(fog_from_t50):.3g}"
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_poly_label_math}}}}}={t50_poly_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
                objective_lines.append(rf"$\mathrm{{FoG}}={fog_math}$")
            else:
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_poly_label_math}}}}}={t50_poly_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
                objective_lines.append(r"$\mathrm{FoG}=\mathrm{NA}$")

            fit_subject_label = "reference" if is_ref_per_polymer else "sample"
            t50_subject_label = t50_ref_label if is_ref_per_polymer else t50_poly_label
            fit_subject_label_math = _escape_mathtext_token(fit_subject_label)
            t50_subject_label_math = _escape_mathtext_token(t50_subject_label)
            if fit_rea is None:
                objective_lines.append(rf"$\mathrm{{{fit_subject_label_math}\ fit:\ unavailable}}$")
            elif not use_exp_rea:
                objective_lines.append(
                    rf"$\mathrm{{{fit_subject_label_math}\ fit:\ low\ confidence}}\ (R^2={float(fit_rea.r2):.3f})$"
                )
            if t50_model is None and t50_poly is not None and np.isfinite(float(t50_poly)):
                objective_lines.append(
                    rf"$t_{{50,\mathrm{{{t50_subject_label_math}}}}}:\ \mathrm{{linear\ crossing}}$"
                )

            if (not is_ref_per_polymer) and (fit_ref_rea is not None) and (not use_exp_ref_rea):
                objective_lines.append(
                    rf"$\mathrm{{{t50_ref_label_math}\ fit:\ low\ confidence}}\ (R^2={float(fit_ref_rea.r2):.3f},\ \mathrm{{dashed}})$"
                )
            if (
                (not is_ref_per_polymer)
                and ref_rea_t50_show is not None
                and np.isfinite(float(ref_rea_t50_show))
                and (
                    fit_ref_rea is None
                    or (not use_exp_ref_rea)
                )
            ):
                objective_lines.append(
                    rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}:\ \mathrm{{linear\ crossing}}$"
                )

            if (not is_ref_per_polymer) and np.isfinite(native_rel_at_0) and (not native_feasible):
                objective_lines.append(
                    rf"$U_{{0}}={float(native_rel_at_0):.3g} < \theta={float(native_activity_min_rel):.2f}\ \mathrm{{at}}\ 0\,\mathrm{{min}}$"
                )

            objective_text = "\n".join(objective_lines)
            info_pad_scalar = float(info_pad[0]) if isinstance(info_pad, (tuple, list)) else float(info_pad)
            txt_obj = ax_func.annotate(
                objective_text,
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                multialignment="left",
                fontsize=7.0,
                bbox=dict(
                    boxstyle=f"round,pad={max(0.16, info_pad_scalar - 0.03)}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.93,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_obj.get_bbox_patch() is not None:
                txt_obj.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())

            ax_func.set_title(f"{pid_label} | REA comparison and FoG ratio")
            ax_func.set_xlabel("Heat time (min)")
            ax_func.set_ylabel("REA (%)")
            ax_func.tick_params(axis="x", which="both", length=0, labelsize=6)
            ax_func.tick_params(axis="y", which="both", length=0, labelsize=6)
            ax_func.set_xlim(0.0, 62.5)
            obj_vals: list[float] = []
            if np.any(np.isfinite(rea)):
                obj_vals.extend(rea[np.isfinite(rea)].tolist())
            if ref_rea.size > 0 and np.any(np.isfinite(ref_rea)):
                obj_vals.extend(ref_rea[np.isfinite(ref_rea)].tolist())
            if obj_vals:
                obj_arr = np.asarray(obj_vals, dtype=float)
                omax = float(np.nanmax(obj_arr))
                omin = float(np.nanmin(obj_arr))
                omargin = (omax - omin) * 0.05 if omax > omin else (omax * 0.05 if omax > 0 else 2.0)
                y_top_obj = omax + omargin
                if not np.isfinite(y_top_obj) or y_top_obj <= 0.0:
                    y_top_obj = 100.0
                ax_func.set_ylim(0.0, y_top_obj)
            else:
                ax_func.set_ylim(0.0, 100.0)

            # Draw L-shaped guide lines (REA panel style): no penetration to top/right.
            y_bottom_obj = float(ax_func.get_ylim()[0])
            guide_x_candidates = [x for x in (t50_poly_guide_x, t50_ref_guide_x) if x is not None]
            if guide_x_candidates:
                x_guide_max = float(max(guide_x_candidates))
                ax_func.plot(
                    [0.0, x_guide_max],
                    [target_line, target_line],
                    linestyle=(0, (3, 2)),
                    color="0.55",
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )
            if t50_poly_guide_x is not None:
                ax_func.plot(
                    [t50_poly_guide_x, t50_poly_guide_x],
                    [y_bottom_obj, target_line],
                    linestyle=(0, (3, 2)),
                    color=color,
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )
            if t50_ref_guide_x is not None:
                ax_func.plot(
                    [t50_ref_guide_x, t50_ref_guide_x],
                    [y_bottom_obj, target_line],
                    linestyle=(0, (3, 2)),
                    color="#808080",
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )
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
                _draw_fit_with_extension(ax_left, fit_abs, t, color, use_dashed_main=False, preserve_gray=is_ref_per_polymer)
                info_text_left = _info_box_text(abs_rhs, float(fit_abs.r2))
            else:
                info_text_left = _info_box_text(r"\mathrm{no\ fit}", None)
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
                _draw_fit_with_extension(ax_right, fit_rea, t, color, use_dashed_main=False, preserve_gray=is_ref_per_polymer)
                t50_show = t50_model if t50_model is not None else t50_lin
                info_text_right = _info_box_text(rea_rhs, float(fit_rea.r2), t50=t50_show)
            else:
                t50_show = t50_lin
                info_text_right = _info_box_text(r"\mathrm{no\ fit}", None, t50=t50_show)
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
            # Save the right panel directly (no post-hoc crop) for high-quality gallery tiling.
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bbox = ax_func.get_tightbbox(renderer=renderer).expanded(1.02, 1.02)
                bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                out_panel_path = out_rea_fog_panel / f"{stem}__{run_id}.png"
                fig.savefig(
                    out_panel_path,
                    dpi=int(dpi),
                    bbox_inches=bbox_inches,
                    pad_inches=0.00,
                    pil_kwargs={"compress_level": 1},
                )
            except Exception:
                pass
            plt.close(fig)

        # --- v2 figure: Abs + REA + U(t) with theta/t_theta (policy_v2 supplementary)
        with plt.rc_context(_style):
            fig_v2, (ax_abs_v2, ax_rea_v2, ax_u_v2) = plt.subplots(1, 3, figsize=(10.4, 2.8))

            # Abs panel
            ax_abs_v2.scatter(t, aa, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=30, clip_on=False)
            if use_exp_abs and fit_abs is not None:
                _draw_fit_with_extension(ax_abs_v2, fit_abs, t, color, use_dashed_main=False, preserve_gray=is_ref_per_polymer)
            if gox_ref_t.size > 0 and gox_ref_y.size > 0:
                ax_abs_v2.scatter(
                    gox_ref_t,
                    gox_ref_y,
                    s=10,
                    marker="s",
                    color="#808080",
                    edgecolors="0.2",
                    linewidths=0.3,
                    alpha=0.85,
                    zorder=24,
                    clip_on=False,
                )
            ax_abs_v2.set_title(f"{pid_label} | Absolute activity")
            ax_abs_v2.set_xlabel("Heat time (min)")
            ax_abs_v2.set_ylabel("Absolute activity (a.u./s)")
            ax_abs_v2.set_xlim(0.0, 62.5)
            y_abs_vals = [v for v in aa.tolist() if np.isfinite(v)]
            if gox_ref_y.size > 0:
                y_abs_vals.extend([v for v in gox_ref_y.tolist() if np.isfinite(v)])
            if y_abs_vals:
                y_max_abs = float(np.nanmax(np.asarray(y_abs_vals, dtype=float)))
                ax_abs_v2.set_ylim(0.0, max(1e-9, y_max_abs * 1.08))
            else:
                ax_abs_v2.set_ylim(0.0, 1.0)
            ax_abs_v2.tick_params(axis="both", which="both", length=0, labelsize=6)
            ax_abs_v2.spines["top"].set_visible(False)
            ax_abs_v2.spines["right"].set_visible(False)
            ax_abs_v2.spines["left"].set_color("0.7")
            ax_abs_v2.spines["bottom"].set_color("0.7")
            ax_abs_v2.spines["left"].set_zorder(-10)
            ax_abs_v2.spines["bottom"].set_zorder(-10)

            # REA panel
            ax_rea_v2.scatter(t, rea, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=30, clip_on=False)
            if use_exp_rea and fit_rea is not None:
                _draw_fit_with_extension(ax_rea_v2, fit_rea, t, color, use_dashed_main=False, preserve_gray=is_ref_per_polymer)
            if ref_rea_t.size > 0 and ref_rea.size > 0:
                ax_rea_v2.scatter(
                    ref_rea_t,
                    ref_rea,
                    s=10,
                    marker="s",
                    color="#808080",
                    edgecolors="0.2",
                    linewidths=0.3,
                    alpha=0.85,
                    zorder=24,
                    clip_on=False,
                )
            t50_show_v2 = t50_model if t50_model is not None else t50_lin
            if t50_show_v2 is not None and np.isfinite(float(t50_show_v2)) and float(t50_show_v2) > 0.0:
                t50_val = float(t50_show_v2)
                y_line = 50.0 if t50_definition == T50_DEFINITION_REA50 else float(t50_target_rea)
                ax_rea_v2.plot([0.0, t50_val], [y_line, y_line], linestyle=(0, (3, 2)), color="0.5", linewidth=0.6, alpha=0.85, zorder=6)
                ax_rea_v2.plot([t50_val, t50_val], [0.0, y_line], linestyle=(0, (3, 2)), color="0.35", linewidth=0.6, alpha=0.85, zorder=6)
            ax_rea_v2.set_title(f"{pid_label} | REA (%)")
            ax_rea_v2.set_xlabel("Heat time (min)")
            ax_rea_v2.set_ylabel("REA (%)")
            ax_rea_v2.set_xlim(0.0, 62.5)
            y_rea_vals = [v for v in rea.tolist() if np.isfinite(v)]
            if ref_rea.size > 0:
                y_rea_vals.extend([v for v in ref_rea.tolist() if np.isfinite(v)])
            if y_rea_vals:
                y_max_rea = float(np.nanmax(np.asarray(y_rea_vals, dtype=float)))
                ax_rea_v2.set_ylim(0.0, max(100.0, y_max_rea * 1.08))
            else:
                ax_rea_v2.set_ylim(0.0, 100.0)
            ax_rea_v2.tick_params(axis="both", which="both", length=0, labelsize=6)
            ax_rea_v2.spines["top"].set_visible(False)
            ax_rea_v2.spines["right"].set_visible(False)
            ax_rea_v2.spines["left"].set_color("0.7")
            ax_rea_v2.spines["bottom"].set_color("0.7")
            ax_rea_v2.spines["left"].set_zorder(-10)
            ax_rea_v2.spines["bottom"].set_zorder(-10)

            # U(t) panel
            ax_u_v2.scatter(t, u_series, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=30, clip_on=False)
            ax_u_v2.plot(t, u_series, color=color, linewidth=0.8, alpha=0.85, zorder=8, clip_on=False)
            ax_u_v2.axhline(
                y=float(native_activity_min_rel),
                color="0.35",
                linestyle=(0, (3, 2)),
                linewidth=0.8,
                alpha=0.9,
                zorder=4,
            )
            if t_theta_val is not None and np.isfinite(float(t_theta_val)):
                ax_u_v2.axvline(
                    x=float(t_theta_val),
                    color=color,
                    linestyle=(0, (3, 2)),
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=4,
                )
            if t_theta_val is not None and np.isfinite(float(t_theta_val)):
                u_note = (
                    rf"$\theta={float(native_activity_min_rel):.2f}$" + "\n"
                    + rf"$t_{{\theta}}={float(t_theta_val):.3g}\,\mathrm{{min}}$"
                )
            else:
                u_note = (
                    rf"$\theta={float(native_activity_min_rel):.2f}$" + "\n"
                    + r"$t_{\theta}=\mathrm{NA}$"
                )
            flag_note = _format_t_theta_flag_note(t_theta_flag)
            if flag_note:
                u_note += f"\n{flag_note}"
            txt_u_v2 = ax_u_v2.annotate(
                u_note,
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=7.0,
                bbox=dict(
                    boxstyle=f"round,pad={max(0.16, (float(info_pad[0]) if isinstance(info_pad, (tuple, list)) else float(info_pad)) - 0.03)}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.93,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_u_v2.get_bbox_patch() is not None:
                txt_u_v2.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())
            ax_u_v2.set_title(f"{pid_label} | U(t)=abs/ref0")
            ax_u_v2.set_xlabel("Heat time (min)")
            ax_u_v2.set_ylabel("U(t) (-)")
            ax_u_v2.set_xlim(0.0, 62.5)
            u_vals_finite = u_series[np.isfinite(u_series)]
            u_top = float(np.nanmax(u_vals_finite)) * 1.1 if u_vals_finite.size > 0 else 1.2
            ax_u_v2.set_ylim(0.0, max(1.2, u_top, float(native_activity_min_rel) * 1.15))
            ax_u_v2.tick_params(axis="both", which="both", length=0, labelsize=6)
            ax_u_v2.spines["top"].set_visible(False)
            ax_u_v2.spines["right"].set_visible(False)
            ax_u_v2.spines["left"].set_color("0.7")
            ax_u_v2.spines["bottom"].set_color("0.7")
            ax_u_v2.spines["left"].set_zorder(-10)
            ax_u_v2.spines["bottom"].set_zorder(-10)

            fig_v2.tight_layout(pad=0.3)
            fig_v2.subplots_adjust(left=0.08, wspace=0.28)
            out_v2_path = out_per_polymer_v2 / f"{stem}__{run_id}.png"
            fig_v2.savefig(
                out_v2_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig_v2)
            # Memory optimization: periodically force garbage collection after plot generation
            if len(t50_rows) % 5 == 0:  # Every 5 polymers
                gc.collect()

        # --- v3 figure: ref-normalized activity only (U_ref(t)=abs/ref(0)) with GOx overlay
        with plt.rc_context(_style):
            fig_ref, ax_ref = plt.subplots(1, 1, figsize=(3.5, 2.8))

            # Polymer curve on reference-at-0 scale.
            finite_poly_u = np.isfinite(u_series)
            if np.any(finite_poly_u):
                ax_ref.plot(
                    t,
                    u_series,
                    color=color,
                    linewidth=0.8,
                    alpha=0.90,
                    zorder=12,
                    clip_on=False,
                )
                ax_ref.scatter(
                    t,
                    u_series,
                    s=12,
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.4,
                    alpha=1.0,
                    zorder=30,
                    clip_on=False,
                    label=pid_label,
                )

            # Reference curve on the same axis: abs_ref(t) / abs_ref(0).
            ref_u_series = np.array([], dtype=float)
            if (
                gox_ref_t.size > 0
                and gox_ref_y.size > 0
                and np.isfinite(gox_ref_at_0)
                and float(gox_ref_at_0) > 0.0
            ):
                ref_u_series = gox_ref_y / float(gox_ref_at_0)
                ax_ref.plot(
                    gox_ref_t,
                    ref_u_series,
                    color="#808080",
                    linewidth=0.8,
                    alpha=0.85,
                    zorder=9,
                    clip_on=False,
                )
                ax_ref.scatter(
                    gox_ref_t,
                    ref_u_series,
                    s=10,
                    marker="s",
                    color="#808080",
                    edgecolors="0.2",
                    linewidths=0.3,
                    alpha=0.90,
                    zorder=24,
                    clip_on=False,
                    label=f"{reference_polymer_id} reference",
                )

            # Feasibility threshold and t_theta on the same normalized scale.
            ax_ref.axhline(
                y=float(native_activity_min_rel),
                color="0.35",
                linestyle=(0, (3, 2)),
                linewidth=0.8,
                alpha=0.95,
                zorder=4,
            )
            if t_theta_val is not None and np.isfinite(float(t_theta_val)):
                ax_ref.axvline(
                    x=float(t_theta_val),
                    color=color,
                    linestyle=(0, (3, 2)),
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=4,
                )

            u0_math = (
                f"{float(native_rel_at_0):.3g}"
                if np.isfinite(native_rel_at_0)
                else r"\mathrm{NA}"
            )
            t_theta_math = (
                f"{float(t_theta_val):.3g}\\,\\mathrm{{min}}"
                if (t_theta_val is not None and np.isfinite(float(t_theta_val)))
                else r"\mathrm{NA}"
            )
            note_lines = [
                rf"$U_{{0}}={u0_math}$",
                rf"$\theta={float(native_activity_min_rel):.2f}$",
                rf"$t_{{\theta}}={t_theta_math}$",
            ]
            flag_note = _format_t_theta_flag_note(t_theta_flag)
            if flag_note:
                note_lines.append(flag_note)
            if not np.any(finite_poly_u):
                note_lines.append("U(t): unavailable")
            if ref_u_series.size == 0:
                note_lines.append("reference profile: unavailable")
            note_lines.append(f"ref source: {gox_ref_source}")
            txt_ref = ax_ref.annotate(
                "\n".join(note_lines),
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=7.0,
                bbox=dict(
                    boxstyle=f"round,pad={max(0.16, (float(info_pad[0]) if isinstance(info_pad, (tuple, list)) else float(info_pad)) - 0.03)}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.93,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_ref.get_bbox_patch() is not None:
                txt_ref.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())

            ax_ref.set_title(f"{pid_label} | Ref-normalized activity")
            ax_ref.set_xlabel("Heat time (min)")
            ax_ref.set_ylabel(r"$U_{\mathrm{ref}}(t)=\mathrm{abs}(t)/\mathrm{ref}(0)$")
            ax_ref.set_xticks(HEAT_TICKS_0_60)
            ax_ref.set_xlim(0.0, 62.5)

            y_vals_ref: list[float] = [float(native_activity_min_rel), 1.0]
            if np.any(finite_poly_u):
                y_vals_ref.extend([float(v) for v in u_series[finite_poly_u]])
            if ref_u_series.size > 0:
                y_vals_ref.extend([float(v) for v in ref_u_series[np.isfinite(ref_u_series)]])
            y_vals_ref_arr = np.asarray([v for v in y_vals_ref if np.isfinite(v)], dtype=float)
            if y_vals_ref_arr.size > 0:
                y_top_ref = float(np.nanmax(y_vals_ref_arr))
                ax_ref.set_ylim(0.0, max(1.2, y_top_ref * 1.10, float(native_activity_min_rel) * 1.15))
            else:
                ax_ref.set_ylim(0.0, 1.2)

            ax_ref.tick_params(axis="both", which="both", length=0, labelsize=6)
            ax_ref.spines["top"].set_visible(False)
            ax_ref.spines["right"].set_visible(False)
            ax_ref.spines["left"].set_visible(True)
            ax_ref.spines["left"].set_color("0.7")
            ax_ref.spines["left"].set_zorder(-10)
            ax_ref.spines["bottom"].set_visible(True)
            ax_ref.spines["bottom"].set_color("0.7")
            ax_ref.spines["bottom"].set_zorder(-10)

            handles_ref, labels_ref = ax_ref.get_legend_handles_labels()
            if handles_ref:
                ax_ref.legend(
                    handles_ref,
                    labels_ref,
                    loc="lower left",
                    fontsize=5.8,
                    frameon=True,
                    framealpha=0.90,
                    borderpad=0.25,
                )

            fig_ref.tight_layout(pad=0.3)
            fig_ref.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.16)
            out_ref_path = out_per_polymer_refnorm / f"{stem}__{run_id}.png"
            fig_ref.savefig(
                out_ref_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig_ref)

        t50_rows.append(
            {
                "run_id": run_id,
                "polymer_id": str(pid),
                "polymer_label": pid_label,
                "reference_polymer_id": str(reference_polymer_id),
                "n_points": int(len(g)),
                "abs_activity_at_0": abs_activity_at_0,
                "native_activity_rel_at_0": native_rel_at_0,
                "native_0": native_rel_at_0,
                "native_activity_min_rel_threshold": float(native_activity_min_rel),
                "native_activity_feasible": int(native_feasible),
                "abs_activity_at_20": abs_activity_at_20,
                "u_at_20": u_at_20,
                "t_theta_min": float(t_theta_val) if (t_theta_val is not None and np.isfinite(float(t_theta_val))) else np.nan,
                "t_theta_censor_flag": str(t_theta_flag),
                "functional_activity_at_20_rel": func_at_20,
                "gox_abs_activity_at_0_ref": gox_ref_at_0,
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
    t50_path = out_t50_csv_dir / f"t50__{run_id}.csv"
    t50_df.to_csv(t50_path, index=False)
    legacy_t50_path = out_t50_dir / f"t50__{run_id}.csv"
    if legacy_t50_path.is_file():
        legacy_t50_path.unlink(missing_ok=True)

    # Gallery: all polymers, right panel only ("REA comparison and FoG ratio"), 5 columns.
    rea_fog_grid_path = out_t50_dir / f"rea_comparison_fog_grid__{run_id}.png"
    rea_fog_panel_pngs_ordered = [
        out_rea_fog_panel / f"{stems.get(pid, safe_stem(pid))}__{run_id}.png"
        for pid in polymer_ids
    ]
    _write_rea_fog_gallery_from_per_polymer(
        per_polymer_png_paths=rea_fog_panel_pngs_ordered,
        out_png=rea_fog_grid_path,
        n_cols=5,
        crop_left_frac=0.0,
        gap_px=10,
        outer_pad_px=10,
    )

    # --- All polymers comparison plots (overlay: Absolute, REA, and objective trade-off)
    # Generate two versions: one with include_in_all_polymers=True only (default), one with all polymers (for debugging)
    import gc
    _style = apply_paper_style()
    
    # Version 1: Filtered (only include_in_all_polymers=True) - this is the default/main output
    # Filter out polymers with include_in_all_polymers=False
    # After _parse_bool_flag, the column should be boolean, but ensure it's bool type for comparison
    df_for_downstream = df[~df["polymer_id"].astype(str).map(_is_background_polymer_id)].copy()
    mask = df_for_downstream["include_in_all_polymers"].astype(bool) == True
    df_filtered = df_for_downstream[mask].copy()
    polymer_ids_filtered = sorted(df_filtered["polymer_id"].astype(str).unique().tolist()) if not df_filtered.empty else []
    
    # Debug: print excluded polymers (can be removed later)
    excluded_polymers = (
        sorted(df_for_downstream[~mask]["polymer_id"].astype(str).unique().tolist())
        if not df_for_downstream[~mask].empty
        else []
    )
    if excluded_polymers:
        print(f"Info: Excluding polymers from all_polymers plot (include_in_all_polymers=False): {excluded_polymers}")
    
    # Version 2: All polymers (for comparison/debugging) - only generated if different from filtered
    df_all = df_for_downstream.copy()
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
                # Reference polymer always uses gray color.
                pid_str = str(pid)
                if _normalize_polymer_id_token(pid_str) == reference_polymer_norm:
                    color = "#808080"  # Medium gray
                else:
                    color = cmap.get(pid_str, "#0072B2")
                pid_label = safe_label(pid_str)

                # Plot data points (high zorder so points appear in front of axes, especially heat time 0)
                # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
                # alpha=1.0 for fully opaque plots
                ax_abs.scatter(t, aa, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)

                # Plot fit curve: same color correspondence as per_polymer.
                # REA fit keeps y0 constrained at 100.
                y0_abs_init = float(aa[0]) if aa.size > 0 and np.isfinite(float(aa[0])) else None
                fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
                is_ref = _normalize_polymer_id_token(pid_str) == reference_polymer_norm
                if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                    _draw_fit_with_extension(ax_abs, fit_abs, t, color, use_dashed_main=True, preserve_gray=is_ref)

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
                # Reference polymer always uses gray color.
                pid_str = str(pid)
                if _normalize_polymer_id_token(pid_str) == reference_polymer_norm:
                    color = "#808080"  # Medium gray
                else:
                    color = cmap.get(pid_str, "#0072B2")
                pid_label = safe_label(pid_str)

                # Plot data points (high zorder so points appear in front of axes, especially heat time 0)
                # clip_on=False to prevent clipping at axes boundary (especially for heat_time=0 points)
                # alpha=1.0 for fully opaque plots
                ax_rea.scatter(t, rea, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)

                # Plot fit curve: same color correspondence as per_polymer.
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
                is_ref = _normalize_polymer_id_token(pid_str) == reference_polymer_norm
                if use_exp_rea:
                    _draw_fit_with_extension(ax_rea, fit_rea, t, color, use_dashed_main=True, preserve_gray=is_ref)

            # Restore per-axis color legend (polymer ID mapping) at upper-right.
            abs_handles, abs_labels = ax_abs.get_legend_handles_labels()
            if abs_handles:
                ax_abs.legend(
                    abs_handles,
                    abs_labels,
                    loc="upper right",
                    frameon=True,
                    fontsize=5.8,
                    ncol=1,
                    columnspacing=0.4,
                    handlelength=1.0,
                    handletextpad=0.3,
                )
            rea_handles, rea_labels = ax_rea.get_legend_handles_labels()
            if rea_handles:
                ax_rea.legend(
                    rea_handles,
                    rea_labels,
                    loc="upper right",
                    frameon=True,
                    fontsize=5.8,
                    ncol=1,
                    columnspacing=0.4,
                    handlelength=1.0,
                    handletextpad=0.3,
                )

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

            # Right: decision chart for the constrained objective.
            # Rule shown in one panel:
            #   1) eligible when native_0 >= theta
            #   2) among eligible polymers, rank by FoG.
            objective_rows: list[dict[str, Any]] = []
            for pid in polymer_ids_plot:
                pid_str = str(pid)
                row_df = t50_df[t50_df["polymer_id"].astype(str) == pid_str]
                if row_df.empty:
                    continue
                row0 = row_df.iloc[0]
                native_rel = pd.to_numeric(row0.get("native_activity_rel_at_0", np.nan), errors="coerce")
                t50_exp = pd.to_numeric(row0.get("t50_exp_min", np.nan), errors="coerce")
                t50_lin = pd.to_numeric(row0.get("t50_linear_min", np.nan), errors="coerce")
                t50_used = np.nan
                if np.isfinite(t50_exp):
                    t50_used = float(t50_exp)
                elif np.isfinite(t50_lin):
                    t50_used = float(t50_lin)
                fog_val = np.nan
                if (
                    np.isfinite(t50_used)
                    and t50_used > 0.0
                    and ref_rea_t50_show is not None
                    and np.isfinite(float(ref_rea_t50_show))
                    and float(ref_rea_t50_show) > 0.0
                ):
                    fog_val = float(t50_used) / float(ref_rea_t50_show)
                if not (np.isfinite(native_rel) and np.isfinite(fog_val)):
                    continue
                is_ref_obj = _normalize_polymer_id_token(pid_str) == reference_polymer_norm
                feasible = bool(float(native_rel) >= float(native_activity_min_rel))
                objective_rows.append(
                    {
                        "polymer_id": pid_str,
                        "label": safe_label(pid_str),
                        "native_rel": float(native_rel),
                        "fog": float(fog_val),
                        "feasible": feasible,
                        "is_ref": is_ref_obj,
                        "color": "#808080" if is_ref_obj else cmap.get(pid_str, "#0072B2"),
                    }
                )

            obj_df = pd.DataFrame(
                columns=["polymer_id", "label", "native_rel", "fog", "feasible", "is_ref", "color"]
            )
            if objective_rows:
                obj_df = pd.DataFrame(objective_rows)
                obj_df = obj_df.sort_values(
                    ["feasible", "fog"],
                    ascending=[False, False],
                    kind="mergesort",
                ).reset_index(drop=True)

                # Keep the panel readable when many polymers are present.
                max_rows = 16
                if len(obj_df) > max_rows:
                    ref_df = obj_df[obj_df["is_ref"]].head(1)
                    keep_n = max(0, max_rows - len(ref_df))
                    non_ref_df = obj_df[~obj_df["is_ref"]].head(keep_n)
                    obj_df = pd.concat([ref_df, non_ref_df], ignore_index=True)
                    obj_df = obj_df.sort_values(
                        ["feasible", "fog"],
                        ascending=[False, False],
                        kind="mergesort",
                    ).reset_index(drop=True)

                y = np.arange(len(obj_df), dtype=float)
                bars = ax_func.barh(
                    y,
                    obj_df["fog"].to_numpy(dtype=float),
                    color=obj_df["color"].tolist(),
                    edgecolor="0.25",
                    linewidth=0.35,
                    height=0.62,
                    alpha=0.9,
                    zorder=12,
                )
                for bar, feasible in zip(bars, obj_df["feasible"].tolist()):
                    if not bool(feasible):
                        bar.set_alpha(0.45)
                        bar.set_hatch("//")
                        bar.set_edgecolor("0.45")

                # FoG reference baseline.
                ax_func.axvline(
                    x=1.0,
                    color="0.45",
                    linestyle=(0, (3, 2)),
                    linewidth=0.75,
                    alpha=0.85,
                    zorder=8,
                )
                ax_func.set_yticks(y)
                ax_func.set_yticklabels(obj_df["label"].tolist(), fontsize=5.4)
                ax_func.invert_yaxis()
                ax_func.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
                ax_func.set_title("Decision chart (native gate + FoG rank)")

                fog_vals = obj_df["fog"].to_numpy(dtype=float)
                fog_max = float(np.nanmax(fog_vals)) if fog_vals.size > 0 else 1.0
                fog_right = max(1.4, fog_max * 1.15)
                ax_func.set_xlim(0.0, fog_right)

                # Overlay native activity on a top x-axis with the gate line.
                ax_native = ax_func.twiny()
                ax_native.scatter(
                    obj_df["native_rel"].to_numpy(dtype=float),
                    y,
                    marker="D",
                    s=12,
                    color="0.12",
                    edgecolors="none",
                    alpha=0.9,
                    zorder=30,
                )
                ax_native.axvline(
                    x=float(native_activity_min_rel),
                    color="0.15",
                    linestyle=(0, (3, 2)),
                    linewidth=0.75,
                    alpha=0.9,
                    zorder=25,
                )
                native_vals = obj_df["native_rel"].to_numpy(dtype=float)
                native_max = float(np.nanmax(native_vals)) if native_vals.size > 0 else 1.0
                native_right = max(1.05, float(native_activity_min_rel) * 1.25, native_max * 1.12)
                ax_native.set_xlim(0.0, native_right)
                ax_native.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
                ax_native.tick_params(axis="x", which="both", labelsize=5.2, length=0)
                ax_native.spines["bottom"].set_visible(False)
                ax_native.spines["left"].set_visible(False)
                ax_native.spines["right"].set_visible(False)

                feasible_n = int(np.sum(obj_df["feasible"].to_numpy(dtype=bool)))
                objective_note = (
                    rf"$\mathrm{{Rule}}:\ U_{{0}}\geq {float(native_activity_min_rel):.2f}$"
                    + "\n"
                    + rf"$\mathrm{{Rank\ by\ FoG\ among\ eligible}}:\ {feasible_n}/{len(obj_df)}$"
                )
                txt_obj_all = ax_func.annotate(
                    objective_note,
                    xy=(0, 1),
                    xycoords="axes fraction",
                    xytext=(6, -6),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    multialignment="left",
                    fontsize=6.0,
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor=INFO_BOX_FACE_COLOR,
                        alpha=0.92,
                        edgecolor="none",
                    ),
                    zorder=35,
                )
                if txt_obj_all.get_bbox_patch() is not None:
                    txt_obj_all.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())
            else:
                ax_func.set_title("Decision chart (native gate + FoG rank)")
                ax_func.text(
                    0.5,
                    0.5,
                    "No valid FoG/native points",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="0.35",
                    transform=ax_func.transAxes,
                )
                ax_func.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
                ax_func.set_yticks([])
                ax_func.set_xlim(0.0, 1.5)
                ax_func.set_ylim(0.0, 1.0)

            ax_func.spines["top"].set_visible(False)
            ax_func.spines["right"].set_visible(False)
            ax_func.spines["left"].set_visible(True)
            ax_func.spines["left"].set_color("0.7")  # Light gray
            ax_func.spines["left"].set_zorder(-10)  # Behind data points
            ax_func.spines["bottom"].set_visible(True)
            ax_func.spines["bottom"].set_color("0.7")  # Light gray
            ax_func.spines["bottom"].set_zorder(-10)  # Behind data points

            fig_all.tight_layout(pad=0.3)
            # Increase left margin to accommodate clip_on=False markers (especially heat_time=0 points)
            fig_all.subplots_adjust(left=0.08, wspace=0.30)
            
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

            # Save an additional split-axis decision figure:
            # left panel = FoG bars, right panel = U0 diamonds (no twinned x-axes).
            fig_split, (ax_fog_split, ax_u0_split) = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
            if not obj_df.empty:
                y_split = np.arange(len(obj_df), dtype=float)
                bars_split = ax_fog_split.barh(
                    y_split,
                    obj_df["fog"].to_numpy(dtype=float),
                    color=obj_df["color"].tolist(),
                    edgecolor="0.25",
                    linewidth=0.35,
                    height=0.62,
                    alpha=0.9,
                    zorder=12,
                )
                for bar, feasible in zip(bars_split, obj_df["feasible"].tolist()):
                    if not bool(feasible):
                        bar.set_alpha(0.45)
                        bar.set_hatch("//")
                        bar.set_edgecolor("0.45")
                ax_fog_split.axvline(
                    x=1.0,
                    color="0.45",
                    linestyle=(0, (3, 2)),
                    linewidth=0.75,
                    alpha=0.85,
                    zorder=8,
                )
                ax_fog_split.set_yticks(y_split)
                ax_fog_split.set_yticklabels(obj_df["label"].tolist(), fontsize=5.4)
                ax_fog_split.invert_yaxis()
                ax_fog_split.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
                ax_fog_split.set_title("FoG ranking")
                fog_vals_split = obj_df["fog"].to_numpy(dtype=float)
                fog_max_split = float(np.nanmax(fog_vals_split)) if fog_vals_split.size > 0 else 1.0
                fog_right_split = max(1.4, fog_max_split * 1.15)
                ax_fog_split.set_xlim(0.0, fog_right_split)

                ax_u0_split.scatter(
                    obj_df["native_rel"].to_numpy(dtype=float),
                    y_split,
                    marker="D",
                    s=14,
                    c=obj_df["color"].tolist(),
                    edgecolors="0.2",
                    linewidths=0.25,
                    alpha=0.9,
                    zorder=20,
                )
                infeasible_mask_split = ~obj_df["feasible"].to_numpy(dtype=bool)
                if np.any(infeasible_mask_split):
                    ax_u0_split.scatter(
                        obj_df.loc[infeasible_mask_split, "native_rel"].to_numpy(dtype=float),
                        y_split[infeasible_mask_split],
                        marker="D",
                        s=16,
                        facecolors="none",
                        edgecolors="0.35",
                        linewidths=0.6,
                        alpha=0.9,
                        zorder=25,
                    )
                ax_u0_split.axvline(
                    x=float(native_activity_min_rel),
                    color="0.15",
                    linestyle=(0, (3, 2)),
                    linewidth=0.75,
                    alpha=0.9,
                    zorder=18,
                )
                native_vals_split = obj_df["native_rel"].to_numpy(dtype=float)
                native_max_split = float(np.nanmax(native_vals_split)) if native_vals_split.size > 0 else 1.0
                native_right_split = max(
                    1.05,
                    float(native_activity_min_rel) * 1.25,
                    native_max_split * 1.12,
                )
                ax_u0_split.set_xlim(0.0, native_right_split)
                ax_u0_split.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
                ax_u0_split.set_title(r"$U_0$ gate")
                ax_u0_split.tick_params(axis="y", left=False, labelleft=False)
            else:
                ax_fog_split.text(
                    0.5,
                    0.5,
                    "No valid FoG points",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="0.35",
                    transform=ax_fog_split.transAxes,
                )
                ax_fog_split.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
                ax_fog_split.set_title("FoG ranking")
                ax_fog_split.set_yticks([])
                ax_fog_split.set_xlim(0.0, 1.5)

                ax_u0_split.text(
                    0.5,
                    0.5,
                    "No valid $U_0$ points",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="0.35",
                    transform=ax_u0_split.transAxes,
                )
                ax_u0_split.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
                ax_u0_split.set_title(r"$U_0$ gate")
                ax_u0_split.set_yticks([])
                ax_u0_split.set_xlim(0.0, 1.05)

            for ax in (ax_fog_split, ax_u0_split):
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_color("0.7")
                ax.spines["left"].set_zorder(-10)
                ax.spines["bottom"].set_visible(True)
                ax.spines["bottom"].set_color("0.7")
                ax.spines["bottom"].set_zorder(-10)

            fig_split.tight_layout(pad=0.3)
            fig_split.subplots_adjust(left=0.20, wspace=0.24)
            out_split_path = out_fit_dir / f"all_polymers_decision_split{suffix}__{run_id}.png"
            fig_split.savefig(
                out_split_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            plt.close(fig_split)
            # Memory optimization: force garbage collection after large plot generation
            gc.collect()
    
    # Generate both versions: filtered (default) and all (for debugging if different)
    _plot_all_polymers_subplot(df_filtered, polymer_ids_filtered, suffix="")  # Default: only include_in_all_polymers=True
    if polymer_ids_filtered != polymer_ids_all:  # Only generate "all" version if different from filtered
        _plot_all_polymers_subplot(df_all, polymer_ids_all, suffix="_all")
    
    # Generate custom pair plot if all_polymers_pair=True polymers are specified
    # Only include polymers with all_polymers_pair=True (exclude all others)
    if "all_polymers_pair" in df_for_downstream.columns and df_for_downstream["all_polymers_pair"].any():
        # Ensure boolean comparison works correctly
        mask_pair = df_for_downstream["all_polymers_pair"].astype(bool) == True
        df_pair = df_for_downstream[mask_pair].copy()
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

    # --- Representative 4 polymers comparison plot
    # Fixed controls: reference polymer and PMPC (only when representative-eligible).
    # PMBTA slots: highest/lowest t50 among representative-eligible PMBTA polymers.
    # representative-eligible:
    #   - REA exponential fit succeeded (fit_model non-empty; t50_exp_min finite)
    #   - native activity at 0 min strictly above 0.70 (exclude <= 0.70)
    #   - include_in_all_polymers is True
    import gc
    representative_pids = []
    
    # Exclusions for representative selection
    excluded_for_t50_selection = {pid for pid in polymer_ids if _normalize_polymer_id_token(pid) == reference_polymer_norm}
    excluded_for_t50_selection.add("PMPC")
    if "include_in_all_polymers" in df.columns:
        # Ensure boolean comparison works correctly
        mask_excluded = df["include_in_all_polymers"].astype(bool) == False
        excluded_pids = set(df[mask_excluded]["polymer_id"].astype(str).unique())
        excluded_for_t50_selection.update(excluded_pids)

    rep_native_theta = 0.70

    def _first_finite_value(vals: pd.Series) -> float:
        arr = pd.to_numeric(vals, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr[0]) if arr.size > 0 else np.nan

    def _is_representative_eligible(pid: str) -> bool:
        pid_str = str(pid).strip()
        pid_df = t50_df[t50_df["polymer_id"].astype(str) == pid_str]
        if pid_df.empty:
            return False
        fit_model = str(pid_df["fit_model"].iloc[0]).strip() if "fit_model" in pid_df.columns else ""
        t50_exp = _first_finite_value(pid_df["t50_exp_min"]) if "t50_exp_min" in pid_df.columns else np.nan
        native0 = _first_finite_value(pid_df["native_activity_rel_at_0"]) if "native_activity_rel_at_0" in pid_df.columns else np.nan
        if not fit_model:
            return False
        if not (np.isfinite(t50_exp) and float(t50_exp) > 0.0):
            return False
        if not (np.isfinite(native0) and float(native0) > rep_native_theta):
            return False
        return True

    # Fixed controls: include only when representative-eligible.
    ref_candidates = [pid for pid in polymer_ids if _normalize_polymer_id_token(pid) == reference_polymer_norm]
    for ref_pid in ref_candidates[:1]:
        if _is_representative_eligible(str(ref_pid)):
            representative_pids.append(str(ref_pid))

    for pid in polymer_ids:
        if _normalize_polymer_id_token(pid) == "PMPC" and _is_representative_eligible(str(pid)):
            if str(pid) not in representative_pids:
                representative_pids.append(str(pid))
            break

    # PMBTA top/bottom (exclude reference/PMPC/include_in_all_polymers=False and ineligible rows).
    pmbta_candidates: list[tuple[str, float]] = []
    for pid in t50_df["polymer_id"].astype(str).unique():
        pid_str = str(pid).strip()
        if pid_str in excluded_for_t50_selection:
            continue
        if not _normalize_polymer_id_token(pid_str).startswith("PMBTA"):
            continue
        if not _is_representative_eligible(pid_str):
            continue
        pid_df = t50_df[t50_df["polymer_id"].astype(str) == pid_str]
        t50_exp = _first_finite_value(pid_df["t50_exp_min"])
        if np.isfinite(t50_exp) and float(t50_exp) > 0.0:
            pmbta_candidates.append((pid_str, float(t50_exp)))

    if pmbta_candidates:
        pmbta_candidates.sort(key=lambda x: x[1])
        bottom_pid = pmbta_candidates[0][0]
        top_pid = pmbta_candidates[-1][0]
        if top_pid not in representative_pids:
            representative_pids.append(top_pid)
        if bottom_pid not in representative_pids:
            representative_pids.append(bottom_pid)
    else:
        print(
            "Warning: No representative-eligible PMBTA polymer found "
            "(requires exp fit success, native_0 > 0.70, include_in_all_polymers=True)."
        )
    
    # Create representative plot if we have at least one polymer
    if representative_pids:
        _style = apply_paper_style()
        with plt.rc_context(_style):
            fig_rep, (ax_abs_rep, ax_rea_rep) = plt.subplots(1, 2, figsize=(10.0, 3.5))
            
            # Left: Absolute activity
            for pid in representative_pids:
                if pid not in df["polymer_id"].values:
                    continue
                g = df[df["polymer_id"] == pid].sort_values("heat_min").reset_index(drop=True)
                t = g["heat_min"].to_numpy(dtype=float)
                aa = g["abs_activity"].to_numpy(dtype=float)
                # Reference polymer always uses gray color (exact match, case-insensitive)
                pid_str = str(pid).strip()
                if _is_reference_polymer_id(pid_str, reference_polymer_id=reference_polymer_id):
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
                is_gox = _is_reference_polymer_id(pid_str, reference_polymer_id=reference_polymer_id)
                if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                    _draw_fit_with_extension(ax_abs_rep, fit_abs, t, color, preserve_gray=is_gox)
            
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
                # Reference polymer always uses gray color (exact match, case-insensitive)
                pid_str = str(pid).strip()
                if _is_reference_polymer_id(pid_str, reference_polymer_id=reference_polymer_id):
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
                is_gox = _is_reference_polymer_id(pid_str, reference_polymer_id=reference_polymer_id)
                if use_exp_rea:
                    _draw_fit_with_extension(ax_rea_rep, fit_rea, t, color, preserve_gray=is_gox)
                    # Also track maximum from fitted curve
                    if fit_rea is not None:
                        t_eval = np.linspace(0.0, 60.0, 200)
                        y_eval = _eval_fit_curve(fit_rea, t_eval)
                        y_eval_finite = y_eval[np.isfinite(y_eval)]
                        if y_eval_finite.size > 0:
                            y_max_rea_rep = max(y_max_rea_rep, float(np.max(y_eval_finite)))
            
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
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    native_activity_min_rel: float = 0.70,
    t50_definition: str = T50_DEFINITION_Y0_HALF,
    dpi: int = 600,
    error_band_suffix: str = "",
) -> Optional[Path]:
    """
    Additional per-polymer time-series plots with error bars (mean ± SEM).

    This is intended for runs where the same polymer_id has replicate wells
    at the same heat_min (n > 1). Output:
      - out_fit_dir/per_polymer_with_error{error_band_suffix}__{run_id}/{polymer_stem}__{run_id}.png
      - out_fit_dir/all_polymers_with_error{error_band_suffix}__{run_id}.png

    Use error_band_suffix="" for stats with outlier exclusion (robust);
    use error_band_suffix="_all" for stats from all replicates (no exclusion).

    Returns output directory path when plots are written, otherwise None.
    """
    summary_stats_path = Path(summary_stats_path)
    out_fit_dir = Path(out_fit_dir)
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if not summary_stats_path.is_file():
        return None

    t50_definition = normalize_t50_definition(t50_definition)
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_norm = _normalize_polymer_id_token(reference_polymer_id)

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

    # Parse include flag in the same way as plot_per_polymer_timeseries().
    if "include_in_all_polymers" in df.columns:
        def _parse_bool_flag(v: Any) -> bool:
            if pd.isna(v):
                return True
            if isinstance(v, bool):
                return v
            s = str(v).strip().upper()
            if s in ("TRUE", "1", "YES"):
                return True
            if s in ("FALSE", "0", "NO", ""):
                return False
            return True

        df["include_in_all_polymers"] = df["include_in_all_polymers"].apply(_parse_bool_flag)
    else:
        df["include_in_all_polymers"] = True

    err_suffix = (error_band_suffix or "").strip()
    # Only add this version when replicate information exists.
    if not (df["n"] > 1).any():
        stale_all_with_error = out_fit_dir / f"all_polymers_with_error{err_suffix}__{run_id}.png"
        if stale_all_with_error.is_file():
            stale_all_with_error.unlink(missing_ok=True)
        return None

    # Keep the same downstream rule as regular per-polymer plots.
    df = df[~df["polymer_id"].astype(str).map(_is_background_polymer_id)].copy()
    if df.empty:
        return None

    polymer_ids = sorted(
        df["polymer_id"].astype(str).unique().tolist(),
        key=_polymer_plot_order_key,
    )
    cmap = ensure_polymer_colors(
        polymer_ids,
        color_map_path=Path(color_map_path),
        reference_polymer_id=reference_polymer_id,
    )

    stems: dict[str, str] = {pid: safe_stem(pid) for pid in polymer_ids}
    stem_counts: dict[str, int] = {}
    for st in stems.values():
        stem_counts[st] = stem_counts.get(st, 0) + 1
    for pid, st in list(stems.items()):
        if stem_counts.get(st, 0) > 1:
            stems[pid] = f"{st}__{_short_hash(pid)}"

    out_dir = out_fit_dir / f"per_polymer_with_error{err_suffix}__{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_pngs = {f"{stems[pid]}__{run_id}.png" for pid in polymer_ids}
    for p in out_dir.glob(f"*__{run_id}.png"):
        if p.name not in expected_pngs:
            try:
                p.unlink()
            except Exception:
                pass

    t50_path = out_fit_dir / "t50" / "csv" / f"t50__{run_id}.csv"
    t50_df = pd.read_csv(t50_path) if t50_path.is_file() else pd.DataFrame()
    if not t50_df.empty and "polymer_id" in t50_df.columns:
        t50_df = t50_df.copy()
        t50_df["polymer_id"] = t50_df["polymer_id"].astype(str)
        for c in [
            "native_activity_rel_at_0",
            "t50_exp_min",
            "t50_linear_min",
            "fit_r2",
            "fog",
            "native_activity_min_rel_threshold",
        ]:
            if c in t50_df.columns:
                t50_df[c] = pd.to_numeric(t50_df[c], errors="coerce")

    from gox_plate_pipeline.fitting.core import (
        apply_paper_style,
        INFO_BOX_MARGIN_PT,
        INFO_BOX_FACE_COLOR,
        INFO_BOX_PAD_PER_POLYMER,
        PAPER_ERRORBAR_COLOR,
        get_info_box_gradient_shadow,
    )
    import matplotlib.pyplot as plt
    import gc
    import colorsys

    def _eval_fit_curve(fit_obj: ExpDecayFit, tt: np.ndarray) -> np.ndarray:
        if fit_obj.model == "exp":
            return fit_obj.y0 * np.exp(-fit_obj.k * tt)
        c = float(fit_obj.c) if fit_obj.c is not None else 0.0
        return c + (fit_obj.y0 - c) * np.exp(-fit_obj.k * tt)

    def _to_fluorescent_color(color_hex: str) -> str:
        hex_str = color_hex.lstrip("#")
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, s * 1.4)
        v = min(1.0, v * 1.1)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def _draw_fit_with_extension(
        ax: Any,
        fit_obj: ExpDecayFit,
        t_obs: np.ndarray,
        color_hex: str,
        *,
        use_dashed_main: bool = False,
        preserve_gray: bool = False,
        low_confidence: bool = False,
        force_dashed_main: bool = False,
    ) -> None:
        t_obs = np.asarray(t_obs, dtype=float)
        t_obs = t_obs[np.isfinite(t_obs)]
        if t_obs.size == 0:
            return
        t_min_obs = float(np.min(t_obs))
        t_max_obs = float(np.max(t_obs))
        if preserve_gray and color_hex.upper() == "#808080":
            color_main = color_hex
        else:
            color_main = _to_fluorescent_color(color_hex)

        tt_main = np.linspace(t_min_obs, t_max_obs, 220)
        yy_main = _eval_fit_curve(fit_obj, tt_main)
        main_alpha = 0.40 if use_dashed_main else 0.50
        ext_alpha = 0.30 if use_dashed_main else 0.40
        main_ls = (0, (2.0, 2.0)) if (low_confidence or force_dashed_main) else "-"
        ext_ls = (0, (1.6, 1.9)) if low_confidence else (0, (2.4, 2.4))
        if low_confidence:
            main_alpha = min(0.65, main_alpha + 0.12)
            ext_alpha = min(0.55, ext_alpha + 0.10)
        ax.plot(
            tt_main,
            yy_main,
            color=color_main,
            linewidth=1.7,
            alpha=main_alpha,
            linestyle=main_ls,
            zorder=8,
        )
        if t_min_obs > 0.0:
            tt_pre = np.linspace(0.0, t_min_obs, 120)
            yy_pre = _eval_fit_curve(fit_obj, tt_pre)
            ax.plot(
                tt_pre,
                yy_pre,
                color=color_main,
                linewidth=1.5,
                alpha=ext_alpha,
                linestyle=ext_ls,
                zorder=7,
            )
        if t_max_obs < 60.0:
            tt_post = np.linspace(t_max_obs, 60.0, 140)
            yy_post = _eval_fit_curve(fit_obj, tt_post)
            ax.plot(
                tt_post,
                yy_post,
                color=color_main,
                linewidth=1.5,
                alpha=ext_alpha,
                linestyle=ext_ls,
                zorder=7,
            )

    ref_g = df[
        df["polymer_id"].astype(str).map(_normalize_polymer_id_token) == reference_norm
    ].copy()
    ref_g = ref_g.sort_values("heat_min").reset_index(drop=True)
    ref_t = ref_g["heat_min"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    ref_abs_mean = (
        ref_g["mean_abs_activity"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    )
    ref_rea_mean = (
        ref_g["mean_REA_percent"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    )
    ref_rea_sem = (
        ref_g["sem_REA_percent"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    )
    ref_n = ref_g["n"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    ref_abs0 = value_at_time_linear(ref_t, ref_abs_mean, at_time_min=0.0) if ref_t.size > 0 else None
    fit_ref_rea: Optional[ExpDecayFit] = None
    use_exp_ref_rea = False
    ref_t50_show: Optional[float] = None
    if ref_t.size > 0 and ref_rea_mean.size > 0:
        fit_ref_rea = fit_exponential_decay(
            ref_t,
            ref_rea_mean,
            y0=100.0,
            fixed_y0=100.0,
            min_points=4,
            t50_definition=t50_definition,
        )
        use_exp_ref_rea = bool(
            fit_ref_rea is not None
            and np.isfinite(float(fit_ref_rea.r2))
            and float(fit_ref_rea.r2) >= 0.70
        )
        ref_t50_lin = t50_linear(
            ref_t,
            ref_rea_mean,
            y0=100.0,
            target_frac=0.5,
            target_value=(50.0 if t50_definition == T50_DEFINITION_REA50 else None),
        )
        ref_t50_model = fit_ref_rea.t50 if (fit_ref_rea is not None and use_exp_ref_rea) else None
        ref_t50_show = ref_t50_model if ref_t50_model is not None else ref_t50_lin

    plot_count = 0
    grouped_by_pid: dict[str, pd.DataFrame] = {
        str(pid): g for pid, g in df.groupby("polymer_id", sort=False)
    }
    for pid in polymer_ids:
        g = grouped_by_pid.get(str(pid))
        if g is None:
            continue
        g = g.sort_values("heat_min").reset_index(drop=True)
        t = g["heat_min"].to_numpy(dtype=float)
        n = g["n"].to_numpy(dtype=float)
        aa = g["mean_abs_activity"].to_numpy(dtype=float)
        aa_sem = g["sem_abs_activity"].to_numpy(dtype=float)
        rea = g["mean_REA_percent"].to_numpy(dtype=float)
        rea_sem = g["sem_REA_percent"].to_numpy(dtype=float)

        pid_str = str(pid)
        pid_norm = _normalize_polymer_id_token(pid_str)
        is_ref_per_polymer = pid_norm == reference_norm
        color = "#808080" if is_ref_per_polymer else cmap.get(pid_str, "#0072B2")
        pid_label = safe_label(pid_str)
        stem = stems.get(pid_str, safe_stem(pid_str))
        info_pad = INFO_BOX_PAD_PER_POLYMER
        n_finite = n[np.isfinite(n)]
        if n_finite.size > 0:
            n_min = int(np.nanmin(n_finite))
            n_max = int(np.nanmax(n_finite))
        else:
            n_min, n_max = 1, 1
        n_label = f"n={n_min}" if n_min == n_max else f"n={n_min}-{n_max}"

        y0_abs_init = float(aa[0]) if aa.size > 0 and np.isfinite(float(aa[0])) else None
        fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
        use_exp_abs = bool(
            fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70
        )
        fit_rea = fit_exponential_decay(
            t,
            rea,
            y0=100.0,
            fixed_y0=100.0,
            min_points=4,
            t50_definition=t50_definition,
        )
        use_exp_rea = bool(
            fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70
        )
        t50_lin = t50_linear(
            t,
            rea,
            y0=100.0,
            target_frac=0.5,
            target_value=(50.0 if t50_definition == T50_DEFINITION_REA50 else None),
        )
        t50_model = fit_rea.t50 if (fit_rea is not None and use_exp_rea) else None
        t50_poly = t50_model if t50_model is not None else t50_lin
        fog_from_t50 = (
            float(t50_poly) / float(ref_t50_show)
            if (
                t50_poly is not None
                and np.isfinite(float(t50_poly))
                and float(t50_poly) > 0.0
                and ref_t50_show is not None
                and np.isfinite(float(ref_t50_show))
                and float(ref_t50_show) > 0.0
            )
            else np.nan
        )
        abs0_poly = value_at_time_linear(t, aa, at_time_min=0.0)
        native_rel_at_0 = (
            float(abs0_poly) / float(ref_abs0)
            if (
                abs0_poly is not None
                and np.isfinite(float(abs0_poly))
                and ref_abs0 is not None
                and np.isfinite(float(ref_abs0))
                and float(ref_abs0) > 0.0
            )
            else np.nan
        )
        native_feasible = bool(
            np.isfinite(native_rel_at_0) and float(native_rel_at_0) >= float(native_activity_min_rel)
        )

        with plt.rc_context(apply_paper_style()):
            fig, (ax_left, ax_right, ax_func) = plt.subplots(1, 3, figsize=(10.4, 2.8))
            err_color = PAPER_ERRORBAR_COLOR
            err_alpha_main = 0.76

            # Left: Absolute activity with SEM error bars.
            ax_left.scatter(
                t, aa, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=6, clip_on=False
            )
            band_ok_abs = np.isfinite(aa_sem) & (aa_sem > 0) & (n >= 2)
            if np.any(band_ok_abs):
                ax_left.errorbar(
                    t[band_ok_abs],
                    aa[band_ok_abs],
                    yerr=aa_sem[band_ok_abs],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=err_alpha_main,
                    zorder=5,
                    clip_on=False,
                )
            if use_exp_abs and fit_abs is not None:
                _draw_fit_with_extension(ax_left, fit_abs, t, color, preserve_gray=is_ref_per_polymer)
            ax_left.set_title(f"{pid_label} | Absolute activity (mean ± SEM, {n_label})")
            ax_left.set_xlabel("Heat time (min)")
            ax_left.set_ylabel("Absolute activity (a.u./s)")
            ax_left.set_xticks(HEAT_TICKS_0_60)
            ax_left.set_xlim(0.0, 62.5)
            if np.isfinite(aa).any():
                ymax = float(np.nanmax(aa + np.where(np.isfinite(aa_sem), aa_sem, 0.0)))
                ax_left.set_ylim(0.0, max(ymax * 1.05, 1e-9))
            else:
                ax_left.set_ylim(0.0, 1.0)
            ax_left.tick_params(axis="both", which="both", length=0, labelsize=6)

            # Middle: REA with SEM error bars.
            ax_right.scatter(
                t, rea, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=6, clip_on=False
            )
            band_ok_rea = np.isfinite(rea_sem) & (rea_sem > 0) & (n >= 2)
            if np.any(band_ok_rea):
                ax_right.errorbar(
                    t[band_ok_rea],
                    rea[band_ok_rea],
                    yerr=rea_sem[band_ok_rea],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=err_alpha_main,
                    zorder=5,
                    clip_on=False,
                )
            if use_exp_rea and fit_rea is not None:
                _draw_fit_with_extension(ax_right, fit_rea, t, color, preserve_gray=is_ref_per_polymer)
            ax_right.set_title(f"{pid_label} | REA (mean ± SEM, {n_label})")
            ax_right.set_xlabel("Heat time (min)")
            ax_right.set_ylabel("REA (%)")
            ax_right.set_xticks(HEAT_TICKS_0_60)
            ax_right.set_xlim(0.0, 62.5)
            if np.isfinite(rea).any():
                ymax = float(np.nanmax(rea + np.where(np.isfinite(rea_sem), rea_sem, 0.0)))
                ax_right.set_ylim(0.0, max(ymax * 1.05, 1.0))
            else:
                ax_right.set_ylim(0.0, 100.0)
            ax_right.tick_params(axis="both", which="both", length=0, labelsize=6)

            # Right: REA comparison and FoG ratio.
            ax_func.scatter(
                t, rea, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=6, clip_on=False
            )
            if np.any(band_ok_rea) and (not is_ref_per_polymer):
                ax_func.errorbar(
                    t[band_ok_rea],
                    rea[band_ok_rea],
                    yerr=rea_sem[band_ok_rea],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=err_alpha_main,
                    zorder=5,
                    clip_on=False,
                )
            should_draw_rea_fit_obj = bool(fit_rea is not None and use_exp_rea)
            if should_draw_rea_fit_obj:
                _draw_fit_with_extension(
                    ax_func,
                    fit_rea,
                    t,
                    color,
                    preserve_gray=is_ref_per_polymer,
                    low_confidence=(not is_ref_per_polymer) and (not native_feasible),
                )

            if ref_t.size > 0 and ref_rea_mean.size > 0 and not is_ref_per_polymer:
                ax_func.scatter(
                    ref_t,
                    ref_rea_mean,
                    s=10,
                    marker="s",
                    color="#808080",
                    edgecolors="0.2",
                    linewidths=0.3,
                    alpha=0.82,
                    zorder=6,
                    clip_on=False,
                )
                if fit_ref_rea is not None:
                    _draw_fit_with_extension(
                        ax_func,
                        fit_ref_rea,
                        ref_t,
                        "#808080",
                        preserve_gray=True,
                        low_confidence=not use_exp_ref_rea,
                        force_dashed_main=True,
                    )
                elif ref_t.size > 1:
                    ax_func.plot(
                        ref_t,
                        ref_rea_mean,
                        color="#808080",
                        linewidth=1.3,
                        alpha=0.75,
                        linestyle=(0, (2.0, 2.0)),
                        zorder=7,
                    )

            target_line = 50.0
            t50_poly_guide_x = (
                float(t50_poly)
                if (t50_poly is not None and np.isfinite(float(t50_poly)) and float(t50_poly) > 0.0)
                else None
            )
            t50_ref_guide_x = (
                float(ref_t50_show)
                if (
                    ref_t50_show is not None
                    and np.isfinite(float(ref_t50_show))
                    and float(ref_t50_show) > 0.0
                    and not is_ref_per_polymer
                )
                else None
            )
            y_vals = []
            if np.any(np.isfinite(rea)):
                y_vals.extend((rea + np.where(np.isfinite(rea_sem), rea_sem, 0.0)).tolist())
            if ref_t.size > 0 and np.any(np.isfinite(ref_rea_mean)):
                y_vals.extend((ref_rea_mean + np.where(np.isfinite(ref_rea_sem), ref_rea_sem, 0.0)).tolist())
            y_top_func = float(np.nanmax(np.asarray(y_vals, dtype=float))) if y_vals else 100.0
            ax_func.set_ylim(0.0, max(100.0, y_top_func * 1.05))
            y_bottom_func = float(ax_func.get_ylim()[0])
            guide_x_candidates = [x for x in (t50_poly_guide_x, t50_ref_guide_x) if x is not None]
            if guide_x_candidates:
                ax_func.plot(
                    [0.0, float(max(guide_x_candidates))],
                    [target_line, target_line],
                    linestyle=(0, (3, 2)),
                    color="0.55",
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )
            if t50_poly_guide_x is not None:
                ax_func.plot(
                    [t50_poly_guide_x, t50_poly_guide_x],
                    [y_bottom_func, target_line],
                    linestyle=(0, (3, 2)),
                    color=color,
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )
            if t50_ref_guide_x is not None:
                ax_func.plot(
                    [t50_ref_guide_x, t50_ref_guide_x],
                    [y_bottom_func, target_line],
                    linestyle=(0, (3, 2)),
                    color="#808080",
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )

            t50_poly_math = (
                f"{float(t50_poly):.3g}\\,\\mathrm{{min}}"
                if (t50_poly is not None and np.isfinite(float(t50_poly)) and float(t50_poly) > 0.0)
                else r"\mathrm{NA}"
            )
            t50_ref_math = (
                f"{float(ref_t50_show):.3g}\\,\\mathrm{{min}}"
                if (ref_t50_show is not None and np.isfinite(float(ref_t50_show)) and float(ref_t50_show) > 0.0)
                else r"\mathrm{NA}"
            )
            t50_poly_label = "sample" if pid_norm.startswith("GOX WITH") else "poly"
            t50_ref_label = "GOx" if reference_norm == "GOX" else safe_label(reference_polymer_id)
            t50_poly_label_math = _escape_mathtext_token(t50_poly_label)
            t50_ref_label_math = _escape_mathtext_token(t50_ref_label)
            objective_lines: list[str] = []
            if is_ref_per_polymer:
                objective_lines.append(rf"$\mathrm{{reference}}:\ {t50_ref_label_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
            elif np.isfinite(fog_from_t50):
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_poly_label_math}}}}}={t50_poly_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
                objective_lines.append(rf"$\mathrm{{FoG}}={float(fog_from_t50):.3g}$")
            else:
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_poly_label_math}}}}}={t50_poly_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
                objective_lines.append(r"$\mathrm{FoG}=\mathrm{NA}$")

            fit_subject_label = "reference" if is_ref_per_polymer else "sample"
            t50_subject_label = t50_ref_label if is_ref_per_polymer else t50_poly_label
            fit_subject_label_math = _escape_mathtext_token(fit_subject_label)
            t50_subject_label_math = _escape_mathtext_token(t50_subject_label)
            if fit_rea is None:
                objective_lines.append(rf"$\mathrm{{{fit_subject_label_math}\ fit:\ unavailable}}$")
            elif not use_exp_rea:
                objective_lines.append(
                    rf"$\mathrm{{{fit_subject_label_math}\ fit:\ low\ confidence}}\ (R^2={float(fit_rea.r2):.3f})$"
                )
            if t50_model is None and t50_poly is not None and np.isfinite(float(t50_poly)):
                objective_lines.append(
                    rf"$t_{{50,\mathrm{{{t50_subject_label_math}}}}}:\ \mathrm{{linear\ crossing}}$"
                )
            if (not is_ref_per_polymer) and (fit_ref_rea is not None) and (not use_exp_ref_rea):
                objective_lines.append(
                    rf"$\mathrm{{{t50_ref_label_math}\ fit:\ low\ confidence}}\ (R^2={float(fit_ref_rea.r2):.3f},\ \mathrm{{dashed}})$"
                )
            if (
                (not is_ref_per_polymer)
                and ref_t50_show is not None
                and np.isfinite(float(ref_t50_show))
                and (fit_ref_rea is None or (not use_exp_ref_rea))
            ):
                objective_lines.append(
                    rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}:\ \mathrm{{linear\ crossing}}$"
                )
            if (not is_ref_per_polymer) and np.isfinite(native_rel_at_0) and (not native_feasible):
                objective_lines.append(
                    rf"$U_{{0}}={float(native_rel_at_0):.3g} < \theta={float(native_activity_min_rel):.2f}\ \mathrm{{at}}\ 0\,\mathrm{{min}}$"
                )

            objective_text = "\n".join(objective_lines)
            txt_obj = ax_func.annotate(
                objective_text,
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                multialignment="left",
                fontsize=7.0,
                bbox=dict(
                    boxstyle=f"round,pad={max(0.16, (float(info_pad[0]) if isinstance(info_pad, (tuple, list)) else float(info_pad)) - 0.03)}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.93,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_obj.get_bbox_patch() is not None:
                txt_obj.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())
            ax_func.set_title(f"{pid_label} | REA comparison and FoG ratio")
            ax_func.set_xlabel("Heat time (min)")
            ax_func.set_ylabel("REA (%)")
            ax_func.set_xticks(HEAT_TICKS_0_60)
            ax_func.set_xlim(0.0, 62.5)
            ax_func.tick_params(axis="both", which="both", length=0, labelsize=6)

            for ax in (ax_left, ax_right, ax_func):
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_color("0.7")
                ax.spines["left"].set_zorder(-10)
                ax.spines["bottom"].set_visible(True)
                ax.spines["bottom"].set_color("0.7")
                ax.spines["bottom"].set_zorder(-10)

            fig.tight_layout(pad=0.3)
            fig.subplots_adjust(left=0.08, wspace=0.28)
            out_path = out_dir / f"{stem}__{run_id}.png"
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

    # Add "all polymers with error bars" plot for runs with replicate wells.
    all_with_error_path = out_fit_dir / f"all_polymers_with_error{err_suffix}__{run_id}.png"
    df_for_downstream = df.copy()
    mask_include = df_for_downstream["include_in_all_polymers"].astype(bool) == True
    df_filtered = df_for_downstream[mask_include].copy()
    polymer_ids_filtered = sorted(
        df_filtered["polymer_id"].astype(str).unique().tolist(),
        key=_polymer_plot_order_key,
    ) if not df_filtered.empty else []

    if not polymer_ids_filtered:
        if all_with_error_path.is_file():
            all_with_error_path.unlink(missing_ok=True)
        return out_dir

    with plt.rc_context(apply_paper_style()):
        fig_all, (ax_abs, ax_rea, ax_func) = plt.subplots(1, 3, figsize=(13.2, 3.5))
        err_color = PAPER_ERRORBAR_COLOR
        err_alpha_main = 0.76

        for pid, g in df_filtered.groupby("polymer_id", sort=False):
            g = g.sort_values("heat_min").reset_index(drop=True)
            t = g["heat_min"].to_numpy(dtype=float)
            n = g["n"].to_numpy(dtype=float)
            aa = g["mean_abs_activity"].to_numpy(dtype=float)
            aa_sem = g["sem_abs_activity"].to_numpy(dtype=float)
            rea = g["mean_REA_percent"].to_numpy(dtype=float)
            rea_sem = g["sem_REA_percent"].to_numpy(dtype=float)

            pid_str = str(pid)
            is_ref = _normalize_polymer_id_token(pid_str) == reference_norm
            color = "#808080" if is_ref else cmap.get(pid_str, "#0072B2")
            pid_label = safe_label(pid_str)

            ax_abs.scatter(t, aa, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)
            band_ok_abs = np.isfinite(aa_sem) & (aa_sem > 0) & (n >= 2)
            if np.any(band_ok_abs):
                ax_abs.errorbar(
                    t[band_ok_abs],
                    aa[band_ok_abs],
                    yerr=aa_sem[band_ok_abs],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.7,
                    capsize=2.0,
                    alpha=err_alpha_main,
                    zorder=24,
                    clip_on=False,
                )

            y0_abs_init = float(aa[0]) if aa.size > 0 and np.isfinite(float(aa[0])) else None
            fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
            if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                _draw_fit_with_extension(ax_abs, fit_abs, t, color, use_dashed_main=True, preserve_gray=is_ref)

            ax_rea.scatter(t, rea, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)
            band_ok_rea = np.isfinite(rea_sem) & (rea_sem > 0) & (n >= 2)
            if np.any(band_ok_rea):
                ax_rea.errorbar(
                    t[band_ok_rea],
                    rea[band_ok_rea],
                    yerr=rea_sem[band_ok_rea],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.7,
                    capsize=2.0,
                    alpha=err_alpha_main,
                    zorder=24,
                    clip_on=False,
                )

            y0_rea_init = float(rea[0]) if rea.size > 0 and np.isfinite(float(rea[0])) else 100.0
            fit_rea = fit_exponential_decay(
                t,
                rea,
                y0=y0_rea_init,
                fixed_y0=100.0,
                min_points=4,
                t50_definition=t50_definition,
            )
            if fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70:
                _draw_fit_with_extension(ax_rea, fit_rea, t, color, use_dashed_main=True, preserve_gray=is_ref)

        abs_handles, abs_labels = ax_abs.get_legend_handles_labels()
        if abs_handles:
            ax_abs.legend(
                abs_handles,
                abs_labels,
                loc="upper right",
                frameon=True,
                fontsize=5.8,
                ncol=1,
                columnspacing=0.4,
                handlelength=1.0,
                handletextpad=0.3,
            )
        rea_handles, rea_labels = ax_rea.get_legend_handles_labels()
        if rea_handles:
            ax_rea.legend(
                rea_handles,
                rea_labels,
                loc="upper right",
                frameon=True,
                fontsize=5.8,
                ncol=1,
                columnspacing=0.4,
                handlelength=1.0,
                handletextpad=0.3,
            )

        ax_abs.set_xlabel("Heat time (min)")
        ax_abs.set_ylabel("Absolute activity (a.u./s)")
        ax_abs.set_xticks(HEAT_TICKS_0_60)
        ax_abs.set_xlim(0.0, 62.5)
        abs_all = df_filtered["mean_abs_activity"].to_numpy(dtype=float)
        abs_sem_all = np.where(np.isfinite(df_filtered["sem_abs_activity"].to_numpy(dtype=float)), df_filtered["sem_abs_activity"].to_numpy(dtype=float), 0.0)
        if np.any(np.isfinite(abs_all)):
            y_top_abs = float(np.nanmax(abs_all + abs_sem_all)) * 1.05
            ax_abs.set_ylim(0.0, max(y_top_abs, 1e-9))
        else:
            ax_abs.set_ylim(0.0, 1.0)

        ax_rea.set_xlabel("Heat time (min)")
        ax_rea.set_ylabel("REA (%)")
        ax_rea.set_xticks(HEAT_TICKS_0_60)
        ax_rea.set_xlim(0.0, 62.5)
        rea_all = df_filtered["mean_REA_percent"].to_numpy(dtype=float)
        rea_sem_all = np.where(np.isfinite(df_filtered["sem_REA_percent"].to_numpy(dtype=float)), df_filtered["sem_REA_percent"].to_numpy(dtype=float), 0.0)
        if np.any(np.isfinite(rea_all)):
            y_top_rea = float(np.nanmax(rea_all + rea_sem_all)) * 1.05
            ax_rea.set_ylim(0.0, max(y_top_rea, 1.0))
        else:
            ax_rea.set_ylim(0.0, 100.0)

        objective_rows: list[dict[str, Any]] = []
        for pid in polymer_ids_filtered:
            pid_str = str(pid)
            row_df = t50_df[t50_df["polymer_id"].astype(str) == pid_str] if ("polymer_id" in t50_df.columns) else pd.DataFrame()
            if row_df.empty:
                continue
            row0 = row_df.iloc[0]
            native_rel = pd.to_numeric(row0.get("native_activity_rel_at_0", np.nan), errors="coerce")
            t50_exp = pd.to_numeric(row0.get("t50_exp_min", np.nan), errors="coerce")
            t50_lin = pd.to_numeric(row0.get("t50_linear_min", np.nan), errors="coerce")
            t50_used = np.nan
            if np.isfinite(t50_exp):
                t50_used = float(t50_exp)
            elif np.isfinite(t50_lin):
                t50_used = float(t50_lin)
            fog_val = np.nan
            if (
                np.isfinite(t50_used)
                and t50_used > 0.0
                and ref_t50_show is not None
                and np.isfinite(float(ref_t50_show))
                and float(ref_t50_show) > 0.0
            ):
                fog_val = float(t50_used) / float(ref_t50_show)
            if not (np.isfinite(native_rel) and np.isfinite(fog_val)):
                continue
            is_ref_obj = _normalize_polymer_id_token(pid_str) == reference_norm
            feasible = bool(float(native_rel) >= float(native_activity_min_rel))
            objective_rows.append(
                {
                    "polymer_id": pid_str,
                    "label": safe_label(pid_str),
                    "native_rel": float(native_rel),
                    "fog": float(fog_val),
                    "feasible": feasible,
                    "is_ref": is_ref_obj,
                    "color": "#808080" if is_ref_obj else cmap.get(pid_str, "#0072B2"),
                }
            )

        obj_df = pd.DataFrame(
            columns=["polymer_id", "label", "native_rel", "fog", "feasible", "is_ref", "color"]
        )
        if objective_rows:
            obj_df = pd.DataFrame(objective_rows)
            obj_df = obj_df.sort_values(
                ["feasible", "fog"],
                ascending=[False, False],
                kind="mergesort",
            ).reset_index(drop=True)

            max_rows = 16
            if len(obj_df) > max_rows:
                ref_df = obj_df[obj_df["is_ref"]].head(1)
                keep_n = max(0, max_rows - len(ref_df))
                non_ref_df = obj_df[~obj_df["is_ref"]].head(keep_n)
                obj_df = pd.concat([ref_df, non_ref_df], ignore_index=True)
                obj_df = obj_df.sort_values(
                    ["feasible", "fog"],
                    ascending=[False, False],
                    kind="mergesort",
                ).reset_index(drop=True)

            y = np.arange(len(obj_df), dtype=float)
            bars = ax_func.barh(
                y,
                obj_df["fog"].to_numpy(dtype=float),
                color=obj_df["color"].tolist(),
                edgecolor="0.25",
                linewidth=0.35,
                height=0.62,
                alpha=0.9,
                zorder=12,
            )
            for bar, feasible in zip(bars, obj_df["feasible"].tolist()):
                if not bool(feasible):
                    bar.set_alpha(0.45)
                    bar.set_hatch("//")
                    bar.set_edgecolor("0.45")

            ax_func.axvline(
                x=1.0,
                color="0.45",
                linestyle=(0, (3, 2)),
                linewidth=0.75,
                alpha=0.85,
                zorder=8,
            )
            ax_func.set_yticks(y)
            ax_func.set_yticklabels(obj_df["label"].tolist(), fontsize=5.4)
            ax_func.invert_yaxis()
            ax_func.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
            ax_func.set_title("Decision chart (native gate + FoG rank)")
            fog_vals = obj_df["fog"].to_numpy(dtype=float)
            fog_max = float(np.nanmax(fog_vals)) if fog_vals.size > 0 else 1.0
            ax_func.set_xlim(0.0, max(1.4, fog_max * 1.15))

            ax_native = ax_func.twiny()
            ax_native.scatter(
                obj_df["native_rel"].to_numpy(dtype=float),
                y,
                marker="D",
                s=12,
                color="0.12",
                edgecolors="none",
                alpha=0.9,
                zorder=30,
            )
            ax_native.axvline(
                x=float(native_activity_min_rel),
                color="0.15",
                linestyle=(0, (3, 2)),
                linewidth=0.75,
                alpha=0.9,
                zorder=25,
            )
            native_vals = obj_df["native_rel"].to_numpy(dtype=float)
            native_max = float(np.nanmax(native_vals)) if native_vals.size > 0 else 1.0
            native_right = max(1.05, float(native_activity_min_rel) * 1.25, native_max * 1.12)
            ax_native.set_xlim(0.0, native_right)
            ax_native.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
            ax_native.tick_params(axis="x", which="both", labelsize=5.2, length=0)
            ax_native.spines["bottom"].set_visible(False)
            ax_native.spines["left"].set_visible(False)
            ax_native.spines["right"].set_visible(False)
        else:
            ax_func.set_title("Decision chart (native gate + FoG rank)")
            ax_func.text(
                0.5,
                0.5,
                "No valid FoG/native points",
                ha="center",
                va="center",
                fontsize=6.4,
                color="0.35",
                transform=ax_func.transAxes,
            )
            ax_func.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
            ax_func.set_yticks([])
            ax_func.set_xlim(0.0, 1.5)
            ax_func.set_ylim(0.0, 1.0)

        for ax in (ax_abs, ax_rea, ax_func):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color("0.7")
            ax.spines["left"].set_zorder(-10)
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color("0.7")
            ax.spines["bottom"].set_zorder(-10)

        fig_all.tight_layout(pad=0.3)
        fig_all.subplots_adjust(left=0.08, wspace=0.30)
        fig_all.savefig(
            all_with_error_path,
            dpi=int(dpi),
            bbox_inches="tight",
            pad_inches=0.02,
            pil_kwargs={"compress_level": 1},
        )
        plt.close(fig_all)

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


def _robust_outlier_mask(
    values: np.ndarray,
    *,
    min_samples: int = 4,
    z_threshold: float = 3.5,
    ratio_low: Optional[float] = 0.33,
    ratio_high: Optional[float] = 3.0,
    min_keep: int = 2,
) -> np.ndarray:
    """
    Robustly detect outliers from a 1D numeric vector.

    Uses MAD-based robust z-score and optional multiplicative ratio bounds to the median.
    If applying exclusions would leave too few points (< min_keep), no points are excluded.
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
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    apply_outlier_filter: bool = True,
    outlier_min_runs: int = 4,
    outlier_z_threshold: float = 3.5,
    outlier_ratio_low: float = 0.33,
    outlier_ratio_high: float = 3.0,
    reference_abs0_outlier_low: float | None = None,
    reference_abs0_outlier_high: float | None = None,
    outlier_min_keep: int = 2,
    native_activity_min_rel: float = 0.70,
    dpi: int = 600,
) -> Optional[Path]:
    """
    Plot per-polymer time-series with error bars across a run group.

    Aggregates runs (auto-discovered by date when same_date_runs is None, or provided
    explicitly via same_date_runs), computes mean ± SEM per polymer_id × heat_min,
    and fits curves to the mean trace.

    Output:
      - out_fit_dir/per_polymer_across_runs__{group_label}__{run_id}/{polymer_stem}__{group_label}.png
      - out_fit_dir/per_polymer_across_runs__{group_label}__{run_id}/rea_comparison_fog_panel__{group_label}/{polymer_stem}__{group_label}.png
      - out_fit_dir/per_polymer_across_runs__{group_label}__{run_id}/rea_comparison_fog_grid__{group_label}.png
      - out_fit_dir/per_polymer_across_runs__{group_label}__{run_id}/csv/outlier_constraints__{group_label}.csv

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
    if combined.empty:
        return None

    outlier_events: list[dict[str, object]] = []

    # Run-level reference abs(0) guard: if reference run is extreme, exclude the whole run.
    if apply_outlier_filter:
        ref_abs0_ratio_low = (
            float(reference_abs0_outlier_low)
            if reference_abs0_outlier_low is not None
            else float(outlier_ratio_low)
        )
        ref_abs0_ratio_high = (
            float(reference_abs0_outlier_high)
            if reference_abs0_outlier_high is not None
            else float(outlier_ratio_high)
        )
        ref_norm = _normalize_polymer_id_token(reference_polymer_id)
        combined["_polymer_norm"] = combined["polymer_id"].astype(str).map(_normalize_polymer_id_token)
        ref_rows = combined[combined["_polymer_norm"] == ref_norm].copy()
        if not ref_rows.empty:
            ref_rows = ref_rows[
                np.isfinite(ref_rows["abs_activity"])
                & (ref_rows["abs_activity"] > 0.0)
            ].copy()
            if not ref_rows.empty:
                # Per-run baseline abs from earliest available heat point of reference polymer.
                ref_rows = ref_rows.sort_values(["run_id", "heat_min"], kind="mergesort")
                ref_per_run = (
                    ref_rows.groupby("run_id", as_index=False)
                    .first()[["run_id", "abs_activity"]]
                    .rename(columns={"abs_activity": "reference_abs0"})
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
                    ratio_low=ref_abs0_ratio_low,
                    ratio_high=ref_abs0_ratio_high,
                    min_keep=max(2, int(outlier_min_keep)),
                )
                if np.any(ref_mask):
                    excluded_runs = sorted(
                        {
                            str(x).strip()
                            for x in ref_per_run.loc[ref_mask, "run_id"].tolist()
                            if str(x).strip()
                        }
                    )
                    if excluded_runs:
                        combined = combined[
                            ~combined["run_id"].astype(str).isin(excluded_runs)
                        ].copy()
                        for _, rr in ref_per_run.loc[ref_mask].iterrows():
                            outlier_events.append(
                                {
                                    "event_type": "reference_run_excluded",
                                    "polymer_id": str(reference_polymer_id),
                                    "heat_min": np.nan,
                                    "excluded_run_id": str(rr["run_id"]).strip(),
                                    "excluded_abs_activity": float(rr["reference_abs0"]),
                                }
                            )
        if "_polymer_norm" in combined.columns:
            combined = combined.drop(columns=["_polymer_norm"])
        if combined.empty:
            return None

    n_group_runs = int(combined["run_id"].astype(str).nunique())

    # Aggregate across runs: group by polymer_id × heat_min
    agg_data = []
    min_runs_for_error_bar = max(2, int(outlier_min_keep))
    for (pid, heat_min), g in combined.groupby(["polymer_id", "heat_min"], dropna=False):
        if _is_background_polymer_id(str(pid)):
            continue
        n_runs_total = int(g["run_id"].astype(str).nunique())
        if n_runs_total < min_runs_for_error_bar:
            continue  # Need at least 2 runs for error bands

        g = g.copy()
        keep_mask = np.ones(len(g), dtype=bool)
        outlier_abs_mask = np.zeros(len(g), dtype=bool)
        outlier_rea_mask = np.zeros(len(g), dtype=bool)
        if apply_outlier_filter:
            outlier_abs_mask = _robust_outlier_mask(
                g["abs_activity"].to_numpy(dtype=float),
                min_samples=outlier_min_runs,
                z_threshold=outlier_z_threshold,
                ratio_low=outlier_ratio_low,
                ratio_high=outlier_ratio_high,
                min_keep=max(2, int(outlier_min_keep)),
            )
            outlier_rea_mask = _robust_outlier_mask(
                g["REA_percent"].to_numpy(dtype=float),
                min_samples=outlier_min_runs,
                z_threshold=outlier_z_threshold,
                ratio_low=outlier_ratio_low,
                ratio_high=outlier_ratio_high,
                min_keep=max(2, int(outlier_min_keep)),
            )
            outlier_row_mask = outlier_abs_mask | outlier_rea_mask
            n_keep = int(np.count_nonzero(~outlier_row_mask))
            if np.any(outlier_row_mask) and n_keep >= max(2, int(outlier_min_keep)):
                keep_mask = ~outlier_row_mask
                excluded_rows = g.loc[outlier_row_mask, ["run_id", "abs_activity", "REA_percent"]].copy()
                excluded_rows["outlier_abs"] = outlier_abs_mask[outlier_row_mask]
                excluded_rows["outlier_rea"] = outlier_rea_mask[outlier_row_mask]
                for _, rr in excluded_rows.iterrows():
                    outlier_events.append(
                        {
                            "event_type": "polymer_heat_outlier_excluded",
                            "polymer_id": str(pid),
                            "heat_min": float(heat_min),
                            "excluded_run_id": str(rr["run_id"]).strip(),
                            "excluded_abs_activity": float(rr["abs_activity"]),
                            "excluded_rea_percent": float(rr["REA_percent"]),
                            "outlier_abs": bool(rr["outlier_abs"]),
                            "outlier_rea": bool(rr["outlier_rea"]),
                        }
                    )
        g_used = g.loc[keep_mask].copy()
        n_runs = int(g_used["run_id"].astype(str).nunique())
        if n_runs < min_runs_for_error_bar:
            continue

        aa_values = g_used["abs_activity"].dropna()
        rea_values = g_used["REA_percent"].dropna()

        if len(aa_values) == 0 or len(rea_values) == 0:
            continue

        agg_data.append({
            "polymer_id": str(pid),
            "heat_min": float(heat_min),
            "n_runs": n_runs,
            "n_runs_total": n_runs_total,
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
    agg_df = agg_df[~agg_df["polymer_id"].astype(str).map(_is_background_polymer_id)].copy()
    if agg_df.empty:
        return None
    polymer_ids = sorted(
        agg_df["polymer_id"].astype(str).unique().tolist(),
        key=_polymer_plot_order_key,
    )
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
    out_csv_dir = out_dir / "csv"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_rea_fog_panel = out_dir / f"rea_comparison_fog_panel__{label}"
    out_rea_fog_panel.mkdir(parents=True, exist_ok=True)

    expected_pngs = {f"{stems[pid]}__{label}.png" for pid in polymer_ids}
    for p in out_dir.glob(f"*__{label}.png"):
        if p.name not in expected_pngs:
            p.unlink(missing_ok=True)
    for p in out_rea_fog_panel.glob(f"*__{label}.png"):
        if p.name not in expected_pngs:
            p.unlink(missing_ok=True)

    from gox_plate_pipeline.fitting.core import (
        apply_paper_style,
        INFO_BOX_MARGIN_PT,
        INFO_BOX_FACE_COLOR,
        INFO_BOX_PAD_PER_POLYMER,
        PAPER_ERRORBAR_COLOR,
        get_info_box_gradient_shadow,
    )
    import matplotlib.pyplot as plt
    import gc

    def _write_rea_fog_grid(
        *,
        panel_png_paths: list[Path],
        out_png: Path,
        n_cols: int = 5,
        gap_px: int = 10,
        outer_pad_px: int = 10,
    ) -> Optional[Path]:
        try:
            from PIL import Image
        except Exception:
            return None
        tiles: list[Any] = []
        for p in panel_png_paths:
            if not p.is_file():
                continue
            try:
                with Image.open(p) as im:
                    tiles.append(im.copy())
            except Exception:
                continue
        if not tiles:
            out_png.unlink(missing_ok=True)
            return None
        cols = max(1, int(n_cols))
        rows = int(math.ceil(len(tiles) / float(cols)))
        tile_w = max(int(im.width) for im in tiles)
        tile_h = max(int(im.height) for im in tiles)
        gap = max(0, int(gap_px))
        outer = max(0, int(outer_pad_px))
        out_w = outer * 2 + cols * tile_w + (cols - 1) * gap
        out_h = outer * 2 + rows * tile_h + (rows - 1) * gap
        bg_rgb = (245, 245, 245)
        canvas = Image.new("RGB", (out_w, out_h), bg_rgb)
        for idx, tile in enumerate(tiles):
            rr = idx // cols
            cc = idx % cols
            x = outer + cc * (tile_w + gap)
            y = outer + rr * (tile_h + gap)
            ox = (tile_w - int(tile.width)) // 2
            oy = (tile_h - int(tile.height)) // 2
            if tile.mode == "RGBA":
                canvas.paste(tile, (x + ox, y + oy), tile)
            else:
                canvas.paste(tile.convert("RGB"), (x + ox, y + oy))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_png, format="PNG", optimize=False, compress_level=1)
        return out_png

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
        s = min(1.0, s * 1.4)
        v = min(1.0, v * 1.1)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def _draw_fit_with_extension(
        ax: Any,
        fit_obj: ExpDecayFit,
        t_obs: np.ndarray,
        color_hex: str,
        *,
        preserve_gray: bool = False,
        low_confidence: bool = False,
        force_dashed_main: bool = False,
    ) -> None:
        t_obs = np.asarray(t_obs, dtype=float)
        t_obs = t_obs[np.isfinite(t_obs)]
        if t_obs.size == 0:
            return
        t_min_obs = float(np.min(t_obs))
        t_max_obs = float(np.max(t_obs))
        color_main = color_hex if preserve_gray else _to_fluorescent_color(color_hex)
        tt_main = np.linspace(t_min_obs, t_max_obs, 220)
        yy_main = _eval_fit_curve(fit_obj, tt_main)
        main_alpha = 0.50
        ext_alpha = 0.40
        main_ls = (0, (2.0, 2.0)) if (low_confidence or force_dashed_main) else "-"
        ext_ls = (0, (1.6, 1.9)) if low_confidence else (0, (2.4, 2.4))
        if low_confidence:
            main_alpha = min(0.65, main_alpha + 0.12)
            ext_alpha = min(0.55, ext_alpha + 0.10)
        ax.plot(
            tt_main,
            yy_main,
            color=color_main,
            linewidth=1.7,
            alpha=main_alpha,
            linestyle=main_ls,
            zorder=8,
        )
        if t_min_obs > 0.0:
            tt_pre = np.linspace(0.0, t_min_obs, 120)
            yy_pre = _eval_fit_curve(fit_obj, tt_pre)
            ax.plot(
                tt_pre,
                yy_pre,
                color=color_main,
                linewidth=1.5,
                alpha=ext_alpha,
                linestyle=ext_ls,
                zorder=7,
            )
        if t_max_obs < 60.0:
            tt_post = np.linspace(t_max_obs, 60.0, 140)
            yy_post = _eval_fit_curve(fit_obj, tt_post)
            ax.plot(
                tt_post,
                yy_post,
                color=color_main,
                linewidth=1.5,
                alpha=ext_alpha,
                linestyle=ext_ls,
                zorder=7,
            )

    reference_norm = _normalize_polymer_id_token(reference_polymer_id)
    ref_g = agg_df[
        agg_df["polymer_id"].astype(str).map(_normalize_polymer_id_token) == reference_norm
    ].copy()
    ref_g = ref_g.sort_values("heat_min").reset_index(drop=True)
    ref_t = ref_g["heat_min"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    ref_abs_mean = (
        ref_g["mean_abs_activity"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    )
    ref_rea_mean = ref_g["mean_REA_percent"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    ref_rea_sem = ref_g["sem_REA_percent"].to_numpy(dtype=float) if not ref_g.empty else np.array([], dtype=float)
    ref_n_runs = ref_g["n_runs"].to_numpy(dtype=int) if not ref_g.empty else np.array([], dtype=int)
    ref_abs0 = value_at_time_linear(ref_t, ref_abs_mean, at_time_min=0.0) if ref_t.size > 0 else None
    fit_ref_rea: Optional[ExpDecayFit] = None
    use_exp_ref_rea = False
    ref_t50_show: Optional[float] = None
    if ref_t.size > 0 and ref_rea_mean.size > 0:
        fit_ref_rea = fit_exponential_decay(ref_t, ref_rea_mean, y0=100.0, fixed_y0=100.0, min_points=4)
        use_exp_ref_rea = bool(
            fit_ref_rea is not None
            and np.isfinite(float(fit_ref_rea.r2))
            and float(fit_ref_rea.r2) >= 0.70
        )
        ref_t50_lin = t50_linear(ref_t, ref_rea_mean, y0=100.0, target_frac=0.5, target_value=50.0)
        ref_t50_model = fit_ref_rea.t50 if (fit_ref_rea is not None and use_exp_ref_rea) else None
        ref_t50_show = ref_t50_model if ref_t50_model is not None else ref_t50_lin

    plot_count = 0
    grouped_by_pid: dict[str, pd.DataFrame] = {
        str(pid): g for pid, g in agg_df.groupby("polymer_id", sort=False)
    }
    for pid in polymer_ids:
        g = grouped_by_pid.get(str(pid))
        if g is None:
            continue
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
        pid_norm = _normalize_polymer_id_token(str(pid))
        is_ref_per_polymer = pid_norm == reference_norm
        info_pad = INFO_BOX_PAD_PER_POLYMER

        y0_abs_init = float(aa_mean[0]) if aa_mean.size > 0 and np.isfinite(float(aa_mean[0])) else None
        fit_abs = fit_exponential_decay(t, aa_mean, y0=y0_abs_init, min_points=4)
        use_exp_abs = bool(
            fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70
        )
        fit_rea = fit_exponential_decay(t, rea_mean, y0=100.0, fixed_y0=100.0, min_points=4)
        use_exp_rea = bool(
            fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70
        )
        t50_lin = t50_linear(t, rea_mean, y0=100.0, target_frac=0.5, target_value=50.0)
        t50_model = fit_rea.t50 if (fit_rea is not None and use_exp_rea) else None
        t50_poly = t50_model if t50_model is not None else t50_lin
        fog_from_t50 = (
            float(t50_poly) / float(ref_t50_show)
            if (
                t50_poly is not None
                and np.isfinite(float(t50_poly))
                and float(t50_poly) > 0.0
                and ref_t50_show is not None
                and np.isfinite(float(ref_t50_show))
                and float(ref_t50_show) > 0.0
            )
            else np.nan
        )
        abs0_poly = value_at_time_linear(t, aa_mean, at_time_min=0.0)
        native_rel_at_0 = (
            float(abs0_poly) / float(ref_abs0)
            if (
                abs0_poly is not None
                and np.isfinite(float(abs0_poly))
                and ref_abs0 is not None
                and np.isfinite(float(ref_abs0))
                and float(ref_abs0) > 0.0
            )
            else np.nan
        )
        native_feasible = bool(
            np.isfinite(native_rel_at_0) and float(native_rel_at_0) >= float(native_activity_min_rel)
        )

        with plt.rc_context(apply_paper_style()):
            fig, (ax_left, ax_right, ax_func) = plt.subplots(1, 3, figsize=(10.4, 2.8))
            # Keep SEM bars neutral and slightly darker than before so they don't blend into GOx gray.
            err_color = PAPER_ERRORBAR_COLOR
            err_alpha_main = 0.76

            # Left: Absolute activity with error bars
            # Match per-run per-polymer style: no point-connecting polyline.
            ax_left.scatter(
                t, aa_mean, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=6, clip_on=False
            )
            band_ok_abs = np.isfinite(aa_sem) & (aa_sem > 0) & (n_runs >= 2)
            if np.any(band_ok_abs):
                ax_left.errorbar(
                    t[band_ok_abs],
                    aa_mean[band_ok_abs],
                    yerr=aa_sem[band_ok_abs],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=err_alpha_main,
                    zorder=5,
                    clip_on=False,
                )
            if use_exp_abs and fit_abs is not None:
                _draw_fit_with_extension(ax_left, fit_abs, t, color, preserve_gray=is_ref_per_polymer)
            ax_left.set_title(f"{pid_label} | Absolute activity (mean ± SEM, n={n_group_runs} runs)")
            ax_left.set_xlabel("Heat time (min)")
            ax_left.set_ylabel("Absolute activity (a.u./s)")
            ax_left.set_xticks(HEAT_TICKS_0_60)
            ax_left.set_xlim(0.0, 62.5)
            if np.isfinite(aa_mean).any():
                abs_upper = aa_mean + np.where(np.isfinite(aa_sem), aa_sem, 0.0)
                ymax = float(np.nanmax(abs_upper))
                ax_left.set_ylim(0.0, max(ymax * 1.05, 1e-9))
            else:
                ax_left.set_ylim(0.0, 1.0)
            ax_left.tick_params(axis="both", which="both", length=0, labelsize=6)

            # Middle: REA with error bars
            # Match per-run per-polymer style: no point-connecting polyline.
            ax_right.scatter(
                t, rea_mean, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=6, clip_on=False
            )
            band_ok_rea = np.isfinite(rea_sem) & (rea_sem > 0) & (n_runs >= 2)
            if np.any(band_ok_rea):
                ax_right.errorbar(
                    t[band_ok_rea],
                    rea_mean[band_ok_rea],
                    yerr=rea_sem[band_ok_rea],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=err_alpha_main,
                    zorder=5,
                    clip_on=False,
                )
            if use_exp_rea and fit_rea is not None:
                _draw_fit_with_extension(ax_right, fit_rea, t, color, preserve_gray=is_ref_per_polymer)
            ax_right.set_title(f"{pid_label} | REA (mean ± SEM, n={n_group_runs} runs)")
            ax_right.set_xlabel("Heat time (min)")
            ax_right.set_ylabel("REA (%)")
            ax_right.set_xticks(HEAT_TICKS_0_60)
            ax_right.set_xlim(0.0, 62.5)
            if np.isfinite(rea_mean).any():
                rea_upper = rea_mean + np.where(np.isfinite(rea_sem), rea_sem, 0.0)
                ymax = float(np.nanmax(rea_upper))
                ax_right.set_ylim(0.0, max(ymax * 1.05, 1.0))
            else:
                ax_right.set_ylim(0.0, 100.0)
            ax_right.tick_params(axis="both", which="both", length=0, labelsize=6)

            # Right: REA comparison and FoG ratio (with error bars)
            ax_func.scatter(
                t, rea_mean, s=12, color=color, edgecolors="0.2", linewidths=0.4, alpha=1.0, zorder=6, clip_on=False
            )
            # In REA comparison panel, GOx is shown as mean trace without error bars.
            if np.any(band_ok_rea) and (not is_ref_per_polymer):
                ax_func.errorbar(
                    t[band_ok_rea],
                    rea_mean[band_ok_rea],
                    yerr=rea_sem[band_ok_rea],
                    fmt="none",
                    ecolor=err_color,
                    elinewidth=0.8,
                    capsize=2.2,
                    alpha=err_alpha_main,
                    zorder=5,
                    clip_on=False,
                )
            should_draw_rea_fit_obj = bool(fit_rea is not None and use_exp_rea)
            if should_draw_rea_fit_obj:
                _draw_fit_with_extension(
                    ax_func,
                    fit_rea,
                    t,
                    color,
                    preserve_gray=is_ref_per_polymer,
                    low_confidence=(not is_ref_per_polymer) and (not native_feasible),
                )

            if ref_t.size > 0 and ref_rea_mean.size > 0 and not is_ref_per_polymer:
                ax_func.scatter(
                    ref_t,
                    ref_rea_mean,
                    s=10,
                    marker="s",
                    color="#808080",
                    edgecolors="0.2",
                    linewidths=0.3,
                    alpha=0.82,
                    zorder=6,
                    clip_on=False,
                )
                if fit_ref_rea is not None:
                    _draw_fit_with_extension(
                        ax_func,
                        fit_ref_rea,
                        ref_t,
                        "#808080",
                        preserve_gray=True,
                        low_confidence=not use_exp_ref_rea,
                        force_dashed_main=True,
                    )
                elif ref_t.size > 1:
                    # Keep reference trace visible even when exponential fit is unavailable.
                    ax_func.plot(
                        ref_t,
                        ref_rea_mean,
                        color="#808080",
                        linewidth=1.3,
                        alpha=0.75,
                        linestyle=(0, (2.0, 2.0)),
                        zorder=7,
                    )

            target_line = 50.0
            t50_poly_guide_x = (
                float(t50_poly)
                if (t50_poly is not None and np.isfinite(float(t50_poly)) and float(t50_poly) > 0.0)
                else None
            )
            t50_ref_guide_x = (
                float(ref_t50_show)
                if (
                    ref_t50_show is not None
                    and np.isfinite(float(ref_t50_show))
                    and float(ref_t50_show) > 0.0
                    and not is_ref_per_polymer
                )
                else None
            )
            y_vals = []
            if np.any(np.isfinite(rea_mean)):
                y_vals.extend((rea_mean + np.where(np.isfinite(rea_sem), rea_sem, 0.0)).tolist())
            if ref_t.size > 0 and np.any(np.isfinite(ref_rea_mean)):
                y_vals.extend((ref_rea_mean + np.where(np.isfinite(ref_rea_sem), ref_rea_sem, 0.0)).tolist())
            y_top_func = float(np.nanmax(np.asarray(y_vals, dtype=float))) if y_vals else 100.0
            ax_func.set_ylim(0.0, max(100.0, y_top_func * 1.05))
            y_bottom_func = float(ax_func.get_ylim()[0])
            guide_x_candidates = [x for x in (t50_poly_guide_x, t50_ref_guide_x) if x is not None]
            if guide_x_candidates:
                ax_func.plot(
                    [0.0, float(max(guide_x_candidates))],
                    [target_line, target_line],
                    linestyle=(0, (3, 2)),
                    color="0.55",
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )
            if t50_poly_guide_x is not None:
                ax_func.plot(
                    [t50_poly_guide_x, t50_poly_guide_x],
                    [y_bottom_func, target_line],
                    linestyle=(0, (3, 2)),
                    color=color,
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )
            if t50_ref_guide_x is not None:
                ax_func.plot(
                    [t50_ref_guide_x, t50_ref_guide_x],
                    [y_bottom_func, target_line],
                    linestyle=(0, (3, 2)),
                    color="#808080",
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=4,
                )

            t50_poly_math = (
                f"{float(t50_poly):.3g}\\,\\mathrm{{min}}"
                if (t50_poly is not None and np.isfinite(float(t50_poly)) and float(t50_poly) > 0.0)
                else r"\mathrm{NA}"
            )
            t50_ref_math = (
                f"{float(ref_t50_show):.3g}\\,\\mathrm{{min}}"
                if (ref_t50_show is not None and np.isfinite(float(ref_t50_show)) and float(ref_t50_show) > 0.0)
                else r"\mathrm{NA}"
            )
            t50_poly_label = "sample" if pid_norm.startswith("GOX WITH") else "poly"
            t50_ref_label = "GOx" if reference_norm == "GOX" else safe_label(reference_polymer_id)
            t50_poly_label_math = _escape_mathtext_token(t50_poly_label)
            t50_ref_label_math = _escape_mathtext_token(t50_ref_label)
            objective_lines: list[str] = []
            if is_ref_per_polymer:
                objective_lines.append(rf"$\mathrm{{reference}}:\ {t50_ref_label_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
            elif np.isfinite(fog_from_t50):
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_poly_label_math}}}}}={t50_poly_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
                objective_lines.append(rf"$\mathrm{{FoG}}={float(fog_from_t50):.3g}$")
            else:
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_poly_label_math}}}}}={t50_poly_math}$")
                objective_lines.append(rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}={t50_ref_math}$")
                objective_lines.append(r"$\mathrm{FoG}=\mathrm{NA}$")

            fit_subject_label = "reference" if is_ref_per_polymer else "sample"
            t50_subject_label = t50_ref_label if is_ref_per_polymer else t50_poly_label
            fit_subject_label_math = _escape_mathtext_token(fit_subject_label)
            t50_subject_label_math = _escape_mathtext_token(t50_subject_label)
            if fit_rea is None:
                objective_lines.append(rf"$\mathrm{{{fit_subject_label_math}\ fit:\ unavailable}}$")
            elif not use_exp_rea:
                objective_lines.append(
                    rf"$\mathrm{{{fit_subject_label_math}\ fit:\ low\ confidence}}\ (R^2={float(fit_rea.r2):.3f})$"
                )
            if t50_model is None and t50_poly is not None and np.isfinite(float(t50_poly)):
                objective_lines.append(
                    rf"$t_{{50,\mathrm{{{t50_subject_label_math}}}}}:\ \mathrm{{linear\ crossing}}$"
                )

            if (not is_ref_per_polymer) and (fit_ref_rea is not None) and (not use_exp_ref_rea):
                objective_lines.append(
                    rf"$\mathrm{{{t50_ref_label_math}\ fit:\ low\ confidence}}\ (R^2={float(fit_ref_rea.r2):.3f},\ \mathrm{{dashed}})$"
                )
            if (
                (not is_ref_per_polymer)
                and ref_t50_show is not None
                and np.isfinite(float(ref_t50_show))
                and (
                    fit_ref_rea is None
                    or (not use_exp_ref_rea)
                )
            ):
                objective_lines.append(
                    rf"$t_{{50,\mathrm{{{t50_ref_label_math}}}}}:\ \mathrm{{linear\ crossing}}$"
                )

            if (not is_ref_per_polymer) and np.isfinite(native_rel_at_0) and (not native_feasible):
                objective_lines.append(
                    rf"$U_{{0}}={float(native_rel_at_0):.3g} < \theta={float(native_activity_min_rel):.2f}\ \mathrm{{at}}\ 0\,\mathrm{{min}}$"
                )

            objective_text = "\n".join(objective_lines)
            txt_obj = ax_func.annotate(
                objective_text,
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
                textcoords="offset points",
                ha="right",
                va="top",
                multialignment="left",
                fontsize=7.0,
                bbox=dict(
                    boxstyle=f"round,pad={max(0.16, (float(info_pad[0]) if isinstance(info_pad, (tuple, list)) else float(info_pad)) - 0.03)}",
                    facecolor=INFO_BOX_FACE_COLOR,
                    alpha=0.93,
                    edgecolor="none",
                ),
                zorder=10,
            )
            if txt_obj.get_bbox_patch() is not None:
                txt_obj.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())
            ax_func.set_title(f"{pid_label} | REA comparison and FoG ratio")
            ax_func.set_xlabel("Heat time (min)")
            ax_func.set_ylabel("REA (%)")
            ax_func.set_xlim(0.0, 62.5)
            ax_func.tick_params(axis="both", which="both", length=0, labelsize=6)

            # Spine styling
            for ax in (ax_left, ax_right, ax_func):
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_color("0.7")
                ax.spines["left"].set_zorder(-10)
                ax.spines["bottom"].set_visible(True)
                ax.spines["bottom"].set_color("0.7")
                ax.spines["bottom"].set_zorder(-10)

            fig.tight_layout(pad=0.3)
            fig.subplots_adjust(left=0.08, wspace=0.28)
            out_path = out_dir / f"{stem}__{label}.png"
            fig.savefig(
                out_path,
                dpi=int(dpi),
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compress_level": 1},
            )
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bbox = ax_func.get_tightbbox(renderer=renderer).expanded(1.02, 1.02)
                bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                out_panel_path = out_rea_fog_panel / f"{stem}__{label}.png"
                fig.savefig(
                    out_panel_path,
                    dpi=int(dpi),
                    bbox_inches=bbox_inches,
                    pad_inches=0.00,
                    pil_kwargs={"compress_level": 1},
                )
            except Exception:
                pass
            plt.close(fig)
            plot_count += 1
            if plot_count % 5 == 0:
                gc.collect()

    panel_paths_ordered = [out_rea_fog_panel / f"{stems.get(pid, safe_stem(pid))}__{label}.png" for pid in polymer_ids]
    _write_rea_fog_grid(
        panel_png_paths=panel_paths_ordered,
        out_png=out_dir / f"rea_comparison_fog_grid__{label}.png",
        n_cols=5,
        gap_px=10,
        outer_pad_px=10,
    )

    outlier_report_cols = [
        "event_type",
        "polymer_id",
        "heat_min",
        "excluded_run_id",
        "excluded_abs_activity",
        "excluded_rea_percent",
        "outlier_abs",
        "outlier_rea",
    ]
    outlier_report = pd.DataFrame(outlier_events, columns=outlier_report_cols)
    outlier_report_path = out_csv_dir / f"outlier_constraints__{label}.csv"
    outlier_report.to_csv(outlier_report_path, index=False)
    legacy_outlier_path = out_dir / f"outlier_constraints__{label}.csv"
    if legacy_outlier_path.is_file():
        legacy_outlier_path.unlink(missing_ok=True)

    return out_dir


def plot_per_polymer_timeseries_across_runs_with_error_bars(
    *,
    run_id: str,
    processed_dir: Path,
    out_fit_dir: Path,
    color_map_path: Path,
    same_date_runs: Optional[list[str]] = None,
    group_label: Optional[str] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    apply_outlier_filter: bool = True,
    outlier_min_runs: int = 4,
    outlier_z_threshold: float = 3.5,
    outlier_ratio_low: float = 0.33,
    outlier_ratio_high: float = 3.0,
    reference_abs0_outlier_low: float | None = None,
    reference_abs0_outlier_high: float | None = None,
    outlier_min_keep: int = 2,
    native_activity_min_rel: float = 0.70,
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
        reference_polymer_id=reference_polymer_id,
        apply_outlier_filter=apply_outlier_filter,
        outlier_min_runs=outlier_min_runs,
        outlier_z_threshold=outlier_z_threshold,
        outlier_ratio_low=outlier_ratio_low,
        outlier_ratio_high=outlier_ratio_high,
        reference_abs0_outlier_low=reference_abs0_outlier_low,
        reference_abs0_outlier_high=reference_abs0_outlier_high,
        outlier_min_keep=outlier_min_keep,
        native_activity_min_rel=native_activity_min_rel,
        dpi=dpi,
    )
