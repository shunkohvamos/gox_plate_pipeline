# src/gox_plate_pipeline/polymer_timeseries.py
from __future__ import annotations

import hashlib
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
    min_points: int = 4,
) -> Optional[ExpDecayFit]:
    """
    Exponential decay fit with y0 estimated from all data points (not fixed to t=0).

    Models (y0 is fitted from all points, not fixed):
      - exp:         y = y0 * exp(-k t), k>=0, y0>0
      - exp_plateau: y = c + (y0-c)*exp(-k t), 0<=c<=y0, k>=0, y0>0

    Picks the model with lower AIC. Returns None when fitting is unsafe.
    y0 parameter is optional: if provided, used as initial guess; otherwise estimated from data.
    """
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

    # Estimate y0 from data if not provided
    # Use the maximum value as initial guess (typically the first or early point)
    if y0 is None or not np.isfinite(float(y0)) or float(y0) <= 0.0:
        y0_init = float(np.max(y))
    else:
        y0_init = float(y0)

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

    # --- Model 1: y = y0 * exp(-k t) - fit both y0 and k
    def _m1(tt: np.ndarray, y0_param: float, k_param: float) -> np.ndarray:
        return y0_param * np.exp(-k_param * tt)

    best: Optional[ExpDecayFit] = None
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
        t50_1 = float(np.log(2.0) / k1) if k1 > 0.0 else None
        best = ExpDecayFit(model="exp", y0=y0_1, k=k1, c=None, r2=float(r21), aic=float(aic1), t50=t50_1)
    except Exception:
        best = None

    # --- Model 2: plateau (only if we have enough points to justify)
    # y = c + (y0-c)*exp(-k t) - fit c, y0, and k
    if t.size >= 5:
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

            # t50 (half of y0): only defined when plateau is below half.
            target = 0.5 * float(y0_2)
            t50_2: Optional[float]
            if k2 <= 0.0:
                t50_2 = None
            elif c2 >= target:
                t50_2 = None
            else:
                frac = (target - c2) / max(float(y0_2) - c2, 1e-12)
                if frac <= 0.0 or frac >= 1.0:
                    t50_2 = None
                else:
                    t50_2 = float(-np.log(frac) / k2)

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
) -> Optional[float]:
    """
    Estimate t50 by linear interpolation on the observed curve.
    Returns None if the curve never crosses target.
    """
    t = np.asarray(t_min, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 2 or not np.isfinite(float(y0)) or float(y0) <= 0.0:
        return None

    target = float(y0) * float(target_frac)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

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
) -> Path:
    """
    Create per-polymer time series plots: one figure per polymer with Absolute (left) and REA (right).

    Outputs (PNG only):
      - out_fit_dir/per_polymer__{run_id}/{polymer_stem}__{run_id}.png  (combined: abs left, REA right)
      - out_fit_dir/t50/t50__{run_id}.csv

    Returns path to the written t50 CSV.
    """
    summary_simple_path = Path(summary_simple_path)
    out_fit_dir = Path(out_fit_dir)
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")

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

    def _draw_fit_with_extension(ax: Any, fit_obj: ExpDecayFit, t_obs: np.ndarray, color_hex: str, *, use_dashed_main: bool = False) -> None:
        """
        Draw fitted curve with extension.
        
        Args:
            ax: Matplotlib axes
            fit_obj: Exponential decay fit object
            t_obs: Observed time points
            color_hex: Base color in hex format
            use_dashed_main: If True, curves have higher transparency (for all_polymers plots).
                            If False, curves have normal transparency (for per_polymer plots).
                            Both use solid line for main curve and dashed line for extensions.
        """
        t_obs = np.asarray(t_obs, dtype=float)
        t_obs = t_obs[np.isfinite(t_obs)]
        if t_obs.size == 0:
            return
        t_min_obs = float(np.min(t_obs))
        t_max_obs = float(np.max(t_obs))

        # Convert to fluorescent color (bright, high-saturation)
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

        # Debug: verify that aa and rea are different
        if len(aa) > 0 and len(rea) > 0:
            print(f"DEBUG {pid}: aa range=[{np.min(aa):.2f}, {np.max(aa):.2f}], rea range=[{np.min(rea):.2f}, {np.max(rea):.2f}]")
            print(f"DEBUG {pid}: aa[0]={aa[0]:.2f}, rea[0]={rea[0]:.2f}")

        # GOx always uses gray color
        pid_str = str(pid)
        if pid_str.upper() == "GOX":
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

        # --- REA (right panel)
        # y0 is optional: if provided, used as initial guess; otherwise estimated from all data points
        y0_rea_init = None
        if rea.size > 0 and np.isfinite(float(rea[0])):
            y0_rea_init = float(rea[0])  # Use first point as initial guess if available
        else:
            y0_rea_init = 100.0  # Default initial guess for REA

        fit_rea = fit_exponential_decay(t, rea, y0=y0_rea_init, min_points=4)
        use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
        r2_rea = float(fit_rea.r2) if (fit_rea is not None and np.isfinite(float(fit_rea.r2))) else None
        
        # Debug: verify that absolute and REA fits are different
        if r2_abs is not None and r2_rea is not None:
            print(f"DEBUG {pid}: R² (Absolute)={r2_abs:.4f}, R² (REA)={r2_rea:.4f}, diff={abs(r2_abs - r2_rea):.4f}")
        if fit_abs is not None and fit_rea is not None:
            k_abs = float(fit_abs.k) if np.isfinite(float(fit_abs.k)) else None
            k_rea = float(fit_rea.k) if np.isfinite(float(fit_rea.k)) else None
            if k_abs is not None and k_rea is not None:
                print(f"DEBUG {pid}: k (Absolute)={k_abs:.6f}, k (REA)={k_rea:.6f}, ratio={k_rea/k_abs:.4f}")
        
        # For t50_linear, use y0 from fit if available, otherwise use initial guess or first point
        y0_rea_for_t50 = float(fit_rea.y0) if (fit_rea is not None and use_exp_rea) else (y0_rea_init if y0_rea_init is not None else (float(rea[0]) if rea.size > 0 and np.isfinite(float(rea[0])) else 100.0))
        t50_lin = t50_linear(t, rea, y0=y0_rea_for_t50, target_frac=0.5)
        t50_model = fit_rea.t50 if (fit_rea is not None and use_exp_rea) else None
        # Keep 'fit' variable for backward compatibility in t50_rows.append
        fit = fit_rea

        # --- One figure: left = Absolute activity, right = REA (%)
        # STIX fontset for mathtext (R², t_{50}, e^{-kt}) so sub/superscripts render correctly
        _style = {**apply_paper_style(), "mathtext.fontset": "stix"}
        with plt.rc_context(_style):
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 2.6))

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
                _draw_fit_with_extension(ax_left, fit_abs, t, color, use_dashed_main=False)
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
            y_margin_abs = (np.max(aa) - np.min(aa)) * 0.05 if aa.size > 0 and np.max(aa) > np.min(aa) else np.max(aa) * 0.05 if aa.size > 0 else 1.0
            # Set x-axis to start at 0 so x and y axes intersect at (0, 0)
            ax_left.set_xlim(0.0, 60 + x_margin_right)
            if aa.size > 0:
                y_min_abs = float(np.min(aa[np.isfinite(aa)])) if np.any(np.isfinite(aa)) else 0.0
                y_max_abs = float(np.max(aa[np.isfinite(aa)])) if np.any(np.isfinite(aa)) else 1.0
                y_top_abs = y_max_abs + y_margin_abs
                ax_left.set_ylim(0.0, y_top_abs)  # Start y-axis at 0
                print(f"DEBUG {pid} (Absolute): ylim=[0.0, {y_top_abs:.2f}], data range=[{y_min_abs:.2f}, {y_max_abs:.2f}]")
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

            # Right: REA (%)
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
                _draw_fit_with_extension(ax_right, fit_rea, t, color, use_dashed_main=False)
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
            y_margin_rea = (np.max(rea) - np.min(rea)) * 0.05 if rea.size > 0 and np.max(rea) > np.min(rea) else np.max(rea) * 0.05 if rea.size > 0 else 2.0
            # Set x-axis to start at 0 so x and y axes intersect at (0, 0)
            ax_right.set_xlim(0.0, 60 + x_margin_right)
            if rea.size > 0:
                y_min_rea = float(np.min(rea[np.isfinite(rea)])) if np.any(np.isfinite(rea)) else 0.0
                y_max_rea = float(np.max(rea[np.isfinite(rea)])) if np.any(np.isfinite(rea)) else 100.0
                y_top_rea = y_max_rea + y_margin_rea
                ax_right.set_ylim(0.0, y_top_rea)  # Start y-axis at 0
                print(f"DEBUG {pid} (REA): ylim=[0.0, {y_top_rea:.2f}], data range=[{y_min_rea:.2f}, {y_max_rea:.2f}]")
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
                # Calculate y value at t50 from fitted curve (should be 50.0 for REA, but use actual curve value)
                # This ensures the intersection is at the center of the fitted curve line
                if use_exp_rea and fit_rea is not None:
                    y_at_t50 = float(_eval_fit_curve(fit_rea, np.array([t50_val]))[0])
                else:
                    y_at_t50 = 50.0  # Fallback to 50.0 if no fit
                # Horizontal line: from left edge (x=0) to t50 intersection (left side only)
                # Use zorder=5 to be behind fitted curve (zorder=8) but still visible
                ax_right.plot([0.0, t50_val], [y_at_t50, y_at_t50], linestyle=(0, (3, 2)), color="0.5", linewidth=0.6, alpha=0.8, zorder=5)
                # Vertical line: from bottom to t50 intersection (bottom side only)
                ax_right.plot([t50_val, t50_val], [y_bottom, y_at_t50], linestyle=(0, (3, 2)), color="0.4", linewidth=0.6, alpha=0.8, zorder=5)

            fig.tight_layout(pad=0.3)
            # Increase left margin to accommodate clip_on=False markers (especially heat_time=0 points)
            fig.subplots_adjust(left=0.12)
            # Ensure spines visibility and color after tight_layout (per_polymer: only x-axis and y-axis, light gray)
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

        t50_rows.append(
            {
                "run_id": run_id,
                "polymer_id": str(pid),
                "polymer_label": pid_label,
                "n_points": int(len(g)),
                "y0_REA_percent": float(y0_rea_for_t50),
                "t50_linear_min": float(t50_lin) if t50_lin is not None else np.nan,
                "t50_exp_min": float(t50_model) if t50_model is not None else np.nan,
                "fit_model": fit_rea.model if (fit_rea is not None and use_exp_rea) else "",
                "fit_k_per_min": float(fit_rea.k) if (fit_rea is not None and use_exp_rea) else np.nan,
                "fit_tau_min": float(1.0 / fit_rea.k) if (fit_rea is not None and use_exp_rea and fit_rea.k > 0) else np.nan,
                "fit_plateau": float(fit_rea.c) if (fit_rea is not None and use_exp_rea and fit_rea.c is not None) else np.nan,
                "fit_r2": float(fit_rea.r2) if (fit_rea is not None and use_exp_rea) else np.nan,
                "rea_connector": "exp" if use_exp_rea else "polyline",
            }
        )

    t50_df = pd.DataFrame(t50_rows)
    # out_t50_dir is already created earlier (before out_per_polymer)
    t50_path = out_t50_dir / f"t50__{run_id}.csv"
    t50_df.to_csv(t50_path, index=False)

    # --- All polymers comparison plots (overlay all polymers in one figure: Absolute left, REA right)
    _style = {**apply_paper_style(), "mathtext.fontset": "stix"}
    with plt.rc_context(_style):
        fig_all, (ax_abs, ax_rea) = plt.subplots(1, 2, figsize=(10.0, 3.5))
        n_polymers = len(polymer_ids)

        # Left: Absolute activity
        for pid, g in df.groupby("polymer_id", sort=False):
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

            # Plot fit curve if available
            # y0 is optional: if provided, used as initial guess; otherwise estimated from all data points
            y0_abs_init = float(aa[0]) if aa.size > 0 and np.isfinite(float(aa[0])) else None
            fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
            if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                _draw_fit_with_extension(ax_abs, fit_abs, t, color, use_dashed_main=True)
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
        for pid, g in df.groupby("polymer_id", sort=False):
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

        # Right: REA (%)
        for pid, g in df.groupby("polymer_id", sort=False):
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

            # Plot fit curve if available
            # y0 is optional: if provided, used as initial guess; otherwise estimated from all data points
            y0_rea_init = float(rea[0]) if rea.size > 0 and np.isfinite(float(rea[0])) else 100.0
            fit_rea = fit_exponential_decay(t, rea, y0=y0_rea_init, min_points=4)
            use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
            if use_exp_rea:
                _draw_fit_with_extension(ax_rea, fit_rea, t, color, use_dashed_main=True)
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
        for pid, g in df.groupby("polymer_id", sort=False):
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

        # Legend: place outside on the right side (shared for both panels)
        if n_polymers > 8:
            # Place legend outside on the right
            ax_rea.legend(
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
            # Place legend inside (upper right of REA panel)
            ax_rea.legend(
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
            fig_all.subplots_adjust(left=0.12)
        
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

        # Save combined figure (replace separate files)
        out_all_path = out_fit_dir / f"all_polymers__{run_id}.png"
        fig_all.savefig(
            out_all_path,
            dpi=int(dpi),
            bbox_inches="tight",
            pad_inches=0.02,
            pil_kwargs={"compress_level": 1},
        )
        plt.close(fig_all)

        # Remove old separate files if they exist
        for old_file in (out_fit_dir / f"all_polymers_abs__{run_id}.png", out_fit_dir / f"all_polymers_rea__{run_id}.png"):
            if old_file.exists():
                try:
                    old_file.unlink()
                except Exception:
                    pass

    # --- Representative 4 polymers comparison plot (GOx, PMPC, t50 top, t50 bottom)
    # Load t50 data to select representative polymers
    representative_pids = []
    
    # Check for GOx
    if "GOx" in polymer_ids:
        representative_pids.append("GOx")
    
    # Check for PMPC
    if "PMPC" in polymer_ids:
        representative_pids.append("PMPC")
    
    # Find t50 top and bottom (only numeric t50 values)
    t50_numeric = t50_df[t50_df["t50_exp_min"].notna() & np.isfinite(t50_df["t50_exp_min"])].copy()
    if t50_numeric.empty:
        # Fallback to linear t50 if exp t50 is not available
        t50_numeric = t50_df[t50_df["t50_linear_min"].notna() & np.isfinite(t50_df["t50_linear_min"])].copy()
        t50_col = "t50_linear_min"
    else:
        t50_col = "t50_exp_min"
    
    if not t50_numeric.empty:
        # Get best t50 per polymer (use exp if available, otherwise linear)
        t50_per_polymer = []
        for pid in t50_df["polymer_id"].unique():
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
            # Top (highest t50)
            top_pid = t50_per_polymer[-1][0]
            if top_pid not in representative_pids:
                representative_pids.append(top_pid)
            # Bottom (lowest t50)
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
                ax_abs_rep.scatter(t, aa, s=10, color=color, edgecolors="0.2", linewidths=0.3, alpha=1.0, zorder=30, label=pid_label, clip_on=False)
                
                # Plot fit curve if available
                y0_abs_init = float(aa[0]) if aa.size > 0 and np.isfinite(float(aa[0])) else None
                fit_abs = fit_exponential_decay(t, aa, y0=y0_abs_init, min_points=4)
                if fit_abs is not None and np.isfinite(float(fit_abs.r2)) and float(fit_abs.r2) >= 0.70:
                    _draw_fit_with_extension(ax_abs_rep, fit_abs, t, color)
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
                # GOx always uses gray color
                pid_str = str(pid)
                if pid_str.upper() == "GOX":
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
                fit_rea = fit_exponential_decay(t, rea, y0=y0_rea_init, min_points=4)
                use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
                if use_exp_rea:
                    _draw_fit_with_extension(ax_rea_rep, fit_rea, t, color)
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
                fit_rea = fit_exponential_decay(t, rea, y0=y0_rea_init, min_points=4)
                use_exp_rea = bool(fit_rea is not None and np.isfinite(float(fit_rea.r2)) and float(fit_rea.r2) >= 0.70)
                t50_model = fit_rea.t50 if (fit_rea is not None and use_exp_rea) else None
                y0_rea_for_t50 = float(fit_rea.y0) if (fit_rea is not None and use_exp_rea) else (y0_rea_init if y0_rea_init is not None else (float(rea[0]) if rea.size > 0 and np.isfinite(float(rea[0])) else 100.0))
                t50_lin = t50_linear(t, rea, y0=y0_rea_for_t50, target_frac=0.5)
                t50_show = t50_model if t50_model is not None else t50_lin
                
                if t50_show is not None and np.isfinite(float(t50_show)) and float(t50_show) > 0.0:
                    t50_val = float(t50_show)
                    # Calculate y value at t50 from fitted curve (ensures intersection at center of fitted curve line)
                    if use_exp_rea and fit_rea is not None:
                        y_at_t50 = float(_eval_fit_curve(fit_rea, np.array([t50_val]))[0])
                    else:
                        y_at_t50 = 50.0  # Fallback to 50.0 if no fit
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

    return out_dir
