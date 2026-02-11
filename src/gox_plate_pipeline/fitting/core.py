# src/gox_plate_pipeline/fitting/core.py
"""
Core data structures and utility functions for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import font_manager as fm
import matplotlib.patheffects as path_effects
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class FitResult:
    slope: float
    intercept: float
    r2: float
    n: int
    t_start: float
    t_end: float


class FitSelectionError(RuntimeError):
    """Raised when no acceptable fitting window can be selected for a well."""


# Distance from plot frame to info box (points); same for both axes so top-right / top-left margins are equal
INFO_BOX_MARGIN_PT = 8

# Info box styling
INFO_BOX_FACE_COLOR = "0.96"  # Very light gray fill
INFO_BOX_PAD_DEFAULT = 0.3  # Default padding for most plots
# For per_polymer: padding on all sides (left, bottom, right, top) for left-aligned text
# Increased padding to reduce cramped appearance
INFO_BOX_PAD_PER_POLYMER = (0.6, 0.5, 0.6, 0.5)  # All sides with larger margin for left-aligned content


def get_info_box_gradient_shadow() -> list:
    """
    Create a gradient shadow effect for info boxes by layering multiple shadows.
    Shadows get progressively further and more transparent to create a smooth gradient.
    """
    return [
        path_effects.SimplePatchShadow(offset=(1.0, -1.0), shadow_rgbFace="0.4", alpha=0.10, rho=0.3),
        path_effects.SimplePatchShadow(offset=(2.0, -2.0), shadow_rgbFace="0.4", alpha=0.07, rho=0.5),
        path_effects.SimplePatchShadow(offset=(3.5, -3.5), shadow_rgbFace="0.4", alpha=0.04, rho=0.7),
        path_effects.SimplePatchShadow(offset=(5.0, -5.0), shadow_rgbFace="0.4", alpha=0.02, rho=0.9),
        path_effects.Normal(),  # Draw the actual box on top
    ]


def apply_paper_style() -> dict:
    """
    Return matplotlib rcParams dict for paper-grade figures.
    
    All figures in this project use this style - no exceptions.
    Settings follow Figure-rules.mdc and Matplotlib-style-implementation.mdc.
    
    Font priority: Arial > Helvetica > DejaVu Sans
    Font size: 6-7 pt (final output, paper-grade)
    Line width: 0.6-0.8 pt
    DPI: 600 (line/text dominant figures)
    Output: PNG only (PDF not required)
    
    Usage:
        with plt.rc_context(apply_paper_style()):
            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            ...
            fig.savefig("out.png")  # dpi/bbox from rcParams
    """
    # Font: Arial is mandatory. Rebuild font cache if needed.
    # Priority: Arial > Helvetica > Liberation Sans > DejaVu Sans
    available = {f.name for f in fm.fontManager.ttflist}
    
    # Check if Arial is available
    if "Arial" not in available:
        # Try to rebuild font cache
        fm._load_fontmanager(try_read_cache=False)
        available = {f.name for f in fm.fontManager.ttflist}
    
    # Determine which font to use (Arial preferred)
    font_priority = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    chosen = next((f for f in font_priority if f in available), "DejaVu Sans")
    
    return {
        # ===== FONT (Arial mandatory) =====
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,  # TrueType embedding (for rare PDF needs)
        "ps.fonttype": 42,
        
        # ===== FONT SIZES (paper-grade: 5-8 pt, target 6-7 pt) =====
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,
        
        # ===== LINES & MARKERS (paper-grade: 0.6-0.9 pt) =====
        "lines.linewidth": 0.7,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.4,
        
        # ===== AXES & SPINES (full frame, uniform width) =====
        "axes.linewidth": 0.6,
        "axes.edgecolor": "0.3",
        "axes.labelcolor": "0.15",
        "axes.titlepad": 4,
        "axes.labelpad": 3,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.facecolor": "white",
        "axes.axisbelow": True,  # grid behind data
        
        # ===== TICKS (subtle, paper-like) =====
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.color": "0.3",
        "ytick.color": "0.3",
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # ===== GRID (minimal, unobtrusive) =====
        "axes.grid": False,  # off by default, enable explicitly if needed
        "grid.linewidth": 0.3,
        "grid.alpha": 0.3,
        "grid.color": "0.7",
        "grid.linestyle": "-",
        
        # ===== LEGEND (clean, no frame clutter) =====
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.7",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.2,
        "legend.handleheight": 0.7,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
        
        # ===== FIGURE =====
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.dpi": 100,  # screen display
        "figure.autolayout": False,  # we use tight_layout explicitly
        
        # ===== SAVEFIG (PNG, 600 dpi for line/text figures) =====
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.format": "png",
        
        # ===== HISTOGRAM / BAR =====
        "hist.bins": 30,
    }


# Backward compatibility alias
_get_arial_rc = apply_paper_style


# Standard figure sizes (1-column width ≈ 90mm ≈ 3.5 in)
PAPER_FIGSIZE_SINGLE = (3.5, 2.6)  # Single panel
PAPER_FIGSIZE_WIDE = (5.0, 2.6)   # Wide single panel
PAPER_FIGSIZE_SQUARE = (3.0, 3.0)  # Square panel


def paper_savefig(fig, path, **kwargs):
    """
    Save figure with paper-grade settings.
    
    Ensures PNG output at 600 dpi with tight bbox.
    Use this instead of fig.savefig() for consistency.
    """
    defaults = {
        "dpi": 600,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
        "edgecolor": "white",
    }
    defaults.update(kwargs)
    fig.savefig(path, **defaults)


def _fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Ordinary least squares linear fit: y = a*x + b
    Returns slope/intercept and R^2.
    """
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return FitResult(
        slope=float(a),
        intercept=float(b),
        r2=float(r2),
        n=int(len(x)),
        t_start=float(np.min(x)),
        t_end=float(np.max(x)),
    )


def _fit_linear_theilsen(x: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Theil-Sen robust linear fit: slope = median of pairwise slopes.
    Outlier-resistant; R^2 computed from residuals for compatibility with selection logic.
    """
    res = scipy_stats.theilslopes(y, x=x, method="separate")
    slope = float(res.slope)
    intercept = float(res.intercept)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return FitResult(
        slope=slope,
        intercept=intercept,
        r2=float(r2),
        n=int(len(x)),
        t_start=float(np.min(x)),
        t_end=float(np.max(x)),
    )


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)  # MAD -> sigma (normal)


def _percentile_range(y: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    lo, hi = np.percentile(y, [p_low, p_high])
    return float(hi - lo)


def _auto_mono_eps(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    rng = _percentile_range(y)
    sigma_dy = _robust_sigma(np.diff(y)) if y.size >= 2 else 0.0
    # eps is the threshold to call a decrease "significant"
    return float(max(1e-12, 0.01 * rng, 3.0 * sigma_dy))


def _auto_min_delta_y(y: np.ndarray, mono_eps: float) -> float:
    y = np.asarray(y, dtype=float)
    rng = _percentile_range(y)
    # NOTE:
    #   sigma(y) is inflated by the reaction trend itself (monotonic increase),
    #   so using it as "noise" makes min_delta_y unrealistically large.
    #   Use scale-based and step-noise-based terms instead.
    # Cap at 10% of range to allow fitting low-slope data (relaxed from 30%)
    base = float(max(1e-12, 0.02 * rng, 3.0 * float(mono_eps)))
    cap = 0.10 * rng
    return float(min(base, cap)) if cap > 0 else base


def _detect_step_jump(
    y: np.ndarray,
    threshold_frac: float = 0.25,
) -> Optional[int]:
    """
    Detect a sustained upward step jump in the data.

    Returns the index of the LAST point BEFORE the jump, or None if no jump detected.

    Guardrails:
      - Ignore single-point spikes.
      - Detect only level-shift-like changes (flat-ish before/after),
        not natural rapidly rising reaction phases.
    """
    y = np.asarray(y, dtype=float)
    n = int(len(y))
    if n < 4:
        return None

    # Robust range to reduce sensitivity to isolated outliers.
    rng = _percentile_range(y)
    if rng < 1e-12:
        return None

    threshold = float(threshold_frac) * float(rng)
    k = 3 if n >= 8 else 2

    for i in range(0, n - k - 1):
        pre_seg = y[max(0, i - k + 1) : i + 1]
        post_seg = y[i + 1 : i + 1 + k]
        pre = float(np.median(pre_seg))
        post = float(np.median(post_seg))

        # Candidate step size
        if (post - pre) <= threshold:
            continue

        # Require level-shift-like local behavior.
        pre_span = float(abs(pre_seg[-1] - pre_seg[0])) if pre_seg.size >= 2 else 0.0
        post_span = float(abs(post_seg[-1] - post_seg[0])) if post_seg.size >= 2 else 0.0
        trend_guard = 0.35 * threshold
        if pre_span > trend_guard or post_span > trend_guard:
            continue

        # Persistency check to avoid one-point spikes.
        post_min = float(np.min(post_seg))
        if post_min > (pre + 0.5 * threshold):
            return int(i)

    return None


def _is_isolated_down_spike(y: np.ndarray, idx: int, mono_eps: float) -> bool:
    """
    Return True when idx looks like an isolated local downward spike.

    This is used to avoid moving start_idx forward because of one noisy dip
    while the trace is already in a rising phase.
    """
    y = np.asarray(y, dtype=float)
    i = int(idx)
    if i <= 0 or i >= (len(y) - 1):
        return False

    rng = _percentile_range(y)
    dip_thr = float(max(3.0 * float(mono_eps), 0.08 * float(rng)))
    bridge_thr = float(max(2.0 * float(mono_eps), 0.10 * float(rng)))

    left = float(y[i - 1])
    mid = float(y[i])
    right = float(y[i + 1])
    deep_dip = (left - mid) > dip_thr and (right - mid) > dip_thr
    bridged = abs(right - left) <= bridge_thr
    return bool(deep_dip and bridged)


def _find_start_index(
    t: np.ndarray,
    y: np.ndarray,
    mono_eps: float,
    max_shift: int = 5,
    window: int = 3,
    allow_down_steps: int = 1,
    min_rise: Optional[float] = None,
) -> int:
    """
    Start-point detection with lag-phase awareness.

    The selector is direction-aware:
      - Increasing traces: find a local minimum before sustained rise.
      - Decreasing traces: mirror the logic on (-y), equivalent to finding
        a local maximum before sustained fall.

    For clearly decreasing traces that already decline from the first few
    points (no lag phase), return index 0 so early windows remain available.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(len(y))
    if n < window:
        return 0

    dy = np.diff(y)
    eps = float(mono_eps)

    # Infer overall trend direction from net change with a noise-aware threshold.
    sigma_dy = _robust_sigma(dy) if dy.size > 0 else 0.0
    trend_thr = float(max(1e-12, 0.5 * sigma_dy))
    net_all = float(y[-1] - y[0]) if n >= 2 else 0.0
    trend_sign = -1 if net_all < -trend_thr else 1
    
    # For increasing traces (normal case), use original logic unchanged.
    # This preserves existing behavior for wells that already work correctly.
    if trend_sign > 0:
        # Original logic for increasing traces - no changes
        y_eval = y
        dy_eval_full = np.diff(y_eval)
    else:
        # Decreasing traces: check if lag phase exists or if decline starts immediately.
        # Design principle: "Rescue locally, keep normal case unchanged"
        
        # Step 1: Check if there's a clear lag phase (initial flat/noisy section)
        # Use a longer probe to assess early behavior more robustly
        probe_len = min(n, max(8, int(window) + 5))
        if probe_len >= 4:
            probe = y[:probe_len]
            d_probe = np.diff(probe)
            step_eps = float(max(1e-12, 0.25 * sigma_dy))
            
            # Count significant steps
            down_steps = int(np.sum(d_probe < -step_eps))
            up_steps = int(np.sum(d_probe > step_eps))
            net_probe = float(probe[-1] - probe[0])
            
            # Guardrail 1: Overall signal range check - avoid rescuing very flat traces (E-row case)
            # E-row wells have very small y_range (1-2), while real signals have larger range
            y_range = float(np.max(y) - np.min(y))
            has_minimum_range = y_range >= max(eps * 1.5, 5.0)  # At least 5.0 or 1.5*eps
            
            # Guardrail 2: Net change check - require meaningful overall change
            # Use a more lenient threshold: either absolute change or relative to range
            min_net_change = max(eps * 2.0, y_range * 0.15, 10.0)  # At least 10.0 or 15% of range or 2*eps
            has_meaningful_change = abs(net_all) >= min_net_change
            
            # Guardrail 3: Early trend check - if first portion shows sustained decline,
            # this is likely "no lag phase" case (B1, B2, B4, A1, C2, D1)
            # Condition: more down steps than up steps, and net decline in probe
            early_decline = (down_steps >= up_steps) and (net_probe < -step_eps)
            
            # Guardrail 4: Avoid rescuing very noisy traces (E-row)
            # Check if early portion is dominated by noise (many small fluctuations)
            noise_ratio = float(np.sum(np.abs(d_probe) < step_eps)) / max(1, len(d_probe))
            is_not_noise_dominated = noise_ratio < 0.65  # Less than 65% of steps are noise-level
            
            # Fast path: return 0 if no lag phase detected AND guardrails pass
            # All guardrails must pass to ensure we don't rescue E-row noisy traces
            if early_decline and has_minimum_range and has_meaningful_change and is_not_noise_dominated:
                return 0
        
        # Step 2: If fast path didn't trigger, use mirrored logic for decreasing traces
        # This handles cases with lag phase in decreasing traces
        y_eval = -y
        dy_eval_full = np.diff(y_eval)

    # Find all local minima in the search range
    search_range = min(int(max_shift) + window + 3, n - window)
    local_minima = []

    for i in range(search_range):
        is_local_min = True
        # Check if point i is a local minimum (lower than neighbors)
        if i > 0 and y_eval[i] >= y_eval[i - 1]:
            is_local_min = False
        if i < n - 1 and y_eval[i] > y_eval[i + 1]:
            is_local_min = False

        # Also consider points where decrease ends and increase begins
        if i > 0 and i < n - 1:
            if dy_eval_full[i - 1] < 0 and dy_eval_full[i] > 0:  # transition from down to up
                is_local_min = True

        if is_local_min:
            local_minima.append(i)

    # If no local minima found, add index 0 and the global minimum
    if not local_minima:
        local_minima = [0]
        if search_range > 1:
            gmin = int(np.argmin(y_eval[:search_range]))
            if gmin not in local_minima:
                local_minima.append(gmin)

    # Evaluate each local minimum: score by quality of subsequent trend
    best_idx = 0
    best_score = -np.inf

    for idx in local_minima:
        if idx >= n - window:
            continue

        # Evaluate quality over a longer window (to assess stability)
        eval_len = min(8, n - idx)
        yw = y_eval[idx : idx + eval_len]
        dy_local = np.diff(yw)

        # Score: ratio of positive steps + bonus for net rise
        up_count = int(np.sum(dy_local > 0))
        down_count = int(np.sum(dy_local < -eps * 0.5))
        net_rise = float(yw[-1] - yw[0])

        # Penalize if there are early decreases (noisy start)
        early_down = int(np.sum(dy_local[:2] < 0)) if len(dy_local) >= 2 else 0

        score = (up_count - down_count) + (net_rise / max(eps, 1.0)) - early_down * 0.5

        if score > best_score:
            best_score = score
            best_idx = idx

    # Validate: ensure there's actually upward trend after best_idx
    if best_idx < n - window:
        yw_after = y_eval[best_idx : best_idx + window + 2]
        if len(yw_after) > 1:
            net = float(yw_after[-1] - yw_after[0])
            if net > 0:
                # For increasing traces: if the chosen "minimum" is already well above y[0],
                # this is not a true lag phase (just a small dip in a rising trace) → prefer 0
                # so that B1/B2/B4-style wells include the first points (user requirement).
                if trend_sign > 0 and best_idx > 0:
                    y_range = float(np.max(y) - np.min(y))
                    # Use relative rise from y[0]: if "minimum" is already >12% of range above start,
                    # treat as shallow dip, not lag (so B1/B2/B4 get first points in).
                    lag_thr = max(1e-6 * y_range, y_range * 0.12)
                    if float(y[best_idx]) > float(y[0]) + lag_thr:
                        return 0
                return int(best_idx)

    # Fallback: find global minimum in search range
    if search_range > 1:
        gmin = int(np.argmin(y_eval[:search_range]))
        return int(gmin)

    return 0
