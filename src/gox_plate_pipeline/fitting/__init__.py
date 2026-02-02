# src/gox_plate_pipeline/fitting/__init__.py
"""
Fitting subpackage - modular organization of fitting functions.

This package provides all fitting-related functionality split into focused modules:
  - core: Basic data structures and utility functions
  - preprocessing: Data preprocessing (add_well_coordinates, add_heat_time)
  - candidates: Candidate window generation and per-well fitting
  - selection: Fit selection algorithms
  - plotting: Diagnostic plotting
  - qc: QC report generation
  - pipeline: Main pipeline (compute_rates_and_rea)
"""

# Core types and utilities
from .core import (
    FitResult,
    FitSelectionError,
    apply_paper_style,
    _get_arial_rc,  # backward compat alias
    _fit_linear,
    _fit_linear_theilsen,
    _robust_sigma,
    _percentile_range,
    _auto_mono_eps,
    _auto_min_delta_y,
    _detect_step_jump,
    PAPER_FIGSIZE_SINGLE,
    PAPER_FIGSIZE_WIDE,
    PAPER_FIGSIZE_SQUARE,
    paper_savefig,
    _find_start_index,
)

# Preprocessing
from .preprocessing import (
    add_well_coordinates,
    add_heat_time,
)

# Candidate generation
from .candidates import (
    generate_candidate_windows,
    fit_initial_rate_one_well,
)

# Selection
from .selection import (
    select_fit,
    try_skip_extend,
    try_extend_fit,
    _enforce_final_safety,
    find_best_short_window,
    find_fit_with_outlier_removal,
    detect_internal_outliers,
    detect_curvature_and_shorten,
    fit_with_outlier_skip_full_range,
)

# Plotting
from .plotting import (
    _format_heat,
    plot_fit_diagnostic,
)

# QC
from .qc import (
    _normalize_exclude_reason,
    write_fit_qc_report,
)

# Pipeline
from .pipeline import (
    compute_rates_and_rea,
)

__all__ = [
    # Core
    "FitResult",
    "FitSelectionError",
    "apply_paper_style",
    "_get_arial_rc",
    "PAPER_FIGSIZE_SINGLE",
    "PAPER_FIGSIZE_WIDE",
    "PAPER_FIGSIZE_SQUARE",
    "paper_savefig",
    "_fit_linear",
    "_fit_linear_theilsen",
    "_robust_sigma",
    "_percentile_range",
    "_auto_mono_eps",
    "_auto_min_delta_y",
    "_detect_step_jump",
    "_find_start_index",
    # Preprocessing
    "add_well_coordinates",
    "add_heat_time",
    # Candidates
    "generate_candidate_windows",
    "fit_initial_rate_one_well",
    # Selection
    "select_fit",
    "try_skip_extend",
    "try_extend_fit",
    "_enforce_final_safety",
    "find_best_short_window",
    "find_fit_with_outlier_removal",
    # Plotting
    "_format_heat",
    "plot_fit_diagnostic",
    # QC
    "_normalize_exclude_reason",
    "write_fit_qc_report",
    # Pipeline
    "compute_rates_and_rea",
]
