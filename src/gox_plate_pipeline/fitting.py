# src/gox_plate_pipeline/fitting.py
"""
Backward-compatible re-export of all fitting functions.

The fitting functionality has been modularized into the fitting/ subpackage.
This file re-exports everything for backward compatibility with existing imports.

New code should import directly from the submodules:
    from gox_plate_pipeline.fitting.core import FitResult
    from gox_plate_pipeline.fitting.pipeline import compute_rates_and_rea

Legacy imports still work:
    from gox_plate_pipeline.fitting import compute_rates_and_rea
"""

# Re-export everything from the fitting subpackage
from gox_plate_pipeline.fitting import (
    # Core
    FitResult,
    FitSelectionError,
    apply_paper_style,
    _get_arial_rc,
    PAPER_FIGSIZE_SINGLE,
    PAPER_FIGSIZE_WIDE,
    PAPER_FIGSIZE_SQUARE,
    paper_savefig,
    _fit_linear,
    _fit_linear_theilsen,
    _robust_sigma,
    _percentile_range,
    _auto_mono_eps,
    _auto_min_delta_y,
    _detect_step_jump,
    _find_start_index,
    # Preprocessing
    add_well_coordinates,
    add_heat_time,
    # Candidates
    generate_candidate_windows,
    fit_initial_rate_one_well,
    # Selection
    select_fit,
    try_skip_extend,
    try_extend_fit,
    _enforce_final_safety,
    # Plotting
    _format_heat,
    plot_fit_diagnostic,
    write_plate_grid,
    # QC
    _normalize_exclude_reason,
    write_fit_qc_report,
    # Pipeline
    compute_rates_and_rea,
)

__all__ = [
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
    "add_well_coordinates",
    "add_heat_time",
    "generate_candidate_windows",
    "fit_initial_rate_one_well",
    "select_fit",
    "try_skip_extend",
    "try_extend_fit",
    "_enforce_final_safety",
    "_format_heat",
    "plot_fit_diagnostic",
    "write_plate_grid",
    "_normalize_exclude_reason",
    "write_fit_qc_report",
    "compute_rates_and_rea",
]
