#!/usr/bin/env python3
"""
Dump default values used in Methods one-page summary (METHODS_ONE_PAGE.md).

Run from repo root:
  python scripts/dump_methods_defaults.py

Output: JSON to stdout. Use to keep research strategy/METHODS_ONE_PAGE.md in sync with code.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _literal_or_none(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_argparse_defaults(script_path: Path, option_names: list[str]) -> dict[str, object]:
    """
    Extract argparse defaults from add_argument calls in a script.

    Supports:
      - positional option strings in add_argument("--foo", ...)
      - default=...
      - action="store_true" / "store_false" (implicit defaults)
    """
    text = script_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    defaults: dict[str, object] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue

        opts: list[str] = []
        for arg in node.args:
            lit = _literal_or_none(arg)
            if isinstance(lit, str) and lit.startswith("--"):
                opts.append(lit)
        if not opts:
            continue

        explicit_default: object | None = None
        has_explicit_default = False
        action: str | None = None
        for kw in node.keywords:
            if kw.arg == "default":
                explicit_default = _literal_or_none(kw.value)
                has_explicit_default = True
            elif kw.arg == "action":
                act = _literal_or_none(kw.value)
                if isinstance(act, str):
                    action = act

        if has_explicit_default:
            default_val = explicit_default
        elif action == "store_true":
            default_val = False
        elif action == "store_false":
            default_val = True
        else:
            default_val = None

        for opt in opts:
            defaults[opt] = default_val

    missing = [name for name in option_names if name not in defaults]
    if missing:
        raise KeyError(f"Missing argparse defaults in {script_path}: {missing}")
    return {name: defaults[name] for name in option_names}


def _get_t50_defaults() -> dict:
    from gox_plate_pipeline.polymer_timeseries import (
        T50_DEFINITION_REA50,
        T50_DEFINITION_Y0_HALF,
    )
    return {
        "canonical_modes": [T50_DEFINITION_Y0_HALF, T50_DEFINITION_REA50],
        "default_mode": T50_DEFINITION_Y0_HALF,
        "y0_half_description": "threshold = 0.5 * fitted y0",
        "rea50_description": "threshold = fixed REA 50%",
        "unit": "min",
    }


def _get_fog_defaults() -> dict:
    from gox_plate_pipeline.fog import (
        NATIVE_ACTIVITY_MIN_REL_DEFAULT,
        T50_UNIT,
    )
    rel = float(NATIVE_ACTIVITY_MIN_REL_DEFAULT)
    return {
        "formula": "FoG = t50_polymer / t50_bare_GOx",
        "t50_unit": T50_UNIT,
        "native_activity_min_rel": rel,
        "native_activity_min_rel_pct": int(round(100 * rel)),
        "objective_column": "log_fog_activity_bonus_penalty",
    }


def _get_initial_rate_defaults() -> dict:
    defaults = _extract_argparse_defaults(
        SCRIPTS_DIR / "fit_initial_rates.py",
        [
            "--min_points",
            "--max_points",
            "--r2_min",
            "--select_method",
            "--find_start",
            "--start_max_shift",
            "--start_window",
            "--mono_min_frac",
            "--min_pos_steps",
            "--min_snr",
            "--max_t_end",
        ],
    )
    return {
        "min_points": int(defaults["--min_points"]),
        "max_points": int(defaults["--max_points"]),
        "r2_min": float(defaults["--r2_min"]),
        "select_method": str(defaults["--select_method"]),
        "find_start": bool(int(defaults["--find_start"])),
        "start_max_shift": int(defaults["--start_max_shift"]),
        "start_window": int(defaults["--start_window"]),
        "mono_min_frac": float(defaults["--mono_min_frac"]),
        "min_pos_steps": int(defaults["--min_pos_steps"]),
        "min_snr": float(defaults["--min_snr"]),
        "max_t_end_s": float(defaults["--max_t_end"]),
        "window_priority": [
            "1. Earlier start (late start penalized)",
            "2. Higher linearity (R²)",
            "3. Sufficient point count",
        ],
    }


def _get_bo_defaults() -> dict:
    defaults = _extract_argparse_defaults(
        SCRIPTS_DIR / "run_bayesian_optimization.py",
        [
            "--objective_column",
            "--acquisition",
            "--exploration_ratio",
            "--n_suggestions",
            "--ei_xi",
            "--ucb_kappa",
            "--anchor_fraction",
            "--replicate_fraction",
            "--anchor_polymer_ids",
            "--min_component",
            "--max_component",
            "--min_distance_between",
            "--min_distance_to_train",
            "--candidate_step",
            "--n_random_candidates",
            "--anchor_correction",
            "--min_anchor_polymers",
            "--sparse_isotropic_max_unique_points",
            "--min_length_scale_sparse_isotropic",
        ],
    )
    return {
        "objective_column": str(defaults["--objective_column"]),
        "acquisition": str(defaults["--acquisition"]),
        "exploration_ratio": float(defaults["--exploration_ratio"]),
        "n_suggestions": int(defaults["--n_suggestions"]),
        "ei_xi": float(defaults["--ei_xi"]),
        "ucb_kappa": float(defaults["--ucb_kappa"]),
        "anchor_fraction": float(defaults["--anchor_fraction"]),
        "replicate_fraction": float(defaults["--replicate_fraction"]),
        "anchor_polymer_ids": str(defaults["--anchor_polymer_ids"]),
        "min_component": float(defaults["--min_component"]),
        "max_component": float(defaults["--max_component"]),
        "min_distance_between": float(defaults["--min_distance_between"]),
        "min_distance_to_train": float(defaults["--min_distance_to_train"]),
        "candidate_step": float(defaults["--candidate_step"]),
        "n_random_candidates": int(defaults["--n_random_candidates"]),
        "anchor_correction_default": bool(defaults["--anchor_correction"]),
        "min_anchor_polymers": int(defaults["--min_anchor_polymers"]),
        "kernel": "Matern-5/2",
        "ard_default": True,
        "sparse_isotropic_max_unique_points": int(defaults["--sparse_isotropic_max_unique_points"]),
        "min_length_scale_sparse_isotropic": float(defaults["--min_length_scale_sparse_isotropic"]),
    }


def get_all_defaults() -> dict:
    """Return all defaults dict (for dump JSON or for update_methods_one_page)."""
    return {
        "t50": _get_t50_defaults(),
        "fog": _get_fog_defaults(),
        "initial_rate": _get_initial_rate_defaults(),
        "bo": _get_bo_defaults(),
    }


def main() -> None:
    print(json.dumps(get_all_defaults(), indent=2))


if __name__ == "__main__":
    main()
