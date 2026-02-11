from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gox_plate_pipeline.fitting.candidates import fit_initial_rate_one_well  # noqa: E402
from gox_plate_pipeline.fitting.core import FitSelectionError, _detect_step_jump  # noqa: E402
from gox_plate_pipeline.fitting.pipeline import (  # noqa: E402
    compute_rates_and_rea,
    _rescue_broad_overestimate,
)


def _build_tidy(plate_id: str, well: str, y: np.ndarray) -> pd.DataFrame:
    t = np.arange(len(y), dtype=float) * 30.0 + 6.0
    return pd.DataFrame(
        {
            "plate_id": plate_id,
            "well": well,
            "time_s": t,
            "signal": y.astype(float),
            "polymer_id": "P1",
        }
    )


def _identity_sel(sel: pd.Series, *args, **kwargs) -> pd.Series:
    return sel


def _identity_cands_sel(cands: pd.DataFrame, sel: pd.Series, *args, **kwargs) -> pd.Series:
    return sel


def _build_neighbor_tidy() -> pd.DataFrame:
    t = np.arange(10, dtype=float) * 30.0 + 6.0
    rows = []
    for well, slope in [("A1", 0.10), ("A2", 0.20)]:
        y = 10.0 + slope * t
        for ti, yi in zip(t, y):
            rows.append(
                {
                    "plate_id": "plateN",
                    "well": well,
                    "time_s": float(ti),
                    "signal": float(yi),
                    "polymer_id": "PN",
                }
            )
    return pd.DataFrame(rows)


def _sel_for_test(slope: float, method: str = "initial_positive") -> pd.Series:
    return pd.Series(
        {
            "t_start": 6.0,
            "t_end": 186.0,
            "n": 7,
            "slope": float(slope),
            "intercept": 10.0,
            "r2": 0.94,
            "start_idx": 0,
            "end_idx": 6,
            "dy": 20.0,
            "mono_frac": 1.0,
            "down_steps": 0,
            "pos_steps": 6,
            "pos_steps_eps": 6,
            "pos_eps": 0.0,
            "rmse": 0.1,
            "snr": 20.0,
            "start_idx_used": 0,
            "skip_indices": "",
            "skip_count": 0,
            "select_method_used": method,
        }
    )


class FittingLogicGuardsTests(unittest.TestCase):
    def test_step_jump_ignores_single_spike(self) -> None:
        y = 100.0 + 2.0 * np.arange(21, dtype=float)
        y[11] += 80.0  # single-point upward spike
        jump_idx = _detect_step_jump(y, threshold_frac=0.25)
        self.assertIsNone(jump_idx)

    def test_step_jump_detects_persistent_level_shift(self) -> None:
        y = np.array([100, 101, 99, 100, 101, 100, 102, 152, 151, 153, 152, 154], dtype=float)
        jump_idx = _detect_step_jump(y, threshold_frac=0.25)
        self.assertEqual(jump_idx, 6)

    def test_isolated_down_spike_keeps_start_idx_zero(self) -> None:
        # Local dip at index=1 should be treated as noise, not lag.
        y = np.array([10, 4, 16, 26, 36, 46, 56, 66, 76, 86], dtype=float)
        tidy = _build_tidy("plate1", "A1", y)
        cands = fit_initial_rate_one_well(
            tidy,
            min_points=6,
            max_points=8,
            find_start=True,
            start_max_shift=5,
            start_window=3,
            start_allow_down_steps=1,
        )
        self.assertFalse(cands.empty)
        self.assertTrue((cands["start_idx_used"] == 0).all())

    def test_outlier_removal_fallback_runs_before_short_window(self) -> None:
        y = 100.0 + 1.5 * np.arange(21, dtype=float)
        tidy = _build_tidy("plate1", "A1", y)

        fake_sel = pd.Series(
            {
                "t_start": 6.0,
                "t_end": 156.0,
                "n": 6,
                "slope": 1.5,
                "intercept": 100.0,
                "r2": 0.95,
                "start_idx": 0,
                "end_idx": 5,
                "dy": 7.5,
                "mono_frac": 1.0,
                "down_steps": 0,
                "pos_steps": 5,
                "pos_steps_eps": 5,
                "pos_eps": 0.1,
                "rmse": 0.1,
                "snr": 75.0,
                "start_idx_used": 0,
                "skip_indices": "7",
                "skip_count": 1,
                "select_method_used": "outlier_removed",
            }
        )

        with patch("gox_plate_pipeline.fitting.pipeline.select_fit", side_effect=FitSelectionError("forced")):
            with patch("gox_plate_pipeline.fitting.pipeline.find_fit_with_outlier_removal", return_value=fake_sel.copy()) as m_out:
                with patch("gox_plate_pipeline.fitting.pipeline.find_best_short_window", return_value=fake_sel.copy()) as m_short:
                    with patch("gox_plate_pipeline.fitting.pipeline.fit_with_outlier_skip_full_range", return_value=None):
                        with patch("gox_plate_pipeline.fitting.pipeline.try_extend_fit", side_effect=_identity_sel):
                            with patch("gox_plate_pipeline.fitting.pipeline.try_skip_extend", side_effect=_identity_sel):
                                with patch("gox_plate_pipeline.fitting.pipeline.detect_internal_outliers", side_effect=_identity_sel):
                                    with patch("gox_plate_pipeline.fitting.pipeline.detect_curvature_and_shorten", side_effect=_identity_sel):
                                        with patch("gox_plate_pipeline.fitting.pipeline.apply_conservative_long_override", side_effect=_identity_sel):
                                            selected, _ = compute_rates_and_rea(
                                                tidy=tidy,
                                                heat_times=[0, 5, 10, 15, 20, 40, 60],
                                            )

        self.assertTrue(m_out.called)
        self.assertFalse(m_short.called)
        row = selected.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertIn("outlier_removed", str(row["select_method_used"]))

    def test_rescue_broad_overestimate_prefers_broad_trim(self) -> None:
        t = np.arange(21, dtype=float) * 30.0 + 6.0
        y = 10.0 + 0.05 * t

        sel = pd.Series(
            {
                "slope": 0.30,
                "r2": 0.82,
                "n": 5,
                "start_idx": 5,
                "end_idx": 9,
                "t_start": float(t[5]),
                "t_end": float(t[9]),
                "select_method_used": "initial_positive",
            }
        )

        full_stats = {
            "t_start": float(t[0]),
            "t_end": float(t[-1]),
            "n": 21,
            "slope": 0.07,
            "intercept": 9.0,
            "r2": 0.62,
            "start_idx": 0,
            "end_idx": 20,
            "dy": 40.0,
            "mono_frac": 1.0,
            "down_steps": 0,
            "pos_steps": 18,
            "pos_steps_eps": 18,
            "pos_eps": 0.0,
            "rmse": 1.0,
            "snr": 8.0,
            "slope_se": 0.01,
            "slope_t": 7.0,
            "outlier_count": 0,
            "slope_half_drop_frac": 0.0,
            "mono_eps": 0.1,
            "min_delta_y": 0.0,
            "start_idx_used": 0,
        }

        broad_trim = pd.Series(
            {
                "t_start": float(t[0]),
                "t_end": float(t[-1]),
                "n": 19,
                "slope": 0.08,
                "intercept": 9.5,
                "r2": 0.64,
                "start_idx": 0,
                "end_idx": 20,
                "dy": 41.0,
                "mono_frac": 1.0,
                "down_steps": 2,
                "pos_steps": 17,
                "pos_steps_eps": 17,
                "pos_eps": 0.0,
                "rmse": 1.0,
                "snr": 9.0,
                "start_idx_used": 0,
                "skip_indices": "3,14",
                "skip_count": 2,
                "select_method_used": "full_range_outlier_skip",
            }
        )

        with patch("gox_plate_pipeline.fitting.pipeline._calc_window_stats", return_value=full_stats):
            with patch(
                "gox_plate_pipeline.fitting.pipeline.fit_with_outlier_skip_full_range",
                return_value=broad_trim,
            ):
                out = _rescue_broad_overestimate(
                    sel=sel,
                    t=t,
                    y=y,
                    min_points=6,
                    r2_gate=0.80,
                    fit_method="ols",
                )

        self.assertLess(float(out["slope"]), float(sel["slope"]))
        self.assertEqual(int(out["n"]), 19)
        self.assertIn("post_broad_overfit_ext", str(out["select_method_used"]))
        self.assertTrue(float(out["r2_min_override"]) >= 0.60)
        self.assertTrue(float(out["r2_min_override"]) < 0.80)
        self.assertEqual(int(out["mono_max_down_steps_override"]), 2)

    def test_rescue_broad_overestimate_does_not_trigger_for_early_window(self) -> None:
        t = np.arange(21, dtype=float) * 30.0 + 6.0
        y = 20.0 + 0.1 * t
        sel = pd.Series(
            {
                "slope": 0.12,
                "r2": 0.93,
                "n": 6,
                "start_idx": 0,
                "end_idx": 5,
                "t_start": float(t[0]),
                "t_end": float(t[5]),
                "select_method_used": "initial_positive",
            }
        )

        out = _rescue_broad_overestimate(
            sel=sel,
            t=t,
            y=y,
            min_points=6,
            r2_gate=0.90,
            fit_method="ols",
        )

        self.assertEqual(float(out["slope"]), float(sel["slope"]))
        self.assertEqual(int(out["n"]), int(sel["n"]))

    def test_neighbor_recheck_trigger_updates_left_well(self) -> None:
        tidy = _build_neighbor_tidy()
        cands = pd.DataFrame({"start_idx": [0], "n": [7], "slope": [0.1], "r2": [0.94], "snr": [20.0], "t_end": [186.0]})

        sel_a1 = _sel_for_test(0.10, "initial_positive")
        sel_a2 = _sel_for_test(0.20, "initial_positive")

        def fake_neighbor(sel: pd.Series, t: np.ndarray, y: np.ndarray, *, neighbor_slope: float, fit_method: str = "ols") -> pd.Series:
            out = sel.copy()
            if float(sel["slope"]) < float(neighbor_slope):
                out["slope"] = float(neighbor_slope) * 1.02
                out["select_method_used"] = f"{str(sel.get('select_method_used', ''))}_neighbor_recheck".strip("_")
            return out

        with patch("gox_plate_pipeline.fitting.pipeline.fit_initial_rate_one_well", return_value=cands):
            with patch("gox_plate_pipeline.fitting.pipeline.select_fit", side_effect=[sel_a1.copy(), sel_a2.copy()]):
                with patch("gox_plate_pipeline.fitting.pipeline._promote_longer_if_similar", side_effect=_identity_cands_sel):
                    with patch("gox_plate_pipeline.fitting.pipeline._prefer_early_steeper", side_effect=_identity_cands_sel):
                        with patch("gox_plate_pipeline.fitting.pipeline._prefer_delayed_steeper_when_short", side_effect=_identity_cands_sel):
                            with patch("gox_plate_pipeline.fitting.pipeline.fit_with_outlier_skip_full_range", return_value=None):
                                with patch("gox_plate_pipeline.fitting.pipeline.try_extend_fit", side_effect=_identity_sel):
                                    with patch("gox_plate_pipeline.fitting.pipeline.try_skip_extend", side_effect=_identity_sel):
                                        with patch("gox_plate_pipeline.fitting.pipeline.detect_internal_outliers", side_effect=_identity_sel):
                                            with patch("gox_plate_pipeline.fitting.pipeline.detect_curvature_and_shorten", side_effect=_identity_sel):
                                                with patch("gox_plate_pipeline.fitting.pipeline.apply_conservative_long_override", side_effect=_identity_sel):
                                                    with patch("gox_plate_pipeline.fitting.pipeline._apply_local_fit_audit", side_effect=_identity_sel):
                                                        with patch("gox_plate_pipeline.fitting.pipeline._enforce_final_safety", return_value=None):
                                                            with patch("gox_plate_pipeline.fitting.pipeline._neighbor_recheck_trigger", side_effect=fake_neighbor):
                                                                selected, _ = compute_rates_and_rea(
                                                                    tidy=tidy,
                                                                    heat_times=[0, 5, 10, 15, 20, 40, 60],
                                                                )

        a1 = selected[selected["well"] == "A1"].iloc[0]
        a2 = selected[selected["well"] == "A2"].iloc[0]
        self.assertGreater(float(a1["slope"]), float(a2["slope"]))
        self.assertIn("neighbor_recheck", str(a1["select_method_used"]))

    def test_plot_uses_final_selection_after_neighbor_recheck(self) -> None:
        tidy = _build_neighbor_tidy()
        cands = pd.DataFrame({"start_idx": [0], "n": [7], "slope": [0.1], "r2": [0.94], "snr": [20.0], "t_end": [186.0]})

        sel_a1 = _sel_for_test(0.10, "initial_positive")
        sel_a2 = _sel_for_test(0.20, "initial_positive")
        plotted_slopes: dict[str, float] = {}

        def fake_neighbor(sel: pd.Series, t: np.ndarray, y: np.ndarray, *, neighbor_slope: float, fit_method: str = "ols") -> pd.Series:
            out = sel.copy()
            if float(sel["slope"]) < float(neighbor_slope):
                out["slope"] = float(neighbor_slope) * 1.02
                out["select_method_used"] = f"{str(sel.get('select_method_used', ''))}_neighbor_recheck".strip("_")
            return out

        def fake_plot(df_well: pd.DataFrame, meta: dict, selected: pd.Series, status: str, exclude_reason: str, out_png: Path) -> None:
            if selected is not None:
                plotted_slopes[str(meta["well"])] = float(selected["slope"])

        with patch("gox_plate_pipeline.fitting.pipeline.fit_initial_rate_one_well", return_value=cands):
            with patch("gox_plate_pipeline.fitting.pipeline.select_fit", side_effect=[sel_a1.copy(), sel_a2.copy()]):
                with patch("gox_plate_pipeline.fitting.pipeline._promote_longer_if_similar", side_effect=_identity_cands_sel):
                    with patch("gox_plate_pipeline.fitting.pipeline._prefer_early_steeper", side_effect=_identity_cands_sel):
                        with patch("gox_plate_pipeline.fitting.pipeline._prefer_delayed_steeper_when_short", side_effect=_identity_cands_sel):
                            with patch("gox_plate_pipeline.fitting.pipeline.fit_with_outlier_skip_full_range", return_value=None):
                                with patch("gox_plate_pipeline.fitting.pipeline.try_extend_fit", side_effect=_identity_sel):
                                    with patch("gox_plate_pipeline.fitting.pipeline.try_skip_extend", side_effect=_identity_sel):
                                        with patch("gox_plate_pipeline.fitting.pipeline.detect_internal_outliers", side_effect=_identity_sel):
                                            with patch("gox_plate_pipeline.fitting.pipeline.detect_curvature_and_shorten", side_effect=_identity_sel):
                                                with patch("gox_plate_pipeline.fitting.pipeline.apply_conservative_long_override", side_effect=_identity_sel):
                                                    with patch("gox_plate_pipeline.fitting.pipeline._apply_local_fit_audit", side_effect=_identity_sel):
                                                        with patch("gox_plate_pipeline.fitting.pipeline._enforce_final_safety", return_value=None):
                                                            with patch("gox_plate_pipeline.fitting.pipeline._neighbor_recheck_trigger", side_effect=fake_neighbor):
                                                                with patch("gox_plate_pipeline.fitting.pipeline.plot_fit_diagnostic", side_effect=fake_plot):
                                                                    with TemporaryDirectory() as td:
                                                                        selected, _ = compute_rates_and_rea(
                                                                            tidy=tidy,
                                                                            heat_times=[0, 5, 10, 15, 20, 40, 60],
                                                                            plot_dir=Path(td),
                                                                            plot_mode="all",
                                                                        )

        a1 = selected[selected["well"] == "A1"].iloc[0]
        self.assertIn("A1", plotted_slopes)
        self.assertAlmostEqual(float(a1["slope"]), plotted_slopes["A1"], places=12)


if __name__ == "__main__":
    unittest.main()
