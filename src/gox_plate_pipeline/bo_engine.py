"""
Bayesian optimization engine for GOx polymer composition search.

Design goals for this project:
- Reuse pipeline outputs (bo_learning + fog_plate_aware) without changing fitting logic.
- Keep BO traceable with bo_run_id, manifest, and proposal-reason logs.
- Generate paper-grade outputs including ternary maps and ranking tables.

Terminology:
- GOx (no polymer) and e.g. PMPC are included every run by the experimenter; BO does not define "control" as GOx.
- Anchor slots: fixed polymer compositions (e.g. PMPC, PMTAC) re-proposed in each batch for comparison.
- Round-to-round correction by anchor polymers is off by default (not appropriate for biological experiments).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import cycle
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.colors as mcolors
import matplotlib.patheffects as mpatheffects
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

from gox_plate_pipeline.bo_data import frac_from_xy as _frac_from_xy_bo_data
from gox_plate_pipeline.fitting.core import apply_paper_style
from gox_plate_pipeline.summary import build_run_manifest_dict


EPS = 1e-12
SQRT3_OVER_2 = np.sqrt(3.0) / 2.0


def _natural_round_key(round_id: str) -> int:
    s = str(round_id)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return -1
    try:
        return int(digits)
    except ValueError:
        return -1


@dataclass(frozen=True)
class BOConfig:
    n_suggestions: int = 8
    exploration_ratio: float = 0.35
    anchor_fraction: float = 0.12
    replicate_fraction: float = 0.12
    anchor_count: Optional[int] = None
    replicate_count: Optional[int] = None
    # Fixed-composition slots (e.g. PMPC): re-proposed each batch. Not GOx (no polymer); GOx/PMPC are run every experiment by the user.
    anchor_polymer_ids: tuple[str, ...] = ("PMPC", "PMTAC")
    use_exact_anchor_compositions: bool = True
    replicate_source: str = "exploit"
    candidate_step: float = 0.02
    min_component: float = 0.02
    min_distance_between: float = 0.06
    min_distance_to_train: float = 0.03
    ei_xi: float = 0.01
    ucb_kappa: float = 2.0
    random_state: int = 42
    write_plots: bool = True
    # Anchor correction disabled by default: few-data correction is inappropriate for enzyme experiments;
    # round-to-round scale is handled experimentally (preparation).
    apply_round_anchor_correction: bool = False
    min_anchor_polymers: int = 2
    enable_heteroskedastic_noise: bool = True
    noise_rel_min: float = 0.35
    noise_rel_max: float = 3.0
    # Priority ranking for next-round practical execution (usually 1-3 polymers).
    # Emphasize predicted FoG/t50, keep a smaller EI component for exploration.
    priority_weight_fog: float = 0.45
    priority_weight_t50: float = 0.45
    priority_weight_ei: float = 0.10
    # If True, fit GP on (frac_MPC, frac_BMA, frac_MTAC) with 3 length scales (simplex); no xy_2x2 panels.
    # NOTE: Not recommended. Ternary has 2 degrees of freedom (sum=1), so 2D (x,y) is optimal.
    # 3D GP introduces redundancy and may lead to suboptimal learning. Use only for experimentation.
    use_simplex_gp: bool = False
    # If True, fit GP on (frac_BMA, frac_MTAC) with 2 length scales; distance and plots use BMA–MTAC plane.
    # Default False: use (x, y) with x = BMA/(BMA+MTAC), y = BMA+MTAC; 2x2 panels in xy plane.
    use_bma_mtac_coords: bool = False
    # For sparse design points (e.g., BO start with ~7 points), force isotropic kernel to avoid unstable ARD stripes.
    sparse_force_isotropic: bool = True
    sparse_isotropic_max_unique_points: int = 10
    # In sparse regime, fit a low-order polynomial trend and let GP model the residual.
    sparse_use_trend: bool = False
    sparse_trend_max_unique_points: int = 8
    trend_ridge: float = 1e-5
    # Dense surrogate-map plotting grid (separate from candidate grid used for BO selection).
    # Finer step → smoother gradient; observed points show as small "basins" (low std / low EI) by GP design.
    ternary_plot_step: float = 0.003  # Reduced from 0.005 for smoother gradient visualization (finer grid)
    # 2x2 xy/BMA-MTAC grid size (one axis). Larger → smoother gradient at basin boundaries.
    surrogate_map_xy_grid: int = 501  # Increased from 361 for smoother gradient visualization (finer grid)
    # When isotropic is applied and unique design points <= this, use min_length_scale_sparse_isotropic so gradient stays smooth.
    sparse_isotropic_apply_min_below_n: int = 18  # Apply smooth gradient for more runs (was 10).
    min_length_scale_sparse_isotropic: float = 0.5  # Larger → wider basins, smoother landscape (was 0.3).
    # Std map color: PowerNorm gamma (None = linear norm). Use linear when gradient looks crushed.
    # Reduced from 8.0 to 2.0 for better gradient visibility (not just yellow).
    std_color_gamma: Optional[float] = 2.0
    # Sparse-data exploration: add distance-to-training bonus to avoid narrow local probing.
    sparse_explore_max_unique_points: int = 15
    sparse_explore_distance_weight: float = 0.75
    sparse_combo_distance_weight: float = 0.40


@dataclass
class GPModel2D:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    X_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    length_scale: np.ndarray
    signal_std: float
    noise_std: float
    obs_noise_rel: np.ndarray
    chol_factor: Any
    alpha: np.ndarray
    jitter: float
    kernel_mode: str
    trend_basis: str
    trend_coef: Optional[np.ndarray]
    trend_ridge: float

    @staticmethod
    def _matern52_ard(
        xa: np.ndarray,
        xb: np.ndarray,
        length_scale: np.ndarray,
        signal_std: float,
    ) -> np.ndarray:
        xa = np.asarray(xa, dtype=float)
        xb = np.asarray(xb, dtype=float)
        ls = np.asarray(length_scale, dtype=float).reshape(1, 1, -1)
        d = (xa[:, None, :] - xb[None, :, :]) / np.maximum(ls, EPS)
        r = np.sqrt(np.sum(d * d, axis=2))
        sqrt5r = np.sqrt(5.0) * r
        base = (1.0 + sqrt5r + (5.0 / 3.0) * (r * r)) * np.exp(-sqrt5r)
        return (signal_std ** 2) * base

    @staticmethod
    def _poly2_basis(X: np.ndarray) -> np.ndarray:
        """
        Quadratic trend basis for 2D coordinates:
        [1, x1, x2, x1*x2, x1^2, x2^2].
        """
        X = np.asarray(X, dtype=float)
        x1 = X[:, 0]
        x2 = X[:, 1]
        return np.column_stack(
            [
                np.ones(X.shape[0], dtype=float),
                x1,
                x2,
                x1 * x2,
                x1 * x1,
                x2 * x2,
            ]
        )

    @classmethod
    def _fit_poly2_trend(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        noise_rel: np.ndarray,
        ridge: float,
    ) -> np.ndarray:
        Phi = cls._poly2_basis(X)
        w = 1.0 / np.maximum(np.asarray(noise_rel, dtype=float), EPS)
        ridge = max(float(ridge), 0.0)
        reg = np.eye(Phi.shape[1], dtype=float) * ridge
        reg[0, 0] = 0.0  # do not penalize intercept
        A = Phi.T @ (w[:, None] * Phi) + reg
        b = Phi.T @ (w * y)
        try:
            coef = np.linalg.solve(A, b)
        except LinAlgError:
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
        return coef.astype(float)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,
        obs_noise_rel: Optional[np.ndarray] = None,
        force_isotropic: bool = False,
        min_length_scale_isotropic: Optional[float] = None,
        use_trend: bool = False,
        trend_ridge: float = 1e-5,
    ) -> "GPModel2D":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must be (n,2), got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be (n,), got {y.shape}")
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 observations for GP fit.")

        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std = np.where(x_std <= EPS, 1.0, x_std)
        Xs = (X - x_mean) / x_std

        n = Xs.shape[0]
        rng = np.random.RandomState(random_state)
        if obs_noise_rel is None:
            noise_rel = np.ones(n, dtype=float)
        else:
            noise_rel = np.asarray(obs_noise_rel, dtype=float).reshape(-1)
            if noise_rel.shape[0] != n:
                raise ValueError(f"obs_noise_rel must have shape ({n},), got {noise_rel.shape}")
            bad = (~np.isfinite(noise_rel)) | (noise_rel <= 0)
            if bad.any():
                raise ValueError("obs_noise_rel must be finite and > 0 for all rows.")

        trend_basis = "none"
        trend_coef: Optional[np.ndarray] = None
        y_residual = y.copy()
        if use_trend:
            trend_coef = cls._fit_poly2_trend(X, y, noise_rel, ridge=trend_ridge)
            trend_pred = cls._poly2_basis(X) @ trend_coef
            y_residual = y - trend_pred
            trend_basis = "poly2"

        y_mean = float(np.mean(y_residual))
        y_std = float(np.std(y_residual))
        if y_std <= EPS:
            y_std = 1.0
        ys = (y_residual - y_mean) / y_std

        # Robust initial guess from pairwise distances in scaled space.
        dmat = np.sqrt(np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2))
        tri_u = np.triu_indices(n, k=1)
        med_dist = float(np.median(dmat[tri_u])) if tri_u[0].size else 1.0
        med_dist = max(med_dist, 0.3)

        # Per-axis length scale estimates: use median distance along each dimension.
        # This prevents one axis from becoming too large (causing striping artifacts).
        x_distances = np.abs(Xs[:, 0:1] - Xs[None, :, 0])
        y_distances = np.abs(Xs[:, 1:2] - Xs[None, :, 1])
        med_dist_x = float(np.median(x_distances[tri_u])) if tri_u[0].size else med_dist
        med_dist_y = float(np.median(y_distances[tri_u])) if tri_u[0].size else med_dist
        med_dist_x = max(med_dist_x, 0.1)
        med_dist_y = max(med_dist_y, 0.1)

        # Initial guess for signal_std: use observed y variation (scaled space).
        # For sparse data, ensure signal_std is large enough to capture variation.
        # Use robust estimate: median absolute deviation or std, whichever is larger.
        y_var_observed = float(np.var(ys)) if len(ys) > 1 else 0.5
        y_mad = float(np.median(np.abs(ys - np.median(ys)))) if len(ys) > 1 else 0.5
        # Use max of std and MAD to be robust to outliers, but ensure minimum
        signal_std_robust = max(float(np.sqrt(y_var_observed)), y_mad * 1.4826)  # MAD to std conversion
        init_signal_std = max(signal_std_robust, 0.5)  # At least 0.5 in scaled space for sparse data
        # Initial guess for noise_std: start smaller to avoid collapsing to all-noise.
        # For sparse data, allow higher noise but keep it reasonable relative to signal.
        init_noise_std = min(0.5, init_signal_std * 0.7)  # Start with noise < signal, but allow more for sparse data

        # Length scale upper bound: prevent extreme anisotropy that causes striping.
        # For 2D design space [0,1]^2, length_scale > 1.5 means almost flat in that direction.
        # Use stricter bound for sparse data to ensure smooth gradients.
        ls_max = 1.5  # Upper bound to prevent striping (stricter than 2.0)

        if force_isotropic:
            ls_min = 0.08
            if min_length_scale_isotropic is not None and min_length_scale_isotropic > ls_min:
                ls_min = float(min_length_scale_isotropic)
            init = np.log(np.array([max(med_dist, ls_min), init_signal_std, init_noise_std], dtype=float))
            # For sparse data, ensure signal_std lower bound is reasonable to prevent collapse to all-noise
            # Lower bound should be at least 0.1 * initial guess to allow signal to be captured
            signal_std_lower = max(1e-3, init_signal_std * 0.1)  # At least 10% of initial guess
            bounds = [
                (np.log(ls_min), np.log(ls_max)),   # isotropic length_scale (prevent striping)
                (np.log(signal_std_lower), np.log(10.0)),   # signal_std: adaptive lower bound for sparse data
                (np.log(1e-6), np.log(3.0)),   # noise_std: increased from 1.0 to 3.0 for sparse data
            ]
        else:
            init = np.log(np.array([med_dist_x, med_dist_y, init_signal_std, init_noise_std], dtype=float))
            # For sparse data, ensure signal_std lower bound is reasonable to prevent collapse to all-noise
            # Lower bound should be at least 0.1 * initial guess to allow signal to be captured
            signal_std_lower = max(1e-3, init_signal_std * 0.1)  # At least 10% of initial guess
            bounds = [
                (np.log(0.05), np.log(ls_max)),   # length_scale_x (prevent striping)
                (np.log(0.05), np.log(ls_max)),   # length_scale_y (prevent striping)
                (np.log(signal_std_lower), np.log(10.0)),  # signal_std: adaptive lower bound for sparse data
                (np.log(1e-6), np.log(3.0)),   # noise_std: increased from 1.0 to 3.0 for sparse data
            ]

        jitter = 1e-9

        def nll(theta_log: np.ndarray) -> float:
            if force_isotropic:
                ls_scalar = float(np.exp(theta_log[0]))
                ls = np.array([ls_scalar, ls_scalar], dtype=float)
                sig = float(np.exp(theta_log[1]))
                noise = float(np.exp(theta_log[2]))
            else:
                ls = np.exp(theta_log[:2])
                sig = float(np.exp(theta_log[2]))
                noise = float(np.exp(theta_log[3]))
            K = cls._matern52_ard(Xs, Xs, ls, sig)
            K[np.diag_indices_from(K)] += ((noise ** 2) * noise_rel + jitter)
            try:
                cho = cho_factor(K, lower=True, check_finite=False)
                alpha = cho_solve(cho, ys, check_finite=False)
            except LinAlgError:
                return 1e12
            ll = -0.5 * float(np.dot(ys, alpha))
            ll -= np.sum(np.log(np.diag(cho[0])))
            ll -= 0.5 * n * np.log(2.0 * np.pi)
            return -ll

        starts = [init]
        for _ in range(7):
            noise = rng.normal(loc=0.0, scale=0.6, size=len(init))
            starts.append(init + noise)

        best = None
        for s in starts:
            res = minimize(nll, s, method="L-BFGS-B", bounds=bounds)
            if (best is None) or (res.fun < best.fun):
                best = res
        if best is None:
            raise RuntimeError("Failed to optimize GP hyperparameters.")

        theta = best.x
        if force_isotropic:
            ls_scalar = float(np.exp(theta[0]))
            ls = np.array([ls_scalar, ls_scalar], dtype=float)
            sig = float(np.exp(theta[1]))
            noise = float(np.exp(theta[2]))
            kernel_mode = "isotropic"
        else:
            ls = np.exp(theta[:2])
            sig = float(np.exp(theta[2]))
            noise = float(np.exp(theta[3]))
            kernel_mode = "ard"
        K = cls._matern52_ard(Xs, Xs, ls, sig)
        K[np.diag_indices_from(K)] += ((noise ** 2) * noise_rel + jitter)
        cho = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve(cho, ys, check_finite=False)

        return cls(
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            X_train_scaled=Xs,
            y_train_scaled=ys,
            length_scale=ls,
            signal_std=sig,
            noise_std=noise,
            obs_noise_rel=noise_rel,
            chol_factor=cho,
            alpha=alpha,
            jitter=jitter,
            kernel_mode=kernel_mode,
            trend_basis=trend_basis,
            trend_coef=trend_coef,
            trend_ridge=float(trend_ridge),
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = (X - self.x_mean) / self.x_std
        K_cross = self._matern52_ard(
            self.X_train_scaled,
            Xs,
            self.length_scale,
            self.signal_std,
        )
        mu_scaled = K_cross.T @ self.alpha
        v = cho_solve(self.chol_factor, K_cross, check_finite=False)
        var_scaled = np.maximum((self.signal_std ** 2) - np.sum(K_cross * v, axis=0), 0.0)
        mu = mu_scaled * self.y_std + self.y_mean
        if self.trend_basis == "poly2" and self.trend_coef is not None:
            mu = mu + (self._poly2_basis(X) @ self.trend_coef)
        std = np.sqrt(var_scaled) * self.y_std
        return mu, std


@dataclass
class GPModelSimplex:
    """
    GP on 3D composition (frac_MPC, frac_BMA, frac_MTAC) with sum=1.
    Uses ARD Matern-5/2 with 3 length scales (one per component). No xy plane; use ternary plots only.
    """
    x_mean: np.ndarray  # shape (3,)
    x_std: np.ndarray   # shape (3,)
    y_mean: float
    y_std: float
    X_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    length_scale: np.ndarray  # shape (3,)
    signal_std: float
    noise_std: float
    obs_noise_rel: np.ndarray
    chol_factor: Any
    alpha: np.ndarray
    jitter: float

    @staticmethod
    def _matern52_ard(
        xa: np.ndarray,
        xb: np.ndarray,
        length_scale: np.ndarray,
        signal_std: float,
    ) -> np.ndarray:
        xa = np.asarray(xa, dtype=float)
        xb = np.asarray(xb, dtype=float)
        ls = np.asarray(length_scale, dtype=float).reshape(1, 1, -1)
        d = (xa[:, None, :] - xb[None, :, :]) / np.maximum(ls, EPS)
        r = np.sqrt(np.sum(d * d, axis=2))
        sqrt5r = np.sqrt(5.0) * r
        base = (1.0 + sqrt5r + (5.0 / 3.0) * (r * r)) * np.exp(-sqrt5r)
        return (signal_std ** 2) * base

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        random_state: int = 42,
        obs_noise_rel: Optional[np.ndarray] = None,
    ) -> "GPModelSimplex":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be (n, 3) [frac_MPC, frac_BMA, frac_MTAC], got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be (n,), got {y.shape}")
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 observations for GP fit.")
        s = X.sum(axis=1)
        if not np.all(np.abs(s - 1.0) < 0.01):
            raise ValueError("Rows of X must sum to 1 (simplex).")

        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std = np.where(x_std <= EPS, 1.0, x_std)
        Xs = (X - x_mean) / x_std

        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        if y_std <= EPS:
            y_std = 1.0
        ys = (y - y_mean) / y_std

        n = Xs.shape[0]
        rng = np.random.RandomState(random_state)
        if obs_noise_rel is None:
            noise_rel = np.ones(n, dtype=float)
        else:
            noise_rel = np.asarray(obs_noise_rel, dtype=float).reshape(-1)
            if noise_rel.shape[0] != n:
                raise ValueError(f"obs_noise_rel must have shape ({n},), got {noise_rel.shape}")
            bad = (~np.isfinite(noise_rel)) | (noise_rel <= 0)
            if bad.any():
                raise ValueError("obs_noise_rel must be finite and > 0 for all rows.")

        dmat = np.sqrt(np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2))
        tri_u = np.triu_indices(n, k=1)
        med_dist = float(np.median(dmat[tri_u])) if tri_u[0].size else 1.0
        med_dist = max(med_dist, 0.3)

        init = np.log(np.array([med_dist, med_dist, med_dist, 1.0, 0.10], dtype=float))
        bounds = [
            (np.log(0.05), np.log(8.0)),
            (np.log(0.05), np.log(8.0)),
            (np.log(0.05), np.log(8.0)),
            (np.log(1e-3), np.log(10.0)),
            (np.log(1e-6), np.log(1.0)),
        ]

        jitter = 1e-9

        def nll(theta_log: np.ndarray) -> float:
            ls = np.exp(theta_log[:3])
            sig = float(np.exp(theta_log[3]))
            noise = float(np.exp(theta_log[4]))
            K = cls._matern52_ard(Xs, Xs, ls, sig)
            K[np.diag_indices_from(K)] += ((noise ** 2) * noise_rel + jitter)
            try:
                cho = cho_factor(K, lower=True, check_finite=False)
                alpha = cho_solve(cho, ys, check_finite=False)
            except LinAlgError:
                return 1e12
            ll = -0.5 * float(np.dot(ys, alpha))
            ll -= np.sum(np.log(np.diag(cho[0])))
            ll -= 0.5 * n * np.log(2.0 * np.pi)
            return -ll

        starts = [init]
        for _ in range(7):
            noise = rng.normal(loc=0.0, scale=0.6, size=5)
            starts.append(init + noise)

        best = None
        for s in starts:
            res = minimize(nll, s, method="L-BFGS-B", bounds=bounds)
            if (best is None) or (res.fun < best.fun):
                best = res
        if best is None:
            raise RuntimeError("Failed to optimize GP hyperparameters.")

        theta = best.x
        ls = np.exp(theta[:3])
        sig = float(np.exp(theta[3]))
        noise = float(np.exp(theta[4]))
        K = cls._matern52_ard(Xs, Xs, ls, sig)
        K[np.diag_indices_from(K)] += ((noise ** 2) * noise_rel + jitter)
        cho = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve(cho, ys, check_finite=False)

        return cls(
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            X_train_scaled=Xs,
            y_train_scaled=ys,
            length_scale=ls,
            signal_std=sig,
            noise_std=noise,
            obs_noise_rel=noise_rel,
            chol_factor=cho,
            alpha=alpha,
            jitter=jitter,
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = (X - self.x_mean) / self.x_std
        K_cross = self._matern52_ard(
            self.X_train_scaled,
            Xs,
            self.length_scale,
            self.signal_std,
        )
        mu_scaled = K_cross.T @ self.alpha
        v = cho_solve(self.chol_factor, K_cross, check_finite=False)
        var_scaled = np.maximum((self.signal_std ** 2) - np.sum(K_cross * v, axis=0), 0.0)
        mu = mu_scaled * self.y_std + self.y_mean
        std = np.sqrt(var_scaled) * self.y_std
        return mu, std


def _ei(mu: np.ndarray, std: np.ndarray, best: float, xi: float) -> np.ndarray:
    std_safe = np.maximum(std, EPS)
    improvement = mu - best - xi
    z = improvement / std_safe
    ei = improvement * norm.cdf(z) + std_safe * norm.pdf(z)
    ei[std <= EPS] = 0.0
    return ei


def _ucb(mu: np.ndarray, std: np.ndarray, kappa: float) -> np.ndarray:
    return mu + kappa * std


def _resolve_ucb_kappa(ucb_kappa: float, ucb_beta: Optional[float]) -> float:
    """
    Resolve UCB exploration scale from either kappa or beta.
    UCB is computed as mu + kappa * std; if beta is given, kappa = sqrt(beta).
    """
    if ucb_beta is not None:
        beta = float(ucb_beta)
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError(f"ucb_beta must be a finite non-negative value, got {ucb_beta!r}")
        return float(np.sqrt(beta))
    kappa = float(ucb_kappa)
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError(f"ucb_kappa must be a finite non-negative value, got {ucb_kappa!r}")
    return kappa


def _format_colorbar_plain(cbar: Any) -> None:
    """Disable offset notation on colorbars to avoid ambiguous labels like '1e-6+...'. """
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_useOffset(False)
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.update_ticks()


# ---------------------------------------------------------------------------
# Pure regression BO (2D x,y): no classifier, no anchor/replicate slots.
# Design: x = BMA/(BMA+MTAC), y = BMA+MTAC; inverse: MPC=1-y, BMA=x*y, MTAC=(1-x)*y.
# ---------------------------------------------------------------------------

_DEFAULT_COMPOSITION_CONSTRAINTS = {
    "min_mpc": 0.05,
    "max_mpc": 0.95,
    "min_bma": 0.05,
    "max_bma": 0.95,
    "min_mtac": 0.05,
    "max_mtac": 0.95,
}
_DEFAULT_DIVERSITY_PARAMS = {"min_fraction_distance": 0.05}
DEFAULT_OBJECTIVE_COLUMN = "log_fog_native_constrained"


def _resolve_objective_column_name(learning_df: pd.DataFrame, objective_column: str) -> str:
    """Resolve objective column with fail-fast behavior (no implicit fallback)."""
    requested = str(objective_column).strip()
    if requested in learning_df.columns:
        return requested
    raise ValueError(
        "learning_df must have objective column "
        f"{requested!r}. Available columns: {list(learning_df.columns)}"
    )


def _ensure_xy_and_objective(learning_df: pd.DataFrame, objective_column: str) -> pd.DataFrame:
    """Ensure learning_df has x, y and one objective column. Mutates copy and returns it."""
    out = learning_df.copy()
    if "x" not in out.columns or "y" not in out.columns:
        if "frac_BMA" in out.columns and "frac_MTAC" in out.columns and "frac_MPC" in out.columns:
            out = _xy_from_frac(out)
        else:
            raise ValueError("learning_df must have (x, y) or (frac_MPC, frac_BMA, frac_MTAC)")
    _resolve_objective_column_name(out, objective_column)
    return out


def _aggregate_pure_bo_training_rows(df: pd.DataFrame, objective_column: str) -> pd.DataFrame:
    """
    Aggregate duplicated compositions for pure BO training.

    BO frequently receives repeated measurements for the same composition across rounds.
    Aggregating these duplicated coordinates stabilizes GP hyperparameter fitting and
    prevents the model from inflating global noise to explain replicate spread.
    """
    work = df.copy()
    if not {"frac_MPC", "frac_BMA", "frac_MTAC"}.issubset(work.columns):
        mpc, bma, mtac = _frac_from_xy_bo_data(
            work["x"].to_numpy(dtype=float),
            work["y"].to_numpy(dtype=float),
        )
        work["frac_MPC"] = mpc
        work["frac_BMA"] = bma
        work["frac_MTAC"] = mtac
    key_cols = ["frac_MPC", "frac_BMA", "frac_MTAC"]
    for c in key_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[objective_column] = pd.to_numeric(work[objective_column], errors="coerce")
    work = work[np.isfinite(work[key_cols]).all(axis=1) & np.isfinite(work[objective_column])].copy()
    for c in key_cols:
        work[f"__key_{c}"] = work[c].round(9)
    agg_spec: Dict[str, Any] = {
        "x": ("x", "mean"),
        "y": ("y", "mean"),
        "frac_MPC": ("frac_MPC", "mean"),
        "frac_BMA": ("frac_BMA", "mean"),
        "frac_MTAC": ("frac_MTAC", "mean"),
        "objective_mean": (objective_column, "mean"),
        "n_replicates": (objective_column, "size"),
    }
    if "n_observations" in work.columns:
        work["n_observations"] = pd.to_numeric(work["n_observations"], errors="coerce")
        agg_spec["n_observations_raw"] = ("n_observations", "sum")
    if "log_fog_mad" in work.columns:
        work["log_fog_mad"] = pd.to_numeric(work["log_fog_mad"], errors="coerce")
        agg_spec["log_fog_mad_raw"] = ("log_fog_mad", "median")

    grouped = (
        work.groupby(["__key_frac_MPC", "__key_frac_BMA", "__key_frac_MTAC"], as_index=False)
        .agg(**agg_spec)
        .rename(columns={"objective_mean": objective_column})
    )

    # Keep per-row uncertainty proxies for heteroskedastic GP fit.
    if "n_observations_raw" in grouped.columns:
        n_obs = pd.to_numeric(grouped["n_observations_raw"], errors="coerce").to_numpy(dtype=float)
        n_obs = np.where(np.isfinite(n_obs) & (n_obs > 0.0), n_obs, np.nan)
        grouped["n_observations"] = n_obs
        grouped = grouped.drop(columns=["n_observations_raw"])
    else:
        grouped["n_observations"] = grouped["n_replicates"].astype(float)
    if "log_fog_mad_raw" in grouped.columns:
        mad = pd.to_numeric(grouped["log_fog_mad_raw"], errors="coerce").to_numpy(dtype=float)
        mad = np.where(np.isfinite(mad) & (mad > 0.0), mad, np.nan)
        grouped["log_fog_mad"] = mad
        grouped = grouped.drop(columns=["log_fog_mad_raw"])
    grouped = grouped.drop(columns=["__key_frac_MPC", "__key_frac_BMA", "__key_frac_MTAC"])
    return grouped.reset_index(drop=True)


def _sample_valid_xy(
    n: int,
    composition_constraints: Dict[str, float],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample (x,y) in [0,1]^2 and filter by composition constraints. Returns (N,2)."""
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = np.random.default_rng(seed)
    min_mpc = float(composition_constraints.get("min_mpc", _DEFAULT_COMPOSITION_CONSTRAINTS["min_mpc"]))
    max_mpc = float(composition_constraints.get("max_mpc", _DEFAULT_COMPOSITION_CONSTRAINTS["max_mpc"]))
    min_bma = float(composition_constraints.get("min_bma", _DEFAULT_COMPOSITION_CONSTRAINTS["min_bma"]))
    max_bma = float(composition_constraints.get("max_bma", _DEFAULT_COMPOSITION_CONSTRAINTS["max_bma"]))
    min_mtac = float(composition_constraints.get("min_mtac", _DEFAULT_COMPOSITION_CONSTRAINTS["min_mtac"]))
    max_mtac = float(composition_constraints.get("max_mtac", _DEFAULT_COMPOSITION_CONSTRAINTS["max_mtac"]))

    if not (min_mpc <= max_mpc and min_bma <= max_bma and min_mtac <= max_mtac):
        raise ValueError(
            "Each composition bound must satisfy min <= max. "
            f"Got mpc=({min_mpc},{max_mpc}), bma=({min_bma},{max_bma}), mtac=({min_mtac},{max_mtac})"
        )
    sum_min = min_mpc + min_bma + min_mtac
    sum_max = max_mpc + max_bma + max_mtac
    if (sum_min > 1.0 + 1e-9) or (sum_max < 1.0 - 1e-9):
        raise ValueError(
            "Infeasible composition constraints: simplex cannot satisfy all bounds. "
            f"sum(min)={sum_min:.6g}, sum(max)={sum_max:.6g}"
        )

    # Oversample and filter (rejection sampling with explicit stop condition).
    buf: List[np.ndarray] = []
    batch = max(4 * int(n), 1024)
    max_batches = 500
    accepted = 0
    for _ in range(max_batches):
        if accepted >= n:
            break
        xy = rng.uniform(0.0, 1.0, size=(batch, 2))
        x, y = xy[:, 0], xy[:, 1]
        mpc, bma, mtac = _frac_from_xy_bo_data(x, y)
        ok = (
            (mpc >= min_mpc - 1e-9) & (mpc <= max_mpc + 1e-9)
            & (bma >= min_bma - 1e-9) & (bma <= max_bma + 1e-9)
            & (mtac >= min_mtac - 1e-9) & (mtac <= max_mtac + 1e-9)
        )
        good = xy[ok]
        if len(good) > 0:
            buf.append(good)
            accepted += len(good)
    if accepted < n:
        raise ValueError(
            "Could not sample enough feasible candidates from composition constraints. "
            f"requested={n}, accepted={accepted}, max_batches={max_batches}"
        )
    xy = np.vstack(buf)
    return xy[:n]


def _distance_fraction_space(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
) -> np.ndarray:
    """L2 distance in (MPC, BMA, MTAC) space. frac_a (N,3), frac_b (M,3) -> (N,M)."""
    diff = frac_a[:, None, :] - frac_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def propose_batch_next_points(
    learning_df: pd.DataFrame,
    q: int,
    acquisition: str,
    diversity_params: Optional[Dict[str, Any]] = None,
    composition_constraints: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    *,
    objective_column: str = DEFAULT_OBJECTIVE_COLUMN,
    ei_xi: float = 0.01,
    ucb_kappa: float = 2.0,
    ucb_beta: Optional[float] = None,
    n_random_candidates: int = 5000,
    sparse_force_isotropic: bool = True,
    sparse_isotropic_max_unique_points: int = 20,  # Increased from 10 to force isotropic for sparse data
    min_length_scale_sparse_isotropic: float = 0.5,
    sparse_use_trend: bool = False,
    sparse_trend_max_unique_points: int = 8,
    trend_ridge: float = 1e-5,
    enable_heteroskedastic_noise: bool = True,
    noise_rel_min: float = 0.35,
    noise_rel_max: float = 3.0,
    enable_diversity: bool = True,
) -> tuple[pd.DataFrame, GPModel2D]:
    """
    Pure regression BO: propose q next points in (x,y) with EI or UCB and optional diversity.

    Design variables: x = BMA/(BMA+MTAC), y = BMA+MTAC. Inverse: MPC=1-y, BMA=x*y, MTAC=(1-x)*y.
    GP is fitted on (x,y) with z-score standardization; objective is maximized.

    Parameters
    ----------
    learning_df : DataFrame
        Must have (x, y) or (frac_MPC, frac_BMA, frac_MTAC) and objective_column.
    q : int
        Number of batch proposals.
    acquisition : str
        "ei" or "ucb".
    diversity_params : dict, optional
        e.g. {"min_fraction_distance": 0.05}. Distance in (MPC,BMA,MTAC) space; points closer than this to known/selected are penalized.
    composition_constraints : dict, optional
        e.g. {"min_mpc": 0.05, "max_mpc": 0.95, "min_bma": 0.05, ...}. Hard bounds on fractions.
    seed : int, optional
        Random seed for GP fit and random candidate sampling.
    objective_column : str
        Column name for objective (maximized).
    ei_xi : float
        Exploration parameter for EI.
    ucb_kappa : float
        Exploration parameter for UCB (ignored when ucb_beta is provided).
    ucb_beta : float, optional
        Optional UCB beta where kappa = sqrt(beta).
    n_random_candidates : int
        Number of random (x,y) points to evaluate acquisition on.
    sparse_force_isotropic : bool
        If True, force isotropic kernel for sparse design sets to reduce unstable ARD striping.
    sparse_isotropic_max_unique_points : int
        Apply sparse isotropic fallback when unique design points are <= this value.
    sparse_use_trend : bool
        If True, use low-order polynomial trend + GP residual in sparse regime.
    sparse_trend_max_unique_points : int
        Apply sparse trend fallback when unique design points are <= this value.
    trend_ridge : float
        Ridge regularization used when fitting sparse polynomial trend.
    enable_heteroskedastic_noise : bool
        If True, estimate per-observation relative noise from n_observations/log_fog_mad.
    noise_rel_min : float
        Lower clip bound for relative observation noise.
    noise_rel_max : float
        Upper clip bound for relative observation noise.
    enable_diversity : bool
        If True, apply diversity penalty (avoid points too close in fraction space).

    Returns
    -------
    candidates_df : DataFrame
        Columns: x, y, pred_mean, pred_std, acq_value, frac_MPC, frac_BMA, frac_MTAC, selection_order.
    gp : GPModel2D
        Fitted model for plotting.
    """
    q = int(q)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    n_random_candidates = int(n_random_candidates)
    if n_random_candidates <= 0:
        raise ValueError(f"n_random_candidates must be positive, got {n_random_candidates}")

    div_params = dict(_DEFAULT_DIVERSITY_PARAMS)
    if diversity_params:
        div_params.update(diversity_params)
    comp_const = dict(_DEFAULT_COMPOSITION_CONSTRAINTS)
    if composition_constraints:
        comp_const.update(composition_constraints)

    rng = np.random.default_rng(seed)
    rs = int(rng.integers(0, 2**31)) if seed is None else seed

    df = _ensure_xy_and_objective(learning_df, objective_column)
    model_df = _aggregate_pure_bo_training_rows(df, objective_column)
    X_train = model_df[["x", "y"]].to_numpy(dtype=float)
    y_train = model_df[objective_column].to_numpy(dtype=float)
    if X_train.shape[0] < 3:
        raise ValueError("Need at least 3 observations to fit GP.")

    n_unique_design_points = int(np.unique(np.round(X_train, decimals=12), axis=0).shape[0])
    # For sparse data (<= 20 points), force isotropic kernel to prevent striping artifacts.
    # ARD can cause extreme anisotropy (e.g., length_scale_x >> length_scale_y) when data is sparse.
    force_isotropic = bool(sparse_force_isotropic) and (
        n_unique_design_points <= int(max(sparse_isotropic_max_unique_points, 20))
    )
    # For very sparse data (<= sparse_trend_max_unique_points), enable trend by default to help GP capture signal
    # This improves GP's ability to learn from limited data
    # Note: max() ensures that if sparse_trend_max_unique_points is set to a value > 8, we use that threshold
    trend_threshold = int(max(sparse_trend_max_unique_points, 8))
    use_trend = bool(sparse_use_trend) or (n_unique_design_points <= trend_threshold)
    min_ls_sparse = None
    if force_isotropic and n_unique_design_points <= 18:
        min_ls_sparse = float(min_length_scale_sparse_isotropic)
    if bool(enable_heteroskedastic_noise):
        obs_noise_rel = _estimate_obs_noise_rel(
            model_df,
            min_rel=float(noise_rel_min),
            max_rel=float(noise_rel_max),
        )
    else:
        obs_noise_rel = np.ones(len(model_df), dtype=float)
    gp = GPModel2D.fit(
        X_train,
        y_train,
        random_state=rs,
        obs_noise_rel=obs_noise_rel,
        force_isotropic=force_isotropic,
        min_length_scale_isotropic=min_ls_sparse,
        use_trend=use_trend,
        trend_ridge=float(trend_ridge),
    )
    best = float(np.nanmax(y_train))
    ucb_kappa_used = _resolve_ucb_kappa(ucb_kappa, ucb_beta)
    
    # Auto-adjust EI xi based on data variance for sparse data
    # For sparse data, use larger xi to allow exploration even when GP is uncertain
    y_std_observed = float(np.std(y_train)) if len(y_train) > 1 else 1.0
    y_range_observed = float(np.nanmax(y_train) - np.nanmin(y_train)) if len(y_train) > 1 else 1.0
    # Use adaptive xi: scale with observed variance, but ensure minimum for sparse data
    if n_unique_design_points <= 15:
        # For sparse data, use xi proportional to observed std, but at least 0.01 * range
        ei_xi_adaptive = max(ei_xi, 0.01 * y_range_observed, 0.005 * y_std_observed)
    else:
        ei_xi_adaptive = ei_xi

    xy_cand = _sample_valid_xy(n_random_candidates, comp_const, seed=rs)
    mu, std = gp.predict(xy_cand)
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    acq_mode_used = acquisition.lower()
    if acquisition.lower() == "ei":
        # Use adaptive xi for sparse data to improve exploration
        acq = _ei(mu, std, best, ei_xi_adaptive)
        if float(np.nanmax(acq)) < 1e-12:
            # Sparse/underfit regimes can collapse EI to ~0 everywhere. Keep behavior auditable
            # and use UCB ranking as a practical fallback for candidate selection only.
            acq = _ucb(mu, std, ucb_kappa_used)
            acq_mode_used = "ei_ucb_fallback"
    elif acquisition.lower() == "ucb":
        acq = _ucb(mu, std, ucb_kappa_used)
    else:
        raise ValueError(f"acquisition must be 'ei' or 'ucb', got {acquisition!r}")

    mpc_c, bma_c, mtac_c = _frac_from_xy_bo_data(xy_cand[:, 0], xy_cand[:, 1])
    frac_cand = np.column_stack([mpc_c, bma_c, mtac_c])
    mpc_t, bma_t, mtac_t = _frac_from_xy_bo_data(X_train[:, 0], X_train[:, 1])
    frac_train = np.column_stack([mpc_t, bma_t, mtac_t])
    min_dist = _distance_fraction_space(frac_cand, frac_train)
    dist_to_known = np.min(min_dist, axis=1)
    threshold = float(div_params.get("min_fraction_distance", 0.05))

    selected_indices: List[int] = []
    selected_set: set[int] = set()
    selected_fracs: List[np.ndarray] = []
    selected_relaxed_flags: List[bool] = []
    acq_values_at_selection: List[float] = []
    acq_raw = acq.copy()
    acq_rank = acq.copy()

    for _ in range(q):
        acq_penalized = acq_rank.copy()
        if enable_diversity:
            for j in range(len(selected_fracs)):
                d = _distance_fraction_space(frac_cand, selected_fracs[j].reshape(1, 3))
                acq_penalized[d.ravel() < threshold] = -np.inf
            acq_penalized[dist_to_known < threshold] = -np.inf
        i = int(np.argmax(acq_penalized))
        if not np.isfinite(acq_penalized[i]) or acq_penalized[i] <= -1e30:
            break
        selected_indices.append(i)
        selected_set.add(i)
        selected_fracs.append(frac_cand[i].copy())
        selected_relaxed_flags.append(False)
        acq_values_at_selection.append(float(acq_raw[i]))
        acq_rank[i] = -np.inf

    # Ensure we propose q points whenever feasible by relaxing only the diversity exclusion.
    if len(selected_indices) < q:
        for i in np.argsort(acq_raw)[::-1]:
            if len(selected_indices) >= q:
                break
            idx = int(i)
            if idx in selected_set:
                continue
            if not np.isfinite(acq_raw[idx]):
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            selected_fracs.append(frac_cand[idx].copy())
            selected_relaxed_flags.append(True)
            acq_values_at_selection.append(float(acq_raw[idx]))

    rows = []
    for order, (idx, acq_val, relaxed) in enumerate(
        zip(selected_indices, acq_values_at_selection, selected_relaxed_flags), start=1
    ):
        x, y = float(xy_cand[idx, 0]), float(xy_cand[idx, 1])
        mpc, bma, mtac = float(frac_cand[idx, 0]), float(frac_cand[idx, 1]), float(frac_cand[idx, 2])
        rows.append({
            "x": x,
            "y": y,
            "pred_mean": float(mu[idx]),
            "pred_std": float(std[idx]),
            "acq_value": acq_val,
            "frac_MPC": mpc,
            "frac_BMA": bma,
            "frac_MTAC": mtac,
            "min_dist_to_known": float(dist_to_known[idx]),
            "acquisition": acquisition.lower(),
            "acq_mode_used": acq_mode_used,
            "ucb_kappa_used": float(ucb_kappa_used),
            "ei_xi_used": float(ei_xi_adaptive) if acquisition.lower() == "ei" else None,
            "diversity_threshold": float(threshold),
            "diversity_relaxed": bool(relaxed),
            "n_unique_design_points": int(n_unique_design_points),
            "force_isotropic_applied": bool(force_isotropic),
            "use_trend_applied": bool(use_trend),
            "selection_order": order,
        })
    candidates_df = pd.DataFrame(rows)
    if len(candidates_df) < q:
        raise RuntimeError(
            f"Could only propose {len(candidates_df)} points (requested q={q}). "
            "Loosen composition/diversity constraints or increase random candidates."
        )
    return candidates_df, gp


def _pure_bo_grid_predict(
    gp: GPModel2D,
    n_grid: int = 200,
    *,
    ei_xi: float = 0.01,
    ucb_kappa: float = 2.0,
    ucb_beta: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X1, X2, Z_mu, Z_std, Z_ei, Z_ucb for heatmaps in [0,1]^2."""
    xg = np.linspace(0.0, 1.0, n_grid)
    yg = np.linspace(0.0, 1.0, n_grid)
    X1, X2 = np.meshgrid(xg, yg)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    mu, std = gp.predict(grid)
    mu = np.asarray(mu).reshape(X1.shape)
    std = np.asarray(std).reshape(X1.shape)
    best = float(np.nanmax(gp.y_train_scaled * gp.y_std + gp.y_mean))
    ucb_kappa_used = _resolve_ucb_kappa(ucb_kappa, ucb_beta)
    ei = _ei(mu.ravel(), std.ravel(), best, ei_xi).reshape(X1.shape)
    ucb = _ucb(mu.ravel(), std.ravel(), ucb_kappa_used).reshape(X1.shape)
    return X1, X2, mu, std, ei, ucb


def _pure_bo_map_quality(
    gp: GPModel2D,
    *,
    ei_xi: float,
    ucb_kappa: float,
    ucb_beta: Optional[float],
    n_grid: int = 200,
) -> Dict[str, Any]:
    """Lightweight map diagnostics for pure-regression BO outputs."""
    _, _, z_mu, z_std, z_ei, z_ucb = _pure_bo_grid_predict(
        gp,
        n_grid=n_grid,
        ei_xi=ei_xi,
        ucb_kappa=ucb_kappa,
        ucb_beta=ucb_beta,
    )
    mu_min = float(np.nanmin(z_mu))
    mu_max = float(np.nanmax(z_mu))
    std_min = float(np.nanmin(z_std))
    std_max = float(np.nanmax(z_std))
    ei_min = float(np.nanmin(z_ei))
    ei_max = float(np.nanmax(z_ei))
    ucb_min = float(np.nanmin(z_ucb))
    ucb_max = float(np.nanmax(z_ucb))
    avg_span_fixed_y_vary_x = float(np.nanmean(np.nanmax(z_mu, axis=1) - np.nanmin(z_mu, axis=1)))
    avg_span_fixed_x_vary_y = float(np.nanmean(np.nanmax(z_mu, axis=0) - np.nanmin(z_mu, axis=0)))
    mu_range = float(mu_max - mu_min)
    ei_range = float(ei_max - ei_min)
    # Flag flat maps (extremely small ranges) for diagnostic purposes
    mu_flat_threshold = 1e-5  # Threshold for considering mu map as "flat"
    ei_flat_threshold = 1e-10  # Threshold for considering EI map as "flat"
    mu_is_flat = mu_range < mu_flat_threshold
    ei_is_flat = ei_range < ei_flat_threshold
    
    return {
        "mu_min": mu_min,
        "mu_max": mu_max,
        "mu_range": mu_range,
        "mu_is_flat": mu_is_flat,  # Flag for flat mu map
        "avg_span_fixed_y_vary_x": avg_span_fixed_y_vary_x,
        "avg_span_fixed_x_vary_y": avg_span_fixed_x_vary_y,
        "anisotropy_ratio_y_over_x": float(avg_span_fixed_x_vary_y / (avg_span_fixed_y_vary_x + EPS)),
        "std_min": std_min,
        "std_max": std_max,
        "std_range": float(std_max - std_min),
        "ei_min": ei_min,
        "ei_max": ei_max,
        "ei_range": ei_range,
        "ei_collapsed": bool(ei_range <= 1e-12),
        "ei_is_flat": ei_is_flat,  # Flag for flat EI map
        "ucb_min": ucb_min,
        "ucb_max": ucb_max,
        "ucb_range": float(ucb_max - ucb_min),
        "grid_size": int(n_grid),
    }


def plot_pure_bo_observed_scatter(
    learning_df: pd.DataFrame,
    objective_column: str,
    out_path: Path,
    *,
    run_id: str = "",
) -> None:
    """1) Observed points in (x,y); color = objective, marker shape = Round."""
    df = _ensure_xy_and_objective(learning_df, objective_column)
    obj_vals = df[objective_column].dropna()
    vmin = float(obj_vals.min()) if len(obj_vals) else 0.0
    vmax = float(obj_vals.max()) if len(obj_vals) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
        rounds = sorted(df["round_id"].astype(str).unique()) if "round_id" in df.columns else ["1"]
        marker_cycle = cycle(["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"])
        mappable = None
        for r in rounds:
            m = next(marker_cycle)
            sub = df[df["round_id"].astype(str) == r] if "round_id" in df.columns else df
            if sub.empty:
                continue
            sc = ax.scatter(
                sub["x"],
                sub["y"],
                c=sub[objective_column],
                s=28,
                marker=m,
                edgecolors="black",
                linewidths=0.5,
                label=f"Round {r}",
                cmap="viridis",
                norm=norm,
            )
            if mappable is None:
                mappable = sc
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r"$x = \mathrm{BMA}/(\mathrm{BMA}+\mathrm{MTAC})$")
        ax.set_ylabel(r"$y = \mathrm{BMA}+\mathrm{MTAC}$")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=6, ncol=2)
        if run_id:
            ax.set_title("Bayesian Optimization: observed points (color = objective)")
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03, label=objective_column)
            _format_colorbar_plain(cbar)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_pure_bo_mu_heatmap(
    gp: GPModel2D,
    out_path: Path,
    *,
    run_id: str = "",
    n_grid: int = 200,
) -> None:
    """2) mu(x,y) heatmap."""
    X1, X2, Z_mu, _, _, _ = _pure_bo_grid_predict(gp, n_grid=n_grid)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
        # Calculate vmin/vmax dynamically from data
        valid_z = Z_mu[np.isfinite(Z_mu)]
        if len(valid_z) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(valid_z))
            vmax = float(np.nanmax(valid_z))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cs = ax.pcolormesh(X1, X2, Z_mu, cmap=_XY_PANEL_CMAPS["mean"], norm=norm, shading="gouraud")
        # Add contour lines with high transparency for better gradient visibility
        # For sparse data, also check relative range to allow contours even when absolute range is small.
        _range = vmax - vmin
        _range_rel = _range / (abs(vmax) + abs(vmin) + EPS) if (vmax != 0 or vmin != 0) else 0.0
        if _range > _SURROGATE_CONTOUR_MIN_RANGE or _range_rel > 1e-6:
            valid_z = Z_mu[np.isfinite(Z_mu)]
            if len(valid_z) > 0:
                # Use moderate number of contour levels (not too dense) so gradient colors are visible
                n_levels = _SURROGATE_CONTOUR_LEVELS  # Use base level count (18), not doubled
                q_levels = np.linspace(0.05, 0.95, n_levels)
                _levs = np.quantile(valid_z, q_levels)
            else:
                n_levels = _SURROGATE_CONTOUR_LEVELS
                _levs = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
            try:
                qc = ax.contour(
                    X1, X2, Z_mu,
                    levels=_levs,
                    colors=_SURROGATE_CONTOUR_COLOR,
                    linewidths=_SURROGATE_CONTOUR_LW,
                    alpha=0.95,  # High transparency
                    zorder=2,
                )
                qc.set_path_effects([
                    mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                    mpatheffects.Normal(),
                ])
            except (ValueError, TypeError):
                pass
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x = \mathrm{BMA}/(\mathrm{BMA}+\mathrm{MTAC})$")
        ax.set_ylabel(r"$y = \mathrm{BMA}+\mathrm{MTAC}$")
        ax.set_aspect("equal")
        cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03, label="Predicted mean (objective)")
        _format_colorbar_plain(cbar)
        if run_id:
            ax.set_title("Bayesian Optimization: predicted mean $\\mu(x,y)$")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_pure_bo_sigma_heatmap(
    gp: GPModel2D,
    out_path: Path,
    *,
    run_id: str = "",
    n_grid: int = 200,
) -> None:
    """3) sigma(x,y) heatmap."""
    X1, X2, _, Z_std, _, _ = _pure_bo_grid_predict(gp, n_grid=n_grid)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
        # Use gentler gamma for std to show gradient better (not just yellow)
        # For std: use percentile-based vmin to avoid "holes" at observed points
        valid_z = Z_std[np.isfinite(Z_std)]
        if len(valid_z) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            # Use 10th percentile as vmin to create smoother gradient visualization (reduced "holes" at observed points)
            vmin = float(np.percentile(valid_z, 10))
            vmax = float(np.nanmax(valid_z))
            if vmin >= vmax:
                vmin = float(np.nanmin(valid_z))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        norm = mcolors.PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)  # Reduced gamma for better gradient
        cs = ax.pcolormesh(X1, X2, Z_std, cmap=_XY_PANEL_CMAPS["std"], norm=norm, shading="gouraud")
        # Add contour lines with high transparency for better gradient visibility
        # For sparse data, also check relative range to allow contours even when absolute range is small.
        _range = vmax - vmin
        _range_rel = _range / (abs(vmax) + abs(vmin) + EPS) if (vmax != 0 or vmin != 0) else 0.0
        if _range > _SURROGATE_CONTOUR_MIN_RANGE or _range_rel > 1e-6:
            valid_z = Z_std[np.isfinite(Z_std)]
            if len(valid_z) > 0:
                # More contour levels for std to show gradient better
                # Use moderate number of contour levels (not too dense) so gradient colors are visible
                n_levels = _SURROGATE_CONTOUR_LEVELS  # Use base level count (18), not doubled
                q_levels = np.linspace(0.05, 0.95, n_levels)
                _levs = np.quantile(valid_z, q_levels)
            else:
                n_levels = _SURROGATE_CONTOUR_LEVELS
                _levs = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
            try:
                # Higher transparency for std contours to show gradient better
                qc = ax.contour(
                    X1, X2, Z_std,
                    levels=_levs,
                    colors=_SURROGATE_CONTOUR_COLOR,
                    linewidths=_SURROGATE_CONTOUR_LW,
                    alpha=0.95,  # High transparency
                    zorder=2,
                )
                qc.set_path_effects([
                    mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                    mpatheffects.Normal(),
                ])
            except (ValueError, TypeError):
                pass
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x = \mathrm{BMA}/(\mathrm{BMA}+\mathrm{MTAC})$")
        ax.set_ylabel(r"$y = \mathrm{BMA}+\mathrm{MTAC}$")
        ax.set_aspect("equal")
        cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03, label="Predictive std $\\sigma(x,y)$")
        _format_colorbar_plain(cbar)
        if run_id:
            ax.set_title("Bayesian Optimization: predictive std $\\sigma(x,y)$")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_pure_bo_acquisition_and_proposals(
    gp: GPModel2D,
    candidates_df: pd.DataFrame,
    acquisition: str,
    out_path: Path,
    *,
    run_id: str = "",
    n_grid: int = 200,
    ei_xi: float = 0.01,
    ucb_kappa: float = 2.0,
    ucb_beta: Optional[float] = None,
    acq_mode_used: Optional[str] = None,
) -> None:
    """4) Acquisition heatmap + next batch proposals overlaid."""
    X1, X2, mu, std, ei, ucb = _pure_bo_grid_predict(
        gp,
        n_grid=n_grid,
        ei_xi=ei_xi,
        ucb_kappa=ucb_kappa,
        ucb_beta=ucb_beta,
    )
    best = float(np.nanmax(gp.y_train_scaled * gp.y_std + gp.y_mean))
    ucb_kappa_used = _resolve_ucb_kappa(ucb_kappa, ucb_beta)
    # Use actual acquisition mode for consistent labeling
    actual_acq = acq_mode_used if acq_mode_used is not None else acquisition.lower()
    if actual_acq == "ucb" or actual_acq == "ei_ucb_fallback":
        Z_acq = _ucb(mu.ravel(), std.ravel(), ucb_kappa_used).reshape(mu.shape)
        if actual_acq == "ei_ucb_fallback":
            acq_label = f"UCB (EI collapsed, kappa={ucb_kappa_used:.3g})"
        else:
            acq_label = f"UCB (kappa={ucb_kappa_used:.3g})"
        acq_cmap = _XY_PANEL_CMAPS["ucb"]
    else:
        Z_acq = _ei(mu.ravel(), std.ravel(), best, ei_xi).reshape(mu.shape)
        acq_label = f"EI (xi={ei_xi:.3g})"
        acq_cmap = _XY_PANEL_CMAPS["ei"]
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
        # Calculate vmin/vmax dynamically from data
        valid_z = Z_acq[np.isfinite(Z_acq)]
        if len(valid_z) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(valid_z))
            vmax = float(np.nanmax(valid_z))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cs = ax.pcolormesh(X1, X2, Z_acq, cmap=acq_cmap, norm=norm, shading="gouraud")
        # Add contour lines with high transparency for better gradient visibility
        # For sparse data, also check relative range to allow contours even when absolute range is small.
        _range = vmax - vmin
        _range_rel = _range / (abs(vmax) + abs(vmin) + EPS) if (vmax != 0 or vmin != 0) else 0.0
        if _range > _SURROGATE_CONTOUR_MIN_RANGE or _range_rel > 1e-6:
            valid_z = Z_acq[np.isfinite(Z_acq)]
            if len(valid_z) > 0:
                # Use moderate number of contour levels (not too dense) so gradient colors are visible
                n_levels = _SURROGATE_CONTOUR_LEVELS  # Use base level count (18), not doubled
                q_levels = np.linspace(0.05, 0.95, n_levels)
                _levs = np.quantile(valid_z, q_levels)
            else:
                n_levels = _SURROGATE_CONTOUR_LEVELS
                _levs = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
            try:
                qc = ax.contour(
                    X1, X2, Z_acq,
                    levels=_levs,
                    colors=_SURROGATE_CONTOUR_COLOR,
                    linewidths=_SURROGATE_CONTOUR_LW,
                    alpha=0.95,  # High transparency
                    zorder=2,
                )
                qc.set_path_effects([
                    mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                    mpatheffects.Normal(),
                ])
            except (ValueError, TypeError):
                pass
        ax.scatter(
            candidates_df["x"],
            candidates_df["y"],
            s=60,
            facecolor="none",
            edgecolor="red",
            linewidths=1.5,
            label="Proposed",
            zorder=3,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x = \mathrm{BMA}/(\mathrm{BMA}+\mathrm{MTAC})$")
        ax.set_ylabel(r"$y = \mathrm{BMA}+\mathrm{MTAC}$")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=6)
        cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03, label=acq_label)
        _format_colorbar_plain(cbar)
        if run_id:
            ax.set_title(f"Bayesian Optimization: {acq_label} + next batch")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_pure_bo_proposal_distance_comparison(
    candidates_with_diversity: pd.DataFrame,
    candidates_without_diversity: pd.DataFrame,
    out_path: Path,
    *,
    run_id: str = "",
) -> None:
    """5) Pairwise distance distribution (fraction space): diversity ON vs OFF."""
    def pairwise_distances_frac(df: pd.DataFrame) -> np.ndarray:
        frac = df[["frac_MPC", "frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
        n = frac.shape[0]
        if n < 2:
            return np.array([])
        d = _distance_fraction_space(frac, frac)
        i, j = np.triu_indices(n, k=1)
        return d[i, j]

    d_on = pairwise_distances_frac(candidates_with_diversity)
    d_off = pairwise_distances_frac(candidates_without_diversity)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
        upper = max(
            float(d_on.max()) if d_on.size else 0.5,
            float(d_off.max()) if d_off.size else 0.5,
            0.5,
        )
        bins = np.linspace(0, upper, 25)
        has_any = False
        if d_on.size:
            ax.hist(d_on, bins=bins, alpha=0.6, label="Diversity ON", color="C0", density=True)
            has_any = True
        if d_off.size:
            ax.hist(d_off, bins=bins, alpha=0.6, label="Diversity OFF", color="C1", density=True)
            has_any = True
        ax.set_xlabel("Pairwise distance (fraction space)")
        ax.set_ylabel("Density")
        if has_any:
            ax.legend(loc="best", fontsize=6)
        else:
            ax.text(0.5, 0.5, "Not enough points for pairwise distance.", ha="center", va="center", transform=ax.transAxes)
        if run_id:
            ax.set_title("Bayesian Optimization: proposal distances")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_pure_bo_learning_curve(
    learning_df: pd.DataFrame,
    objective_column: str,
    out_path: Path,
    *,
    run_id: str = "",
    top_k: int = 3,
) -> None:
    """6) Learning curve: best and top-k mean per round."""
    df = _ensure_xy_and_objective(learning_df, objective_column)
    if "round_id" not in df.columns:
        df = df.assign(round_id="1")
    rounds = sorted(df["round_id"].astype(str).unique(), key=lambda r: (_natural_round_key(r), r))
    best_per_round: List[float] = []
    topk_mean_per_round: List[float] = []
    for r in rounds:
        sub = df[df["round_id"].astype(str) == r]
        vals = sub[objective_column].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            best_per_round.append(np.nan)
            topk_mean_per_round.append(np.nan)
        else:
            best_per_round.append(float(np.nanmax(vals)))
            k = min(top_k, vals.size)
            topk_mean_per_round.append(float(np.nanmean(np.partition(vals, -k)[-k:])))
    cumulative_best = np.array(best_per_round, dtype=float)
    running = -np.inf
    for i, v in enumerate(cumulative_best):
        if np.isfinite(v):
            running = max(running, float(v))
        cumulative_best[i] = running if np.isfinite(running) else np.nan

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
        x_axis = np.arange(len(rounds), dtype=float)
        ax.plot(x_axis, best_per_round, "o-", label="Best (per round)", linewidth=0.8, markersize=4)
        ax.plot(x_axis, cumulative_best, ".-", label="Best (cumulative)", linewidth=0.8, markersize=4)
        ax.plot(x_axis, topk_mean_per_round, "s--", label=f"Top-{top_k} mean", linewidth=0.8, markersize=4)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(rounds, fontsize=6)
        ax.set_xlabel("Round")
        ax.set_ylabel(objective_column)
        ax.legend(loc="best", fontsize=6)
        if run_id:
            ax.set_title("Bayesian Optimization: learning curve")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def run_pure_regression_bo(
    learning_df: pd.DataFrame,
    out_dir: Path,
    q: int,
    acquisition: str,
    *,
    diversity_params: Optional[Dict[str, Any]] = None,
    composition_constraints: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    run_id: str = "",
    objective_column: str = DEFAULT_OBJECTIVE_COLUMN,
    ei_xi: float = 0.01,
    ucb_kappa: float = 2.0,
    ucb_beta: Optional[float] = None,
    n_random_candidates: int = 5000,
    sparse_force_isotropic: bool = True,
    sparse_isotropic_max_unique_points: int = 20,  # Increased from 10 to force isotropic for sparse data
    min_length_scale_sparse_isotropic: float = 0.5,
    sparse_use_trend: bool = False,
    sparse_trend_max_unique_points: int = 8,
    trend_ridge: float = 1e-5,
    enable_heteroskedastic_noise: bool = True,
    noise_rel_min: float = 0.35,
    noise_rel_max: float = 3.0,
    write_plots: bool = True,
    learning_input_path: Optional[Path] = None,
    fog_plate_aware_path: Optional[Path] = None,
    polymer_colors_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Run pure regression BO: propose batch, save candidates/diagnostics CSV+JSON and (optionally) 6 figures.

    Figures: (1) observed scatter, (2) mu heatmap, (3) sigma heatmap,
    (4) acquisition + proposals, (5) proposal distance ON vs OFF, (6) learning curve.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    objective_column = _resolve_objective_column_name(learning_df, objective_column)
    model_df = _aggregate_pure_bo_training_rows(
        _ensure_xy_and_objective(learning_df, objective_column),
        objective_column,
    )

    candidates_df, gp = propose_batch_next_points(
        learning_df,
        q=q,
        acquisition=acquisition,
        diversity_params=diversity_params,
        composition_constraints=composition_constraints,
        seed=seed,
        objective_column=objective_column,
        ei_xi=ei_xi,
        ucb_kappa=ucb_kappa,
        ucb_beta=ucb_beta,
        n_random_candidates=n_random_candidates,
        sparse_force_isotropic=sparse_force_isotropic,
        sparse_isotropic_max_unique_points=sparse_isotropic_max_unique_points,
        min_length_scale_sparse_isotropic=min_length_scale_sparse_isotropic,
        sparse_use_trend=sparse_use_trend,
        sparse_trend_max_unique_points=sparse_trend_max_unique_points,
        trend_ridge=trend_ridge,
        enable_heteroskedastic_noise=enable_heteroskedastic_noise,
        noise_rel_min=noise_rel_min,
        noise_rel_max=noise_rel_max,
        enable_diversity=True,
    )
    candidates_no_div, _ = propose_batch_next_points(
        learning_df,
        q=q,
        acquisition=acquisition,
        diversity_params=diversity_params,
        composition_constraints=composition_constraints,
        seed=seed,
        objective_column=objective_column,
        ei_xi=ei_xi,
        ucb_kappa=ucb_kappa,
        ucb_beta=ucb_beta,
        n_random_candidates=n_random_candidates,
        sparse_force_isotropic=sparse_force_isotropic,
        sparse_isotropic_max_unique_points=sparse_isotropic_max_unique_points,
        min_length_scale_sparse_isotropic=min_length_scale_sparse_isotropic,
        sparse_use_trend=sparse_use_trend,
        sparse_trend_max_unique_points=sparse_trend_max_unique_points,
        trend_ridge=trend_ridge,
        enable_heteroskedastic_noise=enable_heteroskedastic_noise,
        noise_rel_min=noise_rel_min,
        noise_rel_max=noise_rel_max,
        enable_diversity=False,
    )

    candidates_out = csv_dir / f"pure_bo_candidates__{rid}.csv"
    candidates_no_div_out = csv_dir / f"pure_bo_candidates_no_diversity__{rid}.csv"
    cand_out = candidates_df.copy()
    cand_out.insert(0, "run_id", rid)
    cand_out.insert(1, "bo_run_id", rid)
    cand_out.to_csv(candidates_out, index=False)
    legacy_candidates_out = out_dir / f"pure_bo_candidates__{rid}.csv"
    if legacy_candidates_out.is_file():
        legacy_candidates_out.unlink(missing_ok=True)
    cand_no_div_out = candidates_no_div.copy()
    cand_no_div_out.insert(0, "run_id", rid)
    cand_no_div_out.insert(1, "bo_run_id", rid)
    cand_no_div_out.to_csv(candidates_no_div_out, index=False)
    legacy_candidates_no_div_out = out_dir / f"pure_bo_candidates_no_diversity__{rid}.csv"
    if legacy_candidates_no_div_out.is_file():
        legacy_candidates_no_div_out.unlink(missing_ok=True)

    outputs: Dict[str, Path] = {
        "candidates": candidates_out,
        "candidates_no_diversity": candidates_no_div_out,
    }

    # Keep this available even when write_plots=False (used by summary).
    acq_mode_used = (
        str(candidates_df["acq_mode_used"].iloc[0])
        if len(candidates_df) > 0 and "acq_mode_used" in candidates_df.columns
        else str(acquisition).lower()
    )

    if write_plots:
        plot_learning_df = _ensure_xy_and_objective(learning_df, objective_column).copy()
        if not {"frac_MPC", "frac_BMA", "frac_MTAC"}.issubset(plot_learning_df.columns):
            mpc, bma, mtac = _frac_from_xy_bo_data(
                plot_learning_df["x"].to_numpy(dtype=float),
                plot_learning_df["y"].to_numpy(dtype=float),
            )
            plot_learning_df["frac_MPC"] = mpc
            plot_learning_df["frac_BMA"] = bma
            plot_learning_df["frac_MTAC"] = mtac
        if "log_fog_corrected" not in plot_learning_df.columns:
            plot_learning_df["log_fog_corrected"] = pd.to_numeric(
                plot_learning_df[objective_column],
                errors="coerce",
            )
        if "polymer_id" not in plot_learning_df.columns:
            plot_learning_df["polymer_id"] = [f"P{i+1}" for i in range(len(plot_learning_df))]

        plot_pure_bo_observed_scatter(
            learning_df, objective_column,
            out_dir / f"pure_bo_observed_scatter__{rid}.png",
            run_id=rid,
        )
        plot_pure_bo_mu_heatmap(
            gp,
            out_dir / f"pure_bo_mu_heatmap__{rid}.png",
            run_id=rid,
        )
        plot_pure_bo_sigma_heatmap(
            gp,
            out_dir / f"pure_bo_sigma_heatmap__{rid}.png",
            run_id=rid,
        )
        # Get actual acquisition mode used for consistent labeling
        acq_mode_for_plot = candidates_df["acq_mode_used"].iloc[0] if len(candidates_df) > 0 and "acq_mode_used" in candidates_df.columns else acquisition.lower()
        plot_pure_bo_acquisition_and_proposals(
            gp,
            candidates_df,
            acquisition,
            out_dir / f"pure_bo_acquisition_proposals__{rid}.png",
            run_id=rid,
            ei_xi=ei_xi,
            ucb_kappa=ucb_kappa,
            ucb_beta=ucb_beta,
            acq_mode_used=acq_mode_for_plot,
        )
        plot_pure_bo_proposal_distance_comparison(
            candidates_df,
            candidates_no_div,
            out_dir / f"pure_bo_proposal_distance_comparison__{rid}.png",
            run_id=rid,
        )
        plot_pure_bo_learning_curve(
            learning_df,
            objective_column,
            out_dir / f"pure_bo_learning_curve__{rid}.png",
            run_id=rid,
        )

        # Keep legacy BO visualization family (ternary + 2x2 surrogate panels) in pure mode as well.
        plot_cfg = BOConfig(
            ei_xi=float(ei_xi),
            ucb_kappa=float(_resolve_ucb_kappa(ucb_kappa, ucb_beta)),
            ternary_plot_step=0.003,  # Finer grid for smoother gradient visualization
            surrogate_map_xy_grid=501,  # Finer grid for smoother gradient visualization
            std_color_gamma=2.0,  # Reduced from 8.0 to 2.0 for better gradient in std plots
        )
        plot_cand = _build_plot_frame(gp, plot_learning_df, plot_cfg)
        # Group by (polymer_id, round_id) if round_id exists, otherwise by polymer_id only
        # This ensures each round's data is plotted separately
        if "round_id" in plot_learning_df.columns:
            group_cols = ["polymer_id", "round_id"]
        else:
            group_cols = ["polymer_id"]
        observed = (
            plot_learning_df.groupby(group_cols, as_index=False)
            .agg(
                frac_MPC=("frac_MPC", "mean"),
                frac_BMA=("frac_BMA", "mean"),
                frac_MTAC=("frac_MTAC", "mean"),
            )
        )
        # Get actual acquisition mode used from candidates_df for consistent labeling
        acq_mode_used = (
            str(candidates_df["acq_mode_used"].iloc[0])
            if len(candidates_df) > 0 and "acq_mode_used" in candidates_df.columns
            else str(acquisition).lower()
        )
        
        # Get map quality for flat map annotation
        map_quality = _pure_bo_map_quality(
            gp,
            ei_xi=ei_xi,
            ucb_kappa=ucb_kappa,
            ucb_beta=ucb_beta,
            n_grid=200,
        )
        mu_flat_note = " (flat map)" if map_quality.get("mu_is_flat", False) else ""
        ei_flat_note = " (flat map)" if map_quality.get("ei_is_flat", False) else ""
        
        ternary_mean_path = out_dir / f"pure_bo_ternary_mean__{rid}.png"
        ternary_std_path = out_dir / f"pure_bo_ternary_std__{rid}.png"
        ternary_ei_path = out_dir / f"pure_bo_ternary_ei__{rid}.png"
        ternary_ucb_path = out_dir / f"pure_bo_ternary_ucb__{rid}.png"
        xy_2x2_path = out_dir / f"pure_bo_xy_2x2_mean_std_ei_ucb__{rid}.png"
        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="pred_log_fog_mean",
            title=f"Bayesian Optimization: predicted mean log(FoG){mu_flat_note}",
            cbar_label="Predicted mean log(FoG)",
            out_path=ternary_mean_path,
        )
        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="pred_log_fog_std",
            title="Bayesian Optimization: predictive std log(FoG)",
            cbar_label="Predictive std log(FoG)",
            out_path=ternary_std_path,
            std_color_gamma=2.0,  # Reduced from 8.0 to 2.0 for better gradient
        )
        # Use actual acquisition mode in title/label for consistency
        if acq_mode_used == "ei_ucb_fallback":
            ei_title = f"Bayesian Optimization: Expected Improvement (no promising region){ei_flat_note}"
            ei_label = "UCB (EI collapsed)"
        else:
            ei_title = f"Bayesian Optimization: Expected Improvement{ei_flat_note}"
            ei_label = "EI"
        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="ei",
            title=ei_title,
            cbar_label=ei_label,
            out_path=ternary_ei_path,
        )
        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="ucb",
            title="Bayesian Optimization: Upper Confidence Bound",
            cbar_label="UCB",
            out_path=ternary_ucb_path,
        )
        # Pass acq_mode_used to 2x2 panels for consistent labeling
        _plot_xy_2x2_panels(gp, plot_learning_df, plot_cfg, rid, xy_2x2_path, acq_mode_used=acq_mode_used)

        # Generate ternary 2x2 panels (same order as xy_2x2: Mean, Std, EI, UCB)
        ternary_2x2_path = out_dir / f"pure_bo_ternary_2x2_mean_std_ei_ucb__{rid}.png"
        _plot_ternary_2x2_panels(
            plot_cand,
            observed,
            rid,
            ternary_2x2_path,
            std_color_gamma=2.0,
            acq_mode_used=acq_mode_used,
        )

        outputs["observed_scatter"] = out_dir / f"pure_bo_observed_scatter__{rid}.png"
        outputs["mu_heatmap"] = out_dir / f"pure_bo_mu_heatmap__{rid}.png"
        outputs["sigma_heatmap"] = out_dir / f"pure_bo_sigma_heatmap__{rid}.png"
        outputs["acquisition_proposals"] = out_dir / f"pure_bo_acquisition_proposals__{rid}.png"
        outputs["distance_comparison"] = out_dir / f"pure_bo_proposal_distance_comparison__{rid}.png"
        outputs["learning_curve"] = out_dir / f"pure_bo_learning_curve__{rid}.png"
        outputs["ternary_mean"] = ternary_mean_path
        outputs["ternary_std"] = ternary_std_path
        outputs["ternary_ei"] = ternary_ei_path
        outputs["ternary_ucb"] = ternary_ucb_path
        outputs["xy_2x2_panels"] = xy_2x2_path
        outputs["ternary_2x2_panels"] = ternary_2x2_path

        # Generate ranking bar charts if fog_plate_aware_path is provided
        if fog_plate_aware_path is not None:
            fog_path = Path(fog_plate_aware_path)
            print(f"Attempting to generate ranking bar charts from: {fog_path} (exists: {fog_path.exists()}, is_file: {fog_path.is_file() if fog_path.exists() else False})")
            if not fog_path.exists() or not fog_path.is_file():
                import warnings
                warnings.warn(
                    f"fog_plate_aware_path provided but file not found: {fog_path}. "
                    "Skipping ranking bar charts.",
                    UserWarning
                )
            else:
                try:
                    print(f"Loading fog_plate_aware data from: {fog_path}")
                    fog = _load_fog_plate_aware(fog_path)
                    print(f"Loaded {len(fog)} rows from fog_plate_aware")
                    if fog.empty:
                        import warnings
                        warnings.warn(
                            f"fog_plate_aware data is empty after loading from {fog_path}. "
                            "Skipping ranking bar charts.",
                            UserWarning
                        )
                    else:
                        print("Generating ranking tables...")
                        ranking = _rank_tables(fog, bo_run_id=rid, fog_plate_aware_path=fog_path)
                        rank_all = ranking["all_round"]
                        print(f"Ranking tables generated. Found {len(rank_all)} polymers in all_round ranking.")
                        
                        fog_all_tbl = rank_all[
                            ["run_id", "bo_run_id", "polymer_id", "mean_fog", "mean_log_fog", "n_observations", "rounds", "run_ids", "rank_fog_desc"]
                        ].sort_values(["rank_fog_desc", "polymer_id"])
                        t50_all_tbl = rank_all[
                            ["run_id", "bo_run_id", "polymer_id", "mean_t50_min", "n_observations", "rounds", "run_ids", "rank_t50_desc"]
                        ].sort_values(["rank_t50_desc", "polymer_id"])
                        
                        print(f"FoG table shape: {fog_all_tbl.shape}, columns: {list(fog_all_tbl.columns)}")
                        print(f"t50 table shape: {t50_all_tbl.shape}, columns: {list(t50_all_tbl.columns)}")
                        if not t50_all_tbl.empty:
                            print(f"t50 table mean_t50_min stats: min={t50_all_tbl['mean_t50_min'].min()}, max={t50_all_tbl['mean_t50_min'].max()}, finite={t50_all_tbl['mean_t50_min'].notna().sum()}/{len(t50_all_tbl)}")
                        
                        fog_bar_all = out_dir / f"pure_bo_fog_ranking_all__{rid}.png"
                        t50_bar_all = out_dir / f"pure_bo_t50_ranking_all__{rid}.png"
                        print(f"Generating bar charts: {fog_bar_all.name}, {t50_bar_all.name}")
                        bar_color_map = _load_polymer_colors(polymer_colors_path) if polymer_colors_path else None
                        if bar_color_map:
                            print(f"Loaded color map with {len(bar_color_map)} entries")
                        if not fog_all_tbl.empty:
                            _plot_ranking_bar(
                                fog_all_tbl,
                                value_col="mean_fog",
                                label_col="polymer_id",
                                title="FoG ranking (all rounds)",
                                xlabel="Mean FoG",
                                out_path=fog_bar_all,
                                color_map=bar_color_map,
                            )
                        else:
                            print(f"Warning: fog_all_tbl is empty, skipping FoG bar chart")
                        if not t50_all_tbl.empty:
                            # Filter out rows with NaN mean_t50_min
                            t50_all_tbl_valid = t50_all_tbl[t50_all_tbl["mean_t50_min"].notna()].copy()
                            print(f"t50 table after filtering NaN: {t50_all_tbl_valid.shape}")
                            if not t50_all_tbl_valid.empty:
                                _plot_ranking_bar(
                                    t50_all_tbl_valid,
                                    value_col="mean_t50_min",
                                    label_col="polymer_id",
                                    title="t50 ranking (all rounds)",
                                    xlabel="Mean t50 [min]",
                                    out_path=t50_bar_all,
                                    color_map=bar_color_map,
                                )
                            else:
                                print(f"Warning: t50_all_tbl_valid is empty after filtering NaN")
                        else:
                            print(f"Warning: t50_all_tbl is empty")
                        outputs["fog_rank_bar_all"] = fog_bar_all
                        outputs["t50_rank_bar_all"] = t50_bar_all
                        print(f"Successfully generated ranking bar charts: {fog_bar_all.name}, {t50_bar_all.name}")
                except Exception as e:
                    # Log error but don't fail the entire run
                    import warnings
                    import traceback
                    error_msg = f"Failed to generate ranking bar charts: {e}\n{traceback.format_exc()}"
                    print(f"ERROR: {error_msg}")
                    warnings.warn(error_msg, UserWarning)
        else:
            print("fog_plate_aware_path is None, skipping ranking bar charts")

    summary = {
        "run_id": rid,
        "bo_run_id": rid,
        "objective_column": objective_column,
        "acquisition": str(acquisition).lower(),
        "q": int(q),
        "n_training_rows": int(len(learning_df)),
        "n_model_rows_after_aggregate": int(len(model_df)),
        "n_unique_design_points": int(model_df[["x", "y"]].drop_duplicates().shape[0]),
        "n_selected": int(len(candidates_df)),
        "config": {
            "ei_xi": float(ei_xi),
            "ei_xi_adaptive": float(
                candidates_df["ei_xi_used"].iloc[0]
            ) if "ei_xi_used" in candidates_df.columns and len(candidates_df) > 0 and pd.notna(candidates_df["ei_xi_used"].iloc[0]) else None,
            "ucb_kappa": float(ucb_kappa),
            "ucb_beta": (None if ucb_beta is None else float(ucb_beta)),
            "ucb_kappa_used": float(_resolve_ucb_kappa(ucb_kappa, ucb_beta)),
            "n_random_candidates": int(n_random_candidates),
            "sparse_force_isotropic": bool(sparse_force_isotropic),
            "sparse_isotropic_max_unique_points": int(sparse_isotropic_max_unique_points),
            "min_length_scale_sparse_isotropic": float(min_length_scale_sparse_isotropic),
            "sparse_use_trend": bool(sparse_use_trend),
            "sparse_trend_max_unique_points": int(sparse_trend_max_unique_points),
            "trend_ridge": float(trend_ridge),
            "enable_heteroskedastic_noise": bool(enable_heteroskedastic_noise),
            "noise_rel_min": float(noise_rel_min),
            "noise_rel_max": float(noise_rel_max),
            "force_isotropic_applied": bool(
                candidates_df["force_isotropic_applied"].iloc[0]
            ) if "force_isotropic_applied" in candidates_df.columns and len(candidates_df) else False,
            "use_trend_applied": bool(
                candidates_df["use_trend_applied"].iloc[0]
            ) if "use_trend_applied" in candidates_df.columns and len(candidates_df) else False,
            "seed": None if seed is None else int(seed),
            "diversity_params": dict(diversity_params or {}),
            "composition_constraints": dict(composition_constraints or {}),
        },
        "gp_hyperparams": {
            "length_scale_x": float(gp.length_scale[0]),
            "length_scale_y": float(gp.length_scale[1]),
            "signal_std": float(gp.signal_std),
            "noise_std": float(gp.noise_std),
            "kernel_mode": str(gp.kernel_mode),
            "trend_basis": str(getattr(gp, "trend_basis", "none")),
            "trend_ridge": float(getattr(gp, "trend_ridge", 0.0)),
            "obs_noise_rel_min": float(np.nanmin(gp.obs_noise_rel)),
            "obs_noise_rel_median": float(np.nanmedian(gp.obs_noise_rel)),
            "obs_noise_rel_max": float(np.nanmax(gp.obs_noise_rel)),
        },
        "design_coverage": _design_coverage_report(model_df, use_bma_mtac_coords=False),
        "map_quality": _pure_bo_map_quality(
            gp,
            ei_xi=ei_xi,
            ucb_kappa=ucb_kappa,
            ucb_beta=ucb_beta,
            n_grid=200,
        ),
        "acq_mode_used": acq_mode_used,  # Record actual acquisition mode used for consistency
        "selected_top": candidates_df.to_dict(orient="records"),
    }
    summary_path = out_dir / f"pure_bo_summary__{rid}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    outputs["summary"] = summary_path

    input_paths: List[Path] = []
    if learning_input_path is not None:
        input_paths.append(Path(learning_input_path))
    manifest = build_run_manifest_dict(
        run_id=rid,
        input_paths=input_paths,
        git_root=Path.cwd(),
        extra={
            "objective_column": objective_column,
            "acquisition": str(acquisition).lower(),
            "q": int(q),
            "output_files": sorted(p.name for p in outputs.values()),
        },
    )
    manifest_path = out_dir / f"pure_bo_manifest__{rid}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    outputs["manifest"] = manifest_path

    return outputs


def _xy_from_frac(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    inter = out["frac_BMA"].astype(float) + out["frac_MTAC"].astype(float)
    out["y"] = inter
    out["x"] = np.where(inter > EPS, out["frac_BMA"].astype(float) / inter, 0.5)
    return out


def _ternary_xy_from_frac(
    frac_mpc: np.ndarray,
    frac_bma: np.ndarray,
    frac_mtac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tx = frac_bma + 0.5 * frac_mpc
    ty = frac_mpc * SQRT3_OVER_2
    return tx, ty


def _generate_simplex_grid(step: float, min_component: float) -> pd.DataFrame:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"candidate_step must be (0,1], got {step}")
    vals = np.arange(0.0, 1.0 + step / 2.0, step)
    rows: list[dict[str, float]] = []
    for mpc in vals:
        max_bma = 1.0 - mpc
        b_vals = np.arange(0.0, max_bma + step / 2.0, step)
        for bma in b_vals:
            mtac = 1.0 - mpc - bma
            if mtac < -1e-9:
                continue
            mtac = max(mtac, 0.0)
            if min(mpc, bma, mtac) + 1e-12 < min_component:
                continue
            rows.append(
                {
                    "frac_MPC": float(mpc),
                    "frac_BMA": float(bma),
                    "frac_MTAC": float(mtac),
                }
            )
    return pd.DataFrame(rows)


def _load_bo_learning(bo_learning_path: Path) -> pd.DataFrame:
    df = pd.read_csv(bo_learning_path)
    required = ["polymer_id", "round_id", "frac_MPC", "frac_BMA", "frac_MTAC", "log_fog"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"BO learning CSV missing required columns: {missing}")

    out = df.copy()
    out["polymer_id"] = out["polymer_id"].astype(str).str.strip()
    out["round_id"] = out["round_id"].astype(str).str.strip()
    for c in ["frac_MPC", "frac_BMA", "frac_MTAC", "log_fog"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "n_observations" in out.columns:
        out["n_observations"] = pd.to_numeric(out["n_observations"], errors="coerce")
    if "log_fog_mad" in out.columns:
        out["log_fog_mad"] = pd.to_numeric(out["log_fog_mad"], errors="coerce")
    out = out[np.isfinite(out["log_fog"])].copy()
    out = out[np.isfinite(out[["frac_MPC", "frac_BMA", "frac_MTAC"]]).all(axis=1)].copy()
    s = out[["frac_MPC", "frac_BMA", "frac_MTAC"]].sum(axis=1)
    out = out[np.isfinite(s) & (np.abs(s - 1.0) <= 5e-4)].copy()
    out = _xy_from_frac(out)
    return out.reset_index(drop=True)


def _load_fog_plate_aware(fog_plate_aware_path: Path) -> pd.DataFrame:
    df = pd.read_csv(fog_plate_aware_path)
    required = ["round_id", "run_id", "polymer_id", "t50_min", "fog", "log_fog"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FoG plate-aware CSV missing required columns: {missing}")
    out = df.copy()
    out["round_id"] = out["round_id"].astype(str).str.strip()
    out["run_id"] = out["run_id"].astype(str).str.strip()
    out["polymer_id"] = out["polymer_id"].astype(str).str.strip()
    for c in ["t50_min", "fog", "log_fog"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Also load gox_t50_used_min if available (for GOx data extraction)
    if "gox_t50_used_min" in df.columns:
        out["gox_t50_used_min"] = pd.to_numeric(df["gox_t50_used_min"], errors="coerce")
    out = out[np.isfinite(out["t50_min"]) & np.isfinite(out["fog"])].copy()
    out = out[out["fog"] > 0].copy()
    return out.reset_index(drop=True)


def _gp_hyperparams_for_summary(
    gp: Any, use_simplex_gp: bool, use_bma_mtac_coords: bool = False
) -> dict:
    base = {
        "signal_std": float(gp.signal_std),
        "noise_std": float(gp.noise_std),
        "obs_noise_rel_min": float(np.nanmin(gp.obs_noise_rel)),
        "obs_noise_rel_median": float(np.nanmedian(gp.obs_noise_rel)),
        "obs_noise_rel_max": float(np.nanmax(gp.obs_noise_rel)),
        "y_mean": float(gp.y_mean),
        "y_std": float(gp.y_std),
        "kernel_mode": str(getattr(gp, "kernel_mode", "ard")),
        "trend_basis": str(getattr(gp, "trend_basis", "none")),
        "trend_ridge": float(getattr(gp, "trend_ridge", 0.0)),
    }
    if use_simplex_gp and len(gp.length_scale) == 3:
        base["length_scale_frac_MPC"] = float(gp.length_scale[0])
        base["length_scale_frac_BMA"] = float(gp.length_scale[1])
        base["length_scale_frac_MTAC"] = float(gp.length_scale[2])
    elif use_bma_mtac_coords and len(gp.length_scale) == 2:
        base["length_scale_frac_BMA"] = float(gp.length_scale[0])
        base["length_scale_frac_MTAC"] = float(gp.length_scale[1])
    else:
        base["length_scale_x"] = float(gp.length_scale[0])
        base["length_scale_y"] = float(gp.length_scale[1])
    return base


def _collect_referenced_run_ids(learning_df: pd.DataFrame, fog_df: pd.DataFrame) -> list[str]:
    ids: set[str] = set()
    if "run_id" in learning_df.columns:
        ids.update(str(v).strip() for v in learning_df["run_id"].dropna().astype(str))
    if "run_ids" in learning_df.columns:
        for v in learning_df["run_ids"].dropna().astype(str):
            for rid in str(v).split(","):
                rid = rid.strip()
                if rid:
                    ids.add(rid)
    if "run_id" in fog_df.columns:
        ids.update(str(v).strip() for v in fog_df["run_id"].dropna().astype(str))
    return sorted(x for x in ids if x)


def _apply_round_anchor_correction(
    learning_df: pd.DataFrame,
    *,
    enabled: bool,
    min_anchor_polymers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Shift log_fog by round using anchor polymers (intersection across rounds).
    Correction is additive in log-domain, i.e. multiplicative on FoG.
    """
    out = learning_df.copy()
    out["log_fog_raw"] = out["log_fog"].astype(float)
    out["round_shift"] = 0.0
    out["log_fog_corrected"] = out["log_fog_raw"]

    meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "method": "none",
        "anchors": [],
        "global_anchor_stat": None,
        "round_anchor_stat": {},
        "round_shift": {},
    }
    if not enabled:
        return out, meta

    round_groups = {str(r): g.copy() for r, g in out.groupby("round_id")}
    if len(round_groups) < 2:
        meta["method"] = "single_round_no_correction"
        return out, meta

    round_poly_sets = [set(g["polymer_id"].astype(str)) for g in round_groups.values()]
    anchors = sorted(set.intersection(*round_poly_sets)) if round_poly_sets else []
    if len(anchors) < int(min_anchor_polymers):
        meta["method"] = "insufficient_shared_anchors"
        meta["anchors"] = anchors
        return out, meta

    round_anchor_stat: dict[str, float] = {}
    for round_id, g in round_groups.items():
        vals = g[g["polymer_id"].isin(anchors)]["log_fog_raw"].astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < int(min_anchor_polymers):
            meta["method"] = "insufficient_anchor_rows_per_round"
            meta["anchors"] = anchors
            return out, meta
        round_anchor_stat[round_id] = float(np.nanmedian(vals))

    global_stat = float(np.nanmedian(list(round_anchor_stat.values())))
    shift_map = {r: (global_stat - v) for r, v in round_anchor_stat.items()}
    out["round_shift"] = out["round_id"].map(shift_map).fillna(0.0).astype(float)
    out["log_fog_corrected"] = out["log_fog_raw"] + out["round_shift"]

    meta.update(
        {
            "applied": True,
            "method": "median_shift_by_shared_anchors",
            "anchors": anchors,
            "global_anchor_stat": global_stat,
            "round_anchor_stat": round_anchor_stat,
            "round_shift": shift_map,
        }
    )
    return out, meta


def _estimate_obs_noise_rel(
    learning_df: pd.DataFrame,
    *,
    min_rel: float,
    max_rel: float,
) -> np.ndarray:
    """
    Estimate per-observation relative noise for heteroskedastic GP.

    - Starts from 1.0 for all rows.
    - If n_observations exists, scales by 1/sqrt(n_observations).
    - If log_fog_mad exists, scales by MAD relative to median MAD.
    - Re-centered to median 1.0 and clipped to [min_rel, max_rel].
    """
    n = int(len(learning_df))
    rel = np.ones(n, dtype=float)
    if n == 0:
        return rel

    if "n_observations" in learning_df.columns:
        obs_n = pd.to_numeric(learning_df["n_observations"], errors="coerce").to_numpy(dtype=float)
        obs_n = np.where(np.isfinite(obs_n) & (obs_n > 0.0), obs_n, 1.0)
        rel *= 1.0 / np.sqrt(obs_n)

    if "log_fog_mad" in learning_df.columns:
        mad = pd.to_numeric(learning_df["log_fog_mad"], errors="coerce").to_numpy(dtype=float)
        finite_pos = np.isfinite(mad) & (mad > 0.0)
        if np.any(finite_pos):
            ref = float(np.nanmedian(mad[finite_pos]))
            ref = max(ref, 1e-6)
            mad_rel = np.ones(n, dtype=float)
            mad_rel[finite_pos] = mad[finite_pos] / ref
            mad_rel = np.clip(mad_rel, 0.5, 3.0)
            rel *= mad_rel

    finite_rel = np.isfinite(rel) & (rel > 0.0)
    if np.any(finite_rel):
        med = float(np.nanmedian(rel[finite_rel]))
        med = med if med > 0.0 else 1.0
        rel = rel / med
    else:
        rel[:] = 1.0

    lo = float(min(min_rel, max_rel))
    hi = float(max(min_rel, max_rel))
    return np.clip(rel, lo, hi)


def _resolve_batch_counts(cfg: BOConfig) -> dict[str, int]:
    n_total = int(max(cfg.n_suggestions, 1))

    # Keep defaults conservative on small batches; explicit counts override.
    if cfg.anchor_count is None:
        n_anchor = int(max(1, round(n_total * max(cfg.anchor_fraction, 0.0)))) if n_total >= 6 else 0
    else:
        n_anchor = int(max(cfg.anchor_count, 0))
    n_anchor = min(n_anchor, max(n_total - 1, 0))

    if cfg.replicate_count is None:
        n_replicate = int(max(1, round(n_total * max(cfg.replicate_fraction, 0.0)))) if n_total >= 6 else 0
    else:
        n_replicate = int(max(cfg.replicate_count, 0))
    n_replicate = min(n_replicate, max(n_total - n_anchor - 1, 0))

    n_unique = n_total - n_anchor - n_replicate
    n_unique = max(n_unique, 1)

    n_explore = int(round(n_unique * float(np.clip(cfg.exploration_ratio, 0.0, 1.0))))
    if n_unique > 1:
        n_explore = min(n_explore, n_unique - 1)
    else:
        n_explore = 0
    n_exploit = n_unique - n_explore
    n_exploit = max(n_exploit, 1)
    n_explore = max(n_unique - n_exploit, 0)

    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "n_exploit": n_exploit,
        "n_explore": n_explore,
        "n_anchor": n_anchor,
        "n_replicate": n_replicate,
    }


def _anchor_targets_from_learning(
    learning_df: pd.DataFrame,
    *,
    anchor_ids: Iterable[str],
    n_anchors: int,
) -> list[dict[str, float]]:
    if n_anchors <= 0 or learning_df.empty:
        return []
    grouped = (
        learning_df.groupby("polymer_id", as_index=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            frac_MPC=("frac_MPC", "mean"),
            frac_BMA=("frac_BMA", "mean"),
            frac_MTAC=("frac_MTAC", "mean"),
            round_count=("round_id", "nunique"),
        )
    )
    n_rounds = int(learning_df["round_id"].astype(str).nunique()) if "round_id" in learning_df.columns else 1
    present = set(grouped["polymer_id"].astype(str))
    shared_ids = set(grouped[grouped["round_count"] >= n_rounds]["polymer_id"].astype(str)) if n_rounds > 1 else present

    ordered_ids: list[str] = []
    for pid in anchor_ids:
        p = str(pid).strip()
        if p and p in present and p not in ordered_ids:
            ordered_ids.append(p)
    for pid in sorted(shared_ids):
        if pid not in ordered_ids:
            ordered_ids.append(pid)
    for pid in grouped.sort_values(["round_count", "polymer_id"], ascending=[False, True])["polymer_id"].astype(str):
        if pid not in ordered_ids:
            ordered_ids.append(pid)

    targets: list[dict[str, float]] = []
    for pid in ordered_ids:
        if len(targets) >= n_anchors:
            break
        row = grouped[grouped["polymer_id"].astype(str) == pid].iloc[0]
        targets.append(
            {
                "polymer_id": str(pid),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "frac_MPC": float(row["frac_MPC"]),
                "frac_BMA": float(row["frac_BMA"]),
                "frac_MTAC": float(row["frac_MTAC"]),
            }
        )
    return targets


def _build_exact_anchor_rows(
    model: GPModel2D,
    anchor_targets: list[dict[str, float]],
    learning_df: pd.DataFrame,
    cfg: BOConfig,
    *,
    start_order: int,
    cand_template: pd.DataFrame,
) -> pd.DataFrame:
    """Build suggestion rows for exact anchor polymer compositions (no grid rounding)."""
    if not anchor_targets:
        return pd.DataFrame()
    best = float(np.nanmax(learning_df["log_fog_corrected"].to_numpy(dtype=float)))
    rows: list[dict[str, Any]] = []
    for i, target in enumerate(anchor_targets):
        if len(model.length_scale) == 3:
            X_pred = np.array([
                [float(target["frac_MPC"]), float(target["frac_BMA"]), float(target["frac_MTAC"])]
            ], dtype=float)
        elif getattr(cfg, "use_bma_mtac_coords", False):
            X_pred = np.array([
                [float(target["frac_BMA"]), float(target["frac_MTAC"])]
            ], dtype=float)
        else:
            X_pred = np.array([[float(target["x"]), float(target["y"])]], dtype=float)
        mu, std = model.predict(X_pred)
        mu, std = float(mu[0]), float(std[0])
        x, y = float(target["x"]), float(target["y"])
        rows.append({
            "frac_MPC": float(target["frac_MPC"]),
            "frac_BMA": float(target["frac_BMA"]),
            "frac_MTAC": float(target["frac_MTAC"]),
            "x": x,
            "y": y,
            "pred_log_fog_mean": mu,
            "pred_log_fog_std": std,
            "pred_fog_median": np.exp(mu),
            "pred_fog_mean": np.exp(mu + 0.5 * (std ** 2)),
            "ei": _ei(np.array([mu]), np.array([std]), best, cfg.ei_xi)[0],
            "ucb": _ucb(np.array([mu]), np.array([std]), cfg.ucb_kappa)[0],
            "constraint_sum_ok": True,
            "constraint_bounds_ok": True,
            "min_dist_to_train": 0.0,
            "selected": 1,
            "selection_reason": f"anchor_{target['polymer_id']}",
            "selection_order": start_order + i,
        })
    out = pd.DataFrame(rows)
    for c in cand_template.columns:
        if c not in out.columns and c.startswith("policy_"):
            out[c] = cand_template[c].iloc[0]
    return out


def _replicate_pool(selected: pd.DataFrame, source: str) -> pd.DataFrame:
    """Subset of selected to draw replicates from: exploit, explore, or all."""
    base = selected.copy()
    if source == "explore":
        pool = base[base["selection_reason"].astype(str).str.contains("explore", case=False, regex=True)].copy()
    elif source == "exploit":
        pool = base[base["selection_reason"].astype(str).str.contains("exploit", case=False, regex=True)].copy()
    else:
        pool = base.copy()
    return pool if not pool.empty else base


def _append_replicates(
    selected: pd.DataFrame,
    n_replicate: int,
    replicate_source: str = "exploit",
) -> pd.DataFrame:
    if n_replicate <= 0 or selected.empty:
        return selected.copy()
    base = selected.copy().sort_values("selection_order")
    pool = _replicate_pool(base, replicate_source)
    if pool.empty:
        return base

    replicas: list[pd.Series] = []
    next_order = int(np.nanmax(base["selection_order"].to_numpy(dtype=float))) + 1
    for i in range(int(n_replicate)):
        src = pool.iloc[i % len(pool)].copy()
        src["selection_reason"] = f"replicate_of_{int(src['selection_order'])}"
        src["selection_order"] = float(next_order + i)
        src["selected"] = 1
        replicas.append(src)

    rep_df = pd.DataFrame(replicas)
    out = pd.concat([base, rep_df], ignore_index=True)
    out = out.sort_values("selection_order").reset_index(drop=True)
    return out


def _rank_tables(fog_df: pd.DataFrame, bo_run_id: str, fog_plate_aware_path: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    # Round-wise ranking.
    by_round = (
        fog_df.groupby(["round_id", "polymer_id"], as_index=False)
        .agg(
            mean_t50_min=("t50_min", "mean"),
            mean_fog=("fog", "mean"),
            mean_log_fog=("log_fog", "mean"),
            n_observations=("fog", "size"),
            run_ids=("run_id", lambda x: ",".join(sorted(set(map(str, x))))),
        )
    )
    by_round["rank_t50_desc"] = by_round.groupby("round_id")["mean_t50_min"].rank(
        method="dense", ascending=False
    ).astype(int)
    by_round["rank_fog_desc"] = by_round.groupby("round_id")["mean_fog"].rank(
        method="dense", ascending=False
    ).astype(int)
    # Insert run_id and bo_run_id columns (check if they already exist)
    if "run_id" not in by_round.columns:
        by_round.insert(0, "run_id", bo_run_id)
    else:
        by_round["run_id"] = bo_run_id
    if "bo_run_id" not in by_round.columns:
        by_round.insert(1, "bo_run_id", bo_run_id)
    else:
        by_round["bo_run_id"] = bo_run_id

    # All-round ranking.
    all_round = (
        fog_df.groupby(["polymer_id"], as_index=False)
        .agg(
            mean_t50_min=("t50_min", "mean"),
            mean_fog=("fog", "mean"),
            mean_log_fog=("log_fog", "mean"),
            n_observations=("fog", "size"),
            rounds=("round_id", lambda x: ",".join(sorted(set(map(str, x))))),
            run_ids=("run_id", lambda x: ",".join(sorted(set(map(str, x))))),
        )
    )
    
    # Insert run_id and bo_run_id columns before adding GOx (to avoid duplicate insert error)
    if "run_id" not in all_round.columns:
        all_round.insert(0, "run_id", bo_run_id)
    else:
        all_round["run_id"] = bo_run_id
    if "bo_run_id" not in all_round.columns:
        all_round.insert(1, "bo_run_id", bo_run_id)
    else:
        all_round["bo_run_id"] = bo_run_id
    
    # Add GOx data if available
    gox_t50_mean = _get_gox_mean_t50(fog_df, fog_plate_aware_path)
    if gox_t50_mean is not None and np.isfinite(gox_t50_mean):
        # Check if GOx already exists in all_round
        gox_exists = (all_round["polymer_id"].str.upper() == "GOX").any()
        if not gox_exists:
            gox_row = pd.DataFrame({
                "run_id": [bo_run_id],
                "bo_run_id": [bo_run_id],
                "polymer_id": ["GOx"],
                "mean_t50_min": [gox_t50_mean],
                "mean_fog": [1.0],  # FoG = 1.0 for GOx (baseline)
                "mean_log_fog": [0.0],  # log(1.0) = 0.0
                "n_observations": [len(fog_df["round_id"].unique())],
                "rounds": [",".join(sorted(fog_df["round_id"].astype(str).unique()))],
                "run_ids": [",".join(sorted(fog_df["run_id"].astype(str).unique()))],
            })
            all_round = pd.concat([all_round, gox_row], ignore_index=True)
    
    all_round["rank_t50_desc"] = all_round["mean_t50_min"].rank(
        method="dense", ascending=False
    ).astype(int)
    all_round["rank_fog_desc"] = all_round["mean_fog"].rank(
        method="dense", ascending=False
    ).astype(int)
    return {"by_round": by_round, "all_round": all_round}


def _get_gox_mean_t50(fog_df: pd.DataFrame, fog_plate_aware_path: Optional[Path] = None) -> Optional[float]:
    """Extract mean GOx t50 across all rounds from fog_df or fog_plate_aware CSV."""
    # Try to get GOx t50 from gox_t50_used_min column if available in fog_df
    if "gox_t50_used_min" in fog_df.columns:
        gox_t50 = pd.to_numeric(fog_df["gox_t50_used_min"], errors="coerce")
        gox_t50_finite = gox_t50[np.isfinite(gox_t50)]
        if len(gox_t50_finite) > 0:
            return float(gox_t50_finite.mean())
    
    # Try to load from fog_plate_aware CSV directly if fog_plate_aware_path is provided
    # (the CSV might have gox_t50_used_min column even if fog_df doesn't)
    if fog_plate_aware_path is not None and fog_plate_aware_path.is_file():
        try:
            raw_df = pd.read_csv(fog_plate_aware_path)
            if "gox_t50_used_min" in raw_df.columns:
                gox_t50 = pd.to_numeric(raw_df["gox_t50_used_min"], errors="coerce")
                gox_t50_finite = gox_t50[np.isfinite(gox_t50)]
                if len(gox_t50_finite) > 0:
                    return float(gox_t50_finite.mean())
        except Exception:
            pass
    
    return None


def _latest_round_gox_baseline(fog_df: pd.DataFrame) -> dict[str, Any]:
    if fog_df.empty or "round_id" not in fog_df.columns:
        return {
            "last_round_id": "",
            "last_round_gox_t50_min_median": np.nan,
            "last_round_gox_t50_min_mean": np.nan,
            "last_round_gox_t50_n": 0,
        }
    rounds = sorted(fog_df["round_id"].astype(str).unique(), key=_natural_round_key)
    if not rounds:
        return {
            "last_round_id": "",
            "last_round_gox_t50_min_median": np.nan,
            "last_round_gox_t50_min_mean": np.nan,
            "last_round_gox_t50_n": 0,
        }
    last_round = rounds[-1]
    sub = fog_df[fog_df["round_id"].astype(str) == str(last_round)].copy()
    if "gox_t50_used_min" not in sub.columns:
        return {
            "last_round_id": str(last_round),
            "last_round_gox_t50_min_median": np.nan,
            "last_round_gox_t50_min_mean": np.nan,
            "last_round_gox_t50_n": 0,
        }
    vals = pd.to_numeric(sub["gox_t50_used_min"], errors="coerce")
    vals = vals[np.isfinite(vals)].to_numpy(dtype=float)
    if vals.size == 0:
        return {
            "last_round_id": str(last_round),
            "last_round_gox_t50_min_median": np.nan,
            "last_round_gox_t50_min_mean": np.nan,
            "last_round_gox_t50_n": 0,
        }
    return {
        "last_round_id": str(last_round),
        "last_round_gox_t50_min_median": float(np.nanmedian(vals)),
        "last_round_gox_t50_min_mean": float(np.nanmean(vals)),
        "last_round_gox_t50_n": int(vals.size),
    }


def _build_next_experiment_topk_table(
    selected: pd.DataFrame,
    *,
    bo_run_id: str,
    top_k: int,
    baseline: dict[str, Any],
    exploration_ratio: float,
    priority_weight_fog: float,
    priority_weight_t50: float,
    priority_weight_ei: float,
) -> pd.DataFrame:
    sel_full = selected.copy().sort_values("selection_order").copy()
    k = int(max(1, top_k))
    n_avail = len(sel_full)
    if n_avail == 0:
        return sel_full

    # Build a mixed top-k recommendation list (exploit + explore), then fill by priority.
    n_target = min(k, n_avail)
    explore_target = int(round(n_target * float(np.clip(exploration_ratio, 0.0, 1.0))))
    if n_target >= 3 and explore_target == 0:
        explore_target = 1
    explore_target = min(explore_target, n_target - 1) if n_target > 1 else 0
    exploit_target = n_target - explore_target

    is_explore = sel_full["selection_reason"].astype(str).str.contains("explore", case=False, regex=True)
    exploit_pool = sel_full[~is_explore].copy()
    explore_pool = sel_full[is_explore].copy()

    picked = []
    picked.extend(exploit_pool.head(exploit_target).index.tolist())
    picked.extend(explore_pool.head(explore_target).index.tolist())
    if len(picked) < n_target:
        for idx in sel_full.index.tolist():
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= n_target:
                break

    sel = sel_full.loc[picked].copy().sort_values("selection_order")
    if sel.empty:
        return sel

    gox_base = float(baseline.get("last_round_gox_t50_min_median", np.nan))
    if np.isfinite(gox_base):
        sel["pred_t50_min_mean_vs_last_round_gox"] = sel["pred_fog_mean"].astype(float) * gox_base
        sel["pred_t50_min_median_vs_last_round_gox"] = sel["pred_fog_median"].astype(float) * gox_base
    else:
        sel["pred_t50_min_mean_vs_last_round_gox"] = np.nan
        sel["pred_t50_min_median_vs_last_round_gox"] = np.nan

    # Uncertainty-aware estimates (log_fog ~ Normal(mu, sigma)).
    mu = sel["pred_log_fog_mean"].astype(float).to_numpy()
    sd = np.maximum(sel["pred_log_fog_std"].astype(float).to_numpy(), EPS)
    z95 = 1.959963984540054
    sel["pred_fog_lower95"] = np.exp(mu - z95 * sd)
    sel["pred_fog_upper95"] = np.exp(mu + z95 * sd)
    sel["prob_fog_gt_1"] = 1.0 - norm.cdf((0.0 - mu) / sd)
    sel["robust_fog_score"] = sel["pred_fog_mean"].astype(float) * sel["prob_fog_gt_1"].astype(float)

    if np.isfinite(gox_base):
        sel["pred_t50_min_lower95_vs_last_round_gox"] = sel["pred_fog_lower95"].astype(float) * gox_base
        sel["pred_t50_min_upper95_vs_last_round_gox"] = sel["pred_fog_upper95"].astype(float) * gox_base
        sel["robust_t50_score"] = sel["pred_t50_min_mean_vs_last_round_gox"].astype(float) * sel["prob_fog_gt_1"].astype(float)
    else:
        sel["pred_t50_min_lower95_vs_last_round_gox"] = np.nan
        sel["pred_t50_min_upper95_vs_last_round_gox"] = np.nan
        sel["robust_t50_score"] = np.nan

    sel["pred_activity_gain_pct_vs_gox_mean"] = (sel["pred_fog_mean"].astype(float) - 1.0) * 100.0
    sel["pred_activity_gain_pct_vs_gox_median"] = (sel["pred_fog_median"].astype(float) - 1.0) * 100.0

    sel["last_round_id"] = str(baseline.get("last_round_id", ""))
    sel["last_round_gox_t50_min_median"] = float(baseline.get("last_round_gox_t50_min_median", np.nan))
    sel["last_round_gox_t50_min_mean"] = float(baseline.get("last_round_gox_t50_min_mean", np.nan))
    sel["last_round_gox_t50_n"] = int(baseline.get("last_round_gox_t50_n", 0))
    sel["proposal_policy"] = "mixed_exploit_explore"
    sel["selection_priority_note"] = "Low order is higher priority."

    # Practical priority score: prioritize expected FoG/t50 gain, keep EI as secondary.
    def _minmax_norm(v: pd.Series) -> pd.Series:
        x = pd.to_numeric(v, errors="coerce").astype(float)
        finite = np.isfinite(x)
        if not finite.any():
            return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
        vmin = float(np.nanmin(x[finite]))
        vmax = float(np.nanmax(x[finite]))
        if vmax - vmin <= 1e-12:
            out = np.full(len(x), 0.5, dtype=float)
        else:
            out = (x.to_numpy(dtype=float) - vmin) / (vmax - vmin)
            out[~np.isfinite(out)] = 0.0
        return pd.Series(out, index=x.index)

    w_fog = float(max(priority_weight_fog, 0.0))
    w_t50 = float(max(priority_weight_t50, 0.0))
    w_ei = float(max(priority_weight_ei, 0.0))
    w_sum = w_fog + w_t50 + w_ei
    if w_sum <= 0.0:
        w_fog, w_t50, w_ei = 0.45, 0.45, 0.10
        w_sum = 1.0
    w_fog, w_t50, w_ei = (w_fog / w_sum), (w_t50 / w_sum), (w_ei / w_sum)

    norm_fog = _minmax_norm(sel["robust_fog_score"])
    if np.isfinite(sel["robust_t50_score"]).any():
        norm_t50 = _minmax_norm(sel["robust_t50_score"])
    else:
        norm_t50 = norm_fog.copy()
    norm_ei = _minmax_norm(sel["ei"])

    sel["priority_score"] = 100.0 * (
        w_fog * norm_fog + w_t50 * norm_t50 + w_ei * norm_ei
    )
    sel["priority_rank"] = (
        sel["priority_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    sel["recommended_top3"] = (sel["priority_rank"] <= 3).astype(int)
    sel["priority_weight_fog"] = w_fog
    sel["priority_weight_t50"] = w_t50
    sel["priority_weight_ei"] = w_ei

    keep_cols = [
        "frac_MPC",
        "frac_BMA",
        "frac_MTAC",
        "x",
        "y",
        "priority_rank",
        "priority_score",
        "recommended_top3",
        "selection_order",
        "selection_reason",
        "ei",
        "ucb",
        "pred_log_fog_mean",
        "pred_log_fog_std",
        "pred_fog_mean",
        "pred_fog_median",
        "pred_fog_lower95",
        "pred_fog_upper95",
        "prob_fog_gt_1",
        "robust_fog_score",
        "pred_activity_gain_pct_vs_gox_mean",
        "pred_activity_gain_pct_vs_gox_median",
        "pred_t50_min_mean_vs_last_round_gox",
        "pred_t50_min_median_vs_last_round_gox",
        "pred_t50_min_lower95_vs_last_round_gox",
        "pred_t50_min_upper95_vs_last_round_gox",
        "robust_t50_score",
        "last_round_id",
        "last_round_gox_t50_min_median",
        "last_round_gox_t50_min_mean",
        "last_round_gox_t50_n",
        "priority_weight_fog",
        "priority_weight_t50",
        "priority_weight_ei",
        "proposal_policy",
        "selection_priority_note",
    ]
    sel = sel[keep_cols].copy().sort_values(
        ["priority_rank", "selection_order"], ascending=[True, True]
    )
    sel.insert(0, "run_id", bo_run_id)
    sel.insert(1, "bo_run_id", bo_run_id)
    return sel


def _add_gp_predictions_to_simplex_frame(
    df: pd.DataFrame,
    model: Any,
    learning_df: pd.DataFrame,
    cfg: BOConfig,
) -> pd.DataFrame:
    """
    Add GP predictions (pred_log_fog_mean, pred_log_fog_std, ei, ucb, etc.) to a simplex-shaped frame.
    df must have frac_MPC, frac_BMA, frac_MTAC and (for 2D GP) x, y from _xy_from_frac.
    Returns a copy of df with prediction columns added.
    """
    out = df.copy()
    if len(model.length_scale) == 3:
        X_pred = out[["frac_MPC", "frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    elif getattr(cfg, "use_bma_mtac_coords", False):
        X_pred = out[["frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    else:
        X_pred = out[["x", "y"]].to_numpy(dtype=float)
    mu, std = model.predict(X_pred)
    best = float(np.nanmax(learning_df["log_fog_corrected"].to_numpy(dtype=float)))
    out["pred_log_fog_mean"] = mu
    out["pred_log_fog_std"] = std
    out["pred_fog_median"] = np.exp(mu)
    out["pred_fog_mean"] = np.exp(mu + 0.5 * (std ** 2))
    out["ei"] = _ei(mu, std, best, cfg.ei_xi)
    out["ucb"] = _ucb(mu, std, cfg.ucb_kappa)
    return out


def _build_candidate_frame(
    model: Any,
    learning_df: pd.DataFrame,
    cfg: BOConfig,
) -> pd.DataFrame:
    cand = _generate_simplex_grid(cfg.candidate_step, cfg.min_component)
    cand = _xy_from_frac(cand)
    cand = _add_gp_predictions_to_simplex_frame(cand, model, learning_df, cfg)
    # Constraints & distances for audit.
    s = cand[["frac_MPC", "frac_BMA", "frac_MTAC"]].sum(axis=1)
    cand["constraint_sum_ok"] = np.abs(s - 1.0) <= 1e-6
    cand["constraint_bounds_ok"] = (
        (cand["frac_MPC"] >= -1e-9)
        & (cand["frac_BMA"] >= -1e-9)
        & (cand["frac_MTAC"] >= -1e-9)
        & (cand["frac_MPC"] <= 1.0 + 1e-9)
        & (cand["frac_BMA"] <= 1.0 + 1e-9)
        & (cand["frac_MTAC"] <= 1.0 + 1e-9)
    )
    # Distance for min_dist_to_train: match coordinate system (x,y) or (frac_BMA, frac_MTAC).
    if getattr(cfg, "use_bma_mtac_coords", False) and len(model.length_scale) == 2:
        X_train_pt = learning_df[["frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
        cand_pt = cand[["frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    else:
        X_train_pt = learning_df[["x", "y"]].to_numpy(dtype=float)
        cand_pt = cand[["x", "y"]].to_numpy(dtype=float)
    diff = cand_pt[:, None, :] - X_train_pt[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    cand["min_dist_to_train"] = np.min(dist, axis=1)
    cand["selected"] = 0
    cand["selection_reason"] = ""
    cand["selection_order"] = np.nan
    return cand


def _build_plot_frame(
    model: Any,
    learning_df: pd.DataFrame,
    cfg: BOConfig,
) -> pd.DataFrame:
    """
    Build dense simplex predictions for visualization only.
    Keeps BO candidate selection logic unchanged.
    """
    step = float(max(min(cfg.ternary_plot_step, 0.2), 0.002))
    plot_df = _generate_simplex_grid(step=step, min_component=0.0)
    plot_df = _xy_from_frac(plot_df)
    return _add_gp_predictions_to_simplex_frame(plot_df, model, learning_df, cfg)


def _build_map_quality_report(cand: pd.DataFrame, cfg: BOConfig, gp: Any) -> dict:
    report: dict[str, Any] = {}
    z = pd.to_numeric(cand.get("pred_log_fog_mean"), errors="coerce").to_numpy(dtype=float)
    std = pd.to_numeric(cand.get("pred_log_fog_std"), errors="coerce").to_numpy(dtype=float)
    ei = pd.to_numeric(cand.get("ei"), errors="coerce").to_numpy(dtype=float)
    report["z_min"] = float(np.nanmin(z)) if np.isfinite(z).any() else np.nan
    report["z_max"] = float(np.nanmax(z)) if np.isfinite(z).any() else np.nan
    report["z_std"] = float(np.nanstd(z)) if np.isfinite(z).any() else np.nan
    report["std_mean"] = float(np.nanmean(std)) if np.isfinite(std).any() else np.nan
    report["ei_max"] = float(np.nanmax(ei)) if np.isfinite(ei).any() else np.nan
    report["ei_collapsed"] = bool(report["ei_max"] < 1e-10) if np.isfinite(report["ei_max"]) else True

    if getattr(cfg, "use_bma_mtac_coords", False):
        a_spans: list[float] = []
        b_spans: list[float] = []
        for _, sub in cand.groupby("frac_MTAC"):
            if len(sub) > 1:
                a_spans.append(float(sub["pred_log_fog_mean"].max() - sub["pred_log_fog_mean"].min()))
        for _, sub in cand.groupby("frac_BMA"):
            if len(sub) > 1:
                b_spans.append(float(sub["pred_log_fog_mean"].max() - sub["pred_log_fog_mean"].min()))
        report["avg_span_fixed_mtac_vary_bma"] = float(np.nanmean(a_spans)) if a_spans else np.nan
        report["avg_span_fixed_bma_vary_mtac"] = float(np.nanmean(b_spans)) if b_spans else np.nan
    else:
        a_spans = []
        b_spans = []
        for _, sub in cand.groupby("y"):
            if len(sub) > 1:
                a_spans.append(float(sub["pred_log_fog_mean"].max() - sub["pred_log_fog_mean"].min()))
        for _, sub in cand.groupby("x"):
            if len(sub) > 1:
                b_spans.append(float(sub["pred_log_fog_mean"].max() - sub["pred_log_fog_mean"].min()))
        report["avg_span_fixed_y_vary_x"] = float(np.nanmean(a_spans)) if a_spans else np.nan
        report["avg_span_fixed_x_vary_y"] = float(np.nanmean(b_spans)) if b_spans else np.nan

    ls = np.asarray(getattr(gp, "length_scale", []), dtype=float)
    report["length_scale_min"] = float(np.nanmin(ls)) if ls.size else np.nan
    report["length_scale_max"] = float(np.nanmax(ls)) if ls.size else np.nan
    report["length_scale_lower_bound_hit"] = bool(np.any(ls <= 0.051)) if ls.size else False
    report["length_scale_upper_bound_hit"] = bool(np.any(ls >= 7.9)) if ls.size else False
    noise_std = float(getattr(gp, "noise_std", np.nan))
    report["noise_std"] = noise_std
    report["noise_std_upper_bound_hit"] = bool(noise_std >= 0.99) if np.isfinite(noise_std) else False
    return report


def _design_coverage_report(
    learning_df: pd.DataFrame,
    *,
    use_bma_mtac_coords: bool = False,
) -> dict[str, Any]:
    """
    Summarize coverage of the 2D design space used by BO.

    This is useful for sparse-start campaigns (e.g., PMBTA-1..5 ± controls) where
    striping artifacts can appear if one axis is under-sampled at fixed levels
    of the other axis.
    """
    if learning_df.empty:
        return {
            "n_unique_design_points": 0,
            "axis_a_name": None,
            "axis_b_name": None,
            "n_axis_b_levels": 0,
            "n_axis_b_levels_with_single_axis_a": 0,
            "fraction_axis_b_levels_with_single_axis_a": np.nan,
            "n_axis_a_levels": 0,
            "n_axis_a_levels_with_single_axis_b": 0,
            "fraction_axis_a_levels_with_single_axis_b": np.nan,
            "striping_risk_high": False,
        }

    if use_bma_mtac_coords:
        a_col = "frac_BMA"
        b_col = "frac_MTAC"
    else:
        a_col = "x"
        b_col = "y"
    work = learning_df[[a_col, b_col]].copy()
    for c in [a_col, b_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[np.isfinite(work[a_col]) & np.isfinite(work[b_col])].copy()
    if work.empty:
        return {
            "n_unique_design_points": 0,
            "axis_a_name": a_col,
            "axis_b_name": b_col,
            "n_axis_b_levels": 0,
            "n_axis_b_levels_with_single_axis_a": 0,
            "fraction_axis_b_levels_with_single_axis_a": np.nan,
            "n_axis_a_levels": 0,
            "n_axis_a_levels_with_single_axis_b": 0,
            "fraction_axis_a_levels_with_single_axis_b": np.nan,
            "striping_risk_high": False,
        }

    # Round keys to avoid tiny floating jitter creating fake extra levels.
    key = work.round(6).drop_duplicates()
    n_unique = int(len(key))
    axis_b_levels = key.groupby(b_col)[a_col].nunique()
    axis_a_levels = key.groupby(a_col)[b_col].nunique()
    n_b = int(len(axis_b_levels))
    n_a = int(len(axis_a_levels))
    n_b_single = int((axis_b_levels <= 1).sum())
    n_a_single = int((axis_a_levels <= 1).sum())
    frac_b_single = float(n_b_single / n_b) if n_b > 0 else np.nan
    frac_a_single = float(n_a_single / n_a) if n_a > 0 else np.nan
    striping_risk = bool(
        n_unique <= 15
        and n_b >= 3
        and np.isfinite(frac_b_single)
        and (frac_b_single >= 0.6)
    )
    return {
        "n_unique_design_points": n_unique,
        "axis_a_name": a_col,
        "axis_b_name": b_col,
        "n_axis_b_levels": n_b,
        "n_axis_b_levels_with_single_axis_a": n_b_single,
        "fraction_axis_b_levels_with_single_axis_a": frac_b_single,
        "n_axis_a_levels": n_a,
        "n_axis_a_levels_with_single_axis_b": n_a_single,
        "fraction_axis_a_levels_with_single_axis_b": frac_a_single,
        "striping_risk_high": striping_risk,
    }


def _select_diverse_batch(
    cand: pd.DataFrame, learning_df: pd.DataFrame, cfg: BOConfig
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    counts = _resolve_batch_counts(cfg)
    n_total = counts["n_total"]
    n_unique = counts["n_unique"]
    n_explore = counts["n_explore"]
    n_exploit = counts["n_exploit"]
    n_anchor = counts["n_anchor"]
    n_replicate = counts["n_replicate"]

    coord_cols = (
        ["frac_BMA", "frac_MTAC"]
        if getattr(cfg, "use_bma_mtac_coords", False)
        else ["x", "y"]
    )
    n_unique_design_points = int(learning_df[coord_cols].drop_duplicates().shape[0])
    sparse_distance_mode = bool(
        n_unique_design_points
        <= int(max(getattr(cfg, "sparse_explore_max_unique_points", 15), 0))
    )
    explore_dist_weight = float(max(getattr(cfg, "sparse_explore_distance_weight", 0.75), 0.0))
    combo_dist_weight = float(max(getattr(cfg, "sparse_combo_distance_weight", 0.40), 0.0))

    selected_idx: list[int] = []
    selected_xy: list[np.ndarray] = []

    def _can_pick(row: pd.Series) -> bool:
        if not bool(row["constraint_sum_ok"]) or not bool(row["constraint_bounds_ok"]):
            return False
        if float(row["min_dist_to_train"]) < cfg.min_distance_to_train:
            return False
        if not selected_xy:
            return True
        p = row[coord_cols].to_numpy(dtype=float)
        d = [float(np.linalg.norm(p - q)) for q in selected_xy]
        return min(d) >= cfg.min_distance_between

    # Exploit by EI; when EI collapses to ~0, fallback to UCB ranking.
    ei_max = float(np.nanmax(cand["ei"].to_numpy(dtype=float)))
    exploit_metric = "ei" if ei_max >= 1e-8 else "ucb"
    exploit_reason = "exploit_ei" if exploit_metric == "ei" else "exploit_ucb_fallback"
    for idx, row in cand.sort_values(exploit_metric, ascending=False).iterrows():
        if len(selected_idx) >= n_exploit:
            break
        if _can_pick(row):
            selected_idx.append(int(idx))
            selected_xy.append(row[coord_cols].to_numpy(dtype=float))
            cand.loc[idx, "selection_reason"] = exploit_reason

    # Explore by uncertainty (std).
    if sparse_distance_mode:
        std_z = (cand["pred_log_fog_std"] - cand["pred_log_fog_std"].mean()) / (
            cand["pred_log_fog_std"].std(ddof=0) + EPS
        )
        dist_z = (cand["min_dist_to_train"] - cand["min_dist_to_train"].mean()) / (
            cand["min_dist_to_train"].std(ddof=0) + EPS
        )
        explore_score = std_z + explore_dist_weight * dist_z
        explore_ranked = cand.assign(explore_score=explore_score).sort_values("explore_score", ascending=False)
        explore_reason = "explore_std_plus_distance"
    else:
        explore_ranked = cand.sort_values("pred_log_fog_std", ascending=False)
        explore_reason = "explore_std"
    for idx, row in explore_ranked.iterrows():
        if len(selected_idx) >= (n_exploit + n_explore):
            break
        if idx in selected_idx:
            continue
        if _can_pick(row):
            selected_idx.append(int(idx))
            selected_xy.append(row[coord_cols].to_numpy(dtype=float))
            cand.loc[idx, "selection_reason"] = explore_reason

    # Anchor: either exact compositions (added in run_bo) or nearest grid points here.
    anchor_targets = _anchor_targets_from_learning(
        learning_df,
        anchor_ids=cfg.anchor_polymer_ids,
        n_anchors=n_anchor,
    )
    if anchor_targets and not getattr(cfg, "use_exact_anchor_compositions", True):
        for target in anchor_targets:
            if len(selected_idx) >= (n_unique + n_anchor):
                break
            remain = cand.loc[~cand.index.isin(selected_idx)].copy()
            if remain.empty:
                break
            diff = remain[coord_cols].to_numpy(dtype=float) - np.array(
                [float(target[coord_cols[0]]), float(target[coord_cols[1]])], dtype=float
            )
            j = int(np.argmin(np.sum(diff * diff, axis=1)))
            idx = int(remain.index[j])
            selected_idx.append(idx)
            selected_xy.append(cand.loc[idx, coord_cols].to_numpy(dtype=float))
            cand.loc[idx, "selection_reason"] = f"anchor_{target['polymer_id']}"

    # Fill remainder (combined score) if still not enough for n_unique.
    if len(selected_idx) < n_unique:
        z_ei = (cand["ei"] - cand["ei"].mean()) / (cand["ei"].std(ddof=0) + EPS)
        z_std = (cand["pred_log_fog_std"] - cand["pred_log_fog_std"].mean()) / (
            cand["pred_log_fog_std"].std(ddof=0) + EPS
        )
        z_dist = (cand["min_dist_to_train"] - cand["min_dist_to_train"].mean()) / (
            cand["min_dist_to_train"].std(ddof=0) + EPS
        )
        combo = (1.0 - cfg.exploration_ratio) * z_ei + cfg.exploration_ratio * z_std
        if sparse_distance_mode and combo_dist_weight > 0.0:
            combo = combo + combo_dist_weight * z_dist
        ranked = cand.assign(combo=combo).sort_values("combo", ascending=False)
        for idx, row in ranked.iterrows():
            if len(selected_idx) >= n_unique:
                break
            if idx in selected_idx:
                continue
            if _can_pick(row):
                selected_idx.append(int(idx))
                selected_xy.append(row[coord_cols].to_numpy(dtype=float))
                cand.loc[idx, "selection_reason"] = (
                    "balanced_combo_plus_distance" if sparse_distance_mode else "balanced_combo"
                )

    for i, idx in enumerate(selected_idx, start=1):
        cand.loc[idx, "selected"] = 1
        cand.loc[idx, "selection_order"] = i

    cand["policy_exploration_ratio"] = cfg.exploration_ratio
    cand["policy_n_total"] = n_total
    cand["policy_n_unique"] = n_unique
    cand["policy_n_exploit"] = n_exploit
    cand["policy_n_explore"] = n_explore
    cand["policy_n_anchor"] = n_anchor
    cand["policy_n_replicate"] = n_replicate
    cand["policy_sparse_distance_mode"] = bool(sparse_distance_mode)
    cand["policy_sparse_explore_distance_weight"] = float(explore_dist_weight)
    cand["policy_sparse_combo_distance_weight"] = float(combo_dist_weight)
    return cand, anchor_targets


def _plot_ternary_field(
    cand: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    cbar_label: str,
    out_path: Path,
    std_color_gamma: Optional[float] = None,
) -> None:
    mpc = cand["frac_MPC"].to_numpy(dtype=float)
    bma = cand["frac_BMA"].to_numpy(dtype=float)
    mtac = cand["frac_MTAC"].to_numpy(dtype=float)
    z = cand[value_col].to_numpy(dtype=float)
    tx, ty = _ternary_xy_from_frac(mpc, bma, mtac)
    tick_fracs = np.linspace(0.0, 1.0, 6)

    def _pt(frac_mpc: float, frac_bma: float, frac_mtac: float) -> tuple[float, float]:
        px, py = _ternary_xy_from_frac(
            np.array([frac_mpc], dtype=float),
            np.array([frac_bma], dtype=float),
            np.array([frac_mtac], dtype=float),
        )
        return float(px[0]), float(py[0])

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        tri = mtri.Triangulation(tx, ty)
        # Calculate vmin/vmax dynamically from data
        # For std, use percentile-based vmin to avoid "holes" at observed points
        # This makes the visualization smoother and more intuitive
        valid_z = z[np.isfinite(z)]
        if len(valid_z) == 0:
            vmin, vmax = 0.0, 1.0
        elif value_col == "pred_log_fog_std":
            # For std: use 10th percentile as vmin to avoid "holes" at observed points
            # This creates a smoother gradient visualization with gentler contours
            vmin = float(np.percentile(valid_z, 10))
            vmax = float(np.nanmax(valid_z))
            # Ensure vmin < vmax
            if vmin >= vmax:
                vmin = float(np.nanmin(valid_z))
        else:
            # For other plots: use min/max
            vmin = float(np.nanmin(valid_z))
            vmax = float(np.nanmax(valid_z))
        # Ensure vmin/vmax are valid
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        ternary_cmap = {
            "pred_log_fog_mean": _XY_PANEL_CMAPS["mean"],
            "pred_log_fog_std": _XY_PANEL_CMAPS["std"],
            "ei": _XY_PANEL_CMAPS["ei"],
            "ucb": _XY_PANEL_CMAPS["ucb"],
        }
        cmap_name = ternary_cmap.get(value_col, "viridis")
        if value_col == "pred_log_fog_std" and std_color_gamma is not None:
            # Use gentler gamma for std to show gradient better (not just yellow)
            # Default std_color_gamma is 8.0, but we'll use 2.0 for better gradient
            gamma_used = min(std_color_gamma, 2.0) if std_color_gamma > 2.0 else std_color_gamma
            norm = mcolors.PowerNorm(gamma=gamma_used, vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cs = ax.tripcolor(
            tri,
            z,
            cmap=cmap_name,
            norm=norm,
            shading="gouraud",
        )
        # Thin white contour lines with dark stroke (visible on both light and dark; only when range is non-trivial).
        # Use quantile-based levels to ensure contours span the full range, not just near observed points.
        # For all plots, use more levels and higher transparency for better gradient visibility.
        # For sparse data, also check relative range (range / abs(mean)) to allow contours even when absolute range is small.
        # For EI, use a more lenient threshold since EI values can be small but still meaningful.
        _range = vmax - vmin
        _range_rel = _range / (abs(vmax) + abs(vmin) + EPS) if (vmax != 0 or vmin != 0) else 0.0
        # Draw contours if absolute range > threshold OR relative range > 1e-6 (for sparse data with small absolute values)
        # For EI, use a more lenient threshold to ensure contours are drawn even when values are very small
        # EI values can be extremely small (e.g., 1e-265), so we need very lenient thresholds
        if value_col == "ei":
            # For EI: use very lenient thresholds and always try to draw contours if there's any variation
            contour_threshold_abs = 1e-15  # Very lenient absolute threshold for EI
            contour_threshold_rel = 1e-12  # Very lenient relative threshold for EI
        else:
            contour_threshold_abs = _SURROGATE_CONTOUR_MIN_RANGE
            contour_threshold_rel = 1e-6
        if _range > contour_threshold_abs or _range_rel > contour_threshold_rel:
            # Use quantiles to distribute contour levels across the full data range.
            # This prevents contours from clustering only near observed points.
            valid_z = z[np.isfinite(z)]
            if len(valid_z) > 0:
                # Use moderate number of contour levels (not too dense) so gradient colors are visible
                n_levels = _SURROGATE_CONTOUR_LEVELS  # Use base level count (18), not doubled
                q_levels = np.linspace(0.05, 0.95, n_levels)
                _levs = np.quantile(valid_z, q_levels)
            else:
                n_levels = _SURROGATE_CONTOUR_LEVELS
                _levs = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
            _z = np.where(np.isfinite(z), z, np.nanmean(np.where(np.isfinite(z), z, np.nan)))
            try:
                # Higher transparency for all contours to show gradient better
                contour_alpha = 0.95  # High transparency for all plots
                tc = ax.tricontour(
                    tri,
                    _z,
                    levels=_levs,
                    colors=_SURROGATE_CONTOUR_COLOR,
                    linewidths=_SURROGATE_CONTOUR_LW,
                    alpha=contour_alpha,
                    zorder=2,
                )
                # TriContourSet has no .collections; set path_effects on the contour set itself.
                tc.set_path_effects([
                    mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                    mpatheffects.Normal(),
                ])
            except (ValueError, TypeError):
                pass

        # Triangle boundary.
        p_mtac = np.array([0.0, 0.0])
        p_bma = np.array([1.0, 0.0])
        p_mpc = np.array([0.5, SQRT3_OVER_2])
        ax.plot(
            [p_mtac[0], p_bma[0], p_mpc[0], p_mtac[0]],
            [p_mtac[1], p_bma[1], p_mpc[1], p_mtac[1]],
            color="black",
            lw=0.8,
        )

        # Observed points.
        omtpc = observed["frac_MPC"].to_numpy(dtype=float)
        obma = observed["frac_BMA"].to_numpy(dtype=float)
        omtac = observed["frac_MTAC"].to_numpy(dtype=float)
        ox, oy = _ternary_xy_from_frac(omtpc, obma, omtac)
        ax.scatter(
            ox,
            oy,
            s=16,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )

        ax.set_title(title)
        ax.set_xlim(-0.13, 1.13)
        ax.set_ylim(-0.14, SQRT3_OVER_2 + 0.07)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        cbar = fig.colorbar(cs, ax=ax, fraction=0.045, pad=0.08)
        cbar.set_label(cbar_label)
        # Apply math formatter for proper scientific notation (e.g., 10^{-265} instead of 1e-265)
        cbar_formatter = mticker.FuncFormatter(lambda x, p: _math_scalar_formatter(x, p))
        cbar.ax.yaxis.set_major_formatter(cbar_formatter)

        # Corner labels (English only, clean format).
        ax.text(0.5, SQRT3_OVER_2 + 0.03, "PMPC", ha="center", va="bottom")
        ax.text(-0.03, -0.03, "PMTAC", ha="right", va="top")
        ax.text(1.03, -0.03, "PBMA", ha="left", va="top")

        # Base: BMA 0 -> 100 (left to right). Tick labels 0, 20, 40, 60, 80, 100.
        for t in tick_fracs:
            px, py = _pt(0.0, t, 1.0 - t)
            ax.plot([px, px], [py, py - 0.012], color="black", lw=0.45)
            ax.text(px, py - 0.026, f"{int(round(t * 100))}", ha="center", va="top", fontsize=5.5)

        # Left edge: MTAC 100 at bottom \u2192 0 at top (vertex = 100% MTAC).
        for t in tick_fracs[1:-1]:
            px, py = _pt(t, 0.0, 1.0 - t)
            mtac_pct = int(round((1.0 - t) * 100))
            ax.plot([px, px - 0.010], [py, py], color="black", lw=0.45)
            ax.text(px - 0.017, py, f"{mtac_pct}", ha="right", va="center", fontsize=5.5)

        # Right edge: MPC 0 at bottom \u2192 100 at top (vertex = 100% MPC).
        for t in tick_fracs[1:-1]:
            px, py = _pt(t, 1.0 - t, 0.0)
            mpc_pct = int(round(t * 100))
            ax.plot([px, px + 0.010], [py, py], color="black", lw=0.45)
            ax.text(px + 0.017, py, f"{mpc_pct}", ha="left", va="center", fontsize=5.5)

        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


# Colormaps for xy 2x2 panels (same style as Ioka: mean/std/ei/ucb).
_XY_PANEL_CMAPS = {"mean": "viridis", "std": "plasma", "ei": "inferno", "ucb": "cividis"}
_EDGE_MARGIN = 0.02
_N_LEVELS_XY = 24
# Paper-grade: ~90 mm per column → 2 columns ≈ 7.1 inch width (Figure-rules.mdc).
_XY_2X2_FIGSIZE = (7.2, 6.6)
_CONTOUR_LINEWIDTH = 0.35  # 0.25–1.0 pt (Figure-rules.mdc).
# Surrogate map: thin light contour lines (white with thin dark stroke so visible on any colormap).
_SURROGATE_CONTOUR_LEVELS = 9  # Reduced from 18 to make contours less dense (half the original)
_SURROGATE_CONTOUR_COLOR = "white"
_SURROGATE_CONTOUR_LW = 0.5
_SURROGATE_CONTOUR_ALPHA = 0.85
_SURROGATE_CONTOUR_STROKE_LW = 0.65  # Dark outline so white line is visible on light and dark.
_SURROGATE_CONTOUR_STROKE_COLOR = "0.25"
_SURROGATE_CONTOUR_MIN_RANGE = 1e-10  # Only draw contours when (vmax - vmin) > this. Reduced for sparse data.
_SCATTER_MARKER_PT = 4.5  # minimal, distinguishable when printed.
_SCATTER_EDGE_PT = 0.6   # lines 0.6–0.9 pt.
_STD_COLOR_GAMMA = 8.0  # Strongly stretch sparse low-std pockets so std maps are informative.

# Math axis labels: fraction form, [·] = concentration/ratio (English only).
_XY_LABEL_X = r"$x = \frac{[\mathrm{BMA}]}{[\mathrm{BMA}]+[\mathrm{MTAC}]}$"
_XY_LABEL_Y = r"$y = \frac{[\mathrm{BMA}]+[\mathrm{MTAC}]}{[\mathrm{MPC}]+[\mathrm{BMA}]+[\mathrm{MTAC}]}$"

# rcParams for xy 2x2: mathtext (fractions, superscripts) with one font family.
# DejaVu Sans supports mathtext well; use it so text and math match (no Arial for this figure).
_XY_MATH_RC = {
    "mathtext.fontset": "dejavusans",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
}


def _math_scalar_formatter(x: float, pos: Optional[int] = None) -> str:
    """Format scalar for axes/colorbar: avoid '1e-10', use proper exponent notation (e.g. $10^{-10}$)."""
    if not np.isfinite(x):
        return ""
    if x == 0:
        return "$0$"
    abs_x = abs(x)
    if abs_x >= 1e4 or (abs_x <= 1e-3 and abs_x > 0):
        exp = int(np.floor(np.log10(abs_x)))
        mant = x / (10.0 ** exp)
        if abs(mant - 1.0) < 0.01:
            return r"$10^{%d}$" % exp
        if abs(mant + 1.0) < 0.01:
            return r"$-10^{%d}$" % exp
        return r"$%g \times 10^{%d}$" % (mant, exp)
    return "$%g$" % x


def _plot_xy_2x2_panels(
    model: GPModel2D,
    learning_df: pd.DataFrame,
    cfg: BOConfig,
    bo_run_id: str,
    out_path: Path,
    acq_mode_used: Optional[str] = None,
) -> None:
    """
    Plot 2x2 xy-plane panels: Mean (predicted log FoG), Std, EI, UCB.
    Axes: x = BMA/(BMA+MTAC), y = (BMA+MTAC)/(MPC+BMA+MTAC).
    Overlays observed points (per-polymer mean x,y). Paper-grade, English only.
    Follows Figure-rules.mdc and Matplotlib-style-implementation.mdc.
    Finer surrogate_map_xy_grid → smoother gradient at basin boundaries (low std/EI near observed points).
    """
    n_grid = int(getattr(cfg, "surrogate_map_xy_grid", 361))
    x_grid = np.linspace(0.0, 1.0, n_grid)
    y_grid = np.linspace(0.0, 1.0, n_grid)
    # Grid: meshgrid(x,y) gives X1[i,j]=x_grid[j], X2[i,j]=y_grid[i] so (row, col) = (y_idx, x_idx).
    # grid order: (x,y) with x varying slow, y fast -> grid[k] = (x_grid[k//n], y_grid[k%n]).
    # So mu.reshape gives Z[i,j] = value at (x_grid[i], y_grid[j]) with i=row, j=col.
    # pcolormesh: 1st index = vertical (y), 2nd = horizontal (x). So we need Z_plot[y_idx,x_idx].
    # Use (X1, X2) so (i,j) -> (x_grid[j], y_grid[i]); then Z.T[i,j] = Z[j,i] = value at (x_grid[j], y_grid[i]).
    X1, X2 = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([X1.T.ravel(), X2.T.ravel()])  # (x_i, y_j) order for predict

    mu, std = model.predict(grid)
    best = float(np.nanmax(learning_df["log_fog_corrected"].to_numpy(dtype=float)))
    ei = _ei(mu, std, best, cfg.ei_xi)
    ucb = _ucb(mu, std, cfg.ucb_kappa)

    # Z[i,j] = value at (x_grid[i], y_grid[j]). For plot: row=y, col=x -> use Z.T and (X1, X2).
    MU = mu.reshape((n_grid, n_grid))
    SD = std.reshape((n_grid, n_grid))
    EI = ei.reshape((n_grid, n_grid))
    UCB = ucb.reshape((n_grid, n_grid))

    # Group by (polymer_id, round_id) if round_id exists, otherwise by polymer_id only
    # This ensures each round's data is plotted separately
    if "round_id" in learning_df.columns:
        group_cols = ["polymer_id", "round_id"]
    else:
        group_cols = ["polymer_id"]
    observed = (
        learning_df.groupby(group_cols, as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"))
    )
    ox = observed["x"].to_numpy(dtype=float)
    oy = observed["y"].to_numpy(dtype=float)

    def _vmin_vmax(Z: np.ndarray, is_std: bool = False) -> tuple[float, float]:
        """
        Calculate vmin/vmax dynamically from data.
        For std plots, use percentile-based vmin to avoid "holes" at observed points.
        """
        valid = np.isfinite(Z)
        if not np.any(valid):
            return 0.0, 1.0
        z = Z[valid]
        if is_std:
            # For std: use 10th percentile as vmin to avoid "holes" at observed points
            # This creates a smoother gradient visualization with gentler contours
            vmin = float(np.percentile(z, 10))
            vmax = float(np.nanmax(z))
            if vmin >= vmax:
                vmin = float(np.nanmin(z))
            return vmin, vmax
        else:
            return float(np.nanmin(z)), float(np.nanmax(z))

    # Colorbar labels: math notation for log(FoG); [·] = concentration/ratio.
    # Use actual acquisition mode for consistent labeling
    if acq_mode_used == "ei_ucb_fallback":
        ei_title = "EI (no promising region)"
        ei_label = "Expected Improvement (no promising region)"
    else:
        ei_title = "EI"
        ei_label = "Expected Improvement"
    panels = [
        (MU, "Mean", _XY_PANEL_CMAPS["mean"], r"Predicted mean $\log(\mathrm{FoG})$"),
        (SD, "Std", _XY_PANEL_CMAPS["std"], r"Predictive std $\log(\mathrm{FoG})$"),
        (EI, ei_title, _XY_PANEL_CMAPS["ei"], ei_label),
        (UCB, "UCB", _XY_PANEL_CMAPS["ucb"], "Upper Confidence Bound"),
    ]

    style = {**apply_paper_style(), **_XY_MATH_RC}
    with plt.rc_context(style):
        fig, ax = plt.subplots(2, 2, figsize=_XY_2X2_FIGSIZE, constrained_layout=True)
        cbar_formatter = mticker.FuncFormatter(lambda x, p: _math_scalar_formatter(x, p))
        std_gamma = getattr(cfg, "std_color_gamma", 2.0)  # Reduced from 8.0 to 2.0 for better gradient
        for axi, (Z, title, cmap, cbar_label) in zip(ax.flat, panels):
            # Calculate vmin/vmax dynamically, using percentile-based vmin for std
            vmin, vmax = _vmin_vmax(Z, is_std=(title == "Std"))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0
            if title == "Std":
                # Use gentler gamma for std to show gradient better (not just yellow)
                norm = mcolors.PowerNorm(gamma=std_gamma, vmin=vmin, vmax=vmax) if std_gamma is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            # Smooth continuous coloring.
            cs = axi.pcolormesh(X1, X2, Z.T, cmap=cmap, norm=norm, shading="gouraud")
            # Contour lines (white + dark stroke; only when range is non-trivial).
            # Use quantile-based levels to ensure contours span the full range.
            # For all panels, use more levels and higher transparency for better gradient visibility.
            # For sparse data, also check relative range (range / abs(mean)) to allow contours even when absolute range is small.
            _vmin, _vmax = _vmin_vmax(Z, is_std=(title == "Std"))
            _range_abs = _vmax - _vmin
            _range_rel = _range_abs / (abs(_vmax) + abs(_vmin) + EPS) if (_vmax != 0 or _vmin != 0) else 0.0
            # Draw contours if absolute range > threshold OR relative range > 1e-6 (for sparse data with small absolute values)
            # For EI, use very lenient thresholds to ensure contours are drawn even when values are very small
            if title == "EI":
                # For EI: use very lenient thresholds and always try to draw contours if there's any variation
                contour_threshold_abs = 1e-15  # Very lenient absolute threshold for EI
                contour_threshold_rel = 1e-12  # Very lenient relative threshold for EI
            else:
                contour_threshold_abs = _SURROGATE_CONTOUR_MIN_RANGE
                contour_threshold_rel = 1e-6
            if _range_abs > contour_threshold_abs or _range_rel > contour_threshold_rel:
                valid_z = Z[np.isfinite(Z)]
                if len(valid_z) > 0:
                    # Use moderate number of contour levels (not too dense) so gradient colors are visible
                    n_levels = _SURROGATE_CONTOUR_LEVELS  # Use base level count (18), not doubled
                    q_levels = np.linspace(0.05, 0.95, n_levels)
                    _levs = np.quantile(valid_z, q_levels)
                else:
                    n_levels = _SURROGATE_CONTOUR_LEVELS * 2
                    _levs = np.linspace(_vmin, _vmax, n_levels + 2)[1:-1]
                try:
                    # Higher transparency for all contours to show gradient better
                    contour_alpha = 0.95  # High transparency for all panels
                    qc = axi.contour(
                        X1, X2, Z.T,
                        levels=_levs,
                        colors=_SURROGATE_CONTOUR_COLOR,
                        linewidths=_SURROGATE_CONTOUR_LW,
                        alpha=contour_alpha,
                        zorder=2,
                    )
                    qc.set_path_effects([
                        mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                        mpatheffects.Normal(),
                    ])
                except (ValueError, TypeError):
                    pass
            # Minimal markers, paper-grade (Figure-rules: 印刷で判別できる最小限).
            axi.scatter(
                ox, oy,
                s=_SCATTER_MARKER_PT ** 2,
                facecolor="white",
                edgecolor="black",
                linewidths=_SCATTER_EDGE_PT,
                zorder=3,
            )
            axi.set_xlim(-_EDGE_MARGIN, 1.0 + _EDGE_MARGIN)
            axi.set_ylim(-_EDGE_MARGIN, 1.0 + _EDGE_MARGIN)
            axi.set_aspect("equal", adjustable="box")
            axi.set_xlabel(_XY_LABEL_X)
            axi.set_ylabel(_XY_LABEL_Y)
            axi.set_title(title)
            axi.xaxis.set_major_formatter(cbar_formatter)
            axi.yaxis.set_major_formatter(cbar_formatter)
            cbar = fig.colorbar(cs, ax=axi, fraction=0.046, pad=0.03, label=cbar_label)
            # Apply math formatter to colorbar for proper scientific notation
            cbar.ax.yaxis.set_major_formatter(cbar_formatter)

        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_bma_mtac_2x2_panels(
    model: GPModel2D,
    learning_df: pd.DataFrame,
    cfg: BOConfig,
    bo_run_id: str,
    out_path: Path,
) -> None:
    """
    Plot 2x2 (BMA, MTAC) panels: Mean, Std, EI, UCB.
    Valid region: BMA >= 0, MTAC >= 0, BMA + MTAC <= 1. Use with ternary map for composition.
    """
    n_grid = int(getattr(cfg, "surrogate_map_xy_grid", 361))
    bma_grid = np.linspace(0.0, 1.0, n_grid)
    mtac_grid = np.linspace(0.0, 1.0, n_grid)
    B, M = np.meshgrid(bma_grid, mtac_grid)
    valid = (B + M <= 1.0 + 1e-9) & (B >= -1e-9) & (M >= -1e-9)
    X_pred = np.column_stack([B.ravel()[valid.ravel()], M.ravel()[valid.ravel()]])
    mu, std = model.predict(X_pred)
    best = float(np.nanmax(learning_df["log_fog_corrected"].to_numpy(dtype=float)))
    ei = _ei(mu, std, best, cfg.ei_xi)
    ucb = _ucb(mu, std, cfg.ucb_kappa)

    Z_mu = np.full(B.shape, np.nan, dtype=float)
    Z_std = np.full(B.shape, np.nan, dtype=float)
    Z_ei = np.full(B.shape, np.nan, dtype=float)
    Z_ucb = np.full(B.shape, np.nan, dtype=float)
    Z_mu.ravel()[valid.ravel()] = mu
    Z_std.ravel()[valid.ravel()] = std
    Z_ei.ravel()[valid.ravel()] = ei
    Z_ucb.ravel()[valid.ravel()] = ucb

    # Group by (polymer_id, round_id) if round_id exists, otherwise by polymer_id only
    # This ensures each round's data is plotted separately
    if "round_id" in learning_df.columns:
        group_cols = ["polymer_id", "round_id"]
    else:
        group_cols = ["polymer_id"]
    observed = (
        learning_df.groupby(group_cols, as_index=False)
        .agg(frac_BMA=("frac_BMA", "mean"), frac_MTAC=("frac_MTAC", "mean"))
    )
    obma = observed["frac_BMA"].to_numpy(dtype=float)
    omtac = observed["frac_MTAC"].to_numpy(dtype=float)

    def _vmin_vmax(Z: np.ndarray, is_std: bool = False) -> tuple[float, float]:
        """
        Calculate vmin/vmax dynamically from data.
        For std plots, use percentile-based vmin to avoid "holes" at observed points.
        """
        valid_z = np.isfinite(Z)
        if not np.any(valid_z):
            return 0.0, 1.0
        z = Z[valid_z]
        if is_std:
            # For std: use 10th percentile as vmin to avoid "holes" at observed points
            # This creates a smoother gradient visualization with gentler contours
            vmin = float(np.percentile(z, 10))
            vmax = float(np.nanmax(z))
            if vmin >= vmax:
                vmin = float(np.nanmin(z))
            return vmin, vmax
        else:
            return float(np.nanmin(z)), float(np.nanmax(z))

    panels = [
        (Z_mu, "Mean", _XY_PANEL_CMAPS["mean"], r"Predicted mean $\log(\mathrm{FoG})$"),
        (Z_std, "Std", _XY_PANEL_CMAPS["std"], r"Predictive std $\log(\mathrm{FoG})$"),
        (Z_ei, "EI", _XY_PANEL_CMAPS["ei"], "Expected Improvement"),
        (Z_ucb, "UCB", _XY_PANEL_CMAPS["ucb"], "Upper Confidence Bound"),
    ]
    style = {**apply_paper_style(), **_XY_MATH_RC}
    with plt.rc_context(style):
        fig, ax = plt.subplots(2, 2, figsize=_XY_2X2_FIGSIZE, constrained_layout=True)
        cbar_formatter = mticker.FuncFormatter(lambda x, p: _math_scalar_formatter(x, p))
        std_gamma = getattr(cfg, "std_color_gamma", 2.0)  # Reduced from 8.0 to 2.0 for better gradient
        for axi, (Z, title, cmap, cbar_label) in zip(ax.flat, panels):
            vmin, vmax = _vmin_vmax(Z)
            if vmax <= vmin:
                vmin, vmax = 0.0, 1.0
            if title == "Std":
                norm = mcolors.PowerNorm(gamma=std_gamma, vmin=vmin, vmax=vmax) if std_gamma is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            Z_masked = np.ma.masked_invalid(Z)
            cs = axi.pcolormesh(B, M, Z_masked, cmap=cmap, norm=norm, shading="gouraud")
            # For sparse data, also check relative range to allow contours even when absolute range is small.
            _range_abs = vmax - vmin
            _range_rel = _range_abs / (abs(vmax) + abs(vmin) + EPS) if (vmax != 0 or vmin != 0) else 0.0
            # For EI, use very lenient thresholds to ensure contours are drawn even when values are very small
            if title == "EI":
                # For EI: use very lenient thresholds and always try to draw contours if there's any variation
                contour_threshold_abs = 1e-15  # Very lenient absolute threshold for EI
                contour_threshold_rel = 1e-12  # Very lenient relative threshold for EI
            else:
                contour_threshold_abs = _SURROGATE_CONTOUR_MIN_RANGE
                contour_threshold_rel = 1e-6
            if _range_abs > contour_threshold_abs or _range_rel > contour_threshold_rel:
                valid_z = Z_masked.compressed() if hasattr(Z_masked, 'compressed') else Z_masked[np.isfinite(Z_masked)]
                if len(valid_z) > 0:
                    # More contour levels for all panels to show gradient better
                    n_levels = _SURROGATE_CONTOUR_LEVELS * 2
                    q_levels = np.linspace(0.05, 0.95, n_levels)
                    _levs = np.quantile(valid_z, q_levels)
                else:
                    n_levels = _SURROGATE_CONTOUR_LEVELS * 2
                    _levs = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
                try:
                    # Higher transparency for all contours to show gradient better
                    qc = axi.contour(
                        B, M, Z_masked,
                        levels=_levs,
                        colors=_SURROGATE_CONTOUR_COLOR,
                        linewidths=_SURROGATE_CONTOUR_LW,
                        alpha=0.95,  # High transparency
                        zorder=2,
                    )
                    qc.set_path_effects([
                        mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                        mpatheffects.Normal(),
                    ])
                except (ValueError, TypeError):
                    pass
            axi.scatter(
                obma, omtac,
                s=_SCATTER_MARKER_PT ** 2,
                facecolor="white",
                edgecolor="black",
                linewidths=_SCATTER_EDGE_PT,
                zorder=3,
            )
            axi.set_xlim(-_EDGE_MARGIN, 1.0 + _EDGE_MARGIN)
            axi.set_ylim(-_EDGE_MARGIN, 1.0 + _EDGE_MARGIN)
            axi.set_aspect("equal", adjustable="box")
            axi.set_xlabel(r"$[\mathrm{BMA}]$")
            axi.set_ylabel(r"$[\mathrm{MTAC}]$")
            axi.set_title(title)
            axi.xaxis.set_major_formatter(cbar_formatter)
            axi.yaxis.set_major_formatter(cbar_formatter)
            cbar = fig.colorbar(cs, ax=axi, fraction=0.046, pad=0.03, label=cbar_label)
            # Apply math formatter to colorbar for proper scientific notation
            cbar.ax.yaxis.set_major_formatter(cbar_formatter)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_xy_2x2_from_bma_mtac_model(
    model: GPModel2D,
    learning_df: pd.DataFrame,
    cfg: BOConfig,
    bo_run_id: str,
    out_path: Path,
) -> None:
    """
    Plot 2x2 xy-plane panels (Mean, Std, EI, UCB) when GP is fitted in (BMA, MTAC).
    Grid is in (x, y) with x = BMA/(BMA+MTAC), y = BMA+MTAC; each point is converted
    to (frac_BMA, frac_MTAC) for model.predict, then drawn in (x,y) space.
    """
    n_grid = int(getattr(cfg, "surrogate_map_xy_grid", 361))
    x_grid = np.linspace(0.0, 1.0, n_grid)
    y_grid = np.linspace(0.0, 1.0, n_grid)
    X1, X2 = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([X1.T.ravel(), X2.T.ravel()])  # (x, y) order
    x_flat = grid[:, 0]
    y_flat = grid[:, 1]
    frac_BMA = np.where(y_flat > 0, x_flat * y_flat, 0.0)
    frac_MTAC = np.where(y_flat > 0, (1.0 - x_flat) * y_flat, 0.0)
    X_pred = np.column_stack([frac_BMA, frac_MTAC])
    mu, std = model.predict(X_pred)
    best = float(np.nanmax(learning_df["log_fog_corrected"].to_numpy(dtype=float)))
    ei = _ei(mu, std, best, cfg.ei_xi)
    ucb = _ucb(mu, std, cfg.ucb_kappa)
    MU = mu.reshape((n_grid, n_grid))
    SD = std.reshape((n_grid, n_grid))
    EI = ei.reshape((n_grid, n_grid))
    UCB = ucb.reshape((n_grid, n_grid))
    learning_xy = _xy_from_frac(learning_df)
    # Group by (polymer_id, round_id) if round_id exists, otherwise by polymer_id only
    # This ensures each round's data is plotted separately
    if "round_id" in learning_xy.columns:
        group_cols = ["polymer_id", "round_id"]
    else:
        group_cols = ["polymer_id"]
    observed = (
        learning_xy.groupby(group_cols, as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"))
    )
    ox = observed["x"].to_numpy(dtype=float)
    oy = observed["y"].to_numpy(dtype=float)

    def _vmin_vmax(Z: np.ndarray, is_std: bool = False) -> tuple[float, float]:
        """
        Calculate vmin/vmax dynamically from data.
        For std plots, use percentile-based vmin to avoid "holes" at observed points.
        """
        valid = np.isfinite(Z)
        if not np.any(valid):
            return 0.0, 1.0
        z = Z[valid]
        if is_std:
            # For std: use 10th percentile as vmin to avoid "holes" at observed points
            # This creates a smoother gradient visualization with gentler contours
            vmin = float(np.percentile(z, 10))
            vmax = float(np.nanmax(z))
            if vmin >= vmax:
                vmin = float(np.nanmin(z))
            return vmin, vmax
        else:
            return float(np.nanmin(z)), float(np.nanmax(z))

    panels = [
        (MU, "Mean", _XY_PANEL_CMAPS["mean"], r"Predicted mean $\log(\mathrm{FoG})$"),
        (SD, "Std", _XY_PANEL_CMAPS["std"], r"Predictive std $\log(\mathrm{FoG})$"),
        (EI, "EI", _XY_PANEL_CMAPS["ei"], "Expected Improvement"),
        (UCB, "UCB", _XY_PANEL_CMAPS["ucb"], "Upper Confidence Bound"),
    ]
    style = {**apply_paper_style(), **_XY_MATH_RC}
    with plt.rc_context(style):
        fig, ax = plt.subplots(2, 2, figsize=_XY_2X2_FIGSIZE, constrained_layout=True)
        cbar_formatter = mticker.FuncFormatter(lambda x, p: _math_scalar_formatter(x, p))
        std_gamma = getattr(cfg, "std_color_gamma", 2.0)  # Reduced from 8.0 to 2.0 for better gradient
        for axi, (Z, title, cmap, cbar_label) in zip(ax.flat, panels):
            # Calculate vmin/vmax dynamically, using percentile-based vmin for std
            vmin, vmax = _vmin_vmax(Z, is_std=(title == "Std"))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0
            if title == "Std":
                norm = mcolors.PowerNorm(gamma=std_gamma, vmin=vmin, vmax=vmax) if std_gamma is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cs = axi.pcolormesh(X1, X2, Z.T, cmap=cmap, norm=norm, shading="gouraud")
            # For sparse data, also check relative range to allow contours even when absolute range is small.
            _range_abs = vmax - vmin
            _range_rel = _range_abs / (abs(vmax) + abs(vmin) + EPS) if (vmax != 0 or vmin != 0) else 0.0
            # For EI, use very lenient thresholds to ensure contours are drawn even when values are very small
            if title == "EI":
                # For EI: use very lenient thresholds and always try to draw contours if there's any variation
                contour_threshold_abs = 1e-15  # Very lenient absolute threshold for EI
                contour_threshold_rel = 1e-12  # Very lenient relative threshold for EI
            else:
                contour_threshold_abs = _SURROGATE_CONTOUR_MIN_RANGE
                contour_threshold_rel = 1e-6
            if _range_abs > contour_threshold_abs or _range_rel > contour_threshold_rel:
                valid_z = Z[np.isfinite(Z)]
                if len(valid_z) > 0:
                    # More contour levels for all panels to show gradient better
                    n_levels = _SURROGATE_CONTOUR_LEVELS * 2
                    q_levels = np.linspace(0.05, 0.95, n_levels)
                    _levs = np.quantile(valid_z, q_levels)
                else:
                    n_levels = _SURROGATE_CONTOUR_LEVELS * 2
                    _levs = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
                try:
                    # Higher transparency for all contours to show gradient better
                    qc = axi.contour(
                        X1, X2, Z.T,
                        levels=_levs,
                        colors=_SURROGATE_CONTOUR_COLOR,
                        linewidths=_SURROGATE_CONTOUR_LW,
                        alpha=0.95,  # High transparency
                        zorder=2,
                    )
                    qc.set_path_effects([
                        mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                        mpatheffects.Normal(),
                    ])
                except (ValueError, TypeError):
                    pass
            axi.scatter(
                ox, oy,
                s=_SCATTER_MARKER_PT ** 2,
                facecolor="white",
                edgecolor="black",
                linewidths=_SCATTER_EDGE_PT,
                zorder=3,
            )
            axi.set_xlim(-_EDGE_MARGIN, 1.0 + _EDGE_MARGIN)
            axi.set_ylim(-_EDGE_MARGIN, 1.0 + _EDGE_MARGIN)
            axi.set_aspect("equal", adjustable="box")
            axi.set_xlabel(_XY_LABEL_X)
            axi.set_ylabel(_XY_LABEL_Y)
            axi.set_title(title)
            axi.xaxis.set_major_formatter(cbar_formatter)
            axi.yaxis.set_major_formatter(cbar_formatter)
            cbar = fig.colorbar(cs, ax=axi, fraction=0.046, pad=0.03, label=cbar_label)
            # Apply math formatter to colorbar for proper scientific notation
            cbar.ax.yaxis.set_major_formatter(cbar_formatter)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _load_polymer_colors(path: Path) -> Dict[str, str]:
    """Load polymer_id -> hex color from yaml (e.g. meta/polymers/colors.yml)."""
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if "polymer_id" in d and isinstance(d["polymer_id"], dict):
        return {str(k): str(v).strip() for k, v in d["polymer_id"].items()}
    return {str(k): str(v).strip() for k, v in d.items() if isinstance(d.get(k), str)}


def _plot_ranking_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Plot ranking bar chart. Generates two versions:
    1. color_poly_id: Uses polymer_id-specific colors
    2. color_mean: Uses Mean (viridis) gradient from top to bottom
    """
    if df.empty:
        return
    # Filter out rows with NaN or Inf values in value_col
    data = df[df[value_col].notna() & np.isfinite(df[value_col])].copy()
    if data.empty:
        import warnings
        warnings.warn(f"All values in {value_col} are NaN or Inf, skipping bar chart: {out_path}", UserWarning)
        return
    data = data.sort_values(value_col, ascending=False).copy()
    
    # Format xlabel, title, and colorbar label: if they contain "t50", 
    # put the entire string in math mode to ensure consistent font
    # Use \mathrm{} to keep regular font for non-math text within math mode
    if "t50" in xlabel.lower():
        # Replace "t50" with subscript and wrap entire string in math mode
        xlabel_math = xlabel.replace("t50", r"t_{50}")
        xlabel_formatted = rf"$\mathrm{{{xlabel_math}}}$"
    else:
        xlabel_formatted = xlabel
    
    if "t50" in title.lower():
        # Replace "t50" with subscript and wrap entire string in math mode
        title_math = title.replace("t50", r"t_{50}")
        title_formatted = rf"$\mathrm{{{title_math}}}$"
    else:
        title_formatted = title
    
    # Bar width (thinner bars for more stylish look)
    bar_height = 0.6  # Reduced from default ~0.8 for thinner bars
    
    # Version 1: Polymer ID colors
    default_color = "#4C78A8"
    colors = None
    if color_map:
        colors = [color_map.get(str(pid), default_color) for pid in data[label_col]]
    
    # Generate file paths for both versions
    out_path_poly = out_path.parent / f"{out_path.stem}_color_poly_id{out_path.suffix}"
    out_path_mean = out_path.parent / f"{out_path.stem}_color_mean{out_path.suffix}"
    
    # Apply math font settings for proper t50 subscript rendering
    # Use same font (Arial) for both regular text and math text
    paper_style = apply_paper_style()
    style = {
        **paper_style,
        "mathtext.default": "regular",  # Use regular font (Arial) for math text
        "mathtext.fontset": "custom",  # Custom fontset to use regular font
    }
    
    # Version 1: Polymer ID colors
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.2, max(2.0, 0.22 * len(data) + 0.8)))
        ax.barh(
            data[label_col].astype(str),
            data[value_col].astype(float),
            height=bar_height,
            color=colors if colors is not None else default_color,
        )
        ax.invert_yaxis()
        ax.set_title(title_formatted)
        ax.set_xlabel(xlabel_formatted)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        fig.savefig(out_path_poly, dpi=600, bbox_inches="tight")
        plt.close(fig)
    
    # Version 2: Mean (viridis) gradient from top to bottom
    # Normalize values to [0, 1] for colormap
    values = data[value_col].astype(float).values
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmax > vmin:
        normalized_values = (values - vmin) / (vmax - vmin)
    else:
        normalized_values = np.ones_like(values) * 0.5
    
    # Use viridis colormap (same as Mean in surrogate maps)
    try:
        # Try new matplotlib API (3.5+)
        import matplotlib
        if hasattr(matplotlib, 'colormaps'):
            cmap = matplotlib.colormaps["viridis"]
        else:
            cmap = plt.cm.get_cmap("viridis")
    except (AttributeError, KeyError, TypeError):
        # Fallback to old API
        cmap = plt.cm.get_cmap("viridis")
    gradient_colors = [cmap(val) for val in normalized_values]
    
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.2, max(2.0, 0.22 * len(data) + 0.8)))
        bars = ax.barh(
            data[label_col].astype(str),
            data[value_col].astype(float),
            height=bar_height,
            color=gradient_colors,
        )
        ax.invert_yaxis()
        ax.set_title(title_formatted)
        ax.set_xlabel(xlabel_formatted)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        
        # Add colorbar for Mean gradient version
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(xlabel_formatted)
        fig.savefig(out_path_mean, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_observed_vs_predicted(
    learning_df: pd.DataFrame,
    model: Any,
    cfg: BOConfig,
    bo_run_id: str,
    out_path: Path,
) -> None:
    """Scatter: observed log(FoG) vs GP predicted mean at training points (English only)."""
    if learning_df.empty:
        return
    if len(model.length_scale) == 3:
        X = learning_df[["frac_MPC", "frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    elif getattr(cfg, "use_bma_mtac_coords", False):
        X = learning_df[["frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    else:
        X = learning_df[["x", "y"]].to_numpy(dtype=float)
    y_obs = learning_df["log_fog_corrected"].to_numpy(dtype=float)
    mu, _ = model.predict(X)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        ax.scatter(y_obs, mu, s=18, facecolor="#4C78A8", edgecolor="black", linewidths=0.5, zorder=2)
        lim_lo = min(np.nanmin(y_obs), np.nanmin(mu))
        lim_hi = max(np.nanmax(y_obs), np.nanmax(mu))
        margin = 0.05 * (lim_hi - lim_lo) if lim_hi > lim_lo else 0.1
        ax.plot([lim_lo - margin, lim_hi + margin], [lim_lo - margin, lim_hi + margin], color="0.5", ls="--", lw=0.7)
        ax.set_xlabel(r"Observed $\log(\mathrm{FoG})$")
        ax.set_ylabel(r"Predicted mean $\log(\mathrm{FoG})$")
        ax.set_title("Observed vs predicted")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lim_lo - margin, lim_hi + margin)
        ax.set_ylim(lim_lo - margin, lim_hi + margin)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_ternary_2x2_panels(
    cand: pd.DataFrame,
    observed: pd.DataFrame,
    bo_run_id: str,
    out_path: Path,
    *,
    std_color_gamma: Optional[float] = None,
    acq_mode_used: Optional[str] = None,
) -> None:
    """
    Plot 2x2 ternary panels: Mean (predicted log FoG), Std, EI, UCB.
    Same order as xy_2x2 panels. Paper-grade, English only.
    """
    mpc = cand["frac_MPC"].to_numpy(dtype=float)
    bma = cand["frac_BMA"].to_numpy(dtype=float)
    mtac = cand["frac_MTAC"].to_numpy(dtype=float)
    tx, ty = _ternary_xy_from_frac(mpc, bma, mtac)
    tick_fracs = np.linspace(0.0, 1.0, 6)
    tri = mtri.Triangulation(tx, ty)

    def _pt(frac_mpc: float, frac_bma: float, frac_mtac: float) -> tuple[float, float]:
        px, py = _ternary_xy_from_frac(
            np.array([frac_mpc], dtype=float),
            np.array([frac_bma], dtype=float),
            np.array([frac_mtac], dtype=float),
        )
        return float(px[0]), float(py[0])

    # Prepare data for each panel
    z_mean = cand["pred_log_fog_mean"].to_numpy(dtype=float)
    z_std = cand["pred_log_fog_std"].to_numpy(dtype=float)
    z_ei = cand["ei"].to_numpy(dtype=float)
    z_ucb = cand["ucb"].to_numpy(dtype=float)

    # Observed points
    omtpc = observed["frac_MPC"].to_numpy(dtype=float)
    obma = observed["frac_BMA"].to_numpy(dtype=float)
    omtac = observed["frac_MTAC"].to_numpy(dtype=float)
    ox, oy = _ternary_xy_from_frac(omtpc, obma, omtac)

    # Use actual acquisition mode for consistent labeling
    if acq_mode_used == "ei_ucb_fallback":
        ei_title = "EI (no promising region)"
        ei_label = "Expected Improvement (no promising region)"
    else:
        ei_title = "EI"
        ei_label = "Expected Improvement"

    panels = [
        (z_mean, "Mean", _XY_PANEL_CMAPS["mean"], "Predicted mean log(FoG)", r"Predicted mean $\log(\mathrm{FoG})$"),
        (z_std, "Std", _XY_PANEL_CMAPS["std"], "Predictive std log(FoG)", r"Predictive std $\log(\mathrm{FoG})$"),
        (z_ei, ei_title, _XY_PANEL_CMAPS["ei"], ei_title, ei_label),
        (z_ucb, "UCB", _XY_PANEL_CMAPS["ucb"], "Upper Confidence Bound", "Upper Confidence Bound"),
    ]

    style = {**apply_paper_style(), **_XY_MATH_RC}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4), constrained_layout=True)
        cbar_formatter = mticker.FuncFormatter(lambda x, p: _math_scalar_formatter(x, p))
        std_gamma = std_color_gamma if std_color_gamma is not None else 2.0

        for ax, (z, title_key, cmap, panel_title, cbar_label) in zip(axes.flat, panels):
            # Calculate vmin/vmax dynamically, using percentile-based vmin for std
            valid_z = z[np.isfinite(z)]
            if len(valid_z) == 0:
                vmin, vmax = 0.0, 1.0
            elif title_key == "Std":
                # For std: use 5th percentile as vmin to avoid "holes" at observed points
                # This creates a smoother gradient visualization
                vmin = float(np.percentile(valid_z, 5))
                vmax = float(np.nanmax(valid_z))
                if vmin >= vmax:
                    vmin = float(np.nanmin(valid_z))
            else:
                vmin = float(np.nanmin(valid_z))
                vmax = float(np.nanmax(valid_z))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0

            # Normalization
            if title_key == "Std":
                gamma_used = min(std_gamma, 2.0) if std_gamma > 2.0 else std_gamma
                norm = mcolors.PowerNorm(gamma=gamma_used, vmin=vmin, vmax=vmax) if std_gamma is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Plot
            cs = ax.tripcolor(tri, z, cmap=cmap, norm=norm, shading="gouraud")

            # Contours
            _range = vmax - vmin
            _range_rel = _range / (abs(vmax) + abs(vmin) + EPS) if (vmax != 0 or vmin != 0) else 0.0
            # For EI, use very lenient thresholds to ensure contours are drawn even when values are very small
            if title_key == "EI":
                # For EI: use very lenient thresholds and always try to draw contours if there's any variation
                contour_threshold_abs = 1e-15  # Very lenient absolute threshold for EI
                contour_threshold_rel = 1e-12  # Very lenient relative threshold for EI
            else:
                contour_threshold_abs = _SURROGATE_CONTOUR_MIN_RANGE
                contour_threshold_rel = 1e-6
            if _range > contour_threshold_abs or _range_rel > contour_threshold_rel:
                valid_z = z[np.isfinite(z)]
                if len(valid_z) > 0:
                    n_levels = _SURROGATE_CONTOUR_LEVELS
                    q_levels = np.linspace(0.05, 0.95, n_levels)
                    _levs = np.quantile(valid_z, q_levels)
                else:
                    n_levels = _SURROGATE_CONTOUR_LEVELS
                    _levs = np.linspace(vmin, vmax, n_levels + 2)[1:-1]
                _z = np.where(np.isfinite(z), z, np.nanmean(np.where(np.isfinite(z), z, np.nan)))
                try:
                    contour_alpha = 0.95
                    tc = ax.tricontour(
                        tri,
                        _z,
                        levels=_levs,
                        colors=_SURROGATE_CONTOUR_COLOR,
                        linewidths=_SURROGATE_CONTOUR_LW,
                        alpha=contour_alpha,
                        zorder=2,
                    )
                    tc.set_path_effects([
                        mpatheffects.Stroke(linewidth=_SURROGATE_CONTOUR_STROKE_LW, foreground=_SURROGATE_CONTOUR_STROKE_COLOR),
                        mpatheffects.Normal(),
                    ])
                except (ValueError, TypeError):
                    pass

            # Triangle boundary
            p_mtac = np.array([0.0, 0.0])
            p_bma = np.array([1.0, 0.0])
            p_mpc = np.array([0.5, SQRT3_OVER_2])
            ax.plot(
                [p_mtac[0], p_bma[0], p_mpc[0], p_mtac[0]],
                [p_mtac[1], p_bma[1], p_mpc[1], p_mtac[1]],
                color="black",
                lw=0.8,
            )

            # Observed points
            ax.scatter(
                ox,
                oy,
                s=16,
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
            )

            ax.set_title(panel_title)
            ax.set_xlim(-0.13, 1.13)
            ax.set_ylim(-0.14, SQRT3_OVER_2 + 0.07)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            cbar = fig.colorbar(cs, ax=ax, fraction=0.045, pad=0.08)
            cbar.set_label(cbar_label)
            # Colorbar formatter (for vertical colorbar, use yaxis)
            cbar.ax.yaxis.set_major_formatter(cbar_formatter)

            # Corner labels
            ax.text(0.5, SQRT3_OVER_2 + 0.03, "PMPC", ha="center", va="bottom")
            ax.text(-0.03, -0.03, "PMTAC", ha="right", va="top")
            ax.text(1.03, -0.03, "PBMA", ha="left", va="top")

            # Base: BMA 0 -> 100
            for t in tick_fracs:
                px, py = _pt(0.0, t, 1.0 - t)
                ax.plot([px, px], [py, py - 0.012], color="black", lw=0.45)
                ax.text(px, py - 0.026, f"{int(round(t * 100))}", ha="center", va="top", fontsize=5.5)

            # Left edge: MTAC
            for t in tick_fracs[1:-1]:
                px, py = _pt(t, 0.0, 1.0 - t)
                mtac_pct = int(round((1.0 - t) * 100))
                ax.plot([px, px - 0.010], [py, py], color="black", lw=0.45)
                ax.text(px - 0.017, py, f"{mtac_pct}", ha="right", va="center", fontsize=5.5)

            # Right edge: MPC
            for t in tick_fracs[1:-1]:
                px, py = _pt(t, 1.0 - t, 0.0)
                mpc_pct = int(round(t * 100))
                ax.plot([px, px + 0.010], [py, py], color="black", lw=0.45)
                ax.text(px + 0.017, py, f"{mpc_pct}", ha="left", va="center", fontsize=5.5)

        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_ternary_mean_with_top_candidates(
    plot_cand: pd.DataFrame,
    observed: pd.DataFrame,
    top_df: pd.DataFrame,
    bo_run_id: str,
    out_path: Path,
) -> None:
    """Ternary predicted mean log(FoG) with top 1–3 next-experiment candidates highlighted (stars)."""
    top3 = top_df.head(3)
    if top3.empty:
        _plot_ternary_field(plot_cand, observed, value_col="pred_log_fog_mean", title="Predicted mean log(FoG)", cbar_label="Predicted mean log(FoG)", out_path=out_path)
        return
    mpc = plot_cand["frac_MPC"].to_numpy(dtype=float)
    bma = plot_cand["frac_BMA"].to_numpy(dtype=float)
    mtac = plot_cand["frac_MTAC"].to_numpy(dtype=float)
    z = plot_cand["pred_log_fog_mean"].to_numpy(dtype=float)
    tx, ty = _ternary_xy_from_frac(mpc, bma, mtac)
    tick_fracs = np.linspace(0.0, 1.0, 6)

    def _pt(frac_mpc: float, frac_bma: float, frac_mtac: float) -> tuple[float, float]:
        px, py = _ternary_xy_from_frac(np.array([frac_mpc], dtype=float), np.array([frac_bma], dtype=float), np.array([frac_mtac], dtype=float))
        return float(px[0]), float(py[0])

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        tri = mtri.Triangulation(tx, ty)
        vmin, vmax = float(np.nanmin(z)), float(np.nanmax(z))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        cs = ax.tripcolor(tri, z, cmap="viridis", norm=mcolors.Normalize(vmin=vmin, vmax=vmax), shading="gouraud")
        ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_OVER_2, 0], color="black", lw=0.8)
        omtpc = observed["frac_MPC"].to_numpy(dtype=float)
        obma = observed["frac_BMA"].to_numpy(dtype=float)
        omtac = observed["frac_MTAC"].to_numpy(dtype=float)
        ox, oy = _ternary_xy_from_frac(omtpc, obma, omtac)
        ax.scatter(ox, oy, s=16, facecolor="white", edgecolor="black", linewidth=0.5, zorder=4)
        t_mpc = top3["frac_MPC"].to_numpy(dtype=float)
        t_bma = top3["frac_BMA"].to_numpy(dtype=float)
        t_mtac = top3["frac_MTAC"].to_numpy(dtype=float)
        tx_top, ty_top = _ternary_xy_from_frac(t_mpc, t_bma, t_mtac)
        ax.scatter(tx_top, ty_top, s=180, marker="*", facecolor="gold", edgecolor="black", linewidths=0.6, zorder=5)
        for i, (px, py) in enumerate(zip(tx_top, ty_top), start=1):
            ax.text(px, py + 0.04, str(i), ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title("Predicted mean log(FoG) + top candidates")
        ax.set_xlim(-0.13, 1.13)
        ax.set_ylim(-0.14, SQRT3_OVER_2 + 0.07)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        cbar = fig.colorbar(cs, ax=ax, fraction=0.045, pad=0.08)
        cbar.set_label("Predicted mean log(FoG)")
        ax.text(0.5, SQRT3_OVER_2 + 0.03, "PMPC", ha="center", va="bottom")
        ax.text(-0.03, -0.03, "PMTAC", ha="right", va="top")
        ax.text(1.03, -0.03, "PBMA", ha="left", va="top")
        for t in tick_fracs:
            px, py = _pt(0.0, t, 1.0 - t)
            ax.plot([px, px], [py, py - 0.012], color="black", lw=0.45)
            ax.text(px, py - 0.026, f"{int(round(t * 100))}", ha="center", va="top", fontsize=5.5)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_ternary_sampling_by_round(
    learning_df: pd.DataFrame,
    bo_run_id: str,
    out_path: Path,
) -> None:
    """Ternary: sampling positions colored by round_id (one point per polymer per round)."""
    if learning_df.empty or "round_id" not in learning_df.columns:
        return
    agg = learning_df.groupby(["round_id", "polymer_id"], as_index=False).agg(
        frac_MPC=("frac_MPC", "mean"),
        frac_BMA=("frac_BMA", "mean"),
        frac_MTAC=("frac_MTAC", "mean"),
    )
    mpc = agg["frac_MPC"].to_numpy(dtype=float)
    bma = agg["frac_BMA"].to_numpy(dtype=float)
    mtac = agg["frac_MTAC"].to_numpy(dtype=float)
    tx, ty = _ternary_xy_from_frac(mpc, bma, mtac)
    rounds = sorted(agg["round_id"].astype(str).unique(), key=_natural_round_key)
    round_to_idx = {r: i for i, r in enumerate(rounds)}
    c = [round_to_idx[str(r)] for r in agg["round_id"]]
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_OVER_2, 0], color="black", lw=0.8)
        cmap = plt.get_cmap("tab10")
        for i, r in enumerate(rounds):
            mask = np.array(c) == i
            ax.scatter(tx[mask], ty[mask], s=20, c=[cmap(i % 10)], label=str(r), edgecolors="black", linewidths=0.4, zorder=4)
        ax.set_title("Sampling by round")
        ax.set_xlim(-0.13, 1.13)
        ax.set_ylim(-0.14, SQRT3_OVER_2 + 0.07)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.legend(loc="best", fontsize=6)
        ax.text(0.5, SQRT3_OVER_2 + 0.03, "PMPC", ha="center", va="bottom")
        ax.text(-0.03, -0.03, "PMTAC", ha="right", va="top")
        ax.text(1.03, -0.03, "PBMA", ha="left", va="top")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_ternary_residual(
    learning_df: pd.DataFrame,
    model: Any,
    cfg: BOConfig,
    bo_run_id: str,
    out_path: Path,
) -> None:
    """Ternary scatter: training points colored by residual (observed - predicted)."""
    if learning_df.empty:
        return
    if len(model.length_scale) == 3:
        X = learning_df[["frac_MPC", "frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    elif getattr(cfg, "use_bma_mtac_coords", False):
        X = learning_df[["frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    else:
        X = learning_df[["x", "y"]].to_numpy(dtype=float)
    y_obs = learning_df["log_fog_corrected"].to_numpy(dtype=float)
    mu, _ = model.predict(X)
    resid = y_obs - mu
    mpc = learning_df["frac_MPC"].to_numpy(dtype=float)
    bma = learning_df["frac_BMA"].to_numpy(dtype=float)
    mtac = learning_df["frac_MTAC"].to_numpy(dtype=float)
    tx, ty = _ternary_xy_from_frac(mpc, bma, mtac)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_OVER_2, 0], color="black", lw=0.8)
        rmax = max(abs(np.nanmin(resid)), abs(np.nanmax(resid)), 1e-9)
        sc = ax.scatter(tx, ty, c=resid, s=28, cmap="RdBu_r", vmin=-rmax, vmax=rmax, edgecolors="black", linewidths=0.5, zorder=4)
        ax.set_title("Residual (observed $-$ predicted)")
        ax.set_xlim(-0.13, 1.13)
        ax.set_ylim(-0.14, SQRT3_OVER_2 + 0.07)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.08)
        cbar.set_label(r"Residual $\log(\mathrm{FoG})$")
        ax.text(0.5, SQRT3_OVER_2 + 0.03, "PMPC", ha="center", va="bottom")
        ax.text(-0.03, -0.03, "PMTAC", ha="right", va="top")
        ax.text(1.03, -0.03, "PBMA", ha="left", va="top")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_fog_vs_t50_scatter(
    rank_all: pd.DataFrame,
    bo_run_id: str,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    """Scatter: mean FoG vs mean t50 per polymer (English only)."""
    if rank_all.empty or "mean_fog" not in rank_all.columns or "mean_t50_min" not in rank_all.columns:
        return
    data = rank_all.copy()
    default_color = "#4C78A8"
    colors = [color_map.get(str(pid), default_color) for pid in data["polymer_id"]] if color_map else default_color
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        ax.scatter(data["mean_t50_min"], data["mean_fog"], s=24, c=colors if isinstance(colors, list) else colors, edgecolors="black", linewidths=0.5, zorder=2)
        ax.set_xlabel("Mean t50 [min]")
        ax.set_ylabel("Mean FoG")
        ax.set_title("FoG vs t50 (all rounds)")
        for _, row in data.iterrows():
            ax.annotate(str(row["polymer_id"]), (row["mean_t50_min"], row["mean_fog"]), xytext=(4, 4), textcoords="offset points", fontsize=5, ha="left")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_model_diagnostic(
    map_quality: dict,
    gp_hyperparams: dict,
    bo_run_id: str,
    out_path: Path,
) -> None:
    """Simple text panel: key GP and map quality metrics (English only)."""
    lines = [
        f"BO run: {bo_run_id}",
        "",
        "Map quality",
        f"  z_min = {map_quality.get('z_min', '—')}",
        f"  z_max = {map_quality.get('z_max', '—')}",
        f"  ei_collapsed = {map_quality.get('ei_collapsed', '—')}",
        "",
        "GP",
    ]
    for key in ["length_scale_x", "length_scale_y", "length_scale_frac_MPC", "length_scale_frac_BMA", "length_scale_frac_MTAC", "signal_std", "noise_std", "kernel_mode"]:
        if key in gp_hyperparams:
            lines.append(f"  {key} = {gp_hyperparams[key]}")
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.axis("off")
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=6, verticalalignment="top", fontfamily="monospace")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _default_bo_run_id() -> str:
    """Run ID = execution local date and time (e.g. bo_2025-02-04_14-35-22)."""
    now = datetime.now()
    return f"bo_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def run_bo(
    *,
    bo_learning_path: Path,
    fog_plate_aware_path: Path,
    out_root_dir: Path,
    bo_run_id: Optional[str] = None,
    config: Optional[BOConfig] = None,
    polymer_colors_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Run BO from prepared learning data and write artifacts under:
      {out_root_dir}/{bo_run_id}/
    """
    cfg = config or BOConfig()
    if bool(cfg.apply_round_anchor_correction):
        raise ValueError(
            "apply_round_anchor_correction=True is not allowed in this project. "
            "For enzyme experiments, round/day correction must not be applied."
        )
    bo_run_id = str(bo_run_id or _default_bo_run_id()).strip()
    out_dir = Path(out_root_dir) / bo_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    learning = _load_bo_learning(Path(bo_learning_path))
    fog = _load_fog_plate_aware(Path(fog_plate_aware_path))
    if learning.empty:
        raise ValueError("BO learning data is empty after filtering.")
    if fog.empty:
        raise ValueError("FoG plate-aware data is empty after filtering.")

    learning, anchor_meta = _apply_round_anchor_correction(
        learning,
        enabled=cfg.apply_round_anchor_correction,
        min_anchor_polymers=cfg.min_anchor_polymers,
    )
    if "log_fog_corrected" not in learning.columns:
        learning["log_fog_corrected"] = learning["log_fog"].astype(float)

    if cfg.use_simplex_gp:
        X = learning[["frac_MPC", "frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    elif cfg.use_bma_mtac_coords:
        X = learning[["frac_BMA", "frac_MTAC"]].to_numpy(dtype=float)
    else:
        X = learning[["x", "y"]].to_numpy(dtype=float)
    X_unique = np.unique(np.round(X, decimals=12), axis=0)
    n_unique_design_points = int(X_unique.shape[0])
    force_isotropic = (
        (not cfg.use_simplex_gp)
        and bool(cfg.sparse_force_isotropic)
        and (n_unique_design_points <= int(max(cfg.sparse_isotropic_max_unique_points, 0)))
    )
    use_trend = (
        (not cfg.use_simplex_gp)
        and bool(cfg.sparse_use_trend)
        and (n_unique_design_points <= int(max(cfg.sparse_trend_max_unique_points, 0)))
    )
    y = learning["log_fog_corrected"].to_numpy(dtype=float)
    if cfg.enable_heteroskedastic_noise:
        obs_noise_rel = _estimate_obs_noise_rel(
            learning,
            min_rel=cfg.noise_rel_min,
            max_rel=cfg.noise_rel_max,
        )
    else:
        obs_noise_rel = np.ones(len(learning), dtype=float)
    learning["obs_noise_rel"] = obs_noise_rel
    if cfg.use_simplex_gp:
        gp = GPModelSimplex.fit(X, y, random_state=cfg.random_state, obs_noise_rel=obs_noise_rel)
    else:
        min_ls_sparse = None
        if force_isotropic and n_unique_design_points <= getattr(
            cfg, "sparse_isotropic_apply_min_below_n", 10
        ):
            min_ls_sparse = getattr(cfg, "min_length_scale_sparse_isotropic", 0.2)
        gp = GPModel2D.fit(
            X,
            y,
            random_state=cfg.random_state,
            obs_noise_rel=obs_noise_rel,
            force_isotropic=force_isotropic,
            min_length_scale_isotropic=min_ls_sparse,
            use_trend=use_trend,
            trend_ridge=cfg.trend_ridge,
        )

    cand = _build_candidate_frame(gp, learning, cfg)
    cand, anchor_targets = _select_diverse_batch(cand, learning, cfg)
    selected_unique = cand[cand["selected"] == 1].copy().sort_values("selection_order")
    if selected_unique.empty:
        raise RuntimeError("No feasible BO suggestions selected; loosen distance/component constraints.")
    if anchor_targets and cfg.use_exact_anchor_compositions:
        start_order = int(selected_unique["selection_order"].max()) + 1
        anchor_df = _build_exact_anchor_rows(
            gp, anchor_targets, learning, cfg,
            start_order=start_order, cand_template=cand,
        )
        if not anchor_df.empty:
            selected_unique = (
                pd.concat([selected_unique, anchor_df], ignore_index=True)
                .sort_values("selection_order")
                .reset_index(drop=True)
            )
    n_replicate = int(cand["policy_n_replicate"].iloc[0]) if "policy_n_replicate" in cand.columns else 0
    selected = _append_replicates(
        selected_unique, n_replicate=n_replicate, replicate_source=cfg.replicate_source
    )
    plot_cand = _build_plot_frame(gp, learning, cfg)
    map_quality = _build_map_quality_report(plot_cand, cfg, gp)
    design_coverage = _design_coverage_report(
        learning,
        use_bma_mtac_coords=bool(getattr(cfg, "use_bma_mtac_coords", False)),
    )

    ranking = _rank_tables(fog, bo_run_id=bo_run_id, fog_plate_aware_path=Path(fog_plate_aware_path))
    rank_round = ranking["by_round"]
    rank_all = ranking["all_round"]

    # Write outputs.
    learning_out = csv_dir / f"bo_training_data__{bo_run_id}.csv"
    model_log_out = csv_dir / f"bo_candidate_log__{bo_run_id}.csv"
    suggestions_out = csv_dir / f"bo_suggestions__{bo_run_id}.csv"
    fog_rank_round_out = csv_dir / f"fog_ranking_by_round__{bo_run_id}.csv"
    fog_rank_all_out = csv_dir / f"fog_ranking_all__{bo_run_id}.csv"
    t50_rank_round_out = csv_dir / f"t50_ranking_by_round__{bo_run_id}.csv"
    t50_rank_all_out = csv_dir / f"t50_ranking_all__{bo_run_id}.csv"
    next_exp_top5_out = csv_dir / f"next_experiment_top5__{bo_run_id}.csv"
    map_quality_out = out_dir / f"bo_map_quality__{bo_run_id}.json"

    learning_with_run = learning.copy()
    learning_with_run.insert(0, "run_id", bo_run_id)
    learning_with_run.insert(1, "bo_run_id", bo_run_id)
    learning_with_run.to_csv(learning_out, index=False)
    legacy_learning_out = out_dir / f"bo_training_data__{bo_run_id}.csv"
    if legacy_learning_out.is_file():
        legacy_learning_out.unlink(missing_ok=True)

    cand_out = cand.copy()
    cand_out.insert(0, "run_id", bo_run_id)
    cand_out.insert(1, "bo_run_id", bo_run_id)
    cand_out.to_csv(model_log_out, index=False)
    legacy_model_log_out = out_dir / f"bo_candidate_log__{bo_run_id}.csv"
    if legacy_model_log_out.is_file():
        legacy_model_log_out.unlink(missing_ok=True)

    sel_out = selected.copy()
    sel_out.insert(0, "run_id", bo_run_id)
    sel_out.insert(1, "bo_run_id", bo_run_id)
    sel_out.to_csv(suggestions_out, index=False)
    legacy_suggestions_out = out_dir / f"bo_suggestions__{bo_run_id}.csv"
    if legacy_suggestions_out.is_file():
        legacy_suggestions_out.unlink(missing_ok=True)

    fog_round_tbl = rank_round[
        ["run_id", "bo_run_id", "round_id", "polymer_id", "mean_fog", "mean_log_fog", "n_observations", "run_ids", "rank_fog_desc"]
    ].sort_values(["round_id", "rank_fog_desc", "polymer_id"])
    fog_all_tbl = rank_all[
        ["run_id", "bo_run_id", "polymer_id", "mean_fog", "mean_log_fog", "n_observations", "rounds", "run_ids", "rank_fog_desc"]
    ].sort_values(["rank_fog_desc", "polymer_id"])
    t50_round_tbl = rank_round[
        ["run_id", "bo_run_id", "round_id", "polymer_id", "mean_t50_min", "n_observations", "run_ids", "rank_t50_desc"]
    ].sort_values(["round_id", "rank_t50_desc", "polymer_id"])
    t50_all_tbl = rank_all[
        ["run_id", "bo_run_id", "polymer_id", "mean_t50_min", "n_observations", "rounds", "run_ids", "rank_t50_desc"]
    ].sort_values(["rank_t50_desc", "polymer_id"])

    fog_round_tbl.to_csv(fog_rank_round_out, index=False)
    fog_all_tbl.to_csv(fog_rank_all_out, index=False)
    t50_round_tbl.to_csv(t50_rank_round_out, index=False)
    t50_all_tbl.to_csv(t50_rank_all_out, index=False)
    legacy_fog_rank_round_out = out_dir / f"fog_ranking_by_round__{bo_run_id}.csv"
    legacy_fog_rank_all_out = out_dir / f"fog_ranking_all__{bo_run_id}.csv"
    legacy_t50_rank_round_out = out_dir / f"t50_ranking_by_round__{bo_run_id}.csv"
    legacy_t50_rank_all_out = out_dir / f"t50_ranking_all__{bo_run_id}.csv"
    for legacy_path in [
        legacy_fog_rank_round_out,
        legacy_fog_rank_all_out,
        legacy_t50_rank_round_out,
        legacy_t50_rank_all_out,
    ]:
        if legacy_path.is_file():
            legacy_path.unlink(missing_ok=True)

    baseline = _latest_round_gox_baseline(fog)
    next_exp_top5_tbl = _build_next_experiment_topk_table(
        selected_unique,
        bo_run_id=bo_run_id,
        top_k=5,
        baseline=baseline,
        exploration_ratio=cfg.exploration_ratio,
        priority_weight_fog=cfg.priority_weight_fog,
        priority_weight_t50=cfg.priority_weight_t50,
        priority_weight_ei=cfg.priority_weight_ei,
    )
    next_exp_top5_tbl.to_csv(next_exp_top5_out, index=False)
    legacy_next_exp_top5_out = out_dir / f"next_experiment_top5__{bo_run_id}.csv"
    if legacy_next_exp_top5_out.is_file():
        legacy_next_exp_top5_out.unlink(missing_ok=True)
    with open(map_quality_out, "w", encoding="utf-8") as f:
        json.dump(map_quality, f, indent=2, ensure_ascii=False)

    outputs: Dict[str, Path] = {
        "training_data": learning_out,
        "candidate_log": model_log_out,
        "suggestions": suggestions_out,
        "fog_rank_by_round": fog_rank_round_out,
        "fog_rank_all": fog_rank_all_out,
        "t50_rank_by_round": t50_rank_round_out,
        "t50_rank_all": t50_rank_all_out,
        "next_experiment_top5": next_exp_top5_out,
        "map_quality": map_quality_out,
    }

    # Figures.
    if cfg.write_plots:
        _log_col = "log_fog_corrected" if "log_fog_corrected" in learning.columns else "log_fog"
        # Group by (polymer_id, round_id) if round_id exists, otherwise by polymer_id only
        # This ensures each round's data is plotted separately
        if "round_id" in learning.columns:
            group_cols = ["polymer_id", "round_id"]
        else:
            group_cols = ["polymer_id"]
        observed = (
            learning.groupby(group_cols, as_index=False)
            .agg(
                frac_MPC=("frac_MPC", "mean"),
                frac_BMA=("frac_BMA", "mean"),
                frac_MTAC=("frac_MTAC", "mean"),
                mean_log_fog=(_log_col, "mean"),
            )
        )

        ternary_mean = out_dir / f"ternary_mean_log_fog__{bo_run_id}.png"
        ternary_std = out_dir / f"ternary_std_log_fog__{bo_run_id}.png"
        ternary_ei = out_dir / f"ternary_ei__{bo_run_id}.png"
        ternary_ucb = out_dir / f"ternary_ucb__{bo_run_id}.png"

        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="pred_log_fog_mean",
            title="Predicted mean log(FoG)",
            cbar_label="Predicted mean log(FoG)",
            out_path=ternary_mean,
        )
        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="pred_log_fog_std",
            title="Predictive std log(FoG)",
            cbar_label="Predictive std log(FoG)",
            out_path=ternary_std,
            std_color_gamma=getattr(cfg, "std_color_gamma", 8.0),
        )
        # Check if EI is collapsed to determine title
        ei_title = "Expected Improvement"
        if map_quality.get("ei_collapsed", False):
            ei_title = "Expected Improvement (no promising region)"
        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="ei",
            title=ei_title,
            cbar_label="EI",
            out_path=ternary_ei,
        )
        _plot_ternary_field(
            plot_cand,
            observed,
            value_col="ucb",
            title="Upper Confidence Bound",
            cbar_label="UCB",
            out_path=ternary_ucb,
        )

        # Generate ternary 2x2 panels (same order as xy_2x2: Mean, Std, EI, UCB)
        ternary_2x2 = out_dir / f"ternary_2x2_mean_std_ei_ucb__{bo_run_id}.png"
        _plot_ternary_2x2_panels(
            plot_cand,
            observed,
            bo_run_id,
            ternary_2x2,
            std_color_gamma=getattr(cfg, "std_color_gamma", 8.0),
            acq_mode_used=None,  # run_bo doesn't use acq_mode_used parameter
        )

        observed_vs_pred = out_dir / f"observed_vs_predicted__{bo_run_id}.png"
        ternary_top = out_dir / f"ternary_mean_with_top_candidates__{bo_run_id}.png"
        ternary_sampling = out_dir / f"ternary_sampling_by_round__{bo_run_id}.png"
        ternary_resid = out_dir / f"ternary_residual__{bo_run_id}.png"
        fog_t50_scatter = out_dir / f"fog_vs_t50_scatter__{bo_run_id}.png"
        model_diag = out_dir / f"bo_model_diagnostic__{bo_run_id}.png"

        _plot_observed_vs_predicted(learning, gp, cfg, bo_run_id, observed_vs_pred)
        _plot_ternary_mean_with_top_candidates(plot_cand, observed, next_exp_top5_tbl, bo_run_id, ternary_top)
        _plot_ternary_sampling_by_round(learning, bo_run_id, ternary_sampling)
        _plot_ternary_residual(learning, gp, cfg, bo_run_id, ternary_resid)
        polymer_color_map = _load_polymer_colors(polymer_colors_path) if polymer_colors_path else {}
        _plot_fog_vs_t50_scatter(rank_all, bo_run_id, fog_t50_scatter, color_map=polymer_color_map or None)
        gp_hyperparams = _gp_hyperparams_for_summary(gp, cfg.use_simplex_gp, getattr(cfg, "use_bma_mtac_coords", False))
        _plot_model_diagnostic(map_quality, gp_hyperparams, bo_run_id, model_diag)

        # 2x2 panels: (x,y) or (BMA, MTAC) for 2D GP; none for simplex GP.
        xy_2x2 = None
        bma_mtac_2x2 = None
        if not cfg.use_simplex_gp:
            if getattr(cfg, "use_bma_mtac_coords", False):
                bma_mtac_2x2 = out_dir / f"bma_mtac_2x2_mean_std_ei_ucb__{bo_run_id}.png"
                _plot_bma_mtac_2x2_panels(gp, learning, cfg, bo_run_id, bma_mtac_2x2)
                xy_2x2 = out_dir / f"xy_2x2_mean_std_ei_ucb__{bo_run_id}.png"
                _plot_xy_2x2_from_bma_mtac_model(gp, learning, cfg, bo_run_id, xy_2x2)
            else:
                xy_2x2 = out_dir / f"xy_2x2_mean_std_ei_ucb__{bo_run_id}.png"
                _plot_xy_2x2_panels(gp, learning, cfg, bo_run_id, xy_2x2)

        fog_bar_all = out_dir / f"fog_ranking_all__{bo_run_id}.png"
        t50_bar_all = out_dir / f"t50_ranking_all__{bo_run_id}.png"
        bar_color_map = _load_polymer_colors(polymer_colors_path) if polymer_colors_path else None
        if not fog_all_tbl.empty:
            _plot_ranking_bar(
                fog_all_tbl,
                value_col="mean_fog",
                label_col="polymer_id",
                title="FoG ranking (all rounds)",
                xlabel="Mean FoG",
                out_path=fog_bar_all,
                color_map=bar_color_map,
            )
        if not t50_all_tbl.empty:
            # Filter out rows with NaN mean_t50_min
            t50_all_tbl_valid = t50_all_tbl[t50_all_tbl["mean_t50_min"].notna()].copy()
            if not t50_all_tbl_valid.empty:
                _plot_ranking_bar(
                    t50_all_tbl_valid,
                    value_col="mean_t50_min",
                    label_col="polymer_id",
                    title="t50 ranking (all rounds)",
                    xlabel="Mean t50 [min]",
                    out_path=t50_bar_all,
                    color_map=bar_color_map,
                )
        plot_outputs: Dict[str, Path] = {
            "ternary_mean": ternary_mean,
            "ternary_std": ternary_std,
            "ternary_ei": ternary_ei,
            "ternary_ucb": ternary_ucb,
            "ternary_2x2_panels": ternary_2x2,
            "observed_vs_predicted": observed_vs_pred,
            "ternary_mean_with_top_candidates": ternary_top,
            "ternary_sampling_by_round": ternary_sampling,
            "ternary_residual": ternary_resid,
            "fog_vs_t50_scatter": fog_t50_scatter,
            "bo_model_diagnostic": model_diag,
            "fog_rank_bar_all": fog_bar_all,
            "t50_rank_bar_all": t50_bar_all,
        }
        if xy_2x2 is not None:
            plot_outputs["xy_2x2_panels"] = xy_2x2
        if bma_mtac_2x2 is not None:
            plot_outputs["bma_mtac_2x2_panels"] = bma_mtac_2x2
        outputs.update(plot_outputs)

    # BO summary.
    bo_summary = {
        "run_id": bo_run_id,
        "bo_run_id": bo_run_id,
        "n_training_rows": int(len(learning)),
        "n_unique_polymers": int(learning["polymer_id"].nunique()),
        "n_candidates": int(len(cand)),
        "n_selected": int(len(selected)),
        "objective": "log_fog",
        "objective_column": "log_fog_corrected",
        "config": {
            "n_suggestions": cfg.n_suggestions,
            "exploration_ratio": cfg.exploration_ratio,
            "anchor_fraction": cfg.anchor_fraction,
            "replicate_fraction": cfg.replicate_fraction,
            "anchor_count": cfg.anchor_count,
            "replicate_count": cfg.replicate_count,
            "anchor_polymer_ids": list(cfg.anchor_polymer_ids),
            "use_exact_anchor_compositions": cfg.use_exact_anchor_compositions,
            "replicate_source": cfg.replicate_source,
            "candidate_step": cfg.candidate_step,
            "min_component": cfg.min_component,
            "min_distance_between": cfg.min_distance_between,
            "min_distance_to_train": cfg.min_distance_to_train,
            "ei_xi": cfg.ei_xi,
            "ucb_kappa": cfg.ucb_kappa,
            "random_state": cfg.random_state,
            "apply_round_anchor_correction": cfg.apply_round_anchor_correction,
            "min_anchor_polymers": cfg.min_anchor_polymers,
            "enable_heteroskedastic_noise": cfg.enable_heteroskedastic_noise,
            "noise_rel_min": cfg.noise_rel_min,
            "noise_rel_max": cfg.noise_rel_max,
            "priority_weight_fog": cfg.priority_weight_fog,
            "priority_weight_t50": cfg.priority_weight_t50,
            "priority_weight_ei": cfg.priority_weight_ei,
            "use_simplex_gp": cfg.use_simplex_gp,
            "use_bma_mtac_coords": getattr(cfg, "use_bma_mtac_coords", False),
            "sparse_force_isotropic": bool(cfg.sparse_force_isotropic),
            "sparse_isotropic_max_unique_points": int(cfg.sparse_isotropic_max_unique_points),
            "sparse_isotropic_apply_min_below_n": int(getattr(cfg, "sparse_isotropic_apply_min_below_n", 10)),
            "min_length_scale_sparse_isotropic": float(getattr(cfg, "min_length_scale_sparse_isotropic", 0.2)),
            "sparse_use_trend": bool(cfg.sparse_use_trend),
            "sparse_trend_max_unique_points": int(cfg.sparse_trend_max_unique_points),
            "trend_ridge": float(cfg.trend_ridge),
            "sparse_explore_max_unique_points": int(getattr(cfg, "sparse_explore_max_unique_points", 15)),
            "sparse_explore_distance_weight": float(getattr(cfg, "sparse_explore_distance_weight", 0.75)),
            "sparse_combo_distance_weight": float(getattr(cfg, "sparse_combo_distance_weight", 0.40)),
            "ternary_plot_step": float(cfg.ternary_plot_step),
            "surrogate_map_xy_grid": int(getattr(cfg, "surrogate_map_xy_grid", 361)),
            "n_unique_design_points": n_unique_design_points,
            "force_isotropic_applied": bool(force_isotropic),
            "use_trend_applied": bool(use_trend),
        },
        "anchor_correction": anchor_meta,
        "design_coverage": design_coverage,
        "map_quality": map_quality,
        "gp_hyperparams": _gp_hyperparams_for_summary(
            gp, cfg.use_simplex_gp, getattr(cfg, "use_bma_mtac_coords", False)
        ),
        "selected_top": selected[
            [
                "selection_order",
                "selection_reason",
                "frac_MPC",
                "frac_BMA",
                "frac_MTAC",
                "pred_log_fog_mean",
                "pred_log_fog_std",
                "pred_fog_mean",
                "ei",
                "ucb",
            ]
        ].to_dict(orient="records"),
        "latest_round_gox_baseline": baseline,
    }
    summary_path = out_dir / f"bo_summary__{bo_run_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(bo_summary, f, indent=2, ensure_ascii=False)
    outputs["bo_summary"] = summary_path

    referenced_run_ids = _collect_referenced_run_ids(learning, fog)
    manifest_extra = {
        "referenced_run_ids": referenced_run_ids,
        "objective": "log_fog_corrected",
        "anchor_correction": anchor_meta,
        "output_files": sorted(p.name for p in outputs.values()),
        "bo_learning_path": str(Path(bo_learning_path).resolve()),
        "fog_plate_aware_path": str(Path(fog_plate_aware_path).resolve()),
    }
    manifest = build_run_manifest_dict(
        run_id=bo_run_id,
        input_paths=[Path(bo_learning_path), Path(fog_plate_aware_path)],
        git_root=Path.cwd(),
        extra=manifest_extra,
    )
    manifest_path = out_dir / f"bo_manifest__{bo_run_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    outputs["manifest"] = manifest_path
    return outputs
