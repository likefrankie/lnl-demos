"""Simulation helpers shared by the demos.

Notation follows the lectures: X is n x (p+1) with a column of ones, p is the
number of independent variables, and the residual degrees of freedom are n - p - 1.
"""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Model:
    key: str
    formula: str
    names: List[str]             # coefficient labels, intercept first
    x_names: List[str]           # independent-variable labels
    design: Callable[..., np.ndarray]


def design_two_x(n: int, rho: float = 0.3, seed: int = 42) -> np.ndarray:
    """Fixed design with two correlated regressors: x1 ~ 5 + 2 z1, x2 ~ 10 + 3 z2, corr(z1, z2) = rho."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    x1 = 5.0 + 2.0 * z[:, 0]
    x2 = 10.0 + 3.0 * z[:, 1]
    return np.column_stack([np.ones(n), x1, x2])


MODELS = {
    "mlr2": Model(
        key="mlr2",
        formula="y = β₀ + β₁x₁ + β₂x₂ + ε",
        names=["β₀", "β₁", "β₂"],
        x_names=["x₁", "x₂"],
        design=design_two_x,
    ),
}


def critical_value(level: float, df: int, kind: str = "t") -> float:
    """Upper-tail critical value for a two-sided interval: t_{alpha/2, df} or z_{alpha/2}."""
    alpha = 1.0 - level
    if kind == "t":
        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


@dataclass
class Draw:
    eps: np.ndarray
    y: np.ndarray
    b: np.ndarray       # beta_hat
    se: np.ndarray      # SE(beta_hat) with sigma estimated
    L: np.ndarray
    U: np.ndarray
    hit: np.ndarray     # L <= beta <= U
    sigma_hat: float


def draw_and_fit(X: np.ndarray, beta: np.ndarray, sigma: float, rng: np.random.Generator, crit: float) -> Draw:
    """Draw a new error vector for the fixed X, refit OLS, and build the intervals."""
    n, k = X.shape
    eps = rng.normal(0.0, sigma, n)
    y = X @ beta + eps
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    resid = y - X @ b
    sigma2_hat = resid @ resid / (n - k)          # SSR / (n - p - 1)
    se = np.sqrt(sigma2_hat * np.diag(XtX_inv))
    L, U = b - crit * se, b + crit * se
    return Draw(eps=eps, y=y, b=b, se=se, L=L, U=U, hit=(L <= beta) & (beta <= U), sigma_hat=float(np.sqrt(sigma2_hat)))
