"""Fractal dimension of a 1-D geometry signal.

Port of `ffractal01.m`: the polygon-length-vs-scale method. For a signal `y`,
the polygon length is computed at increasing scales (lambda); the slope of
length-vs-lambda in three lambda bands gives three fractal descriptors `b1,b2,b3`.
`b1` (the short-scale slope) is what the quality index uses.
"""
from __future__ import annotations

import numpy as np

from ..config import Config


def _slope(x: np.ndarray, y: np.ndarray, logspace: bool = False) -> float:
    """Least-squares slope of y ~ a*x + b (MATLAB regress, slope term)."""
    if x.size < 2:
        return np.nan
    if logspace:
        x, y = np.log10(x), np.log10(y)
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def ffractal(y: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """Return [b1, b2, b3] fractal slopes for signal `y` (a row vector)."""
    cfg = cfg or Config()
    landa1, landa2 = cfg.geom_fractal_landa
    y = np.asarray(y, dtype=np.float64).ravel()
    N = y.size
    if N < 4:
        return np.array([np.nan, np.nan, np.nan])

    polygonlength = np.empty(N)
    landa = np.empty(N)
    idx = np.arange(1, N + 1)  # 1-based positions like MATLAB
    for k in range(1, N + 1):
        lk = (N - 1) / k
        xk = np.arange(1, N + 1e-9, lk)          # 1 : lk : N
        xk1 = np.floor(xk).astype(int)
        xk2 = np.ceil(xk).astype(int)
        xk1c = np.clip(xk1, 1, N) - 1
        xk2c = np.clip(xk2, 1, N) - 1
        yk = y[xk1c] + (y[xk2c] - y[xk1c]) * (xk - xk1)
        seg = np.sqrt(np.diff(yk) ** 2 + lk ** 2)
        polygonlength[k - 1] = seg.sum()
        landa[k - 1] = lk * 0.25

    b1 = _slope(landa[landa <= landa1], polygonlength[landa <= landa1])
    m2 = (landa > landa1) & (landa <= landa2)
    b2 = _slope(landa[m2], polygonlength[m2], logspace=True)  # log-log (Hyslip)
    m3 = landa > landa2
    b3 = _slope(landa[m3], polygonlength[m3])
    return np.array([b1, b2, b3])
