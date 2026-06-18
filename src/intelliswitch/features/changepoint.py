"""Changepoint detection — replacement for MATLAB `findchangepts`.

Used in `ProBallast_dataGen.m` for QC:
    chpij = findchangepts(xij,'Statistic','linear','MinThreshold',170,'MinDistance',500)
A wheel-group window is rejected if any changepoint is found (signal is scrambled
/ has level or trend jumps).

MATLAB semantics -> ruptures:
- 'MinThreshold' is the minimum total residual-error improvement required to add a
  changepoint == the PELT penalty `pen`.
- 'MinDistance' is the minimum samples between changepoints == PELT `min_size`.
- 'Statistic','linear' models each segment with mean+slope; 'l2' models mean only.

`cp_statistic="linear"` approximates the MATLAB default used in dataGen; `"l2"` is a
faster mean-shift detector. The penalty almost certainly needs re-tuning against
MATLAB on sample windows — see scripts/validate_features.py.
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt

from ..config import Config


def find_changepoints(x: np.ndarray, cfg: Config | None = None) -> list[int]:
    """Return indices of detected changepoints (excluding the final endpoint)."""
    cfg = cfg or Config()
    x = np.asarray(x, dtype=np.float64).ravel()

    # optional decimation (speed knob — see Config.cp_decimate)
    dec = max(1, int(cfg.cp_decimate))
    if dec > 1:
        x = x[::dec]
    min_size = max(2, cfg.cp_min_distance // dec)

    if x.size < 2 * min_size:
        return []

    # KernelCPD with a linear kernel is an exact (C-accelerated dynamic-program)
    # mean-change detector — fast and the closest practical match to MATLAB's
    # findchangepts. PELT in pure Python is ~100x slower at these window sizes.
    kernel = "rbf" if cfg.cp_statistic == "rbf" else "linear"
    algo = rpt.KernelCPD(kernel=kernel, min_size=min_size).fit(x.reshape(-1, 1))
    bkps = algo.predict(pen=cfg.cp_min_threshold)
    # ruptures appends the series length as the last "breakpoint"; drop it,
    # and map indices back to the original sample rate.
    return [b * dec for b in bkps if b < x.size]


def has_changepoint(x: np.ndarray, cfg: Config | None = None) -> bool:
    """True if the window contains at least one changepoint (i.e. reject for PSD)."""
    return len(find_changepoints(x, cfg)) > 0
