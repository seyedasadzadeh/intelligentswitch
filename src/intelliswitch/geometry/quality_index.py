"""Track quality index at given locations.

Port of `fractaldimention2.m`. For each location in `y_locations` (m from the
switch-blade tip), take a `window`-metre window of the longitudinal level and
compute a quality value for each rail, then keep the worse (min) of the two:

- method "std":     value = -std(longitudinal level in window)   (more negative = worse)
- method "fractal": value =  b1 from ffractal (first fractal slope)

Returns one quality value per requested location.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from .fractal import ffractal


def quality_index_at(
    xgrid_m: np.ndarray,
    long_level_RL: np.ndarray,   # shape (2, n): row0 = right rail, row1 = left rail
    y_locations_m,
    cfg: Config | None = None,
) -> np.ndarray:
    """Quality index per location (worst rail). Mirrors fractaldimention2's Y1."""
    cfg = cfg or Config()
    xgrid_m = np.asarray(xgrid_m, dtype=np.float64).ravel()
    yloc = np.atleast_1d(np.asarray(y_locations_m, dtype=np.float64))
    half = cfg.geom_window_m / 2.0
    method = cfg.geom_method

    out = np.full(yloc.size, np.nan)
    for i, xg in enumerate(yloc):
        sel = (xgrid_m >= xg - half) & (xgrid_m < xg + half)
        if sel.sum() < 4:
            continue
        wr = long_level_RL[0, sel]
        wl = long_level_RL[1, sel]
        if method == "fractal":
            br = ffractal(wr, cfg)[0]
            bl = ffractal(wl, cfg)[0]
        else:  # "std"
            br = -np.std(wr)
            bl = -np.std(wl)
        out[i] = min(br, bl)   # worst (most negative / lowest) of the two rails
    return out
