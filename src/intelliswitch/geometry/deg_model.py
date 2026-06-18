"""Degradation model and label assignment.

Replaces `degmodelGen.m`. From the aligned campaigns, compute the quality index
at each sensor location for every campaign -> a `DegModel` of (days, quality).
A label for a vibration recording is then `quality interpolated at the recording's
day`, for the matching sensor location.

Deliberate simplification vs. the MATLAB: `degmodelGen.m` prepends three synthetic
points hand-modelling the 2016-09-30 tamping. That is experiment-specific label
engineering; here the baseline uses measured campaigns only. Add a tamping
augmentation later if required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from .align import AlignedSC
from .quality_index import quality_index_at


@dataclass
class DegModel:
    days: np.ndarray          # (n_camp,) negative days before reference_date, ascending
    quality: np.ndarray       # (n_loc, n_camp) quality index per location per campaign
    locations_m: np.ndarray   # (n_loc,) sensor locations (m from tip)


def build_deg_model(aligned: AlignedSC, cfg: Config | None = None) -> DegModel:
    cfg = cfg or Config()
    locs = np.asarray(cfg.geom_sensor_locations_m, dtype=float)
    n_camp = aligned.days.size
    Q = np.full((locs.size, n_camp), np.nan)
    for c in range(n_camp):
        xgrid = aligned.x_blade_m[:, c]
        ll = np.vstack([aligned.long_R[:, c], aligned.long_L[:, c]])
        Q[:, c] = quality_index_at(xgrid, ll, locs, cfg)

    order = np.argsort(aligned.days)
    return DegModel(aligned.days[order], Q[:, order], locs)


def label_for_day(deg: DegModel, day: float, location_idx: int) -> float:
    """Quality index at `day` for one sensor location (linear interp, clamped)."""
    d, q = deg.days, deg.quality[location_idx]
    good = np.isfinite(q)
    if good.sum() < 2:
        return np.nan
    return float(np.interp(day, d[good], q[good]))


def attach_labels(features: pd.DataFrame, deg: DegModel, cfg: Config | None = None):
    """Add a `y` column to a feature table using each row's day + location_idx."""
    if features.empty:
        features = features.copy()
        features["y"] = []
        return features
    out = features.copy()
    out["y"] = [
        label_for_day(deg, float(r.days), int(r.location_idx))
        for r in out.itertuples()
    ]
    return out
