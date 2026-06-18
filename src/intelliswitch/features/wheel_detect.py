"""Wheel detection, train speed, axle distances, and wheel grouping.

Port of the wheel-detector / geometry block of `ProBallast_dataGen.m`:
    whssW1 = find(xw1>-500 & xw1<501);  whssW2 = ...
    wh1W1 = whssW1(1)/fs;  wh1W2 = whssW2(1)/fs;
    speed = 68.75/(wh1W2-wh1W1)*3.6;
    ... axle time-distances -> grouping (<6 m) -> per-sensor group arrival times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config


@dataclass
class WheelGeometry:
    speed_kmh: float
    n_wheels: int
    n_groups: int
    axle_td: np.ndarray          # inter-axle time distances (s), first = 0
    axle_td_cum: np.ndarray      # cumulative axle times at W1 (s)
    axle_d: np.ndarray           # inter-axle distances (m), first = 0
    group_td_cum: np.ndarray     # cumulative group-center times at W1 (s)
    timelag_g: float             # half the min inter-group spacing (s) -> window half-width
    wh1_w1: float                # time of first wheel at detector W1 (s)
    groups: list                 # list of member-index lists


def detect_wheels(w1: np.ndarray, w2: np.ndarray, cfg: Config | None = None):
    """Return a WheelGeometry, or None if the detectors are unreliable."""
    cfg = cfg or Config()
    fs = cfg.fs

    idx1 = np.where((w1 > cfg.det_valid_lo) & (w1 < cfg.det_valid_hi))[0]
    idx2 = np.where((w2 > cfg.det_valid_lo) & (w2 < cfg.det_valid_hi))[0]
    if idx1.size == 0 or idx2.size == 0:
        return None

    wh1_w1 = idx1[0] / fs
    wh1_w2 = idx2[0] / fs
    idx1 = idx1[:-1]            # MATLAB drops the last sample of each detector
    if idx1.size < 2 or (wh1_w2 - wh1_w1) == 0:
        return None

    speed_kmh = cfg.det_distance_m / (wh1_w2 - wh1_w1) * 3.6

    d1 = np.diff(idx1) / fs
    axle_td0 = d1[d1 > cfg.axle_min_dt_s]            # inter-axle times
    n_wheels = axle_td0.size + 1
    wheel_diameter = np.sum(np.diff(idx1) == 1) / n_wheels   # contact-duration proxy
    axle_td0 = axle_td0 + wheel_diameter / fs

    axle_td = np.concatenate([[0.0], axle_td0])
    axle_td_cum = np.cumsum(axle_td)
    axle_d = axle_td * speed_kmh / 3.6
    # axle_d_cum = np.cumsum(axle_d)  # available if needed

    # ---- group adjacent wheels closer than group_max_dist_m -----------------
    groups: list[list[int]] = [[0]]
    for w in range(1, axle_d.size):
        if axle_d[w] < cfg.group_max_dist_m:
            groups[-1].append(w)
        else:
            groups.append([w])

    group_td_cum = np.array([axle_td_cum[m].mean() for m in groups])
    if group_td_cum.size < 2:
        return None
    gtd = np.diff(group_td_cum)
    timelag_g = float(gtd.min() / 2.0)

    return WheelGeometry(
        speed_kmh=float(speed_kmh),
        n_wheels=int(n_wheels),
        n_groups=int(group_td_cum.size),
        axle_td=axle_td,
        axle_td_cum=axle_td_cum,
        axle_d=axle_d,
        group_td_cum=group_td_cum,
        timelag_g=timelag_g,
        wh1_w1=float(wh1_w1),
        groups=groups,
    )


def sensor_group_times(geom: WheelGeometry, cfg: Config | None = None) -> dict:
    """Arrival time (s) of each wheel-group centre at each sensor.

    GsA(i) = wh1_w1 + (km_W1 - km_sensor)/speed*3600 + group_td_cum
    Returns {sensor_name: np.ndarray of group-centre times}.
    """
    cfg = cfg or Config()
    km = {"A1": cfg.km_A1, "A2": cfg.km_A2, "A7": cfg.km_A7, "A11": cfg.km_A11}
    out = {}
    for name in cfg.sensor_names:
        wh1_a = geom.wh1_w1 + (cfg.km_W1 - km[name]) / geom.speed_kmh * 3600.0
        out[name] = wh1_a + geom.group_td_cum
    return out
