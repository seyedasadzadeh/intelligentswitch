"""End-to-end feature extraction for one `.log` recording.

Combines the `ProBallast_dataGen.m` pipeline (read -> filter -> detect wheels ->
window each wheel-group -> changepoint QC -> PSD) with the wheel-group acceptance
QC from `ProBallast_PLSR_02.m` (sleeper-/wheel-passing-frequency power thresholds).

Output: one feature row per *accepted* wheel-group, with the full PSD band as the
feature vector plus metadata (sensor, speed, days, QC powers). This is the
inference-facing function that the serving app will call on a single recording.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ..config import Config
from ..io.log_reader import parse_days_before
from ..io.raw_cache import RawRecord, decode_log
from .changepoint import has_changepoint
from .psd import psd_on_grid
from .wheel_detect import detect_wheels, sensor_group_times, WheelGeometry


@dataclass
class FeatureResult:
    features: pd.DataFrame          # one row per accepted wheel-group
    geom: WheelGeometry | None
    days: float | None
    accepted: bool                  # did the recording pass train-level QC?
    reason: str = ""                # why rejected (if not accepted)
    sensor_ok: dict = field(default_factory=dict)


def _bandpass(x: np.ndarray, cfg: Config) -> np.ndarray:
    b, a = butter(cfg.bp_order, np.asarray(cfg.bp_band_hz) / (cfg.fs / 2), btype="band")
    return filtfilt(b, a, x)


def _band_max(p1: np.ndarray, grid: np.ndarray, centre: float, half: float) -> float:
    """Max PSD within [centre-half, centre+half] Hz on the feature grid."""
    sel = (grid >= centre - half) & (grid <= centre + half)
    vals = p1[sel]
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else np.nan


def _sensor_ok(x: np.ndarray, wh1_a: float, speed_kmh: float, cfg: Config) -> bool:
    """Per-sensor QC from dataGen (timing/saturation/strength)."""
    ax = np.abs(x)
    thr = np.quantile(ax, cfg.firstvib_quantile)
    hits = np.where(ax > thr)[0]
    if hits.size == 0:
        return False
    frstvib = hits[0] / cfg.fs
    tol = cfg.sig_pow_threshold_base / speed_kmh
    if abs(frstvib - wh1_a) > tol:
        return False
    if np.max(ax) > cfg.sat_abs_max:
        return False
    if np.quantile(x, 0.999) < cfg.weak_q999_min:
        return False
    return True


def extract_features(source: str | Path | RawRecord,
                     cfg: Config | None = None) -> FeatureResult:
    """Extract features from a `.log` path or a pre-decoded RawRecord.

    The serving app passes a `.log` path (decoded inline); the training pipeline
    may pass a cached RawRecord to skip re-reading text.
    """
    cfg = cfg or Config()
    rec = source if isinstance(source, RawRecord) else decode_log(source, cfg)
    channels, w1, w2 = rec.channels, rec.w1, rec.w2
    days = parse_days_before(rec.name, cfg)
    N = rec.n_samples

    geom = detect_wheels(w1, w2, cfg)
    empty = pd.DataFrame()
    if geom is None:
        return FeatureResult(empty, None, days, False, "wheel detectors unreliable")

    # ---- train-level QUALITY gate ------------------------------------------
    # Only genuine data-quality / consistency checks reject here. Train SPEED is
    # NOT a quality issue (and the model normalizes by speed), so it is recorded
    # as metadata and filtered downstream via pipeline.select_trains. n_groups
    # (train type) is also a selection concern, kept here only to ensure the
    # wheel grouping is self-consistent.
    if geom.n_wheels != 2 * geom.n_groups:
        return FeatureResult(empty, geom, days, False,
                             f"n_wheels({geom.n_wheels}) != 2*n_groups({geom.n_groups})")
    if cfg.n_groups_required and geom.n_groups not in cfg.n_groups_required:
        return FeatureResult(empty, geom, days, False,
                             f"n_groups {geom.n_groups} not in required {cfg.n_groups_required}")

    # band-pass filter each sensor channel
    xf = {name: _bandpass(sig, cfg) for name, sig in channels.items()}
    gs_times = sensor_group_times(geom, cfg)

    km = {"A1": cfg.km_A1, "A2": cfg.km_A2, "A7": cfg.km_A7, "A11": cfg.km_A11}
    grid = cfg.freq_grid
    win_min = geom.timelag_g * cfg.fs

    sleeper_f = geom.speed_kmh / 3.6 / cfg.sleeper_spacing_m
    wheel_f = geom.speed_kmh / 3.6 / cfg.wheel_spacing_m

    rows = []
    sensor_ok = {}
    for li, name in enumerate(cfg.sensor_names):
        x = xf[name]
        wh1_a = geom.wh1_w1 + (cfg.km_W1 - km[name]) / geom.speed_kmh * 3600.0
        ok = _sensor_ok(x, wh1_a, geom.speed_kmh, cfg)
        sensor_ok[name] = ok
        if not ok:
            continue
        for gi, tc in enumerate(gs_times[name]):
            lo = max(1000, int(np.ceil((tc - geom.timelag_g) * cfg.fs)))
            hi = min(N, int(np.floor((tc + geom.timelag_g) * cfg.fs)))
            xij = x[lo:hi]
            if xij.size < win_min:
                continue
            xij = xij - xij.mean()
            if has_changepoint(xij, cfg):           # scrambled signal -> drop
                continue
            p1 = psd_on_grid(xij, cfg)
            p_sleeper = _band_max(p1, grid, sleeper_f, cfg.sleeper_band_hz)
            p_wheel = _band_max(p1, grid, wheel_f, cfg.wheel_band_hz)
            # _02 acceptance test
            if not (p_sleeper < cfg.accept_pSleeper_max
                    and p_wheel < cfg.accept_pWheel_max):
                continue
            row = {f"f{int(round(f))}": v for f, v in zip(grid, p1)}
            row.update(
                sensor=name, location_idx=li, group_idx=gi,
                speed_kmh=geom.speed_kmh, n_groups=geom.n_groups,
                n_wheels=geom.n_wheels, days=days,
                p_sleeper=p_sleeper, p_wheel=p_wheel,
                source=rec.name,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    return FeatureResult(df, geom, days, not df.empty,
                         "" if not df.empty else "no accepted wheel-groups",
                         sensor_ok)
