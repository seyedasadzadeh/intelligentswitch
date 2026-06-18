"""Read raw trackside `.log` recordings.

Port of the data-loading part of `ProBallast_dataGen.m`:
    DTA = importdata(datee);
    DTA(:,1:24) = DTA(:,1:24)*10/32768/0.01;   % accelerometer counts -> g

A `.log` file is whitespace-separated, one sample per row, `n_cols` columns:
cols 1..24 = accelerometers, cols 28/29 = wheel detectors W1/W2 (counts, not g).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config


def read_log(path: str | Path, cfg: Config | None = None) -> np.ndarray:
    """Load a `.log` file and convert accelerometer channels to g.

    Returns the full (N, n_cols) array with cols 0..n_accel_cols-1 scaled to g;
    wheel-detector columns are left as raw counts.
    """
    cfg = cfg or Config()
    # pandas C parser is ~10x faster than np.loadtxt for these large files.
    raw = pd.read_csv(
        path, sep=r"\s+", header=None, dtype=np.float64, engine="c"
    ).to_numpy()
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < cfg.n_cols:
        raise ValueError(
            f"{path}: expected >= {cfg.n_cols} columns, got {raw.shape[1]}"
        )
    arr = raw.astype(np.float64, copy=True)
    arr[:, : cfg.n_accel_cols] *= cfg.adc_to_g
    return arr


def extract_channels(arr: np.ndarray, cfg: Config | None = None):
    """Split a loaded array into sensor signals (g) and the two wheel detectors.

    Returns (channels, w1, w2) where `channels` is a dict name -> 1-D signal.
    """
    cfg = cfg or Config()
    channels = {
        name: arr[:, idx]
        for name, idx in zip(cfg.sensor_names, cfg.accel_channel_idx)
    }
    w1 = arr[:, cfg.wheel_det_idx[0]]
    w2 = arr[:, cfg.wheel_det_idx[1]]
    return channels, w1, w2


def parse_days_before(path: str | Path, cfg: Config | None = None) -> float:
    """Parse acquisition date from a `yymmdd_HHMMSS.log` filename and return the
    (negative) number of days before `cfg.reference_date`.

    Approximates MATLAB `-days365(date, '01-01-2019')` with a calendar-day diff;
    refine to the 365-day convention if exact label alignment requires it.
    """
    cfg = cfg or Config()
    stem = Path(path).stem
    yymmdd = stem.split("_")[0]
    date = datetime.strptime("20" + yymmdd, "%Y%m%d")
    ref = datetime.fromisoformat(cfg.reference_date)
    return -float((ref - date).days)
