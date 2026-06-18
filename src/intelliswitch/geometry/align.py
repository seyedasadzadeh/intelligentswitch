"""Spatial alignment of repeated geometry campaigns for one S&C.

Port of the per-S&C body of `GdataAlign.m` (defaults to Tommerup, KnudeRef k=15).

Pipeline per campaign CSV:
  read -> filter to this S&C (BTR, Spor, Km window) -> apply the type-based
  orientation transform -> collect longitudinal level (R/L).
Then across campaigns:
  sort by date -> cross-correlate each campaign's right-rail level against the
  previous one over a +/- search range -> cumulative shift -> origin set so the
  max-gauge point (crossing tip) sits at a fixed x.

Out of scope (BallDeg05 territory): tamping detection, ALC tonnage, seasonal
degradation-rate modelling.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config

# Danish column names in the campaign CSVs.
COL_KM = "Km"
COL_LONG_R, COL_LONG_L = "Højd_R_D1", "Højd_L_D1"
COL_CORG_R = "Corrugation_30-300mm_DepthRight"
COL_CORG_L = "Corrugation_30-300mm_DepthLeft"
COL_GAUGE = "Sporvidde"


@dataclass
class AlignedSC:
    X_km: np.ndarray          # (n, n_camp) aligned spatial coordinate (km)
    long_R: np.ndarray        # (n, n_camp) right-rail longitudinal level
    long_L: np.ndarray        # (n, n_camp) left-rail longitudinal level
    gauge: np.ndarray         # (n, n_camp)
    days: np.ndarray          # (n_camp,) negative days before reference_date
    dates: list               # campaign datetimes, sorted ascending
    x_blade_m: np.ndarray     # (n, n_camp) coordinate in m from SWITCH-BLADE tip
                              # (gauge origin -> crossing nose sits near +37.7 m)


def _campaign_date(name: str) -> datetime:
    """'Bladenr 8252, Campaign 2016 27 02.csv' -> datetime (format 'yyyy dd MM')."""
    token = name.split("Campaign ")[-1].rsplit(".csv", 1)[0].strip()
    return datetime.strptime(token, "%Y %d %m")


def _spor_int(series: pd.Series) -> np.ndarray:
    """Leading integer of the 'Spor' column (handles values like '3a')."""
    return (
        series.astype(str).str.extract(r"(\d+)", expand=False).astype(float).to_numpy()
    )


def align_campaigns(
    campaign_dir: str | Path, cfg: Config | None = None
) -> AlignedSC:
    cfg = cfg or Config()
    ref = datetime.fromisoformat(cfg.reference_date)

    # type-based window adjustment (GdataAlign lines 10-19)
    km_lo, km_hi = cfg.sc_kmbar - 0.060, cfg.sc_kmbar + 0.060  # generous; refined below
    if cfg.sc_type in (1, 3):
        km_hi = cfg.sc_kmbar + 0.059
        km_lo = cfg.sc_kmbar - 0.060
    else:  # 5, 7
        km_lo = cfg.sc_kmbar - 0.059
        km_hi = cfg.sc_kmbar + 0.060

    files = sorted(Path(campaign_dir).glob(f"Bladenr {cfg.sc_bladenr}*.csv"))
    tip_off_km = cfg.geom_tip_offset_m / 1000.0

    cols = []  # list of dicts per accepted campaign
    for f in files:
        df = pd.read_csv(f, sep=cfg.geom_sep, encoding="utf-8-sig")
        spor = _spor_int(df["Spor"])
        mask = (
            (df["BTR"].to_numpy() == cfg.sc_btr)
            & (spor == cfg.sc_spor)
            & (df[COL_KM].to_numpy() >= km_lo)
            & (df[COL_KM].to_numpy() <= km_hi)
        )
        sub = df[mask]
        if len(sub) == 0:
            continue
        x = sub[COL_KM].to_numpy(dtype=float)
        lr = sub[COL_LONG_R].to_numpy(dtype=float)
        ll = sub[COL_LONG_L].to_numpy(dtype=float)
        g = sub[COL_GAUGE].to_numpy(dtype=float)
        if np.all(lr == cfg.geom_null) or np.all(lr == 0):
            continue

        # type-based orientation transform (GdataAlign lines 104-121)
        if cfg.sc_type == 1:
            x = x - (cfg.sc_kmbar - tip_off_km)
        elif cfg.sc_type == 3:
            x = x - (cfg.sc_kmbar - tip_off_km)
            lr, ll = ll, lr           # swap rails
        elif cfg.sc_type == 5:
            x = -x + (cfg.sc_kmbar + tip_off_km)
            x, lr, ll, g = x[::-1], lr[::-1], ll[::-1], g[::-1]
            lr, ll = ll, lr
        elif cfg.sc_type == 7:
            x = -x + (cfg.sc_kmbar + tip_off_km)
            x, lr, ll, g = x[::-1], lr[::-1], ll[::-1], g[::-1]

        g = np.where(g == cfg.geom_null, 0.0, g)
        cols.append(dict(date=_campaign_date(f.name), x=x, lr=lr, ll=ll, g=g))

    if not cols:
        raise ValueError("no campaigns matched this S&C")

    # stack to a common length (truncate to the shortest campaign)
    n = min(c["x"].size for c in cols)
    cols.sort(key=lambda c: c["date"])          # ascending in time
    X = np.column_stack([c["x"][:n] for c in cols])
    YR = np.column_stack([c["lr"][:n] for c in cols])
    YL = np.column_stack([c["ll"][:n] for c in cols])
    G = np.column_stack([c["g"][:n] for c in cols])
    dates = [c["date"] for c in cols]
    days = np.array([-float((ref - d).days) for d in dates])

    # ---- cross-correlation alignment on the right-rail level ----------------
    sig_len = min(cfg.geom_signal_length, n)
    step = cfg.geom_sample_m                      # 0.25 m
    per_m = int(round(1.0 / step))                # samples per metre (4)
    neib = cfg.geom_align_neib_m
    shifts = np.zeros(X.shape[1])
    for gc in range(1, X.shape[1]):
        best_corr, best_s = -np.inf, 0.0
        for s in np.arange(-neib, neib + step, step):
            d = int(round(per_m * s))
            try:
                if s <= 0:
                    a = YR[0 : sig_len + d, gc - 1]
                    b = YR[-d : sig_len, gc]
                else:
                    a = YR[0 : sig_len - d, gc]
                    b = YR[d:sig_len, gc - 1]
            except Exception:  # noqa: BLE001
                continue
            if a.size < 8 or a.size != b.size:
                continue
            c = np.corrcoef(a, b)[0, 1]
            if np.isfinite(c) and c > best_corr:
                best_corr, best_s = c, s
        shifts[gc] = best_s
    cumshift = np.cumsum(shifts)                  # metres
    shiftedX = X + cumshift[np.newaxis, :] / 1000.0  # km

    # origin: put the max-gauge point (crossing tip) at geom_gauge_tip_x
    gmax = G.max()
    col = int(np.argmax(G.max(axis=0)))
    pts = shiftedX[G[:, col] >= gmax, col]
    shift_for_gauge = cfg.geom_gauge_tip_x - (pts.mean() if pts.size else 0.0)
    X_km = shiftedX + shift_for_gauge
    # knude.X is referenced so the switch-blade tip is at x=0 and the crossing
    # nose (max gauge) at geom_gauge_tip_x. BallDeg05 uses xgrid = 1000*knude.X,
    # i.e. metres from the switch-blade tip. Sensor locations are in this frame.
    x_blade_m = X_km * 1000.0

    return AlignedSC(X_km, YR, YL, G, days, dates, x_blade_m)
