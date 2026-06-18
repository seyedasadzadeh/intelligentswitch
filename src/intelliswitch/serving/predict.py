"""Single-recording inference — the streaming serving entry point.

Prediction granularity & aggregation hierarchy (per recording):

    (sensor x wheel-set) PSDs  ->  one prediction each   (up to 4 sensors x 5 sets)
        |  level 1: aggregate wheel-sets per sensor  (cfg.agg_wheelsets)
        v
    LOCATION prediction (per sensor)                     <- the informative output
        |  level 2: aggregate locations               (cfg.agg_locations)
        v
    S&C-level indicator                                  <- convenience roll-up

Flow: decode + extract + QC-gate (using the model's own cfg, so serving matches
training). If the recording fails the quality gate -> `rejected` with a reason
(the app shows "prediction not possible"). Otherwise every level is returned, so
the caller can use per-wheel-set, per-location, or the single S&C number, and an
out-of-domain flag marks recordings whose speed is outside the training range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..features import extract_features
from ..io.raw_cache import RawRecord
from ..model.core import QualityModel


def _aggregate(values, method: str) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float({"median": np.median, "mean": np.mean,
                  "min": np.min, "max": np.max}.get(method, np.median)(a))


@dataclass
class WheelSetPrediction:
    sensor: str
    group_idx: int
    quality: float
    speed_kmh: float


@dataclass
class LocationPrediction:
    sensor: str
    quality: float                  # level-1 aggregate of wheel-set predictions
    n_wheelsets: int
    speed_kmh: float
    out_of_domain: bool
    wheelsets: list = field(default_factory=list)   # the individual predictions

    # backward-compatible alias
    @property
    def n_groups_used(self) -> int:
        return self.n_wheelsets


@dataclass
class InferenceResult:
    status: str                       # "ok" | "rejected"
    source: str = ""
    locations: list = field(default_factory=list)   # list[LocationPrediction]
    sc_quality: float | None = None   # level-2 S&C indicator
    out_of_domain: bool = False
    reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    # backward-compatible aliases
    @property
    def predictions(self) -> list:
        return self.locations

    @property
    def aggregate(self) -> float | None:
        return self.sc_quality


def predict_recording(source: str | Path | RawRecord,
                      model: QualityModel) -> InferenceResult:
    """Predict track quality for one recording, or reject it with a reason."""
    cfg: Config = model.cfg
    name = source.name if isinstance(source, RawRecord) else Path(source).name

    res = extract_features(source, cfg)
    if not res.accepted:
        return InferenceResult(
            status="rejected", source=name,
            reason=res.reason or "no wheel-group passed the quality gate",
            meta={"speed_kmh": getattr(res.geom, "speed_kmh", None),
                  "sensor_ok": res.sensor_ok},
        )

    df = res.features.copy()
    df["pred"] = model.predict(df)           # one prediction per (sensor, wheel-set)
    lo, hi = model.train_speed_range

    locations = []
    for sensor, g in df.groupby("sensor"):
        ws = [WheelSetPrediction(str(sensor), int(r.group_idx),
                                 float(r.pred), float(r.speed_kmh))
              for r in g.itertuples()]
        spd = float(g["speed_kmh"].iloc[0])
        locations.append(LocationPrediction(
            sensor=str(sensor),
            quality=_aggregate([w.quality for w in ws], cfg.agg_wheelsets),
            n_wheelsets=len(ws), speed_kmh=spd,
            out_of_domain=(spd < lo or spd > hi), wheelsets=ws,
        ))

    sc = _aggregate([l.quality for l in locations], cfg.agg_locations)
    return InferenceResult(
        status="ok", source=name, locations=locations, sc_quality=sc,
        out_of_domain=any(l.out_of_domain for l in locations),
        meta={"speed_kmh": float(df["speed_kmh"].iloc[0]),
              "n_locations": len(locations),
              "n_wheelsets": int(len(df)),
              "agg_wheelsets": cfg.agg_wheelsets, "agg_locations": cfg.agg_locations,
              "model_kind": model.kind},
    )
