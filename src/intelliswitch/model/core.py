"""Backend-agnostic training, evaluation, and the serializable model artifact.

`QualityModel` is the serving artifact: it bundles the fitted backend with the
exact preprocessing contract (cfg, feature columns, transform) and training
metadata (speed range for out-of-domain checks). Train it with `train(df, kind)`;
predict with `model.predict(df)`; persist with `model.save(path)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from ..config import Config
from .backends import make_backend


# ----------------------------------------------------------------- features
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Raw-PSD feature columns (`f<hz>`)."""
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def build_X(df: pd.DataFrame, cfg: Config, fcols: list[str] | None = None) -> np.ndarray:
    """Location-agnostic model input: sqrt(PSD) / speed**power."""
    fcols = fcols or feature_columns(df)
    P = df[fcols].to_numpy(float)
    X = np.sqrt(np.clip(P, 0.0, None)) if cfg.feature_sqrt else P
    p = cfg.feature_speed_norm_power
    if p:
        X = X / np.power(df["speed_kmh"].to_numpy(float)[:, None], p)
    return X


def build_xy(df: pd.DataFrame, cfg: Config, fcols: list[str] | None = None):
    return build_X(df, cfg, fcols), df["y"].to_numpy(float)


def _mask_by_ranges(days: np.ndarray, ranges) -> np.ndarray:
    m = np.zeros(days.shape, dtype=bool)
    for lo, hi in ranges:
        m |= (days >= lo) & (days <= hi)
    return m


def _clean(df: pd.DataFrame, fcols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=fcols + ["y", "days", "speed_kmh"])


# ----------------------------------------------------------------- artifact
@dataclass
class QualityModel:
    kind: str
    backend: object                 # fitted backend (picklable)
    cfg: Config                     # the exact preprocessing contract
    feature_cols: list
    n_train: int
    train_r2: float
    train_speed_range: tuple        # (min, max) speed seen in training -> OOD check
    info: dict = field(default_factory=dict)
    label: str = ""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.backend.predict(build_X(df, self.cfg, self.feature_cols))

    def save(self, path: str) -> str:
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str) -> "QualityModel":
        return joblib.load(path)


# ----------------------------------------------------------------- training
def train(df: pd.DataFrame, cfg: Config | None = None,
          kind: str | None = None, label: str = "") -> QualityModel:
    """Fit one model on the training day-ranges of `df` (pooled across locations)."""
    cfg = cfg or Config()
    kind = kind or cfg.model_kind
    fcols = feature_columns(df)
    d = _clean(df, fcols)
    tr = d[_mask_by_ranges(d["days"].to_numpy(), cfg.train_day_ranges)]
    if len(tr) < 3:
        raise ValueError(f"too few training rows ({len(tr)}) for '{label or kind}'")
    X, y = build_xy(tr, cfg, fcols)
    backend = make_backend(kind, cfg).fit(X, y)
    yhat = backend.predict(X)
    r2 = float(1 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() or 1))
    spd = tr["speed_kmh"].to_numpy(float)
    return QualityModel(kind, backend, cfg, fcols, len(tr), r2,
                        (float(spd.min()), float(spd.max())), backend.info, label or kind)


def fit_pooled(df, cfg=None, kind=None) -> QualityModel:
    """Train the single location-agnostic model (PLSR or LightGBM)."""
    return train(df, cfg, kind, label="pooled")


# ----------------------------------------------------------------- evaluation
def evaluate_testpoints(model: QualityModel, df: pd.DataFrame,
                        cfg: Config | None = None) -> dict:
    """Per-test-window mean prediction vs. mean reference, MAPE and correlation."""
    cfg = cfg or model.cfg
    d = _clean(df, model.feature_cols)
    days = d["days"].to_numpy()
    preds, refs, mids = [], [], []
    for lo, hi in cfg.test_day_ranges:
        sel = (days >= lo) & (days <= hi)
        if sel.sum() == 0:
            continue
        sub = d[sel]
        preds.append(float(model.predict(sub).mean()))
        refs.append(float(sub["y"].mean()))
        mids.append((lo + hi) / 2.0)
    pred, ref = np.array(preds), np.array(refs)
    if pred.size == 0:
        return dict(n_testpoints=0, mape=np.nan, corr=np.nan,
                    pred=pred, ref=ref, mid_days=np.array(mids))
    mape = 100.0 * np.mean(np.abs(pred - ref) / np.abs(ref))
    corr = float(np.corrcoef(pred, ref)[0, 1]) if pred.size > 1 else np.nan
    return dict(n_testpoints=pred.size, mape=float(mape), corr=corr,
                pred=pred, ref=ref, mid_days=np.array(mids))


def evaluate(model: QualityModel, df: pd.DataFrame, cfg: Config | None = None) -> dict:
    """Overall test-point metrics plus a per-location breakdown (monitoring only)."""
    cfg = cfg or model.cfg
    per = {}
    for li, name in enumerate(cfg.sensor_names):
        sub = df[df["location_idx"] == li]
        if not sub.empty:
            per[name] = evaluate_testpoints(model, sub, cfg)
    return dict(overall=evaluate_testpoints(model, df, cfg), per_location=per)


def leave_one_location_out(df: pd.DataFrame, cfg: Config | None = None,
                           kind: str | None = None) -> dict:
    """Spatial generalization: train on other locations, test on the held-out one."""
    cfg = cfg or Config()
    out = {}
    for li, name in enumerate(cfg.sensor_names):
        tr, te = df[df["location_idx"] != li], df[df["location_idx"] == li]
        if tr.empty or te.empty:
            continue
        try:
            m = train(tr, cfg, kind, label=f"LOLO-not-{name}")
        except ValueError as e:  # noqa: BLE001
            out[name] = {"error": str(e)}
            continue
        out[name] = evaluate_testpoints(m, te, cfg)
    return out


def fit_per_location(df: pd.DataFrame, cfg: Config | None = None,
                     kind: str | None = None) -> dict:
    """Diagnostic/monitoring only: one model per location (not for production)."""
    cfg = cfg or Config()
    out = {}
    for li, name in enumerate(cfg.sensor_names):
        loc = df[df["location_idx"] == li]
        if loc.empty:
            continue
        try:
            m = train(loc, cfg, kind, label=name)
        except ValueError as e:  # noqa: BLE001
            out[name] = {"model": None, "eval": None, "error": str(e)}
            continue
        out[name] = {"model": m, "eval": evaluate_testpoints(m, loc, cfg)}
    return out
