"""Estimator backends behind a common interface.

Each backend takes a plain (X, y) matrix and exposes `.fit(X, y) -> self`,
`.predict(X) -> 1d array`, and `.info` (backend-specific metadata). They are
picklable so the trained model can be serialized as part of the serving artifact.

Swapping PLSR <-> LightGBM is just `make_backend("plsr"|"lgbm", cfg)`.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from ..config import Config


def select_n_components(X: np.ndarray, y: np.ndarray, cfg: Config | None = None) -> int:
    """Smallest NOC in [min, max] whose cumulative training y-variance-explained
    exceeds the threshold (port of `_02`'s `cumsum(100*PCTVAR(2,:)) > 55`).

    PLS components are nested (sequential deflation), so the training prediction
    with the first k components equals `x_scores[:, :k] @ y_loadings[:, :k].T`.
    We fit ONCE at the max count and read R^2 at every k from that — instead of
    refitting per candidate (which made this the pipeline's bottleneck).
    """
    cfg = cfg or Config()
    n_samples, n_features = X.shape
    kmax = min(cfg.pls_max_components, n_features, max(1, n_samples - 1))
    kmin = max(1, min(cfg.pls_min_components, kmax))

    pls = PLSRegression(n_components=kmax, scale=False).fit(X, y)
    yc = y - y.mean()
    ss_tot = float((yc ** 2).sum()) or 1.0
    # per-component contribution to the centered prediction, then cumulative.
    contrib = pls.x_scores_ * pls.y_loadings_.ravel()[None, :]   # (n, kmax)
    yhat_cum = np.cumsum(contrib, axis=1)                        # (n, kmax)
    r2 = 1.0 - ((yc[:, None] - yhat_cum) ** 2).sum(axis=0) / ss_tot

    for k in range(kmin, kmax + 1):
        if r2[k - 1] > cfg.pls_y_var_threshold:
            return k
    return kmax


class PLSRBackend:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: PLSRegression | None = None
        self.n_components: int | None = None

    def fit(self, X, y):
        self.n_components = select_n_components(X, y, self.cfg)
        self.model = PLSRegression(n_components=self.n_components, scale=False).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).ravel()

    @property
    def info(self) -> dict:
        return {"backend": "plsr", "n_components": self.n_components}


class LGBMBackend:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None

    def fit(self, X, y):
        import lightgbm as lgb  # imported lazily so PLSR-only installs work
        self.model = lgb.LGBMRegressor(**self.cfg.lgbm_params).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()

    @property
    def info(self) -> dict:
        return {"backend": "lgbm", "n_estimators": self.cfg.lgbm_params.get("n_estimators")}


def make_backend(kind: str, cfg: Config):
    kind = (kind or "plsr").lower()
    if kind == "plsr":
        return PLSRBackend(cfg)
    if kind in ("lgbm", "lightgbm"):
        return LGBMBackend(cfg)
    raise ValueError(f"unknown model kind: {kind!r} (use 'plsr' or 'lgbm')")
