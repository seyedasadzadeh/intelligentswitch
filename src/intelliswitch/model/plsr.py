"""Backwards-compatible shim. The model API now lives in `core` (backend-agnostic)
and `backends` (PLSR/LightGBM). `PLSRModel`/`LocationModel` map to `QualityModel`.
"""
from .core import (  # noqa: F401
    feature_columns, build_X, build_xy,
    QualityModel, train, fit_pooled, fit_per_location,
    evaluate, evaluate_test, leave_one_location_out,
)
from .backends import select_n_components  # noqa: F401

PLSRModel = QualityModel
LocationModel = QualityModel
