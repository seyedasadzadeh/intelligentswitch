from .core import (
    feature_columns,
    build_X,
    build_xy,
    QualityModel,
    train,
    fit_pooled,
    fit_per_location,
    evaluate,
    evaluate_test,
    leave_one_location_out,
)
from .backends import select_n_components, make_backend, PLSRBackend, LGBMBackend

# legacy aliases
PLSRModel = QualityModel
LocationModel = QualityModel

__all__ = [
    "feature_columns", "build_X", "build_xy",
    "QualityModel", "train", "fit_pooled", "fit_per_location",
    "evaluate", "evaluate_test", "leave_one_location_out",
    "select_n_components", "make_backend", "PLSRBackend", "LGBMBackend",
    "PLSRModel", "LocationModel",
]
