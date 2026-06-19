"""Centralized data & model locations (medallion tiers).

Single source of truth for where the pipeline reads and writes, so notebooks,
scripts, and Phase-2 jobs never hard-code paths. Defaults sit under the repo's
`data/` lake; override the roots with environment variables for cluster / CI runs:

    INTELLISWITCH_DATA_DIR    -> base of the data lake   (default: <repo>/data)
    INTELLISWITCH_MODELS_DIR  -> trained model artifacts (default: <data>/models)

Tiers:  raw  ->  interim (raw_cache)  ->  processed (feature_store);  models alongside.
Raw source recordings are large and typically live on an external drive — point
a notebook's `DATA_ROOT` at them, or symlink/copy a subset into `RAW`.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- roots (env-overridable) -----------------------------------------------
DATA_DIR = Path(os.environ.get("INTELLISWITCH_DATA_DIR", _REPO_ROOT / "data"))
MODELS = Path(os.environ.get("INTELLISWITCH_MODELS_DIR", DATA_DIR / "models"))

# ---- medallion tiers --------------------------------------------------------
RAW = DATA_DIR / "raw"                        # source .log / geometry .csv
INTERIM = DATA_DIR / "interim"                # decoded intermediates
RAW_CACHE = INTERIM / "raw_cache"             # decoded .npz (stage 1)
PROCESSED = DATA_DIR / "processed"            # model-ready artifacts
FEATURE_STORE = PROCESSED / "feature_store"   # extracted PSD features (parquet)


def ensure() -> None:
    """Create the lake directories if missing (idempotent)."""
    for p in (RAW, INTERIM, RAW_CACHE, PROCESSED, FEATURE_STORE, MODELS):
        p.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DATA_DIR", "MODELS", "RAW", "INTERIM", "RAW_CACHE",
    "PROCESSED", "FEATURE_STORE", "ensure",
]
