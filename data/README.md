# Data lake

All data used by the notebooks lives here, in stages. Contents are **git-ignored**
(regenerable and/or large); only this README and the `.gitkeep` files are tracked.

| Folder | Stage | Contents | Produced by |
|--------|-------|----------|-------------|
| `raw/` | source | Vibration `.log` and geometry `.csv`. The full set (~7,800 logs) is large — point `DATA_ROOT` at the external drive, or symlink/copy the source here. | (external) |
| `interim/` | decoded | `raw_cache/` — decoded `.npz` per recording (4 sensors + 2 wheel detectors). | `io.build_raw_cache` |
| `processed/` | features | `feature_store/` — extracted PSD features (parquet, config-fingerprinted); labelled training tables. | `pipeline.build_feature_store` |
| `models/` | artifacts | Trained model artifacts (`.joblib`), each bundling its preprocessing contract. | `model.QualityModel.save` |

These locations are the single source of truth in **`intelliswitch.paths`**
(`paths.RAW`, `paths.RAW_CACHE`, `paths.FEATURE_STORE`, `paths.MODELS`); notebooks and
jobs import from there rather than hard-coding paths. Override the roots with
`INTELLISWITCH_DATA_DIR` / `INTELLISWITCH_MODELS_DIR` (e.g. on a cluster). Nothing here
is needed at inference time except a saved model in `models/` — the serving path takes a
`.log` straight to a prediction.
