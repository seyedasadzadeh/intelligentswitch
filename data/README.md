# Data lake

All data used by the notebooks lives here, in stages. Contents are **git-ignored**
(regenerable and/or large); only this README and the `.gitkeep` files are tracked.

| Folder | Stage | Contents | Produced by |
|--------|-------|----------|-------------|
| `raw/` | source | Vibration `.log` and geometry `.csv`. The full set (~7,800 logs) is large — point `DATA_ROOT` at the external drive, or symlink/copy the source here. | (external) |
| `intermediate/` | decoded | `raw_cache/` — decoded `.npz` per recording (4 sensors + 2 wheel detectors). | `io.build_raw_cache` |
| `processed/` | features | `feature_store/` — extracted PSD features (parquet, config-fingerprinted); labelled training tables. | `pipeline.build_feature_store` |
| `models/` | artifacts | Trained model artifacts (`.joblib`), each bundling its preprocessing contract. | `model.QualityModel.save` |

Notebooks resolve these as `DATA = Path("../data")` with `RAW_CACHE`, `STORE`, and
`MODELS` underneath. Nothing here is needed at inference time except a saved model in
`models/` — the serving path takes a `.log` straight to a prediction.
