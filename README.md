# Intelliswitch — track-irregularity prediction (Python refactor)

Python port of the MATLAB PLSR pipeline that predicts a **track quality index**
(fractal value or std of longitudinal level) from **trackside vibration recordings**.

The MATLAB → Python port is complete: vibration features, geometry labels, model
training (PLSR + LightGBM), and a streaming serving path all exist. What remains is
wrapping the `serving` layer in an application (Phase 2).

## Scope (ported from MATLAB)

| Stage | MATLAB source | Python module |
|-------|---------------|---------------|
| Read raw `.log` | `ProBallast_dataGen.m` | `intelliswitch.io.log_reader` |
| Decode-once binary cache | (new) | `intelliswitch.io.raw_cache` |
| Wheel detection / speed / grouping | `ProBallast_dataGen.m` | `intelliswitch.features.wheel_detect` |
| Changepoint QC (`findchangepts`) | `ProBallast_dataGen.m`, `vib_detrend.m` | `intelliswitch.features.changepoint` |
| PSD (Welch + FFT) + freq-grid interp | `ProBallast_dataGen.m` | `intelliswitch.features.psd` |
| Feature extraction (per wheel-group) | `ProBallast_dataGen.m` + `ProBallast_PLSR_02.m` | `intelliswitch.features.extract` |
| Geometry → labels (training only) | `GdataAlign.m` → `fractaldimention2.m` → `BallDeg05.m` → `degmodelGen.m` | `intelliswitch.geometry.*` |
| PLSR train / test / inference | `ProBallast_PLSR_02.m` | `intelliswitch.model.*` |

**Canonical model script:** `ProBallast_PLSR_02.m` (keeps each accepted wheel-group as a
separate observation, with sleeper-/wheel-passing-frequency power QC).

## Key design decisions
- **Quality-index target** (`fractal` | `std`) is configurable, not hardcoded.
- **PSD frequency range** is a hyperparameter (default 10–1000 Hz, to match `_02`).
- **Model-agnostic features**: `features/` emits a plain RAW-PSD feature matrix; the
  model-input transform (`X = sqrt(PSD) / speed**power`) is applied at model-build time,
  so tuning it never forces re-extraction. Backends are pluggable — PLSR and LightGBM
  share one interface (`model.backends.make_backend`).
- **Location-agnostic by design**: no sensor/location identity is ever a feature, so a
  new or relocated sensor is not out-of-distribution (see notebook 07).
- **Two-level caching** so model iteration never pays for extraction:
  - *Stage 1* — decode each `.log` text to a binary `.npz` once (`io.raw_cache`).
  - *Stage 2* — extract PSD features to a parquet **feature store**, keyed on a
    fingerprint of the extraction config; changing it invalidates the store, while
    downstream knobs (labels, PLSR, train/test ranges, `min_speed_kmh`) do not.
- The giant `sensor*.mat` caches are **not** used — features are regenerated from raw `.log`.

## Inference path (what becomes the serving app)
`io.log_reader` → `features.extract` → QC gate → trained `model` → `serving.predict_recording`.
No geometry code is needed at inference.

Predictions roll up through an aggregation hierarchy (configurable via `agg_*`):

```
(sensor × wheel-set) PSDs  ->  one prediction each   (up to 4 sensors × 5 wheel-sets)
    | level 1: aggregate wheel-sets per sensor       (cfg.agg_wheelsets)
    v
LOCATION prediction (per sensor)                     <- the informative output
    | level 2: aggregate locations                   (cfg.agg_locations)
    v
S&C-level indicator                                  <- convenience roll-up
```

A recording that fails the quality gate is returned as `rejected` with a reason; an
`out_of_domain` flag marks recordings whose speed falls outside the training range.

## Layout
```
src/intelliswitch/
  config.py            # all parameters / hyperparameters
  paths.py             # data-lake locations (medallion tiers; env-overridable)
  io/
    log_reader.py      # parse raw .log text
    raw_cache.py       # decode-once binary (.npz) cache
  features/
    wheel_detect.py    # wheel detection, speed, axle grouping
    changepoint.py     # findchangepts QC (the per-log bottleneck)
    psd.py             # Welch / FFT PSD on a fixed frequency grid
    extract.py         # per-wheel-group feature rows + acceptance QC
  geometry/            # training labels only (not used at inference)
    align.py           # spatial alignment of geometry campaigns (GdataAlign)
    fractal.py         # fractal dimension of a geometry signal (ffractal01)
    quality_index.py   # std / fractal quality index at locations
    deg_model.py       # degradation model -> per-day label
  model/
    backends.py        # PLSR / LightGBM behind a common interface
    core.py            # training, evaluation, serializable QualityModel artifact
    plsr.py            # PLSR specifics
  pipeline/
    build_dataset.py   # parallel feature store: build / load / select_trains / summary
  serving/
    predict.py         # predict_recording: single-recording streaming inference
notebooks/             # experiment here (preferred entry point)
  01_features.ipynb            # extract & inspect vibration PSD features
  02_geometry_labels.ipynb     # geometry alignment -> quality-index labels
  03_train_plsr.ipynb          # per-location PLSR (faithful _02 port) + validation plot
  04_two_stage_cache.ipynb     # decode -> cache -> feature store (the dataset builder)
  05_pooled_model.ipynb        # pooled, location-agnostic PLSR + leave-one-location-out
  06_models_and_serving.ipynb  # PLSR vs LightGBM + save artifact + streaming serving
scripts/               # validation flows as plain .py (reference)
  validate_features.py
  validate_geometry.py
data/                  # data lake (contents git-ignored; see data/README.md)
  raw/                 # source .log / geometry .csv (or point DATA_ROOT at the drive)
  interim/             # raw_cache/ — decoded .npz (stage 1)
  processed/           # feature_store/ (parquet) + training tables
  models/              # trained model artifacts (.joblib)
```

Lake locations come from `intelliswitch.paths` (`paths.RAW_CACHE`, `paths.FEATURE_STORE`,
`paths.MODELS`), so notebooks and jobs never hard-code them and they resolve regardless of
working dir; override the roots with `INTELLISWITCH_DATA_DIR` / `INTELLISWITCH_MODELS_DIR`.
Each notebook still sets `DATA_ROOT` for the large source `.log`/`.csv` (kept on the
external drive or symlinked into `data/raw`).
Run `pip install -e .` once so `import intelliswitch` works.
