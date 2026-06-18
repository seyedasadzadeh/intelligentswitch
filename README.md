# Intelliswitch — track-irregularity prediction (Python refactor)

Python port of the MATLAB PLSR pipeline that predicts a **track quality index**
(fractal value or std of longitudinal level) from **trackside vibration recordings**.

This is **Phase 1** (MATLAB → Python). Phase 2 builds the serving app on top of the
`features` + `model` layers.

## Scope (ported from MATLAB)

| Stage | MATLAB source | Python module |
|-------|---------------|---------------|
| Read raw `.log` | `ProBallast_dataGen.m` | `intelliswitch.io.log_reader` |
| Wheel detection / speed / grouping | `ProBallast_dataGen.m` | `intelliswitch.features.wheel_detect` |
| Changepoint QC (`findchangepts`) | `ProBallast_dataGen.m`, `vib_detrend.m` | `intelliswitch.features.changepoint` |
| PSD (Welch + FFT) + freq-grid interp | `ProBallast_dataGen.m` | `intelliswitch.features.psd` |
| Feature extraction (per wheel-group) | `ProBallast_dataGen.m` + `ProBallast_PLSR_02.m` | `intelliswitch.features.extract` |
| Geometry → labels (training only) | `GdataAlign.m` → `fractaldimention2.m` → `BallDeg05.m` → `degmodelGen.m` | `intelliswitch.geometry.*` (TODO) |
| PLSR train / test / inference | `ProBallast_PLSR_02.m` | `intelliswitch.model.plsr` (TODO) |

**Canonical model script:** `ProBallast_PLSR_02.m` (keeps each accepted wheel-group as a
separate observation, with sleeper-/wheel-passing-frequency power QC).

## Key design decisions
- **Quality-index target** (`fractal` | `std`) is configurable, not hardcoded.
- **PSD frequency range** is a hyperparameter (default 10–1000 Hz, to match `_02`).
- **Model-agnostic features**: `features/` emits a plain feature matrix so PLSR can later be
  swapped for XGBoost / CatBoost / LightGBM.
- The giant `sensor*.mat` caches are **not** used — features are regenerated from raw `.log`.

## Inference path (what becomes the serving app)
`io.log_reader` → `features.extract` → trained `model` → prediction.
No geometry code is needed at inference.

## Layout
```
src/intelliswitch/
  config.py            # all parameters / hyperparameters
  io/log_reader.py
  features/{psd,wheel_detect,changepoint,extract}.py
  geometry/{fractal,quality_index,align,deg_model}.py   # labels
  model/               # TODO (PLSR)
notebooks/             # experiment here (preferred entry point)
  01_features.ipynb        # vibration feature extraction
  02_geometry_labels.ipynb # geometry alignment + quality-index labels
  03_build_dataset.ipynb   # features + labels -> training table
scripts/               # same flows as plain .py (reference)
```

Notebooks expect the data under `DATA_ROOT` (set at the top of each notebook).
Run `pip install -e .` once so `import intelliswitch` works.
