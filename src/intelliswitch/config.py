"""Configuration / hyperparameters for the Intelliswitch pipeline.

All magic numbers from the MATLAB scripts (`ProBallast_dataGen.m`,
`ProBallast_PLSR_02.m`) live here so they are explicit and tunable.
MATLAB uses 1-based column indices; Python indices below are 0-based.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    # ---- acquisition --------------------------------------------------------
    fs: int = 20_000                      # sample rate (Hz)
    n_cols: int = 32                      # columns in a .log row
    n_accel_cols: int = 24                # cols 1:24 are accelerometers (MATLAB)
    adc_to_g: float = 10.0 / 32768 / 0.01  # ADC counts -> g  (dataGen line ~30)

    # Accelerometer channels used (MATLAB col -> 0-based idx).
    # Physical names A1,A2,A7,A11 sit at MATLAB cols 2,4,14,22.
    sensor_names: tuple = ("A1", "A2", "A7", "A11")
    accel_channel_idx: tuple = (1, 3, 13, 21)
    # Wheel-detector channels (MATLAB cols 28,29 -> idx 27,28). NOT g-scaled.
    wheel_det_idx: tuple = (27, 28)

    # ---- band-pass filter (Butterworth, dataGen) ---------------------------
    bp_order: int = 3
    bp_band_hz: tuple = (5.0, 1000.0)
    # NOTE: the original dataGen comments out the filter on the first channel
    # (x2/A1). We filter all channels by default; set to False to reproduce the
    # original asymmetry.
    filter_all_channels: bool = True

    # ---- wheel detection / speed (dataGen) ---------------------------------
    det_valid_lo: float = -500.0          # wheel-detector valid range
    det_valid_hi: float = 501.0
    det_distance_m: float = 68.75         # distance between detectors W1 and W2
    axle_min_dt_s: float = 0.009          # min inter-axle time to count as separate axle
    group_max_dist_m: float = 6.0         # group adjacent wheels closer than this

    # ---- sensor positions (km from CPH reference, dataGen) -----------------
    km_W1: float = 175.212436
    km_W2: float = 175.143696
    # A1 was relocated after train index 6990; default = newer location.
    km_A1: float = 175.2055               # old: 175.202 (t < 6990)
    km_A2: float = 175.195676
    km_A7: float = 175.162066
    km_A11: float = 175.148896

    # ---- QC thresholds (dataGen) -------------------------------------------
    firstvib_quantile: float = 0.96       # quantile for first-vibration detection
    sat_abs_max: float = 300.0            # saturation: |x| above this -> reject
    weak_q999_min: float = 1.0            # too weak: quantile(.999) below this -> reject
    sig_pow_threshold_base: float = 0.3 * 200  # /speed -> timing tolerance (s)

    # ---- changepoint QC (findchangepts) ------------------------------------
    cp_statistic: str = "l2"              # "l2" (mean shift) | "linear" (mean+slope)
    cp_min_threshold: float = 170.0       # MATLAB MinThreshold (PELT penalty)
    cp_min_distance: int = 500            # MATLAB MinDistance (PELT min_size)
    # Speed knob: decimate the window by this factor before changepoint detection
    # (1 = off). The changepoint QC is ~88% of per-log cost; 4-8x decimation cuts
    # it ~10x with minor effect on which wheel-groups pass QC. Opt-in.
    cp_decimate: int = 1

    # ---- PSD / feature grid ------------------------------------------------
    psd_scale_g_to_ms2: float = 9.8       # dataGen multiplies window by 9.8 before PSD
    psd_method: str = "welch"             # "welch" | "fft"
    # Feature frequency grid (Hz). Default 10..1000 inclusive to match _02
    # (freq = 10:1000). Exposed as a hyperparameter per design decision.
    freq_min_hz: float = 10.0
    freq_max_hz: float = 1000.0
    freq_step_hz: float = 1.0

    # ---- _02 wheel-group acceptance QC -------------------------------------
    sleeper_spacing_m: float = 0.6        # sleeper passing freq = v/0.6
    wheel_spacing_m: float = 2.5          # wheel passing freq  = v/2.5
    sleeper_band_hz: float = 10.0         # +/- band around sleeper freq
    wheel_band_hz: float = 5.0            # +/- band around wheel freq
    # Train SELECTION (not extraction QC). Applied downstream by
    # pipeline.select_trains, so changing it does NOT invalidate the feature cache.
    min_speed_kmh: float = 135.0          # keep trains faster than this (None = all)
    # n_groups (train type) still gates extraction; size(sensor.wheel,2)==5 counts
    # wheel-GROUPS (=10 axles), not axles. Set None to extract all train types.
    n_groups_required: tuple = (4, 5)
    accept_pSleeper_max: float = 0.06     # keep wheel if sleeper-freq power below this
    accept_pWheel_max: float = 5000.001   # keep wheel if wheel-freq power below this

    # ---- target / model ----------------------------------------------------
    quality_index: str = "fractal"        # "fractal" | "std"
    label_field: str = "label_std_5m"     # which geometry label column to map to y

    # ---- feature transform (model input) -----------------------------------
    # Location-agnostic features: X = sqrt(PSD) / speed**power. Computed at
    # model-build time (the feature store keeps RAW PSD), so tuning this never
    # forces re-extraction. No location identity is ever a feature.
    feature_sqrt: bool = True
    feature_speed_norm_power: float = 1.0   # 0 = no speed normalization

    # Estimator backend: "plsr" | "lgbm"
    model_kind: str = "plsr"

    # ---- inference aggregation hierarchy -----------------------------------
    # Per recording: (sensor x wheel-set) PSDs -> a prediction each.
    #   level 1: wheel-sets -> location (per sensor)  [the informative output]
    #   level 2: locations  -> one S&C-level indicator [convenience roll-up]
    # methods: "median" | "mean" | "min" (worst-case) | "max".
    agg_wheelsets: str = "median"   # robust to a single bad wheel
    agg_locations: str = "median"

    # PLSR (ProBallast_PLSR_02). NOC = smallest #components (within
    # [pls_min_components, pls_max_components]) whose cumulative y-variance-
    # explained on training exceeds the threshold (MATLAB used >55%). The search
    # is done from a single max-component fit (components are nested), so the
    # range is cheap to widen.
    pls_min_components: int = 5
    pls_max_components: int = 40
    pls_y_var_threshold: float = 0.55

    # LightGBM hyperparameters (modest defaults; small, high-dim data).
    lgbm_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.5, min_child_samples=20,
        random_state=0, n_jobs=-1, verbosity=-1,
    ))
    # train/test split by day ranges (inclusive), from _02 lines 123/169-176.
    train_day_ranges: tuple = (
        (-864, -824), (-823, -700), (-669, -460),
        (-414, -336), (-314, -261), (-219, -181),
    )
    test_day_ranges: tuple = (
        (-900, -865), (-699, -670), (-459, -430), (-429, -415),
        (-335, -315), (-260, -220), (-180, -150),
    )

    reference_date: str = "2019-01-01"    # 'days' are negative days before this

    # ---- geometry / labels (GdataAlign, fractaldimention2, degmodelGen) -----
    geom_sep: str = ";"                    # Danish CSV delimiter
    geom_null: float = 999999.0           # missing-value sentinel
    geom_sample_m: float = 0.25           # spatial sampling of geometry (m)
    geom_window_m: float = 10.0           # window for std/fractal quality index
    geom_signal_length: int = 341         # samples used for campaign alignment
    geom_align_neib_m: float = 40.0       # +/- search range for cross-correlation (m)
    geom_tip_offset_m: float = 16.5       # km-bar -> crossing-tip distance (m)
    geom_gauge_tip_x: float = 0.0377      # x (km) where max gauge marks the tip
    # quality-index method for the label: "std" (negative std of long. level) or
    # "fractal" (first fractal slope). Independent of model input `quality_index`.
    geom_method: str = "std"
    # Sensor locations (m from switch-blade tip) where the label is evaluated,
    # mapped to (A1, A2, A7, A11). NOTE: the MATLAB encodes these inconsistently
    # across versions (degmodelGen indexes rows of an Ygrid=-6:50). CONFIRM these
    # against the physical sensor layout before trusting the labels.
    geom_sensor_locations_m: tuple = (4.0, 9.0, 37.0, 50.0)
    geom_fractal_landa: tuple = (1.0, 10.0)  # ffractal01 landa1, landa2 breakpoints

    # Target S&C (Tommerup, KnudeRef k=15) — defaults so the pipeline runs
    # without KnudeRef. (BTR, Spor, Type, Bladenr, km-bar km).
    sc_btr: int = 224045
    sc_spor: int = 2
    sc_type: int = 5
    sc_bladenr: int = 8252
    sc_kmbar: float = 175.183

    @property
    def freq_grid(self) -> np.ndarray:
        """Integer/float Hz grid the per-group PSD is interpolated onto."""
        n = int(round((self.freq_max_hz - self.freq_min_hz) / self.freq_step_hz)) + 1
        return self.freq_min_hz + self.freq_step_hz * np.arange(n)
