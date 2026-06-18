"""Feature store: extract PSD features for many .log files, once, in parallel.

Feature extraction is ~2-4 s/log and dominated by the changepoint QC, so we cache
the results to parquet and never recompute them while iterating on the model.

Design:
- **Parallel**: extraction is independent per log -> multiprocessing Pool.
- **Resumable**: a `manifest.csv` records every processed log (accepted + reason);
  re-running skips logs already in the manifest. Features are written as parquet
  shards under `parts/`, so an interrupted run keeps its completed shards.
- **Labels are NOT stored here.** Features are raw; attach labels at load time
  (geometry config can change without invalidating the expensive feature cache).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ..config import Config
from ..features import extract_features
from ..io.raw_cache import load_raw

# Config fields that affect *feature extraction*. If any of these change, the
# cached features are stale and the store is rebuilt. Downstream-only settings
# (geometry/labels, PLSR, train/test ranges) deliberately do NOT invalidate the
# expensive feature cache — that's the point of caching features separately.
FEATURE_CONFIG_FIELDS = (
    "fs", "n_cols", "n_accel_cols", "adc_to_g", "sensor_names",
    "accel_channel_idx", "wheel_det_idx", "bp_order", "bp_band_hz",
    "filter_all_channels", "det_valid_lo", "det_valid_hi", "det_distance_m",
    "axle_min_dt_s", "group_max_dist_m", "km_W1", "km_W2", "km_A1", "km_A2",
    "km_A7", "km_A11", "firstvib_quantile", "sat_abs_max", "weak_q999_min",
    "sig_pow_threshold_base", "cp_statistic", "cp_min_threshold",
    "cp_min_distance", "cp_decimate", "psd_scale_g_to_ms2", "psd_method",
    "freq_min_hz", "freq_max_hz", "freq_step_hz", "sleeper_spacing_m",
    "wheel_spacing_m", "sleeper_band_hz", "wheel_band_hz",
    "n_groups_required", "reference_date",
)
# NOTE: min_speed_kmh is intentionally absent — speed is a downstream selection
# concern (select_trains), not an extraction parameter, so changing it must NOT
# invalidate the feature cache.


def _feature_config(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in FEATURE_CONFIG_FIELDS}


def _fingerprint(cfg: Config) -> str:
    blob = json.dumps(_feature_config(cfg), sort_keys=True, default=list)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]

# Worker config is set once per process to avoid pickling cfg on every task.
_CFG: Config | None = None
_RAW_DIR: str | None = None


def _init_worker(cfg: Config, raw_dir: str | None = None):
    global _CFG, _RAW_DIR
    _CFG, _RAW_DIR = cfg, raw_dir


def _process_one(path: str):
    cfg = _CFG or Config()
    try:
        src = path
        if _RAW_DIR:                       # use stage-1 binary cache when present
            npz = Path(_RAW_DIR) / (Path(path).stem + ".npz")
            if npz.exists():
                src = load_raw(npz)
        r = extract_features(src, cfg)
        df = r.features if r.accepted else None
        rec = dict(source=Path(path).name, accepted=bool(r.accepted),
                   reason=r.reason, n_rows=0 if df is None else len(df))
        return rec, df
    except Exception as e:  # noqa: BLE001
        rec = dict(source=Path(path).name, accepted=False,
                   reason=f"ERROR {type(e).__name__}: {e}", n_rows=0)
        return rec, None


def build_feature_store(
    log_dir: str | Path,
    store_dir: str | Path,
    cfg: Config | None = None,
    n_workers: int | None = None,
    shard_size: int = 400,
    limit: int | None = None,
    resume: bool = True,
    force: bool = False,
    raw_cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Extract features for every `.log` in `log_dir` into a parquet store.

    The store is keyed on a fingerprint of the extraction config. If the config
    changes (e.g. `cp_decimate`, frequency range, QC thresholds), or `force=True`,
    the store is wiped and rebuilt from scratch. Otherwise an interrupted run
    resumes where it left off. Returns the store directory.
    """
    cfg = cfg or Config()
    store = Path(store_dir)
    (store / "parts").mkdir(parents=True, exist_ok=True)
    manifest_path = store / "manifest.csv"
    config_path = store / "config.json"

    # ---- cache invalidation on config change --------------------------------
    fp = _fingerprint(cfg)
    prev_fp = None
    if config_path.exists():
        try:
            prev_fp = json.loads(config_path.read_text()).get("fingerprint")
        except Exception:  # noqa: BLE001
            prev_fp = None
    stale = prev_fp is not None and prev_fp != fp

    if force or stale:
        if verbose:
            why = "force=True" if force else f"config changed ({prev_fp} -> {fp})"
            print(f"rebuilding feature store from scratch: {why}")
        if (store / "parts").exists():
            shutil.rmtree(store / "parts")
        (store / "parts").mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest_path.unlink()
    config_path.write_text(json.dumps(
        {"fingerprint": fp, "fields": _feature_config(cfg)}, indent=1, default=list
    ))

    processed: set[str] = set()
    if resume and manifest_path.exists():
        processed = set(pd.read_csv(manifest_path)["source"].astype(str))

    logs = sorted(Path(log_dir).glob("*.log"))
    if limit is not None:
        logs = logs[:limit]
    todo = [p for p in logs if p.name not in processed]
    n_workers = n_workers or os.cpu_count() or 1
    part_idx = len(list((store / "parts").glob("part-*.parquet")))

    if verbose:
        print(f"{len(logs)} logs, {len(processed)} already done, "
              f"{len(todo)} to process on {n_workers} workers")

    done = 0
    for start in range(0, len(todo), shard_size):
        batch = [str(p) for p in todo[start : start + shard_size]]
        recs, dfs = [], []
        raw_dir = str(raw_cache_dir) if raw_cache_dir else None
        with Pool(n_workers, initializer=_init_worker, initargs=(cfg, raw_dir)) as pool:
            for rec, df in pool.imap_unordered(_process_one, batch, chunksize=4):
                recs.append(rec)
                if df is not None and len(df):
                    dfs.append(df)
        if dfs:
            part = store / "parts" / f"part-{part_idx:04d}.parquet"
            pd.concat(dfs, ignore_index=True).to_parquet(part)
            part_idx += 1
        pd.DataFrame(recs).to_csv(
            manifest_path, mode="a", header=not manifest_path.exists(), index=False
        )
        done += len(batch)
        if verbose:
            acc = sum(r["accepted"] for r in recs)
            print(f"  {done}/{len(todo)} processed | batch accepted {acc}/{len(batch)}")
    return store


def select_trains(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Select which recordings to use for analysis (NOT a quality filter).

    Speed and train type are choices about *which trains we care about*, separate
    from whether a recording is clean. Applied here so the same cached feature
    store can be re-sliced without re-extraction. Returns a filtered copy.
    """
    cfg = cfg or Config()
    out = df
    if cfg.min_speed_kmh is not None and "speed_kmh" in out:
        out = out[out["speed_kmh"] > cfg.min_speed_kmh]
    if cfg.n_groups_required is not None and "n_groups" in out:
        out = out[out["n_groups"] == cfg.n_groups_required]
    return out.reset_index(drop=True)


def load_feature_store(store_dir: str | Path) -> pd.DataFrame:
    """Concatenate all feature shards into one DataFrame."""
    parts = sorted((Path(store_dir) / "parts").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def store_summary(store_dir: str | Path) -> dict:
    """Quick stats from the manifest (processed, accepted, rejection reasons)."""
    store = Path(store_dir)
    mpath = store / "manifest.csv"
    if not mpath.exists():
        return {"processed": 0}
    m = pd.read_csv(mpath)
    reasons = (
        m.loc[~m["accepted"], "reason"]
        .str.replace(r"\d+", "N", regex=True)
        .value_counts()
        .head(10)
        .to_dict()
    )
    return {
        "processed": len(m),
        "accepted": int(m["accepted"].sum()),
        "feature_rows": int(m["n_rows"].sum()),
        "top_reject_reasons": reasons,
    }
