"""Stage-1 cache: decode raw `.log` text into compact binary records.

Reading a `.log` (~82k rows x 32 cols of text) costs ~0.3 s and dominates the
pipeline at scale (it is disk-I/O bound on the source drive). This module decodes
each log **once** into a small `.npz` holding only the channels we use (4
accelerometers in g + 2 wheel detectors, float32 ~2 MB vs ~16 MB text). Feature
extraction can then read the binary, so re-tuning feature config (cp_decimate,
frequency range, QC thresholds) never re-reads the text.

PRODUCTION NOTE: this is a *training-time* optimization. The serving app receives
only `.log` files and calls `extract_features(path)` directly — `decode_log` runs
inline there, no cache involved.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .log_reader import read_log, extract_channels


@dataclass
class RawRecord:
    """Decoded channels for one recording (the minimal input to extraction)."""
    name: str                 # original filename, e.g. '160715_012329.log'
    channels: dict            # sensor name -> float32 signal (g)
    w1: np.ndarray            # wheel detector W1 (float32, raw counts)
    w2: np.ndarray            # wheel detector W2

    @property
    def n_samples(self) -> int:
        return int(self.w1.shape[0])


def decode_log(path: str | Path, cfg: Config | None = None) -> RawRecord:
    """Read a `.log` and return its compact RawRecord (channels in g, float32)."""
    cfg = cfg or Config()
    arr = read_log(path, cfg)
    channels, w1, w2 = extract_channels(arr, cfg)
    channels = {k: np.asarray(v, dtype=np.float32) for k, v in channels.items()}
    return RawRecord(Path(path).name, channels,
                     np.asarray(w1, dtype=np.float32),
                     np.asarray(w2, dtype=np.float32))


def save_raw(rec: RawRecord, out_path: str | Path) -> None:
    np.savez_compressed(
        out_path, name=rec.name, w1=rec.w1, w2=rec.w2,
        **{f"ch_{k}": v for k, v in rec.channels.items()},
    )


def load_raw(npz_path: str | Path) -> RawRecord:
    d = np.load(npz_path, allow_pickle=False)
    channels = {k[3:]: d[k] for k in d.files if k.startswith("ch_")}
    return RawRecord(str(d["name"]), channels, d["w1"], d["w2"])


# ----------------------------------------------------------------- batch build
_CFG: Config | None = None
_RAW_DIR: str | None = None


def _init(cfg: Config, raw_dir: str):
    global _CFG, _RAW_DIR
    _CFG, _RAW_DIR = cfg, raw_dir


def _decode_one(path: str):
    try:
        rec = decode_log(path, _CFG)
        save_raw(rec, os.path.join(_RAW_DIR, Path(path).stem + ".npz"))
        return Path(path).name, True, ""
    except Exception as e:  # noqa: BLE001
        return Path(path).name, False, f"{type(e).__name__}: {e}"


def build_raw_cache(
    log_dir: str | Path,
    raw_dir: str | Path,
    cfg: Config | None = None,
    n_workers: int | None = None,
    limit: int | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> Path:
    """Decode every `.log` in `log_dir` into `raw_dir/<name>.npz` (parallel, resumable)."""
    cfg = cfg or Config()
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    logs = sorted(Path(log_dir).glob("*.log"))
    if limit is not None:
        logs = logs[:limit]
    existing = {p.stem for p in raw.glob("*.npz")} if resume else set()
    todo = [str(p) for p in logs if p.stem not in existing]
    n_workers = n_workers or os.cpu_count() or 1
    if verbose:
        print(f"{len(logs)} logs, {len(existing)} already decoded, "
              f"{len(todo)} to decode on {n_workers} workers")

    recs = []
    if todo:
        with Pool(n_workers, initializer=_init, initargs=(cfg, str(raw))) as pool:
            for i, rec in enumerate(pool.imap_unordered(_decode_one, todo, chunksize=8), 1):
                recs.append(rec)
                if verbose and i % 200 == 0:
                    print(f"  decoded {i}/{len(todo)}")
        man = pd.DataFrame(recs, columns=["source", "ok", "error"])
        man.to_csv(raw / "decode_manifest.csv", mode="a",
                   header=not (raw / "decode_manifest.csv").exists(), index=False)
        if verbose:
            print(f"done: {man['ok'].sum()}/{len(man)} decoded OK")
    return raw
