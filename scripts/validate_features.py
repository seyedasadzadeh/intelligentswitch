"""Smoke-test feature extraction on real W10v140 .log recordings.

Usage:
    python scripts/validate_features.py /path/to/W10v140 [N]

Prints per-recording diagnostics (speed, wheel/group counts, QC outcome, accepted
feature rows) and aggregate stats, so we can sanity-check the port before wiring
up the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelliswitch import Config
from intelliswitch.features import extract_features


def main(log_dir: str, n: int = 25):
    cfg = Config()
    logs = sorted(Path(log_dir).glob("*.log"))[:n]
    if not logs:
        print(f"no .log files in {log_dir}")
        return

    n_accepted = 0
    total_rows = 0
    speeds = []
    for p in logs:
        try:
            res = extract_features(p, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"{p.name:24s} ERROR {type(e).__name__}: {e}")
            continue
        spd = f"{res.geom.speed_kmh:6.1f}" if res.geom else "   n/a"
        nw = res.geom.n_wheels if res.geom else "-"
        ng = res.geom.n_groups if res.geom else "-"
        tag = f"OK rows={len(res.features)}" if res.accepted else f"skip ({res.reason})"
        print(f"{p.name:24s} v={spd} km/h  wheels={nw} groups={ng}  {tag}")
        if res.geom:
            speeds.append(res.geom.speed_kmh)
        if res.accepted:
            n_accepted += 1
            total_rows += len(res.features)

    print("\n--- summary ---")
    print(f"logs scanned     : {len(logs)}")
    print(f"accepted         : {n_accepted}")
    print(f"feature rows     : {total_rows}")
    if speeds:
        print(f"speed (km/h)     : min={min(speeds):.0f} med={np.median(speeds):.0f} max={max(speeds):.0f}")
    print(f"feature dim      : {cfg.freq_grid.size} ({cfg.freq_min_hz:.0f}-{cfg.freq_max_hz:.0f} Hz)")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    main(d, n)
