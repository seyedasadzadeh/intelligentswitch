"""Validate the geometry -> label pipeline on Tommerup (KnudeRef k=15).

Usage:
    python scripts/validate_geometry.py /path/to/Bladenr8252

Checks that campaign alignment runs, prints the data span, and shows the quality
index over time at each sensor location (it should worsen over time and jump at a
tamping/renewal).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelliswitch import Config
from intelliswitch.geometry import align_campaigns, build_deg_model


def main(campaign_dir: str):
    cfg = Config()
    aligned = align_campaigns(campaign_dir, cfg)
    n, ncamp = aligned.long_R.shape
    print(f"campaigns aligned : {ncamp}")
    print(f"points/campaign   : {n}")
    print(f"x (m from tip)    : {np.nanmin(aligned.x_blade_m):.1f} .. {np.nanmax(aligned.x_blade_m):.1f}")
    print(f"dates             : {aligned.dates[0].date()} .. {aligned.dates[-1].date()}")
    print(f"days (before {cfg.reference_date}) : {aligned.days.min():.0f} .. {aligned.days.max():.0f}")

    deg = build_deg_model(aligned, cfg)
    print(f"\nquality method    : {cfg.geom_method}")
    print(f"sensor locations  : {tuple(cfg.geom_sensor_locations_m)} m from tip")
    print("\nquality index over time (rows=campaigns, cols=A1/A2/A7/A11):")
    print("  days   " + "  ".join(f"{n_:>8}" for n_ in cfg.sensor_names))
    for j in np.argsort(deg.days):
        vals = "  ".join(
            f"{deg.quality[i, j]:8.3f}" if np.isfinite(deg.quality[i, j]) else "     nan"
            for i in range(deg.quality.shape[0])
        )
        print(f"  {deg.days[j]:6.0f}  {vals}")

    finite = np.isfinite(deg.quality).mean()
    print(f"\nfinite quality values: {finite*100:.0f}%")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    main(d)
