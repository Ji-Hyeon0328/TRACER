#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracer_core.highlevel.objective_selector_v1 import ObjectiveSelectorV1


def main():
    selector = ObjectiveSelectorV1()

    terrains = [
        "flat_normal",
        "rough_mid",
        "slope_5deg",
        "slippery_downslope_5deg_forward",
        "sponge_firm_downslope_5deg_forward",
        "unknown_terrain",
    ]

    missions = ["motion", "deploy", "stability"]

    for terrain in terrains:
        print(f"[{terrain}]")
        for mission in missions:
            out = selector.select(
                terrain=terrain,
                mission_mode=mission,
                ram_risk=0.0,
                recovery_needed=0.0,
            )
            print(
                f"  mission={mission:9s} "
                f"beta={out.selected_beta_name:10s} "
                f"β=[{out.beta_motion:.2f},{out.beta_stability:.2f},{out.beta_energy:.2f}] "
                f"style={out.predicted_style:14s} "
                f"score={out.score:+.3f} conf={out.confidence:.3f} "
                f"reason={out.reason}"
            )

        risk_out = selector.select(
            terrain=terrain,
            mission_mode="motion",
            ram_risk=0.70,
            recovery_needed=0.20,
        )
        print(
            f"  mission=motion+risk "
            f"beta={risk_out.selected_beta_name:10s} "
            f"β=[{risk_out.beta_motion:.2f},{risk_out.beta_stability:.2f},{risk_out.beta_energy:.2f}] "
            f"style={risk_out.predicted_style:14s} "
            f"score={risk_out.score:+.3f} conf={risk_out.confidence:.3f} "
            f"reason={risk_out.reason}"
        )
        print()


if __name__ == "__main__":
    main()
