#!/usr/bin/env python3
from __future__ import annotations

import argparse
from tracer_core.highlevel.objective_conditioned_selector import ObjectiveConditionedStyleSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json",
    )
    args = ap.parse_args()

    selector = ObjectiveConditionedStyleSelector(args.model)

    betas = {
        "motion": {"motion": 0.70, "stability": 0.20, "energy": 0.10},
        "stability": {"motion": 0.15, "stability": 0.75, "energy": 0.10},
        "deploy": {"motion": 0.34, "stability": 0.56, "energy": 0.10},
    }

    for terrain in ["flat_normal", "rough_mid", "slope_5deg"]:
        print(f"[{terrain}]")
        for name, beta in betas.items():
            sel = selector.select(terrain, beta, min_confidence=0.0)
            print(
                f"  beta={name:9s} "
                f"objective={sel['objective_name']:20s} "
                f"style={sel['selected_style']:15s} "
                f"semantic={sel['semantic_mode']:28s} "
                f"conf={sel['vote_confidence']:.3f} "
                f"active={int(sel['active'])}"
            )


if __name__ == "__main__":
    main()
