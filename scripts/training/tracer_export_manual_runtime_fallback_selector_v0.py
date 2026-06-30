from __future__ import annotations

import argparse
import json
from pathlib import Path


PROFILE_BETA = {
    "balanced": [0.34, 0.33, 0.33],
    "motion": [0.65, 0.20, 0.15],
    "stability": [0.20, 0.65, 0.15],
    "energy": [0.20, 0.20, 0.60],
    "motion_extreme": [0.85, 0.10, 0.05],
    "stability_extreme": [0.05, 0.90, 0.05],
    "energy_extreme": [0.05, 0.10, 0.85],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--theta", nargs=8, type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    theta = [float(x) for x in args.theta]

    global_default = {
        "profile": "runtime_fallback",
        "preset": args.name,
        "theta_norm": theta,
        "feasible_frac": 1.0,
        "runtime_score": 1.0,
        "zmin_mean": 0.0,
        "distance_mean": 0.0,
    }

    selector = {
        "name": "tracer_manual_runtime_fallback_selector_v0",
        "source": "manual_runtime_fallback_candidate",
        "selection_rule": {
            "global_default": "manual fallback theta",
            "profile_candidate": (
                "all requested objective profiles are blocked and fall back "
                "to the manual runtime_fallback theta"
            ),
        },
        "thresholds": {
            "min_default_feasible_frac": 0.0,
            "min_profile_feasible_frac": 1.0,
        },
        "global_default": global_default,
        "profiles": {},
    }

    # Block all normal objective profiles so every request falls back to
    # the manual runtime fallback candidate.
    for profile in PROFILE_BETA:
        selector["profiles"][profile] = {
            "profile": profile,
            "preset": f"blocked_{profile}",
            "theta_norm": theta,
            "feasible_frac": 0.0,
            "runtime_score": 0.0,
            "zmin_mean": 0.0,
            "distance_mean": 0.0,
            "runtime_allowed": False,
            "fallback_profile": "runtime_fallback",
            "fallback_preset": args.name,
        }

    selector["profiles"]["runtime_fallback"] = {
        "profile": "runtime_fallback",
        "preset": args.name,
        "theta_norm": theta,
        "feasible_frac": 1.0,
        "runtime_score": 1.0,
        "zmin_mean": 0.0,
        "distance_mean": 0.0,
        "runtime_allowed": True,
        "fallback_profile": None,
        "fallback_preset": None,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selector, indent=2))

    print(f"[TRACER] wrote {out}")
    print(json.dumps(global_default, indent=2))


if __name__ == "__main__":
    main()
