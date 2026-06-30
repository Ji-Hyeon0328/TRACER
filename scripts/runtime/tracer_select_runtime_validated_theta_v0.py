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


def infer_profile_from_beta(beta):
    best_name = "balanced"
    best_dist = float("inf")
    for name, ref in PROFILE_BETA.items():
        dist = sum((float(a) - float(b)) ** 2 for a, b in zip(beta, ref))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, best_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selector",
        default="configs/runtime/tracer_runtime_validated_theta_selector_v0.json",
    )
    ap.add_argument("--profile", default="")
    ap.add_argument("--beta", nargs=3, type=float, default=None)
    args = ap.parse_args()

    selector = json.loads(Path(args.selector).read_text())

    if args.profile:
        requested_profile = args.profile
        reason = "profile_arg"
    elif args.beta is not None:
        requested_profile, dist = infer_profile_from_beta(args.beta)
        reason = f"nearest_beta_profile dist={dist:.6f}"
    else:
        requested_profile = "balanced"
        reason = "default_profile"

    profiles = selector.get("profiles", {})
    global_default = selector["global_default"]

    candidate = profiles.get(requested_profile)
    if candidate is None:
        selected = global_default
        selection_reason = f"unknown_profile:{requested_profile}->global_default"
    elif bool(candidate.get("runtime_allowed", False)):
        selected = candidate
        selection_reason = f"runtime_allowed:{requested_profile}"
    else:
        selected = global_default
        selection_reason = (
            f"runtime_blocked:{requested_profile}"
            f"(feasible={candidate.get('feasible_frac')})"
            f"->global_default:{global_default['profile']}"
        )

    out = {
        "requested_profile": requested_profile,
        "request_reason": reason,
        "selection_reason": selection_reason,
        "selected_profile": selected["profile"],
        "selected_preset": selected["preset"],
        "theta_norm": selected["theta_norm"],
        "feasible_frac": selected["feasible_frac"],
        "runtime_score": selected["runtime_score"],
        "zmin_mean": selected["zmin_mean"],
        "distance_mean": selected["distance_mean"],
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
