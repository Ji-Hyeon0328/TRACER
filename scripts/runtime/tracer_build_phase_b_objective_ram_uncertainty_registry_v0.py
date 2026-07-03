#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def ff(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def std_like(xs):
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def key_string(world, profile, vx, near, slow, h, clr):
    return (
        f"{world}::{profile}"
        f"::vx={round(ff(vx), 4):.4f}"
        f"::near={round(ff(near), 4):.4f}"
        f"::slow={round(ff(slow), 4):.4f}"
        f"::h={round(ff(h), 4):.4f}"
        f"::clr={round(ff(clr), 4):.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.score_csv)))
    groups = defaultdict(list)

    for r in rows:
        k = key_string(
            r.get("world_name"),
            r.get("theta5_profile_name"),
            r.get("vx_far"),
            r.get("vx_near"),
            r.get("goal_slow_distance"),
            r.get("body_height"),
            r.get("swing_clearance"),
        )
        groups[k].append(r)

    out_groups = {}

    for k, rs in groups.items():
        sem = Counter(r.get("actual_semantic") for r in rs)
        final_vals = [ff(r.get("actual_final_rel_dist")) for r in rs]
        dx_vals = [ff(r.get("actual_odom_x_delta")) for r in rs]
        risk_vals = [ff(r.get("actual_future_risk")) for r in rs]

        n = len(rs)
        majority_semantic, majority_count = sem.most_common(1)[0]
        semantic_uncertainty = 1.0 - majority_count / max(n, 1)

        std_final = std_like(final_vals)
        std_dx = std_like(dx_vals)
        std_risk = std_like(risk_vals)

        future_uncertainty = min(
            1.0,
            0.50 * semantic_uncertainty
            + 0.25 * min(std_final / 1.0, 1.0)
            + 0.20 * min(std_dx / 1.0, 1.0)
            + 0.05 * min(std_risk / 0.5, 1.0),
        )

        high_variability = (
            semantic_uncertainty > 0.0
            or std_final > 0.50
            or std_dx > 0.50
            or std_risk > 0.20
        )

        first = rs[0]

        out_groups[k] = {
            "key_string": k,
            "key": {
                "world_name": first.get("world_name"),
                "theta5_profile_name": first.get("theta5_profile_name"),
                "vx_far": round(ff(first.get("vx_far")), 4),
                "vx_near": round(ff(first.get("vx_near")), 4),
                "goal_slow_distance": round(ff(first.get("goal_slow_distance")), 4),
                "body_height": round(ff(first.get("body_height")), 4),
                "swing_clearance": round(ff(first.get("swing_clearance")), 4),
            },
            "n": n,
            "semantic_counts": dict(sem),
            "majority_semantic": majority_semantic,
            "semantic_uncertainty": semantic_uncertainty,
            "avg_final_rel_dist": mean(final_vals),
            "std_like_final_rel_dist": std_final,
            "avg_odom_x_delta": mean(dx_vals),
            "std_like_odom_x_delta": std_dx,
            "avg_future_risk": mean(risk_vals),
            "std_like_future_risk": std_risk,
            "future_uncertainty": future_uncertainty,
            "high_variability": high_variability,
        }

    registry = {
        "registry_type": "phase_b_objective_ram_uncertainty_registry_v0",
        "source_score_csv": args.score_csv,
        "num_groups": len(out_groups),
        "groups": out_groups,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)

    print(json.dumps({
        "out_json": args.out_json,
        "num_groups": len(out_groups),
        "groups": list(out_groups.values()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
