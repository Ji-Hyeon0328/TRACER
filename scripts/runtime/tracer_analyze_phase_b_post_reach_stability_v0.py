#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def ff(x, default=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--stable-final-dist", type=float, default=0.30)
    ap.add_argument("--stable-post-max-dist", type=float, default=0.50)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    with open(args.summary_json, "r") as f:
        s = json.load(f)

    csv_path = s.get("csv_path")
    if not csv_path or not Path(csv_path).is_file():
        raise RuntimeError(f"missing csv_path: {csv_path}")

    rows = list(csv.DictReader(open(csv_path)))

    stop = ff(s.get("goal_stop_distance"), 0.15)
    final_dist = ff(s.get("final_rel_dist"))
    min_dist = ff(s.get("min_rel_dist"))
    reached = bool(s.get("reached_stop_distance"))

    first_reach_t = None
    post_reach_dists = []

    for r in rows:
        rel_enable = ff(r.get("rel_enable"), 0.0)
        dist = ff(r.get("rel_dist"))
        t = ff(r.get("t_rel"))

        if not math.isfinite(dist):
            continue

        if rel_enable > 0.5 and dist <= stop and first_reach_t is None:
            first_reach_t = t

        if first_reach_t is not None and t >= first_reach_t:
            post_reach_dists.append(dist)

    post_max = max(post_reach_dists) if post_reach_dists else float("nan")
    post_final = post_reach_dists[-1] if post_reach_dists else float("nan")

    stable_reached = (
        reached
        and math.isfinite(final_dist)
        and final_dist <= args.stable_final_dist
        and math.isfinite(post_max)
        and post_max <= args.stable_post_max_dist
    )

    out = {
        "summary_json": args.summary_json,
        "csv_path": csv_path,
        "world_name": s.get("world_name"),
        "vx_far": s.get("vx_far"),
        "vx_near": s.get("vx_near"),
        "body_height": s.get("body_height"),
        "swing_clearance": s.get("swing_clearance"),
        "goal_stop_distance": stop,
        "reached_stop_distance": reached,
        "stable_reached": stable_reached,
        "first_reach_t": first_reach_t,
        "min_rel_dist": min_dist,
        "final_rel_dist": final_dist,
        "post_reach_max_dist": post_max,
        "post_reach_final_dist": post_final,
        "stable_final_dist_threshold": args.stable_final_dist,
        "stable_post_max_dist_threshold": args.stable_post_max_dist,
        "max_abs_mpc_yaw_rate": s.get("max_abs_mpc_yaw_rate"),
        "odom_x_delta": s.get("odom_x_delta"),
        "progress_initial_minus_min": s.get("progress_initial_minus_min"),
    }

    text = json.dumps(out, indent=2, sort_keys=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(text + "\n")

    print(text)


if __name__ == "__main__":
    main()
