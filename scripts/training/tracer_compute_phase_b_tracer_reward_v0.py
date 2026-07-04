#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def beta_for_world(cfg, world):
    table = cfg.get("default_beta_by_world", {})
    b = table.get(world, table.get("default", {"beta_v": 0.70, "beta_s": 0.25, "beta_e": 0.05}))
    return {
        "beta_v": float(b.get("beta_v", 0.70)),
        "beta_s": float(b.get("beta_s", 0.25)),
        "beta_e": float(b.get("beta_e", 0.05)),
    }


def maybe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def compute_time_to_reach(summary):
    """
    Best-effort CSV reader. The existing summary already gives min/reached,
    but for reach-terminated evaluation we also want first time-to-reach
    if CSV columns are available.
    """
    csv_path = summary.get("csv_path")
    stop = maybe_float(summary.get("stop_distance", summary.get("goal_stop_distance", 0.15)), 0.15)
    if not csv_path:
        return None

    p = Path(csv_path)
    if not p.exists():
        return None

    try:
        with open(p, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None

    if not rows:
        return None

    time_keys = ["t", "time", "time_sec", "elapsed_sec", "stamp_sec"]
    dist_keys = ["rel_dist", "relative_goal_dist", "goal_dist", "rel_goal_dist"]

    time_key = next((k for k in time_keys if k in rows[0]), None)
    dist_key = next((k for k in dist_keys if k in rows[0]), None)

    # If rel_dist is absent but rel_x/rel_y exist, reconstruct.
    has_xy = "rel_x" in rows[0] and "rel_y" in rows[0]

    if not dist_key and not has_xy:
        return None

    t0 = maybe_float(rows[0].get(time_key, 0.0), 0.0) if time_key else 0.0

    for idx, r in enumerate(rows):
        if dist_key:
            d = maybe_float(r.get(dist_key), 999.0)
        else:
            rx = maybe_float(r.get("rel_x"), 999.0)
            ry = maybe_float(r.get("rel_y"), 0.0)
            d = math.sqrt(rx * rx + ry * ry)

        if d <= stop:
            if time_key:
                return max(0.0, maybe_float(r.get(time_key), 0.0) - t0)
            # fallback: infer from duration and row index
            dur = maybe_float(summary.get("duration_sec"), 0.0)
            return dur * idx / max(1, len(rows) - 1)

    return None


def compute_tracer_proxy_reward(summary, cfg):
    world = summary.get("world_name", "default")
    beta = beta_for_world(cfg, world)
    norm = cfg.get("normalization", {})

    goal_distance = float(norm.get("goal_distance", summary.get("goal_distance_ahead", 0.50)))
    stop_distance = float(norm.get("stop_distance", summary.get("stop_distance", 0.15)))
    yaw_limit = float(norm.get("yaw_limit", 0.30))
    vx_ref = float(norm.get("vx_ref", 0.16))
    large_final_dist = float(norm.get("large_final_dist", 3.0))

    reached = bool(summary.get("reached_stop_distance", False))
    min_rel_dist = maybe_float(summary.get("min_rel_dist"), goal_distance)
    final_rel_dist = maybe_float(summary.get("final_rel_dist"), goal_distance)
    initial_rel_dist = maybe_float(summary.get("initial_rel_dist"), goal_distance)
    progress = maybe_float(summary.get("progress_initial_minus_min"), max(0.0, initial_rel_dist - min_rel_dist))
    abs_odom_x_delta = abs(maybe_float(summary.get("odom_x_delta"), 0.0))
    max_abs_yaw = abs(maybe_float(summary.get("max_abs_mpc_yaw_rate"), 0.0))
    max_mpc_vx = abs(maybe_float(summary.get("max_mpc_vx"), 0.0))

    time_to_reach = compute_time_to_reach(summary)

    # R_v: reach/progress/traversal.
    reach_term = 1.0 if reached else 0.0
    min_dist_term = clamp(1.0 - min_rel_dist / max(goal_distance, 1e-6), 0.0, 1.0)
    progress_term = clamp(progress / max(goal_distance - stop_distance, 1e-6), 0.0, 1.0)

    if time_to_reach is None:
        time_term = 0.0
    else:
        duration = max(maybe_float(summary.get("duration_sec"), 35.0), 1e-6)
        time_term = clamp(1.0 - time_to_reach / duration, 0.0, 1.0)

    R_v = (
        4.0 * reach_term
        + 3.0 * min_dist_term
        + 2.0 * progress_term
        + 1.0 * time_term
    )

    # R_s: stability/safety proxy. This intentionally remains an observed outcome metric.
    final_stability = clamp(1.0 - final_rel_dist / max(large_final_dist, 1e-6), -1.0, 1.0)
    yaw_stability = clamp(1.0 - max_abs_yaw / max(yaw_limit, 1e-6), -1.0, 1.0)

    # If the robot reaches but later drifts far away, penalize as post-reach drift.
    post_reach_drift = max(0.0, final_rel_dist - min_rel_dist)
    post_reach_drift_penalty = clamp(post_reach_drift / max(large_final_dist, 1e-6), 0.0, 1.0)

    # Large odom displacement after small goal can indicate sliding/drift in sponge.
    odom_drift_penalty = clamp(max(0.0, abs_odom_x_delta - goal_distance) / max(large_final_dist, 1e-6), 0.0, 1.0)

    R_s = (
        2.0 * final_stability
        + 1.0 * yaw_stability
        - 2.0 * post_reach_drift_penalty
        - 1.0 * odom_drift_penalty
    )

    # R_e: energy/smoothness proxy from command magnitude.
    vx_cost = (max_mpc_vx / max(vx_ref, 1e-6)) ** 2
    yaw_cost = (max_abs_yaw / max(yaw_limit, 1e-6)) ** 2
    R_e = -(0.7 * vx_cost + 0.3 * yaw_cost)

    R = beta["beta_v"] * R_v + beta["beta_s"] * R_s + beta["beta_e"] * R_e

    return {
        "schema": "phase_b_tracer_proxy_reward_result_v0",
        "world_name": world,
        "beta": beta,
        "reward": R,
        "terms": {
            "R_v": R_v,
            "R_s": R_s,
            "R_e": R_e
        },
        "weighted_terms": {
            "beta_v_R_v": beta["beta_v"] * R_v,
            "beta_s_R_s": beta["beta_s"] * R_s,
            "beta_e_R_e": beta["beta_e"] * R_e
        },
        "metrics": {
            "reached_stop_distance": reached,
            "min_rel_dist": min_rel_dist,
            "final_rel_dist": final_rel_dist,
            "progress_initial_minus_min": progress,
            "abs_odom_x_delta": abs_odom_x_delta,
            "max_abs_mpc_yaw_rate": max_abs_yaw,
            "max_mpc_vx": max_mpc_vx,
            "time_to_reach": time_to_reach
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--config", default="configs/phase_b_reward_v0/tracer_slide_reward_proxy_v0.json")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    summary = load_json(args.summary_json)
    cfg = load_json(args.config)
    reward = compute_tracer_proxy_reward(summary, cfg)

    print(json.dumps(reward, indent=2, sort_keys=True))

    if args.out_json:
        save_json(Path(args.out_json), reward)


if __name__ == "__main__":
    main()
