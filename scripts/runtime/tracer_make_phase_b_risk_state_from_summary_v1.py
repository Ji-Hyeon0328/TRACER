#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


RISK_KEYS = [
    "ram_mismatch",
    "ram_uncertainty",
    "recent_slip_score",
    "body_stability_score",
]


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def fget(d, k, default=0.0):
    try:
        v = d.get(k, default)
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def bget(d, k, default=False):
    v = d.get(k, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def default_low_risk():
    return {
        "ram_mismatch": 0.05,
        "ram_uncertainty": 0.05,
        "recent_max_yaw_rate": 0.05,
        "recent_slip_score": 0.02,
        "body_stability_score": 0.95,
        "source": "default_low_risk_bootstrap_v1",
        "reason": ["no previous summary; bootstrap low-risk flat-earth assumption"],
    }


def raw_risk_from_summary(summary):
    reached = bget(summary, "reached_stop_distance")
    yaw = fget(summary, "max_abs_mpc_yaw_rate", 0.0)
    min_dist = fget(summary, "min_rel_dist", 1.0)
    final_dist = fget(summary, "final_rel_dist", 1.0)
    initial_dist = max(fget(summary, "initial_rel_dist", 0.5), 1e-6)
    progress = fget(summary, "progress_initial_minus_min", 0.0)
    odom_dx = fget(summary, "odom_x_delta", 0.0)
    max_vx = fget(summary, "max_mpc_vx", 0.0)

    progress_ratio = clamp(progress / initial_dist)
    final_ratio = clamp(final_dist / initial_dist)

    reason = []

    mismatch = 0.05
    uncertainty = 0.05
    slip_score = 0.02
    stability = 0.95

    if reached:
        reason.append("previous episode reached stop distance")
        mismatch += 0.05 * final_ratio
        uncertainty += 0.05 * final_ratio
    else:
        reason.append("previous episode did not reach stop distance")
        mismatch += 0.25 + 0.20 * final_ratio
        uncertainty += 0.30 + 0.20 * final_ratio
        slip_score += 0.10
        stability -= 0.10

    if progress_ratio < 0.45:
        reason.append("low progress ratio")
        mismatch += 0.20
        uncertainty += 0.15
    elif progress_ratio > 0.65:
        reason.append("good progress ratio")
        mismatch -= 0.03
        uncertainty -= 0.02

    if odom_dx < 0.15:
        reason.append("low odom forward progress")
        mismatch += 0.15
        uncertainty += 0.10

    if yaw > 0.28:
        reason.append("yaw saturation/high yaw activity")
        mismatch += 0.20
        uncertainty += 0.20
        slip_score += 0.25
        stability -= 0.20
    elif yaw > 0.18:
        reason.append("moderate yaw activity")
        mismatch += 0.10
        uncertainty += 0.08
        slip_score += 0.10
        stability -= 0.10
    else:
        reason.append("low yaw activity")
        mismatch -= 0.03
        uncertainty -= 0.03
        slip_score -= 0.01

    if max_vx >= 0.15 and yaw < 0.24 and reached:
        reason.append("aggressive speed succeeded without severe yaw")
        mismatch -= 0.04
        uncertainty -= 0.04
        stability += 0.03

    return {
        "ram_mismatch": clamp(mismatch),
        "ram_uncertainty": clamp(uncertainty),
        "recent_max_yaw_rate": yaw,
        "recent_slip_score": clamp(slip_score),
        "body_stability_score": clamp(stability),
        "source": "phase_b_summary_raw_risk_v1",
        "previous_summary": summary.get("summary_path", summary.get("csv_path", "")),
        "previous_selected_profile_name": summary.get("selected_profile_name", summary.get("selected_profile", "")),
        "previous_reached_stop_distance": reached,
        "previous_min_rel_dist": min_dist,
        "previous_final_rel_dist": final_dist,
        "previous_progress_ratio": progress_ratio,
        "previous_odom_x_delta": odom_dx,
        "previous_max_mpc_vx": max_vx,
        "reason": reason,
    }


def ema(prev, raw, alpha):
    # alpha = how much we trust the newest summary.
    return (1.0 - alpha) * prev + alpha * raw


def smooth_with_previous(prev, raw, alpha):
    out = dict(raw)
    out["source"] = "phase_b_summary_pseudo_ram_ema_v1"
    out["ema_alpha"] = alpha
    out["raw_risk_state"] = raw
    out["previous_risk_state"] = prev
    out["reason"] = list(raw.get("reason", [])) + ["EMA-smoothed with previous pseudo-RAM risk state"]

    for k in RISK_KEYS:
        out[k] = clamp(ema(fget(prev, k, fget(raw, k)), fget(raw, k), alpha))

    prev_yaw = fget(prev, "recent_max_yaw_rate", 0.0)
    raw_yaw = fget(raw, "recent_max_yaw_rate", 0.0)

    # Risk should rise immediately when yaw spikes,
    # but decay gradually after stable episodes.
    if raw_yaw > prev_yaw:
        out["recent_max_yaw_rate"] = raw_yaw
        out["reason"].append("yaw risk rose immediately")
    else:
        out["recent_max_yaw_rate"] = ema(prev_yaw, raw_yaw, alpha)
        out["reason"].append("yaw risk decayed with EMA")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--prev-risk-json", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--ema-alpha", type=float, default=0.45)
    args = ap.parse_args()

    alpha = clamp(args.ema_alpha, 0.0, 1.0)

    if args.summary_json:
        with open(args.summary_json, "r") as f:
            summary = json.load(f)
        summary["summary_path"] = args.summary_json
        raw = raw_risk_from_summary(summary)
    else:
        raw = default_low_risk()

    if args.prev_risk_json and Path(args.prev_risk_json).is_file():
        with open(args.prev_risk_json, "r") as f:
            prev = json.load(f)
        risk = smooth_with_previous(prev, raw, alpha)
    else:
        risk = raw
        risk["ema_alpha"] = alpha

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(risk, f, indent=2, sort_keys=True)

    print(json.dumps(risk, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
