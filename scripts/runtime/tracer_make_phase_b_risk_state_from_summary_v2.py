#!/usr/bin/env python3

import argparse
import csv
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
        if v is None or v == "":
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


def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return 0.0
    idx = int(round((len(vals) - 1) * q))
    idx = max(0, min(len(vals) - 1, idx))
    return vals[idx]


def default_low_risk():
    return {
        "ram_mismatch": 0.05,
        "ram_uncertainty": 0.05,
        "recent_max_yaw_rate": 0.05,
        "raw_max_abs_mpc_yaw_rate": 0.05,
        "yaw_p95_abs_rate": 0.05,
        "yaw_mean_abs_rate": 0.02,
        "yaw_saturation_fraction": 0.0,
        "recent_slip_score": 0.02,
        "body_stability_score": 0.95,
        "source": "default_low_risk_bootstrap_v2",
        "reason": ["no previous summary; bootstrap low-risk flat-earth assumption"],
    }


def load_yaw_stats_from_csv(summary):
    csv_path = summary.get("csv_path", "")
    fallback_max = fget(summary, "max_abs_mpc_yaw_rate", 0.0)

    base = {
        "yaw_column": "",
        "raw_max_abs_mpc_yaw_rate": fallback_max,
        "yaw_p95_abs_rate": fallback_max,
        "yaw_mean_abs_rate": fallback_max,
        "yaw_saturation_fraction": 1.0 if fallback_max >= 0.285 else 0.0,
        "yaw_metric_source": "summary_fallback",
    }

    if not csv_path or not Path(csv_path).is_file():
        return base

    try:
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return base

    if not rows:
        return base

    fields = rows[0].keys()

    preferred = [
        "mpc_yaw_rate",
        "mpc_ref_yaw_rate",
        "cmd_yaw_rate",
        "yaw_rate_cmd",
        "yaw_rate",
    ]

    candidates = []
    for name in preferred:
        if name in fields:
            candidates.append(name)

    if not candidates:
        for name in fields:
            low = name.lower()
            if "yaw" in low and ("rate" in low or "cmd" in low or "mpc" in low or "ref" in low):
                candidates.append(name)

    best = None
    best_vals = []

    for c in candidates:
        vals = []
        for r in rows:
            try:
                vals.append(abs(float(r.get(c, 0.0))))
            except Exception:
                pass

        if not vals:
            continue

        m = max(vals)

        # Prefer a yaw-rate-like column whose max is close to summary max.
        # If summary is unreliable, still accept the largest yaw-related signal.
        closeness = abs(m - fallback_max)
        score = -closeness + 0.01 * m

        if best is None or score > best[0]:
            best = (score, c)
            best_vals = vals

    if best is None or not best_vals:
        return base

    sat_threshold = 0.285
    raw_max = max(best_vals)
    p95 = percentile(best_vals, 0.95)
    mean = sum(best_vals) / len(best_vals)
    sat_frac = sum(1 for v in best_vals if v >= sat_threshold) / max(len(best_vals), 1)

    return {
        "yaw_column": best[1],
        "raw_max_abs_mpc_yaw_rate": raw_max,
        "yaw_p95_abs_rate": p95,
        "yaw_mean_abs_rate": mean,
        "yaw_saturation_fraction": sat_frac,
        "yaw_metric_source": "csv",
    }


def effective_yaw_risk(yaw_stats):
    raw_max = yaw_stats["raw_max_abs_mpc_yaw_rate"]
    p95 = yaw_stats["yaw_p95_abs_rate"]
    mean = yaw_stats["yaw_mean_abs_rate"]
    sat_frac = yaw_stats["yaw_saturation_fraction"]

    # Sustained saturation is high risk.
    if sat_frac > 0.20 or p95 > 0.26:
        return 0.30, "sustained yaw saturation"

    # Short saturation spikes are moderate, not automatically high risk.
    if raw_max > 0.28 and sat_frac > 0.03:
        return 0.22, "short yaw saturation bursts"

    if raw_max > 0.28:
        return 0.17, "isolated yaw saturation spike"

    if p95 > 0.18 or mean > 0.12:
        return 0.20, "moderate yaw activity"

    return max(p95, mean), "low yaw activity"


def raw_risk_from_summary(summary):
    reached = bget(summary, "reached_stop_distance")
    min_dist = fget(summary, "min_rel_dist", 1.0)
    final_dist = fget(summary, "final_rel_dist", 1.0)
    initial_dist = max(fget(summary, "initial_rel_dist", 0.5), 1e-6)
    progress = fget(summary, "progress_initial_minus_min", 0.0)
    odom_dx = fget(summary, "odom_x_delta", 0.0)
    max_vx = fget(summary, "max_mpc_vx", 0.0)

    yaw_stats = load_yaw_stats_from_csv(summary)
    yaw_risk, yaw_reason = effective_yaw_risk(yaw_stats)

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

    reason.append(yaw_reason)

    if yaw_risk >= 0.28:
        mismatch += 0.20
        uncertainty += 0.20
        slip_score += 0.25
        stability -= 0.20
    elif yaw_risk >= 0.18:
        mismatch += 0.10
        uncertainty += 0.08
        slip_score += 0.10
        stability -= 0.10
    else:
        mismatch -= 0.03
        uncertainty -= 0.03
        slip_score -= 0.01

    if max_vx >= 0.15 and yaw_risk < 0.24 and reached:
        reason.append("aggressive speed succeeded without sustained severe yaw")
        mismatch -= 0.04
        uncertainty -= 0.04
        stability += 0.03

    out = {
        "ram_mismatch": clamp(mismatch),
        "ram_uncertainty": clamp(uncertainty),
        "recent_max_yaw_rate": clamp(yaw_risk, 0.0, 0.3),
        "recent_slip_score": clamp(slip_score),
        "body_stability_score": clamp(stability),
        "source": "phase_b_summary_raw_risk_v2",
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

    out.update(yaw_stats)
    return out


def ema(prev, raw, alpha):
    return (1.0 - alpha) * prev + alpha * raw


def smooth_with_previous(prev, raw, alpha):
    out = dict(raw)
    out["source"] = "phase_b_summary_pseudo_ram_ema_v2"
    out["ema_alpha"] = alpha
    out["raw_risk_state"] = raw
    out["previous_risk_state"] = prev
    out["reason"] = list(raw.get("reason", [])) + ["EMA-smoothed with previous pseudo-RAM risk state"]

    for k in RISK_KEYS:
        out[k] = clamp(ema(fget(prev, k, fget(raw, k)), fget(raw, k), alpha))

    prev_yaw = fget(prev, "recent_max_yaw_rate", 0.0)
    raw_yaw = fget(raw, "recent_max_yaw_rate", 0.0)

    if raw_yaw > prev_yaw:
        out["recent_max_yaw_rate"] = raw_yaw
        out["reason"].append("effective yaw risk rose immediately")
    else:
        out["recent_max_yaw_rate"] = ema(prev_yaw, raw_yaw, alpha)
        out["reason"].append("effective yaw risk decayed with EMA")

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
