#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm_to_minus1_plus1(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def theta_from_row(row):
    terrain = row["terrain"]
    vx = float(row["mpc_vx"])
    h = float(row["mpc_body_height"])
    clr = float(row["mpc_clearance"])
    beta_s = 0.25

    gate_level = float(row.get("gate_level_code", 0.0))
    gate_action = float(row.get("gate_action_code", 0.0))
    sigma = float(row.get("ram_sigma_mean", 0.0))
    rho = float(row.get("ram_rho_norm", 0.0))

    # Nominal/reference values from current rule-based behavior.
    # These are teacher labels for warm-start, not final optimal labels.
    if terrain == "flat_normal":
        gait_period_scale = 1.00
        duty_delta = 0.00
        step_length_scale = 1.00
        stance_width_delta = 0.00
        body_height_delta = h - 0.295
        clearance_delta = clr - 0.030
        impedance_scale = 0.95
        residual_scale = 0.90
        beta = [0.55, 0.25, 0.20]
        semantic = "validated_locomotion"
        style = "fast"

    elif terrain == "rough_mid":
        gait_period_scale = 1.22
        duty_delta = 0.12
        step_length_scale = 0.45
        stance_width_delta = 0.03
        body_height_delta = h - 0.295
        clearance_delta = clr - 0.030
        impedance_scale = 1.35
        residual_scale = 1.10
        beta = [0.30, 0.50, 0.20]
        semantic = "cautious_probe"
        style = "cautious"

    elif terrain == "slope_5deg":
        gait_period_scale = 1.25
        duty_delta = 0.14
        step_length_scale = 0.40
        stance_width_delta = 0.035
        body_height_delta = h - 0.295
        clearance_delta = clr - 0.030
        impedance_scale = 1.40
        residual_scale = 1.15
        beta = [0.25, 0.55, 0.20]
        semantic = "high_clearance_slow_probe"
        style = "high_clearance"

    else:
        gait_period_scale = 1.0
        duty_delta = 0.0
        step_length_scale = 1.0
        stance_width_delta = 0.0
        body_height_delta = h - 0.295
        clearance_delta = clr - 0.030
        impedance_scale = 1.0
        residual_scale = 0.5
        beta = [0.40, 0.40, 0.20]
        semantic = "unknown"
        style = "unknown"

    # If gate is unstable/conservative, teacher becomes more conservative.
    if gate_level >= 2.0 or gate_action >= 2.0:
        gait_period_scale = max(gait_period_scale, 1.25)
        duty_delta = max(duty_delta, 0.14)
        step_length_scale = min(step_length_scale, 0.40)
        stance_width_delta = max(stance_width_delta, 0.04)
        body_height_delta = max(body_height_delta, 0.055)
        clearance_delta = max(clearance_delta, 0.080)
        impedance_scale = max(impedance_scale, 1.45)
        residual_scale = max(residual_scale, 1.25)

    # Normalize according to configs/meta_gait/theta_action_space_v0.yaml.
    theta_physical = [
        gait_period_scale,
        duty_delta,
        step_length_scale,
        stance_width_delta,
        body_height_delta,
        clearance_delta,
        impedance_scale,
        residual_scale,
    ]

    ranges = [
        (0.75, 1.35),
        (-0.12, 0.16),
        (0.35, 1.25),
        (-0.04, 0.06),
        (-0.04, 0.06),
        (0.00, 0.09),
        (0.70, 1.60),
        (0.00, 1.50),
    ]

    theta_norm = [
        norm_to_minus1_plus1(x, lo, hi)
        for x, (lo, hi) in zip(theta_physical, ranges)
    ]

    # Compact feature vector v0.
    # Later c_t and full rho vector can be appended.
    terrain_onehot = [
        1.0 if terrain == "flat_normal" else 0.0,
        1.0 if terrain == "rough_mid" else 0.0,
        1.0 if terrain == "slope_5deg" else 0.0,
    ]

    x = []
    x.extend(terrain_onehot)
    x.extend(beta)
    x.extend([
        float(row.get("ram_future_risk", 0.0)),
        float(row.get("ram_future_slip", 0.0)),
        float(row.get("ram_future_invalid", 0.0)),
        float(row.get("ram_run_fallen", 0.0)),
        float(row.get("ram_recovery_needed", 0.0)),
        sigma,
        min(rho / 10.0, 1.0),
        float(row.get("gate_level_code", 0.0)) / 2.0,
        float(row.get("gate_action_code", 0.0)) / 2.0,
        float(row.get("gate_would_override", 0.0)),
        float(row.get("mpc_vx", 0.0)),
        float(row.get("mpc_body_height", 0.0)),
        float(row.get("mpc_clearance", 0.0)),
    ])

    return {
        "terrain": terrain,
        "semantic": semantic,
        "style": style,
        "x": x,
        "theta": theta_norm,
        "theta_physical": theta_physical,
        "beta": beta,
        "gate_level_code": gate_level,
        "gate_action_code": gate_action,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w") as f:
        for path in args.csv:
            with open(path) as fcsv:
                reader = csv.DictReader(fcsv)
                for row in reader:
                    sample = theta_from_row(row)
                    f.write(json.dumps(sample) + "\n")
                    n += 1

    print(f"[TRACER] wrote {n} theta warm-start samples: {out}")


if __name__ == "__main__":
    main()
