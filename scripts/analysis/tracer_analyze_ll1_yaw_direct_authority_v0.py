#!/usr/bin/env python3
from pathlib import Path
import csv
import json
import math
import statistics
import sys

PATTERN = ["Z", "P", "Z", "N", "Z"]


def finite(value):
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None


def mean(values):
    values = list(values)
    if not values:
        raise RuntimeError("mean of empty values")
    return statistics.mean(values)


def slope(rows):
    times = [row["sim_time"] for row in rows]
    values = [row["yaw_unwrapped"] for row in rows]
    t0 = times[0]
    x = [value - t0 for value in times]
    x_mean = mean(x)
    y_mean = mean(values)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 1e-12:
        raise RuntimeError("invalid regression denominator")
    return sum(
        (xi - x_mean) * (yi - y_mean)
        for xi, yi in zip(x, values)
    ) / denominator


def command_label(command):
    if abs(command) <= 0.010:
        return "Z"
    if command >= 0.035:
        return "P"
    if command <= -0.035:
        return "N"
    return None


def unwrap_rows(rows):
    unwrapped = rows[0]["yaw_rad"]
    rows[0]["yaw_unwrapped"] = unwrapped
    previous = rows[0]["yaw_rad"]

    for row in rows[1:]:
        current = row["yaw_rad"]
        delta = current - previous
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        unwrapped += delta
        row["yaw_unwrapped"] = unwrapped
        previous = current


def trim_segment(rows, margin=0.75):
    start = rows[0]["sim_time"]
    end = rows[-1]["sim_time"]
    selected = [
        row
        for row in rows
        if row["sim_time"] >= start + margin
        and row["sim_time"] <= end - margin
    ]
    if len(selected) < 20:
        raise RuntimeError("insufficient trimmed segment rows")
    return selected


def segment_metrics(label, rows):
    selected = trim_segment(rows)
    return {
        "label": label,
        "n": len(selected),
        "duration_sim_s": (
            selected[-1]["sim_time"] - selected[0]["sim_time"]
        ),
        "cmd_mean": mean(row["ref_yaw_rate"] for row in selected),
        "wz_mean": mean(row["wz_world_rad_s"] for row in selected),
        "wz_median": statistics.median(
            row["wz_world_rad_s"] for row in selected
        ),
        "yaw_slope_rad_s": slope(selected),
        "heading_delta_rad": (
            selected[-1]["yaw_unwrapped"]
            - selected[0]["yaw_unwrapped"]
        ),
        "positive_wz_fraction": mean(
            1.0 if row["wz_world_rad_s"] > 0.0 else 0.0
            for row in selected
        ),
        "negative_wz_fraction": mean(
            1.0 if row["wz_world_rad_s"] < 0.0 else 0.0
            for row in selected
        ),
        "max_abs_roll_deg": max(
            abs(row["roll_deg"]) for row in selected
        ),
        "max_abs_pitch_deg": max(
            abs(row["pitch_deg"]) for row in selected
        ),
        "min_z": min(row["z"] for row in selected),
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: analyzer.py INPUT.csv SUMMARY.txt DECISION.json"
        )

    csv_path = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])
    decision_path = Path(sys.argv[3])

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))

    required = {
        "sim_time",
        "x",
        "y",
        "z",
        "roll_deg",
        "pitch_deg",
        "yaw_rad",
        "wz_world_rad_s",
        "ref_vx",
        "ref_yaw_rate",
        "ref_body_h",
        "ref_clearance",
        "ref_enable",
        "ref_age_sim_s",
    }

    if not raw_rows:
        raise RuntimeError("empty yaw authority CSV")

    missing = required - set(raw_rows[0].keys())
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")

    rows = []
    numeric_fields = sorted(required)

    for raw in raw_rows:
        row = {}
        valid = True
        for field in numeric_fields:
            value = finite(raw.get(field))
            if value is None:
                valid = False
                break
            row[field] = value

        if not valid:
            continue
        if row["ref_enable"] < 0.5:
            continue
        if row["ref_age_sim_s"] > 0.5:
            continue
        rows.append(row)

    if len(rows) < 200:
        raise RuntimeError(f"insufficient valid rows: {len(rows)}")

    last_reset_index = 0
    for index in range(1, len(rows)):
        if rows[index]["sim_time"] < rows[index - 1]["sim_time"] - 0.1:
            last_reset_index = index
    rows = rows[last_reset_index:]

    unwrap_rows(rows)

    segments = []
    current_label = None
    current_rows = []

    for row in rows:
        label = command_label(row["ref_yaw_rate"])

        if label is None:
            if current_rows:
                segments.append((current_label, current_rows))
            current_label = None
            current_rows = []
            continue

        if not current_rows:
            current_label = label
            current_rows = [row]
            continue

        time_gap = row["sim_time"] - current_rows[-1]["sim_time"]
        if label != current_label or time_gap > 0.5:
            segments.append((current_label, current_rows))
            current_label = label
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        segments.append((current_label, current_rows))

    long_segments = []
    for label, segment_rows in segments:
        duration = (
            segment_rows[-1]["sim_time"] - segment_rows[0]["sim_time"]
        )
        if duration >= 2.0:
            long_segments.append((label, segment_rows))

    selected = None
    for index in range(0, len(long_segments) - 4):
        labels = [
            long_segments[index + offset][0]
            for offset in range(5)
        ]
        if labels == PATTERN:
            selected = long_segments[index:index + 5]
            break

    if selected is None:
        observed = [
            (
                label,
                segment_rows[-1]["sim_time"]
                - segment_rows[0]["sim_time"],
            )
            for label, segment_rows in long_segments
        ]
        raise RuntimeError(
            "Z/P/Z/N/Z command sequence not found; "
            f"observed={observed}"
        )

    names = ["Z0", "P", "Z1", "N", "Z2"]
    metrics = {
        name: segment_metrics(name, segment_rows)
        for name, (_, segment_rows) in zip(names, selected)
    }

    plus_neutral_wz = mean([
        metrics["Z0"]["wz_mean"],
        metrics["Z1"]["wz_mean"],
    ])
    minus_neutral_wz = mean([
        metrics["Z1"]["wz_mean"],
        metrics["Z2"]["wz_mean"],
    ])
    plus_neutral_slope = mean([
        metrics["Z0"]["yaw_slope_rad_s"],
        metrics["Z1"]["yaw_slope_rad_s"],
    ])
    minus_neutral_slope = mean([
        metrics["Z1"]["yaw_slope_rad_s"],
        metrics["Z2"]["yaw_slope_rad_s"],
    ])

    plus_wz_effect = metrics["P"]["wz_mean"] - plus_neutral_wz
    minus_wz_effect = metrics["N"]["wz_mean"] - minus_neutral_wz
    plus_slope_effect = (
        metrics["P"]["yaw_slope_rad_s"] - plus_neutral_slope
    )
    minus_slope_effect = (
        metrics["N"]["yaw_slope_rad_s"] - minus_neutral_slope
    )

    selected_rows = [
        row
        for _, segment_rows in selected
        for row in segment_rows
    ]
    worst_roll = max(abs(row["roll_deg"]) for row in selected_rows)
    worst_pitch = max(abs(row["pitch_deg"]) for row in selected_rows)
    worst_min_z = min(row["z"] for row in selected_rows)

    rules = []

    def rule(name, value, operator, threshold, passed):
        rules.append({
            "name": name,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        })

    rule(
        "plus_command_transport",
        metrics["P"]["cmd_mean"],
        "abs(value-0.050)<=",
        0.002,
        abs(metrics["P"]["cmd_mean"] - 0.050) <= 0.002,
    )
    rule(
        "minus_command_transport",
        metrics["N"]["cmd_mean"],
        "abs(value+0.050)<=",
        0.002,
        abs(metrics["N"]["cmd_mean"] + 0.050) <= 0.002,
    )

    for neutral in ["Z0", "Z1", "Z2"]:
        rule(
            f"{neutral}_neutral_transport",
            metrics[neutral]["cmd_mean"],
            "abs(value)<=",
            0.002,
            abs(metrics[neutral]["cmd_mean"]) <= 0.002,
        )

    # robot_odom_flat has no timestamp, so command phases use odom-fresh
    # active wall time. These gates require adequate realized Gazebo
    # simulation-time coverage before authority can be accepted.
    for neutral in ["Z0", "Z1", "Z2"]:
        rule(
            f"{neutral}_duration_sim_s",
            metrics[neutral]["duration_sim_s"],
            ">=",
            1.50,
            metrics[neutral]["duration_sim_s"] >= 1.50,
        )

    for signed in ["P", "N"]:
        rule(
            f"{signed}_duration_sim_s",
            metrics[signed]["duration_sim_s"],
            ">=",
            2.50,
            metrics[signed]["duration_sim_s"] >= 2.50,
        )

    rule(
        "plus_realized_wz_effect",
        plus_wz_effect,
        ">=",
        0.005,
        plus_wz_effect >= 0.005,
    )
    rule(
        "minus_realized_wz_effect",
        minus_wz_effect,
        "<=",
        -0.005,
        minus_wz_effect <= -0.005,
    )
    rule(
        "plus_unwrapped_yaw_slope_effect",
        plus_slope_effect,
        ">=",
        0.005,
        plus_slope_effect >= 0.005,
    )
    rule(
        "minus_unwrapped_yaw_slope_effect",
        minus_slope_effect,
        "<=",
        -0.005,
        minus_slope_effect <= -0.005,
    )
    rule(
        "plus_heading_delta",
        metrics["P"]["heading_delta_rad"],
        ">=",
        0.030,
        metrics["P"]["heading_delta_rad"] >= 0.030,
    )
    rule(
        "minus_heading_delta",
        metrics["N"]["heading_delta_rad"],
        "<=",
        -0.030,
        metrics["N"]["heading_delta_rad"] <= -0.030,
    )
    rule(
        "plus_positive_wz_fraction",
        metrics["P"]["positive_wz_fraction"],
        ">=",
        0.60,
        metrics["P"]["positive_wz_fraction"] >= 0.60,
    )
    rule(
        "minus_negative_wz_fraction",
        metrics["N"]["negative_wz_fraction"],
        ">=",
        0.60,
        metrics["N"]["negative_wz_fraction"] >= 0.60,
    )
    rule(
        "hard_safety_roll",
        worst_roll,
        "<=",
        20.0,
        worst_roll <= 20.0,
    )
    rule(
        "hard_safety_pitch",
        worst_pitch,
        "<=",
        20.0,
        worst_pitch <= 20.0,
    )
    rule(
        "hard_safety_min_z",
        worst_min_z,
        ">",
        0.20,
        worst_min_z > 0.20,
    )

    authority_rules = [
        row
        for row in rules
        if row["name"].startswith((
            "plus_realized",
            "minus_realized",
            "plus_unwrapped",
            "minus_unwrapped",
            "plus_heading",
            "minus_heading",
            "plus_positive",
            "minus_negative",
        ))
    ]
    transport_pass = all(
        row["passed"] for row in rules if "transport" in row["name"]
    )
    duration_pass = all(
        row["passed"]
        for row in rules
        if row["name"].endswith("_duration_sim_s")
    )
    authority_pass = all(row["passed"] for row in authority_rules)
    safety_pass = all(
        row["passed"]
        for row in rules
        if row["name"].startswith("hard_safety")
    )
    all_pass = (
        transport_pass
        and duration_pass
        and authority_pass
        and safety_pass
    )

    classification = (
        "valid_direct_signed_yaw_authority_smoke_v0"
        if all_pass
        else "direct_signed_yaw_authority_not_yet_proven_v0"
    )

    payload = {
        "classification": classification,
        "all_pass": all_pass,
        "transport_pass": transport_pass,
        "duration_pass": duration_pass,
        "authority_pass": authority_pass,
        "safety_pass": safety_pass,
        "plus_wz_effect_rad_s": plus_wz_effect,
        "minus_wz_effect_rad_s": minus_wz_effect,
        "plus_yaw_slope_effect_rad_s": plus_slope_effect,
        "minus_yaw_slope_effect_rad_s": minus_slope_effect,
        "worst_roll_deg": worst_roll,
        "worst_pitch_deg": worst_pitch,
        "worst_min_z": worst_min_z,
        "segments": metrics,
        "rules": rules,
    }

    decision_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "============================================================",
        "LL1 DIRECT SIGNED YAW AUTHORITY SMOKE RESULT",
        "============================================================",
        f"classification={classification}",
        f"all_pass={all_pass}",
        f"transport_pass={transport_pass}",
        f"duration_pass={duration_pass}",
        f"authority_pass={authority_pass}",
        f"safety_pass={safety_pass}",
        "",
        "segment metrics:",
    ]

    for name in names:
        item = metrics[name]
        lines.append(
            f"{name}: "
            f"duration={item['duration_sim_s']:.4f}s "
            f"cmd={item['cmd_mean']:+.6f} "
            f"wz_mean={item['wz_mean']:+.6f} "
            f"yaw_slope={item['yaw_slope_rad_s']:+.6f} "
            f"heading_delta={item['heading_delta_rad']:+.6f} "
            f"pos_frac={item['positive_wz_fraction']:.4f} "
            f"neg_frac={item['negative_wz_fraction']:.4f}"
        )

    lines.extend([
        "",
        f"plus_wz_effect_rad_s={plus_wz_effect:+.8f}",
        f"minus_wz_effect_rad_s={minus_wz_effect:+.8f}",
        f"plus_yaw_slope_effect_rad_s={plus_slope_effect:+.8f}",
        f"minus_yaw_slope_effect_rad_s={minus_slope_effect:+.8f}",
        "",
        f"worst_roll_deg={worst_roll:.6f}",
        f"worst_pitch_deg={worst_pitch:.6f}",
        f"worst_min_z={worst_min_z:.6f}",
        "",
        "rules:",
    ])

    for item in rules:
        lines.append(
            f"{'PASS' if item['passed'] else 'FAIL'} "
            f"{item['name']}: "
            f"value={item['value']:.9f} "
            f"requirement={item['operator']} {item['threshold']}"
        )

    lines.extend([
        "",
        "scope:",
        "- direct ROS2-to-UDP-to-ROS1-to-LL1 route",
        "- selector phases use odom-fresh active wall time",
        "- duration gates use realized ROS1/Gazebo simulation time",
        "- J4/J7 bypassed",
        "- authority only",
        "- no lateral-performance or generalization claim",
        "- commanded mechanical work is not evaluated here",
    ])

    text = "\n".join(lines) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
