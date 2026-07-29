#!/usr/bin/env python3

import csv
import json
import math
import statistics
import sys

from pathlib import Path


EXPECTED_BLOCKS = [1, 2, 3, 4, 5]


def finite(value):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return None


def mean(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return statistics.mean(values)


def std(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def fmt(value, digits=4):
    if value is None:
        return "NA"

    return f"{value:.{digits}f}"


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: evaluator.py RUN_METRICS.csv OUTPUT_DIR"
        )

    run_metrics_path = Path(
        sys.argv[1]
    ).expanduser().resolve()

    output_dir = Path(
        sys.argv[2]
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with run_metrics_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(
            csv.DictReader(stream)
        )

    by_block = {}

    for row in rows:
        condition = row["condition"]

        if condition not in {"A", "B"}:
            continue

        block = int(row["block"])

        by_block.setdefault(
            block,
            {},
        )[condition] = row

    missing = [
        block
        for block in EXPECTED_BLOCKS
        if (
            block not in by_block
            or "A" not in by_block[block]
            or "B" not in by_block[block]
        )
    ]

    if missing:
        raise RuntimeError(
            "missing paired A/B blocks: {}".format(
                missing
            )
        )

    metric_names = [
        "goal_time_sim_s",
        "goal_entry_abs_y",
        "max_abs_y_pre_goal",
        "rough_abs_dy",
        "goal_work_abs_J_proxy",
        "work_abs_per_m_J_proxy",
        "realized_vx_body_mean",
        "mean_abs_vx_tracking_error",
        "max_abs_roll_deg",
        "max_abs_pitch_deg",
        "min_base_z",
    ]

    effects = {
        metric: []
        for metric in metric_names
    }

    paired_rows = []

    for block in EXPECTED_BLOCKS:
        a = by_block[block]["A"]
        b = by_block[block]["B"]

        paired_row = {
            "block": block,
        }

        for metric in metric_names:
            a_value = finite(
                a.get(metric)
            )

            b_value = finite(
                b.get(metric)
            )

            if (
                a_value is None
                or b_value is None
            ):
                raise RuntimeError(
                    "missing finite metric {} in block {}".format(
                        metric,
                        block,
                    )
                )

            effect = b_value - a_value

            effects[metric].append(
                effect
            )

            paired_row[
                metric + "_A"
            ] = a_value

            paired_row[
                metric + "_B"
            ] = b_value

            paired_row[
                metric + "_B_minus_A"
            ] = effect

        paired_rows.append(
            paired_row
        )

    b_rows = [
        by_block[block]["B"]
        for block in EXPECTED_BLOCKS
    ]

    rules = []

    def add_rule(
        name,
        value,
        operator,
        threshold,
        passed,
        category,
        explanation,
    ):
        rules.append({
            "name": name,
            "category": category,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
            "explanation": explanation,
        })

    time_effects = effects[
        "goal_time_sim_s"
    ]

    work_effects = effects[
        "goal_work_abs_J_proxy"
    ]

    work_m_effects = effects[
        "work_abs_per_m_J_proxy"
    ]

    vx_effects = effects[
        "realized_vx_body_mean"
    ]

    add_rule(
        "goal_time_mean_effect",
        mean(time_effects),
        "<=",
        -3.0,
        mean(time_effects) <= -3.0,
        "core",
        "B must improve paired mean mission time by at least 3.0 simulated seconds.",
    )

    add_rule(
        "goal_time_improved_blocks",
        sum(
            value < 0.0
            for value in time_effects
        ),
        ">=",
        4,
        sum(
            value < 0.0
            for value in time_effects
        ) >= 4,
        "core",
        "B must be faster in at least four of five paired blocks.",
    )

    add_rule(
        "work_mean_effect",
        mean(work_effects),
        "<=",
        -300.0,
        mean(work_effects) <= -300.0,
        "core",
        "B must reduce commanded-work proxy by at least 300 units on paired mean.",
    )

    add_rule(
        "work_improved_blocks",
        sum(
            value < 0.0
            for value in work_effects
        ),
        ">=",
        4,
        sum(
            value < 0.0
            for value in work_effects
        ) >= 4,
        "core",
        "B must reduce commanded-work proxy in at least four of five blocks.",
    )

    add_rule(
        "work_per_meter_mean_effect",
        mean(work_m_effects),
        "<=",
        -35.0,
        mean(work_m_effects) <= -35.0,
        "core",
        "B must reduce work-per-meter proxy by at least 35 units on paired mean.",
    )

    add_rule(
        "work_per_meter_improved_blocks",
        sum(
            value < 0.0
            for value in work_m_effects
        ),
        ">=",
        4,
        sum(
            value < 0.0
            for value in work_m_effects
        ) >= 4,
        "core",
        "B must reduce work-per-meter proxy in at least four of five blocks.",
    )

    add_rule(
        "realized_vx_mean_effect",
        mean(vx_effects),
        ">=",
        0.003,
        mean(vx_effects) >= 0.003,
        "supporting",
        "B must increase realized mean body-frame VX by at least 0.003 m/s.",
    )

    add_rule(
        "realized_vx_improved_blocks",
        sum(
            value > 0.0
            for value in vx_effects
        ),
        ">=",
        4,
        sum(
            value > 0.0
            for value in vx_effects
        ) >= 4,
        "supporting",
        "B must increase realized VX in at least four of five blocks.",
    )

    relative_rules = [
        (
            "goal_entry_abs_y_mean_penalty",
            "goal_entry_abs_y",
            0.15,
            "lateral",
            "Mean paired goal-entry lateral penalty must not exceed 0.15 m.",
        ),
        (
            "max_abs_y_mean_penalty",
            "max_abs_y_pre_goal",
            0.15,
            "lateral",
            "Mean paired maximum lateral-deviation penalty must not exceed 0.15 m.",
        ),
        (
            "rough_abs_dy_mean_penalty",
            "rough_abs_dy",
            0.10,
            "lateral",
            "Mean paired rough-segment drift penalty must not exceed 0.10 m.",
        ),
        (
            "max_roll_mean_penalty",
            "max_abs_roll_deg",
            1.0,
            "safety",
            "Mean paired maximum-roll penalty must not exceed 1 degree.",
        ),
        (
            "max_pitch_mean_penalty",
            "max_abs_pitch_deg",
            1.0,
            "safety",
            "Mean paired maximum-pitch penalty must not exceed 1 degree.",
        ),
        (
            "tracking_error_mean_penalty",
            "mean_abs_vx_tracking_error",
            0.003,
            "safety",
            "Mean paired VX tracking-error penalty must not exceed 0.003 m/s.",
        ),
    ]

    for (
        rule_name,
        metric,
        threshold,
        category,
        explanation,
    ) in relative_rules:
        value = mean(
            effects[metric]
        )

        add_rule(
            rule_name,
            value,
            "<=",
            threshold,
            value <= threshold,
            category,
            explanation,
        )

    min_z_effect = mean(
        effects["min_base_z"]
    )

    add_rule(
        "min_base_z_mean_effect",
        min_z_effect,
        ">=",
        -0.005,
        min_z_effect >= -0.005,
        "safety",
        "Mean paired minimum-base-height loss must not exceed 5 mm.",
    )

    b_max_roll = max(
        finite(row["max_abs_roll_deg"])
        for row in b_rows
    )

    b_max_pitch = max(
        finite(row["max_abs_pitch_deg"])
        for row in b_rows
    )

    b_min_z = min(
        finite(row["min_base_z"])
        for row in b_rows
    )

    add_rule(
        "B_absolute_max_roll",
        b_max_roll,
        "<=",
        20.0,
        b_max_roll <= 20.0,
        "hard_safety",
        "Every B rollout must remain below the preregistered 20-degree roll guard.",
    )

    add_rule(
        "B_absolute_max_pitch",
        b_max_pitch,
        "<=",
        20.0,
        b_max_pitch <= 20.0,
        "hard_safety",
        "Every B rollout must remain below the preregistered 20-degree pitch guard.",
    )

    add_rule(
        "B_absolute_min_base_z",
        b_min_z,
        ">",
        0.20,
        b_min_z > 0.20,
        "hard_safety",
        "Every B rollout must remain above the preregistered 0.20 m base-height guard.",
    )

    core_pass = all(
        row["passed"]
        for row in rules
        if row["category"] == "core"
    )

    supporting_pass = all(
        row["passed"]
        for row in rules
        if row["category"] == "supporting"
    )

    safety_pass = all(
        row["passed"]
        for row in rules
        if row["category"] in {
            "safety",
            "hard_safety",
            "lateral",
        }
    )

    all_pass = (
        core_pass
        and supporting_pass
        and safety_pass
    )

    if all_pass:
        decision = (
            "CONFIRM_SCREENING_PREFERRED_B_SCHEDULE_WORLD_V5_V0"
        )

        interpretation = (
            "The added paired blocks confirm B as the "
            "screening-preferred terrain-scheduled VX candidate "
            "for world-v5. This does not constitute final "
            "cross-world or Objective-Selector promotion."
        )

    elif not safety_pass:
        decision = (
            "DO_NOT_CONFIRM_B_SCHEDULE_SAFETY_OR_LATERAL_PENALTY_V0"
        )

        interpretation = (
            "B is not confirmed because one or more preregistered "
            "safety or lateral-deviation limits failed."
        )

    else:
        decision = (
            "DO_NOT_CONFIRM_B_SCHEDULE_INSUFFICIENT_REPEATABILITY_V0"
        )

        interpretation = (
            "B remains executable, but the five-block performance "
            "advantage did not meet the preregistered repeatability criteria."
        )

    rules_path = (
        output_dir
        / "confirmatory_rules_v0.csv"
    )

    with rules_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "name",
                "category",
                "value",
                "operator",
                "threshold",
                "passed",
                "explanation",
            ],
        )

        writer.writeheader()
        writer.writerows(rules)

    paired_path = (
        output_dir
        / "confirmatory_paired_blocks_v0.csv"
    )

    fieldnames = list(
        paired_rows[0].keys()
    )

    with paired_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(paired_rows)

    payload = {
        "decision": decision,
        "all_pass": all_pass,
        "core_pass": core_pass,
        "supporting_pass": supporting_pass,
        "safety_pass": safety_pass,
        "interpretation": interpretation,
        "rules": rules,
        "paired_blocks": paired_rows,
    }

    (
        output_dir
        / "confirmatory_decision_v0.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# LL1 VX A/B Confirmatory Decision",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        interpretation,
        "",
        "## Preregistered rule results",
        "",
        "| Rule | Category | Value | Requirement | PASS |",
        "|---|---|---:|---:|:---:|",
    ]

    for row in rules:
        lines.append(
            "| {name} | {category} | {value} | "
            "{operator} {threshold} | {passed} |".format(
                name=row["name"],
                category=row["category"],
                value=fmt(
                    finite(row["value"]),
                    5,
                ),
                operator=row["operator"],
                threshold=row["threshold"],
                passed=(
                    "PASS"
                    if row["passed"]
                    else "FAIL"
                ),
            )
        )

    lines.extend([
        "",
        "## Five-block paired effects",
        "",
        "| Block | Δ time [s] | Δ work | Δ work/m | "
        "Δ body VX | Δ goal |y| | Δ max |y| | "
        "Δ roll [deg] | Δ pitch [deg] | Δ min z [m] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for row in paired_rows:
        lines.append(
            "| {block} | {time} | {work} | {work_m} | "
            "{vx} | {goal_y} | {max_y} | {roll} | "
            "{pitch} | {z} |".format(
                block=row["block"],
                time=fmt(
                    row[
                        "goal_time_sim_s_B_minus_A"
                    ]
                ),
                work=fmt(
                    row[
                        "goal_work_abs_J_proxy_B_minus_A"
                    ],
                    2,
                ),
                work_m=fmt(
                    row[
                        "work_abs_per_m_J_proxy_B_minus_A"
                    ],
                    2,
                ),
                vx=fmt(
                    row[
                        "realized_vx_body_mean_B_minus_A"
                    ],
                    5,
                ),
                goal_y=fmt(
                    row[
                        "goal_entry_abs_y_B_minus_A"
                    ]
                ),
                max_y=fmt(
                    row[
                        "max_abs_y_pre_goal_B_minus_A"
                    ]
                ),
                roll=fmt(
                    row[
                        "max_abs_roll_deg_B_minus_A"
                    ]
                ),
                pitch=fmt(
                    row[
                        "max_abs_pitch_deg_B_minus_A"
                    ]
                ),
                z=fmt(
                    row[
                        "min_base_z_B_minus_A"
                    ],
                    5,
                ),
            )
        )

    lines.extend([
        "",
        "## Scope guard",
        "",
        "- This is five-block balanced engineering confirmation in world-v5.",
        "- Work is a commanded mechanical-work proxy, not battery energy.",
        "- Confirmation does not establish cross-world generalization.",
        "- Confirmation does not create a final Objective Selector teacher label.",
        "",
    ])

    report_path = (
        output_dir
        / "confirmatory_decision_v0.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))

    print()
    print("decision={}".format(decision))
    print("report={}".format(report_path))
    print("rules_csv={}".format(rules_path))
    print("paired_csv={}".format(paired_path))


if __name__ == "__main__":
    main()
