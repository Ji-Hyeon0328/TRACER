#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import csv
import json
import math
import statistics
import sys


CONDITION_NAMES = {
    "A": "fixed_low",
    "B": "abrupt_schedule",
    "C": "transition_aware",
}

DISPLAY_NAMES = {
    "A": "A fixed-low",
    "B": "B abrupt",
    "C": "T transition-aware",
}

METRICS = [
    "goal_time_sim_s",
    "goal_entry_abs_y",
    "max_abs_y_pre_goal",
    "rough_abs_dy",
    "rough_max_abs_y",
    "goal_work_abs_J_proxy",
    "work_abs_per_m_J_proxy",
    "realized_vx_body_mean",
    "mean_abs_vx_tracking_error",
    "max_abs_roll_deg",
    "max_abs_pitch_deg",
    "min_base_z",
    "upslope_entry_vx_body",
    "upslope_entry_abs_y",
    "upslope_entry_abs_roll_deg",
    "upslope_entry_abs_pitch_deg",
    "rough_entry_vx_body",
    "rough_entry_abs_y",
]


def finite(value: Any) -> float | None:
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return None


def mean(values: Iterable[float]) -> float:
    values = list(values)

    if not values:
        raise RuntimeError("mean of empty list")

    return statistics.mean(values)


def std(values: Iterable[float]) -> float:
    values = list(values)

    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(
            csv.DictReader(
                stream,
                delimiter="\t",
            )
        )

    if len(rows) != 9:
        raise RuntimeError(
            f"expected 9 manifest rows, found {len(rows)}"
        )

    counts = Counter(
        row["condition"]
        for row in rows
    )

    if counts != Counter({
        "A": 3,
        "B": 3,
        "C": 3,
    }):
        raise RuntimeError(
            f"unexpected condition counts: {dict(counts)}"
        )

    if any(
        int(row["rc"]) != 0
        for row in rows
    ):
        raise RuntimeError(
            "manifest contains nonzero rollout rc"
        )

    return rows


def read_valid_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        raw_rows = list(
            csv.DictReader(stream)
        )

    required = {
        "sim_time",
        "valid",
        "x",
        "y",
        "z",
        "roll_deg",
        "pitch_deg",
        "vx_body",
        "context",
        "vx_cmd",
        "work_abs_J_cmd",
    }

    if not raw_rows:
        raise RuntimeError(
            f"empty CSV: {path}"
        )

    missing = required - set(raw_rows[0].keys())

    if missing:
        raise RuntimeError(
            f"missing columns in {path}: {sorted(missing)}"
        )

    rows: List[Dict[str, Any]] = []

    numeric_columns = [
        "sim_time",
        "x",
        "y",
        "z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "vx_world",
        "vy_world",
        "vx_body",
        "vy_body",
        "vx_cmd",
        "work_abs_J_cmd",
    ]

    for raw in raw_rows:
        valid = finite(
            raw.get("valid")
        )

        if valid is None or valid < 0.5:
            continue

        row: Dict[str, Any] = {
            "context": raw.get(
                "context",
                "",
            ),
        }

        bad = False

        for column in numeric_columns:
            number = finite(
                raw.get(column)
            )

            if number is None:
                if column in {
                    "yaw_deg",
                    "vx_world",
                    "vy_world",
                    "vy_body",
                }:
                    number = 0.0
                else:
                    bad = True
                    break

            row[column] = number

        if bad:
            continue

        rows.append(row)

    if len(rows) < 100:
        raise RuntimeError(
            f"insufficient valid rows in {path}: {len(rows)}"
        )

    return rows


def first_context(
    rows: List[Dict[str, Any]],
    context: str,
) -> Dict[str, Any]:
    for row in rows:
        if row["context"] == context:
            return row

    raise RuntimeError(
        f"missing context {context}"
    )


def analyze_run(
    manifest_row: Dict[str, str],
) -> Dict[str, Any]:
    log_dir = Path(
        manifest_row["log_dir"]
    )

    csv_path = (
        log_dir
        / "ros1_joint_energy_proxy.csv"
    )

    if not csv_path.is_file():
        raise RuntimeError(
            f"missing run CSV: {csv_path}"
        )

    rows = read_valid_rows(
        csv_path
    )

    goal_index = None

    for index, row in enumerate(rows):
        if row["context"] == "goal_flat":
            goal_index = index
            break

    if goal_index is None:
        raise RuntimeError(
            f"goal_flat not reached: {csv_path}"
        )

    window = rows[
        : goal_index + 1
    ]

    start = window[0]
    goal = window[-1]

    rough_rows = [
        row
        for row in window
        if row["context"] == "rough"
    ]

    if len(rough_rows) < 2:
        raise RuntimeError(
            f"insufficient rough rows: {csv_path}"
        )

    upslope_entry = first_context(
        window,
        "upslope",
    )

    rough_entry = first_context(
        window,
        "rough",
    )

    distance = (
        goal["x"]
        - start["x"]
    )

    if distance <= 0.1:
        raise RuntimeError(
            f"invalid traveled distance: {distance}"
        )

    work = (
        goal["work_abs_J_cmd"]
        - start["work_abs_J_cmd"]
    )

    if work <= 0.0:
        raise RuntimeError(
            f"invalid commanded-work proxy: {work}"
        )

    condition = manifest_row[
        "condition"
    ]

    result: Dict[str, Any] = {
        "run_index": int(
            manifest_row["run_index"]
        ),
        "block": int(
            manifest_row["block"]
        ),
        "condition": condition,
        "condition_name": CONDITION_NAMES[
            condition
        ],
        "log_dir": str(log_dir),
        "csv_path": str(csv_path),

        "start_x": start["x"],
        "goal_x": goal["x"],
        "distance_m": distance,

        "goal_time_sim_s": (
            goal["sim_time"]
            - start["sim_time"]
        ),

        "goal_entry_abs_y": abs(
            goal["y"]
        ),

        "max_abs_y_pre_goal": max(
            abs(row["y"])
            for row in window
        ),

        "rough_abs_dy": abs(
            rough_rows[-1]["y"]
            - rough_rows[0]["y"]
        ),

        "rough_max_abs_y": max(
            abs(row["y"])
            for row in rough_rows
        ),

        "goal_work_abs_J_proxy": work,

        "work_abs_per_m_J_proxy": (
            work
            / distance
        ),

        "realized_vx_body_mean": mean(
            row["vx_body"]
            for row in window
        ),

        "mean_abs_vx_tracking_error": mean(
            abs(
                row["vx_cmd"]
                - row["vx_body"]
            )
            for row in window
        ),

        "max_abs_roll_deg": max(
            abs(row["roll_deg"])
            for row in window
        ),

        "max_abs_pitch_deg": max(
            abs(row["pitch_deg"])
            for row in window
        ),

        "min_base_z": min(
            row["z"]
            for row in window
        ),

        "upslope_entry_vx_body":
            upslope_entry["vx_body"],

        "upslope_entry_abs_y": abs(
            upslope_entry["y"]
        ),

        "upslope_entry_abs_roll_deg": abs(
            upslope_entry["roll_deg"]
        ),

        "upslope_entry_abs_pitch_deg": abs(
            upslope_entry["pitch_deg"]
        ),

        "rough_entry_vx_body":
            rough_entry["vx_body"],

        "rough_entry_abs_y": abs(
            rough_entry["y"]
        ),
    }

    return result


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: List[str] | None = None,
) -> None:
    if not rows:
        raise RuntimeError(
            f"cannot write empty CSV: {path}"
        )

    if fieldnames is None:
        fieldnames = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: analyzer.py RUN_ROOT"
        )

    run_root = Path(
        sys.argv[1]
    ).expanduser().resolve()

    manifest_path = (
        run_root
        / "manifest.tsv"
    )

    analysis_dir = (
        run_root
        / "analysis_vx_transition_abt_v0"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = read_manifest(
        manifest_path
    )

    run_metrics = [
        analyze_run(row)
        for row in manifest
    ]

    run_metrics.sort(
        key=lambda row: row[
            "run_index"
        ]
    )

    write_csv(
        analysis_dir
        / "run_metrics_v0.csv",
        run_metrics,
    )

    condition_rows = []

    by_condition: Dict[
        str,
        List[Dict[str, Any]],
    ] = {
        condition: [
            row
            for row in run_metrics
            if row["condition"]
            == condition
        ]
        for condition in [
            "A",
            "B",
            "C",
        ]
    }

    for condition, rows in by_condition.items():
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in rows
            ]

            condition_rows.append({
                "condition": condition,
                "condition_name":
                    CONDITION_NAMES[condition],
                "metric": metric,
                "n": len(values),
                "mean": mean(values),
                "std": std(values),
                "min": min(values),
                "max": max(values),
            })

    write_csv(
        analysis_dir
        / "condition_summary_long_v0.csv",
        condition_rows,
    )

    by_block: Dict[
        int,
        Dict[str, Dict[str, Any]],
    ] = {}

    for row in run_metrics:
        by_block.setdefault(
            int(row["block"]),
            {},
        )[row["condition"]] = row

    for block in [1, 2, 3]:
        conditions = set(
            by_block.get(
                block,
                {},
            ).keys()
        )

        if conditions != {
            "A",
            "B",
            "C",
        }:
            raise RuntimeError(
                f"incomplete block {block}: {conditions}"
            )

    contrasts: List[
        Tuple[str, str, str]
    ] = [
        (
            "T_minus_A",
            "C",
            "A",
        ),
        (
            "T_minus_B",
            "C",
            "B",
        ),
        (
            "B_minus_A",
            "B",
            "A",
        ),
    ]

    paired_rows = []

    for (
        contrast,
        left,
        right,
    ) in contrasts:
        for block in [
            1,
            2,
            3,
        ]:
            left_row = by_block[
                block
            ][left]

            right_row = by_block[
                block
            ][right]

            for metric in METRICS:
                left_value = float(
                    left_row[metric]
                )

                right_value = float(
                    right_row[metric]
                )

                paired_rows.append({
                    "contrast": contrast,
                    "block": block,
                    "metric": metric,
                    "left_condition": left,
                    "right_condition": right,
                    "left_value": left_value,
                    "right_value": right_value,
                    "effect": (
                        left_value
                        - right_value
                    ),
                })

    write_csv(
        analysis_dir
        / "paired_block_effects_v0.csv",
        paired_rows,
    )

    paired_summary = []

    for (
        contrast,
        left,
        right,
    ) in contrasts:
        for metric in METRICS:
            selected = [
                row
                for row in paired_rows
                if (
                    row["contrast"]
                    == contrast
                    and row["metric"]
                    == metric
                )
            ]

            effects = [
                float(row["effect"])
                for row in selected
            ]

            paired_summary.append({
                "contrast": contrast,
                "metric": metric,
                "n": len(effects),
                "mean_effect": mean(effects),
                "std_effect": std(effects),
                "positive_count": sum(
                    value > 0.0
                    for value in effects
                ),
                "negative_count": sum(
                    value < 0.0
                    for value in effects
                ),
                "zero_count": sum(
                    value == 0.0
                    for value in effects
                ),
            })

    write_csv(
        analysis_dir
        / "paired_effect_summary_v0.csv",
        paired_summary,
    )

    paired_lookup = {
        (
            row["contrast"],
            row["metric"],
        ): float(
            row["mean_effect"]
        )
        for row in paired_summary
    }

    def effect(
        contrast: str,
        metric: str,
    ) -> float:
        return paired_lookup[
            (
                contrast,
                metric,
            )
        ]

    t_rows = by_condition["C"]

    rules = []

    def add_rule(
        name: str,
        category: str,
        value: float,
        operator: str,
        threshold: float,
        passed: bool,
        explanation: str,
    ) -> None:
        rules.append({
            "name": name,
            "category": category,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
            "explanation": explanation,
        })

    # Preregistered T-minus-A requirements.
    value = effect(
        "T_minus_A",
        "goal_time_sim_s",
    )

    add_rule(
        "T_minus_A_goal_time",
        "baseline_gain",
        value,
        "<=",
        -3.0,
        value <= -3.0,
        "T must remain at least 3 simulated seconds faster than A.",
    )

    value = effect(
        "T_minus_A",
        "goal_work_abs_J_proxy",
    )

    add_rule(
        "T_minus_A_work",
        "baseline_gain",
        value,
        "<=",
        -300.0,
        value <= -300.0,
        "T must reduce commanded-work proxy by at least 300 relative to A.",
    )

    value = effect(
        "T_minus_A",
        "max_abs_y_pre_goal",
    )

    add_rule(
        "T_minus_A_max_abs_y_penalty",
        "baseline_lateral",
        value,
        "<=",
        0.15,
        value <= 0.15,
        "T maximum lateral-deviation penalty relative to A must not exceed 0.15 m.",
    )

    value = effect(
        "T_minus_A",
        "rough_abs_dy",
    )

    add_rule(
        "T_minus_A_rough_abs_dy_penalty",
        "baseline_lateral",
        value,
        "<=",
        0.10,
        value <= 0.10,
        "T rough-segment lateral-displacement penalty relative to A must not exceed 0.10 m.",
    )

    # Preregistered T-minus-B requirements.
    value = effect(
        "T_minus_B",
        "max_abs_y_pre_goal",
    )

    add_rule(
        "T_minus_B_max_abs_y_improvement",
        "transition_hypothesis",
        value,
        "<=",
        -0.05,
        value <= -0.05,
        "T must reduce maximum lateral deviation by at least 0.05 m relative to abrupt B.",
    )

    value = effect(
        "T_minus_B",
        "rough_abs_dy",
    )

    add_rule(
        "T_minus_B_rough_abs_dy_improvement",
        "transition_hypothesis",
        value,
        "<=",
        -0.05,
        value <= -0.05,
        "T must reduce rough lateral displacement by at least 0.05 m relative to abrupt B.",
    )

    value = effect(
        "T_minus_B",
        "goal_time_sim_s",
    )

    add_rule(
        "T_minus_B_goal_time_penalty",
        "transition_cost",
        value,
        "<=",
        3.0,
        value <= 3.0,
        "T mission-time penalty relative to B must not exceed 3 simulated seconds.",
    )

    value = effect(
        "T_minus_B",
        "goal_work_abs_J_proxy",
    )

    add_rule(
        "T_minus_B_work_penalty",
        "transition_cost",
        value,
        "<=",
        300.0,
        value <= 300.0,
        "T commanded-work penalty relative to B must not exceed 300.",
    )

    t_worst_roll = max(
        float(row["max_abs_roll_deg"])
        for row in t_rows
    )

    t_worst_pitch = max(
        float(row["max_abs_pitch_deg"])
        for row in t_rows
    )

    t_worst_min_z = min(
        float(row["min_base_z"])
        for row in t_rows
    )

    add_rule(
        "T_absolute_max_roll",
        "hard_safety",
        t_worst_roll,
        "<=",
        20.0,
        t_worst_roll <= 20.0,
        "Every T rollout must remain below 20 degrees absolute roll.",
    )

    add_rule(
        "T_absolute_max_pitch",
        "hard_safety",
        t_worst_pitch,
        "<=",
        20.0,
        t_worst_pitch <= 20.0,
        "Every T rollout must remain below 20 degrees absolute pitch.",
    )

    add_rule(
        "T_absolute_min_base_z",
        "hard_safety",
        t_worst_min_z,
        ">",
        0.20,
        t_worst_min_z > 0.20,
        "Every T rollout must remain above 0.20 m base height.",
    )

    write_csv(
        analysis_dir
        / "decision_rules_v0.csv",
        rules,
    )

    hard_safety_pass = all(
        row["passed"]
        for row in rules
        if row["category"]
        == "hard_safety"
    )

    baseline_pass = all(
        row["passed"]
        for row in rules
        if row["category"] in {
            "baseline_gain",
            "baseline_lateral",
        }
    )

    transition_hypothesis_pass = all(
        row["passed"]
        for row in rules
        if row["category"]
        == "transition_hypothesis"
    )

    transition_cost_pass = all(
        row["passed"]
        for row in rules
        if row["category"]
        == "transition_cost"
    )

    all_pass = all(
        row["passed"]
        for row in rules
    )

    if all_pass:
        decision = (
            "SCREENING_SUPPORTS_TRANSITION_AWARE_"
            "ORACLE_CANDIDATE_WORLD_V5_V0"
        )

        interpretation = (
            "T retained useful benefit relative to A, improved both "
            "preregistered lateral metrics relative to abrupt B, and "
            "stayed within the preregistered time/work and hard-safety bounds."
        )

    elif not hard_safety_pass:
        decision = (
            "REJECT_TRANSITION_AWARE_CANDIDATE_"
            "HARD_SAFETY_FAIL_WORLD_V5_V0"
        )

        interpretation = (
            "T violated at least one preregistered hard-safety bound."
        )

    elif not baseline_pass:
        decision = (
            "DO_NOT_PROMOTE_TRANSITION_AWARE_"
            "INSUFFICIENT_BASELINE_TRADEOFF_WORLD_V5_V0"
        )

        interpretation = (
            "T did not retain the preregistered mission/work/lateral "
            "trade-off relative to conservative A."
        )

    elif not transition_hypothesis_pass:
        decision = (
            "TRANSITION_HYPOTHESIS_NOT_SUPPORTED_"
            "LATERAL_IMPROVEMENT_FAILED_WORLD_V5_V0"
        )

        interpretation = (
            "T did not reduce both preregistered lateral metrics "
            "relative to abrupt B by the required amount."
        )

    else:
        decision = (
            "DO_NOT_PROMOTE_TRANSITION_AWARE_"
            "TIME_OR_WORK_COST_FAILED_WORLD_V5_V0"
        )

        interpretation = (
            "T improved lateral behavior but exceeded the preregistered "
            "time or commanded-work penalty relative to B."
        )

    decision_payload = {
        "decision": decision,
        "all_pass": all_pass,
        "hard_safety_pass":
            hard_safety_pass,
        "baseline_pass":
            baseline_pass,
        "transition_hypothesis_pass":
            transition_hypothesis_pass,
        "transition_cost_pass":
            transition_cost_pass,
        "interpretation":
            interpretation,
        "rules":
            rules,
    }

    (
        analysis_dir
        / "decision_v0.json"
    ).write_text(
        json.dumps(
            decision_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_metrics = [
        "goal_time_sim_s",
        "goal_entry_abs_y",
        "max_abs_y_pre_goal",
        "rough_abs_dy",
        "goal_work_abs_J_proxy",
        "work_abs_per_m_J_proxy",
        "realized_vx_body_mean",
        "max_abs_roll_deg",
        "max_abs_pitch_deg",
        "min_base_z",
        "upslope_entry_vx_body",
        "upslope_entry_abs_y",
        "rough_entry_vx_body",
        "rough_entry_abs_y",
    ]

    lines = [
        "# LL1 VX Transition A/B/T Balanced Analysis",
        "",
        "## Collection validity",
        "",
        "- balanced order: `ABT / TAB / BTA`",
        "- conditions: A/B/T = `3/3/3`",
        "- collection return codes: `9/9 rc=0`",
        "- analysis ends at first `goal_flat` entry",
        "- post-goal hold is excluded",
        "- work is commanded mechanical-work proxy, not battery energy",
        "",
        "## Per-run mission metrics",
        "",
        "| run | block | condition | time | goal |y| | max |y| | rough |Δy| | work | mean vx | roll | pitch | min z |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in run_metrics:
        lines.append(
            "| {run} | {block} | {condition} | "
            "{time} | {goal_y} | {max_y} | {rough_dy} | "
            "{work} | {vx} | {roll} | {pitch} | {z} |".format(
                run=row["run_index"],
                block=row["block"],
                condition=DISPLAY_NAMES[
                    row["condition"]
                ],
                time=fmt(
                    row["goal_time_sim_s"]
                ),
                goal_y=fmt(
                    row["goal_entry_abs_y"]
                ),
                max_y=fmt(
                    row["max_abs_y_pre_goal"]
                ),
                rough_dy=fmt(
                    row["rough_abs_dy"]
                ),
                work=fmt(
                    row["goal_work_abs_J_proxy"],
                    1,
                ),
                vx=fmt(
                    row["realized_vx_body_mean"],
                    5,
                ),
                roll=fmt(
                    row["max_abs_roll_deg"]
                ),
                pitch=fmt(
                    row["max_abs_pitch_deg"]
                ),
                z=fmt(
                    row["min_base_z"]
                ),
            )
        )

    lines.extend([
        "",
        "## Condition summary",
        "",
        "| condition | time | max |y| | rough |Δy| | work | work/m | mean body vx | max roll | max pitch | min z |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for condition in [
        "A",
        "B",
        "C",
    ]:
        rows = by_condition[
            condition
        ]

        def summary(metric: str, digits: int = 4) -> str:
            values = [
                float(row[metric])
                for row in rows
            ]

            return (
                f"{mean(values):.{digits}f} "
                f"± {std(values):.{digits}f}"
            )

        lines.append(
            "| {condition} | {time} | {max_y} | {rough_dy} | "
            "{work} | {work_m} | {vx} | {roll} | {pitch} | {z} |".format(
                condition=DISPLAY_NAMES[
                    condition
                ],
                time=summary(
                    "goal_time_sim_s"
                ),
                max_y=summary(
                    "max_abs_y_pre_goal"
                ),
                rough_dy=summary(
                    "rough_abs_dy"
                ),
                work=summary(
                    "goal_work_abs_J_proxy",
                    1,
                ),
                work_m=summary(
                    "work_abs_per_m_J_proxy",
                    1,
                ),
                vx=summary(
                    "realized_vx_body_mean",
                    5,
                ),
                roll=summary(
                    "max_abs_roll_deg"
                ),
                pitch=summary(
                    "max_abs_pitch_deg"
                ),
                z=summary(
                    "min_base_z"
                ),
            )
        )

    lines.extend([
        "",
        "## Terrain-entry state",
        "",
        "| condition | upslope entry vx | upslope entry |y| | rough entry vx | rough entry |y| |",
        "|---|---:|---:|---:|---:|",
    ])

    for condition in [
        "A",
        "B",
        "C",
    ]:
        rows = by_condition[
            condition
        ]

        lines.append(
            "| {condition} | {up_vx} | {up_y} | {rough_vx} | {rough_y} |".format(
                condition=DISPLAY_NAMES[
                    condition
                ],
                up_vx=(
                    f"{mean(float(row['upslope_entry_vx_body']) for row in rows):.5f} "
                    f"± {std(float(row['upslope_entry_vx_body']) for row in rows):.5f}"
                ),
                up_y=(
                    f"{mean(float(row['upslope_entry_abs_y']) for row in rows):.4f} "
                    f"± {std(float(row['upslope_entry_abs_y']) for row in rows):.4f}"
                ),
                rough_vx=(
                    f"{mean(float(row['rough_entry_vx_body']) for row in rows):.5f} "
                    f"± {std(float(row['rough_entry_vx_body']) for row in rows):.5f}"
                ),
                rough_y=(
                    f"{mean(float(row['rough_entry_abs_y']) for row in rows):.4f} "
                    f"± {std(float(row['rough_entry_abs_y']) for row in rows):.4f}"
                ),
            )
        )

    lines.extend([
        "",
        "## Paired mean effects",
        "",
        "Effects are left minus right. Negative time, work, and lateral effects are preferable.",
        "",
        "| contrast | metric | mean effect | std | signs (+/-) |",
        "|---|---|---:|---:|---:|",
    ])

    report_metrics = [
        "goal_time_sim_s",
        "goal_work_abs_J_proxy",
        "work_abs_per_m_J_proxy",
        "max_abs_y_pre_goal",
        "rough_abs_dy",
        "goal_entry_abs_y",
        "realized_vx_body_mean",
        "max_abs_roll_deg",
        "max_abs_pitch_deg",
        "min_base_z",
        "upslope_entry_vx_body",
        "upslope_entry_abs_y",
        "rough_entry_vx_body",
        "rough_entry_abs_y",
    ]

    for row in paired_summary:
        if row["metric"] not in report_metrics:
            continue

        lines.append(
            "| {contrast} | {metric} | {effect} | {std_effect} | "
            "{positive}/{negative} |".format(
                contrast=row["contrast"],
                metric=row["metric"],
                effect=fmt(
                    float(
                        row["mean_effect"]
                    ),
                    5,
                ),
                std_effect=fmt(
                    float(
                        row["std_effect"]
                    ),
                    5,
                ),
                positive=row[
                    "positive_count"
                ],
                negative=row[
                    "negative_count"
                ],
            )
        )

    lines.extend([
        "",
        "## Preregistered decision",
        "",
        f"`{decision}`",
        "",
        interpretation,
        "",
        "| rule | category | value | requirement | result |",
        "|---|---|---:|---:|:---:|",
    ])

    for row in rules:
        lines.append(
            "| {name} | {category} | {value} | "
            "{operator} {threshold} | {result} |".format(
                name=row["name"],
                category=row["category"],
                value=fmt(
                    float(row["value"]),
                    6,
                ),
                operator=row[
                    "operator"
                ],
                threshold=row[
                    "threshold"
                ],
                result=(
                    "PASS"
                    if row["passed"]
                    else "FAIL"
                ),
            )
        )

    lines.extend([
        "",
        "## Scope guard",
        "",
        "- This is balanced descriptive screening with `n=3` per condition.",
        "- Passing supports only an oracle transition-aware candidate in world-v5.",
        "- It does not create a final Objective Selector label.",
        "- It does not establish cross-world generalization.",
        "- It does not establish battery-energy optimality.",
        "",
    ])

    summary_path = (
        analysis_dir
        / "summary_v0.md"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))

    print()
    print(
        f"decision={decision}"
    )

    failed = [
        row
        for row in rules
        if not row["passed"]
    ]

    print(
        f"failed_rule_count={len(failed)}"
    )

    for row in failed:
        print(
            "FAIL {name}: value={value:.8f} "
            "requirement={operator} {threshold}".format(
                **row
            )
        )

    print(
        f"analysis_dir={analysis_dir}"
    )

    print(
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
