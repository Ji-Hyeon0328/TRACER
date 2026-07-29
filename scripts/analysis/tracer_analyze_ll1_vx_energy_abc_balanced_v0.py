#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import statistics
import sys
import tarfile

from collections import defaultdict
from pathlib import Path


LABEL = {
    "A": "A (fixed_low)",
    "B": "B (terrain_scheduled)",
    "C": "C (fixed_high)",
}

ACTIVE = (
    "flat",
    "upslope",
    "rough",
    "downslope",
)


def f(value):
    try:
        value = float(value)

        return (
            value
            if math.isfinite(value)
            else None
        )

    except Exception:
        return None


def mean_std(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None, None

    return (
        statistics.mean(values),
        (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
    )


def median_or_none(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return statistics.median(values)


def fmt(value, digits=3):
    if (
        value is None
        or not math.isfinite(value)
    ):
        return "NA"

    return (
        f"{value:.{digits}f}"
    )


def sha(path):
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        for block in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_csv(path, rows):
    keys = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))

    rows = [
        row
        for row in rows
        if (
            row.get("valid") == "1"
            and f(row.get("sim_time"))
            is not None
        )
    ]

    if len(rows) < 500:
        raise RuntimeError(
            f"{path}: only {len(rows)} valid rows"
        )

    return rows


def integrate(rows, stop):
    total = defaultdict(float)

    segment = defaultdict(
        lambda: defaultdict(float)
    )

    fields = {
        "work_abs": "power_abs_W_cmd",
        "work_pos": "power_positive_W_cmd",
        "tau2": "tau_sq_sum",
        "dq2": "dq_sq_sum",
    }

    for index in range(1, stop + 1):
        previous = rows[index - 1]
        current = rows[index]

        previous_time = f(
            previous["sim_time"]
        )

        current_time = f(
            current["sim_time"]
        )

        dt = (
            current_time
            - previous_time
        )

        if not 0.0 < dt <= 0.20:
            continue

        previous_context = previous.get(
            "context",
            "unknown",
        )

        current_context = current.get(
            "context",
            "unknown",
        )

        total["duration"] += dt

        if previous_context == current_context:
            segment[
                previous_context
            ]["duration"] += dt
        else:
            segment[
                previous_context
            ]["duration"] += dt / 2.0

            segment[
                current_context
            ]["duration"] += dt / 2.0

        for output_name, source_name in (
            fields.items()
        ):
            previous_value = f(
                previous.get(source_name)
            )

            current_value = f(
                current.get(source_name)
            )

            if (
                previous_value is None
                or current_value is None
            ):
                continue

            integral = (
                (
                    previous_value
                    + current_value
                )
                * 0.5
                * dt
            )

            total[output_name] += integral

            if previous_context == current_context:
                segment[
                    previous_context
                ][output_name] += integral
            else:
                segment[
                    previous_context
                ][output_name] += (
                    integral / 2.0
                )

                segment[
                    current_context
                ][output_name] += (
                    integral / 2.0
                )

    return (
        dict(total),
        {
            key: dict(value)
            for key, value in segment.items()
        },
    )


def kinematic_metrics(rows):
    vx_body = [
        f(row.get("vx_body"))
        for row in rows
    ]

    vx_body = [
        value
        for value in vx_body
        if value is not None
    ]

    tracking_error = []

    for row in rows:
        realized = f(
            row.get("vx_body")
        )

        commanded = f(
            row.get("vx_cmd")
        )

        if (
            realized is not None
            and commanded is not None
        ):
            tracking_error.append(
                abs(realized - commanded)
            )

    roll = [
        abs(value)
        for value in (
            f(row.get("roll_deg"))
            for row in rows
        )
        if value is not None
    ]

    pitch = [
        abs(value)
        for value in (
            f(row.get("pitch_deg"))
            for row in rows
        )
        if value is not None
    ]

    z = [
        value
        for value in (
            f(row.get("z"))
            for row in rows
        )
        if value is not None
    ]

    return {
        "realized_vx_body_mean": (
            statistics.mean(vx_body)
            if vx_body
            else None
        ),
        "realized_vx_body_median": (
            statistics.median(vx_body)
            if vx_body
            else None
        ),
        "mean_abs_vx_tracking_error": (
            statistics.mean(tracking_error)
            if tracking_error
            else None
        ),
        "median_abs_vx_tracking_error": (
            statistics.median(
                tracking_error
            )
            if tracking_error
            else None
        ),
        "max_abs_roll_deg": (
            max(roll)
            if roll
            else None
        ),
        "max_abs_pitch_deg": (
            max(pitch)
            if pitch
            else None
        ),
        "min_base_z": (
            min(z)
            if z
            else None
        ),
    }


def analyze(manifest_row):
    path = (
        Path(manifest_row["log_dir"])
        / "ros1_joint_energy_proxy.csv"
    )

    rows = read_rows(path)

    goal_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("context")
            == "goal_flat"
        ),
        None,
    )

    if goal_index is None:
        raise RuntimeError(
            "run {}: no goal_flat".format(
                manifest_row["run_index"]
            )
        )

    pre_goal = rows[:goal_index + 1]

    start_time = f(
        pre_goal[0]["sim_time"]
    )

    goal_time = f(
        pre_goal[-1]["sim_time"]
    )

    x_values = [
        f(row.get("x"))
        for row in pre_goal
    ]

    y_values = [
        f(row.get("y"))
        for row in pre_goal
    ]

    x_values = [
        value
        for value in x_values
        if value is not None
    ]

    y_values = [
        value
        for value in y_values
        if value is not None
    ]

    total, energy_segments = integrate(
        rows,
        goal_index,
    )

    contexts = set(
        row.get(
            "context",
            "unknown",
        )
        for row in pre_goal
    )

    geometry = {}
    kinematics = {}

    for context in contexts:
        context_rows = [
            row
            for row in pre_goal
            if row.get(
                "context",
                "unknown",
            ) == context
        ]

        context_y = [
            f(row.get("y"))
            for row in context_rows
        ]

        context_y = [
            value
            for value in context_y
            if value is not None
        ]

        context_x = [
            f(row.get("x"))
            for row in context_rows
        ]

        context_x = [
            value
            for value in context_x
            if value is not None
        ]

        geometry[context] = {
            "abs_dy": (
                abs(
                    context_y[-1]
                    - context_y[0]
                )
                if len(context_y) > 1
                else None
            ),
            "max_abs_y": (
                max(
                    abs(value)
                    for value in context_y
                )
                if context_y
                else None
            ),
            "dx": (
                context_x[-1]
                - context_x[0]
                if len(context_x) > 1
                else None
            ),
        }

        kinematics[context] = (
            kinematic_metrics(
                context_rows
            )
        )

    segment_rows = []

    for context in sorted(
        set(energy_segments)
        | set(geometry)
        | set(kinematics)
    ):
        energy = energy_segments.get(
            context,
            {},
        )

        duration = energy.get(
            "duration"
        )

        segment_row = {
            "run_index": int(
                manifest_row["run_index"]
            ),
            "block": int(
                manifest_row["block"]
            ),
            "condition": (
                manifest_row["condition"]
            ),
            "context": context,
            "duration_s": duration,
            "abs_dy": geometry.get(
                context,
                {},
            ).get("abs_dy"),
            "max_abs_y": geometry.get(
                context,
                {},
            ).get("max_abs_y"),
            "work_abs_J_proxy": energy.get(
                "work_abs"
            ),
            "work_positive_J_proxy": (
                energy.get("work_pos")
            ),
            "tau_sq_integral": energy.get(
                "tau2"
            ),
            "dq_sq_integral": energy.get(
                "dq2"
            ),
            "mean_power_abs_W_proxy": (
                energy.get("work_abs")
                / duration
                if (
                    duration
                    and energy.get(
                        "work_abs"
                    ) is not None
                )
                else None
            ),
        }

        segment_row.update(
            kinematics.get(
                context,
                {},
            )
        )

        segment_rows.append(
            segment_row
        )

    by_context = {
        row["context"]: row
        for row in segment_rows
    }

    rough = by_context.get(
        "rough",
        {},
    )

    nonrough_work = sum(
        (
            by_context.get(
                context,
                {},
            ).get(
                "work_abs_J_proxy"
            )
            or 0.0
        )
        for context in (
            "flat",
            "upslope",
            "downslope",
        )
    )

    distance = (
        x_values[-1]
        - x_values[0]
    )

    run_kinematics = (
        kinematic_metrics(
            pre_goal
        )
    )

    run = {
        "run_index": int(
            manifest_row["run_index"]
        ),
        "block": int(
            manifest_row["block"]
        ),
        "condition": (
            manifest_row["condition"]
        ),
        "condition_name": (
            manifest_row[
                "condition_name"
            ]
        ),
        "rc": int(
            manifest_row["rc"]
        ),
        "log_dir": (
            manifest_row["log_dir"]
        ),
        "goal_time_sim_s": (
            goal_time - start_time
        ),
        "goal_entry_x": x_values[-1],
        "goal_entry_y": y_values[-1],
        "goal_entry_abs_y": abs(
            y_values[-1]
        ),
        "max_abs_y_pre_goal": max(
            abs(value)
            for value in y_values
        ),
        "rough_abs_dy": rough.get(
            "abs_dy"
        ),
        "goal_work_abs_J_proxy": (
            total.get("work_abs")
        ),
        "goal_work_positive_J_proxy": (
            total.get("work_pos")
        ),
        "goal_tau_sq_integral": (
            total.get("tau2")
        ),
        "goal_dq_sq_integral": (
            total.get("dq2")
        ),
        "work_abs_per_m_J_proxy": (
            total.get("work_abs")
            / distance
            if distance > 1e-9
            else None
        ),
        "rough_work_abs_J_proxy": (
            rough.get(
                "work_abs_J_proxy"
            )
        ),
        "nonrough_work_abs_J_proxy": (
            nonrough_work
        ),
    }

    run.update(run_kinematics)

    return run, segment_rows


RUN_METRICS = (
    "goal_time_sim_s",
    "goal_entry_abs_y",
    "max_abs_y_pre_goal",
    "rough_abs_dy",
    "goal_work_abs_J_proxy",
    "goal_work_positive_J_proxy",
    "goal_tau_sq_integral",
    "goal_dq_sq_integral",
    "work_abs_per_m_J_proxy",
    "rough_work_abs_J_proxy",
    "nonrough_work_abs_J_proxy",
    "realized_vx_body_mean",
    "realized_vx_body_median",
    "mean_abs_vx_tracking_error",
    "median_abs_vx_tracking_error",
    "max_abs_roll_deg",
    "max_abs_pitch_deg",
    "min_base_z",
)


def summarize(runs):
    output = []

    for condition in "ABC":
        condition_runs = [
            run
            for run in runs
            if run["condition"]
            == condition
        ]

        row = {
            "condition": condition,
            "condition_label": (
                LABEL[condition]
            ),
            "n": len(condition_runs),
        }

        for metric in RUN_METRICS:
            mean, std = mean_std([
                f(run.get(metric))
                for run in condition_runs
            ])

            row[
                metric + "_mean"
            ] = mean

            row[
                metric + "_std"
            ] = std

        output.append(row)

    return output


def paired(runs):
    blocks = defaultdict(dict)

    for run in runs:
        blocks[
            run["block"]
        ][
            run["condition"]
        ] = run

    output = []

    contrasts = (
        (
            "B_minus_A",
            "B",
            "A",
        ),
        (
            "C_minus_A",
            "C",
            "A",
        ),
        (
            "B_minus_C",
            "B",
            "C",
        ),
    )

    for (
        contrast,
        left,
        right,
    ) in contrasts:
        for metric in RUN_METRICS:
            effects = []
            relative = []

            for block in sorted(blocks):
                if (
                    left not in blocks[block]
                    or right
                    not in blocks[block]
                ):
                    continue

                left_value = f(
                    blocks[block][left].get(
                        metric
                    )
                )

                right_value = f(
                    blocks[block][right].get(
                        metric
                    )
                )

                if (
                    left_value is None
                    or right_value is None
                ):
                    continue

                effects.append(
                    left_value
                    - right_value
                )

                if abs(right_value) > 1e-12:
                    relative.append(
                        100.0
                        * (
                            left_value
                            - right_value
                        )
                        / right_value
                    )

            mean, std = mean_std(
                effects
            )

            relative_mean, relative_std = (
                mean_std(relative)
            )

            output.append({
                "contrast": contrast,
                "metric": metric,
                "n_blocks": len(effects),
                "mean_effect": mean,
                "std_effect": std,
                "mean_relative_effect_pct": (
                    relative_mean
                ),
                "std_relative_effect_pct": (
                    relative_std
                ),
                "positive_count": sum(
                    value > 0
                    for value in effects
                ),
                "negative_count": sum(
                    value < 0
                    for value in effects
                ),
            })

    return output


def segment_summary(segments):
    output = []

    segment_metrics = (
        "duration_s",
        "abs_dy",
        "max_abs_y",
        "work_abs_J_proxy",
        "work_positive_J_proxy",
        "tau_sq_integral",
        "dq_sq_integral",
        "mean_power_abs_W_proxy",
        "realized_vx_body_mean",
        "realized_vx_body_median",
        "mean_abs_vx_tracking_error",
        "max_abs_roll_deg",
        "max_abs_pitch_deg",
        "min_base_z",
    )

    for condition in "ABC":
        for context in ACTIVE:
            rows = [
                row
                for row in segments
                if (
                    row["condition"]
                    == condition
                    and row["context"]
                    == context
                )
            ]

            summary = {
                "condition": condition,
                "condition_label": (
                    LABEL[condition]
                ),
                "context": context,
                "n": len(rows),
            }

            for metric in segment_metrics:
                mean, std = mean_std([
                    f(row.get(metric))
                    for row in rows
                ])

                summary[
                    metric + "_mean"
                ] = mean

                summary[
                    metric + "_std"
                ] = std

            output.append(summary)

    return output


def markdown(
    runs,
    condition_summary,
    paired_effects,
    segment_condition_summary,
):
    lines = [
        (
            "# TRACER LL1 VX Energy "
            "A/B/C Balanced Analysis"
        ),
        "",
        "## Collection validity",
        "",
        "- runs: {}".format(
            len(runs)
        ),
        (
            "- balanced conditions: "
            "A/B/C = 3/3/3"
        ),
        (
            "- all runs rc=0: {}/{}".format(
                sum(
                    run["rc"] == 0
                    for run in runs
                ),
                len(runs),
            )
        ),
        (
            "- analysis window ends at "
            "first `goal_flat` entry; "
            "post-goal hold is excluded."
        ),
        (
            "- work values are commanded "
            "mechanical-work/effort proxies, "
            "not battery electrical energy."
        ),
        "",
        "## Per-run mission outcome",
        "",
        (
            "| run | block | condition | "
            "goal time | goal |y| | max |y| | "
            "rough |Δy| | work |τq̇| | work/m | "
            "mean body vx | vx MAE | max roll | "
            "max pitch | min z |"
        ),
        (
            "|---:|---:|---|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|"
            "---:|---:|"
        ),
    ]

    for run in sorted(
        runs,
        key=lambda value: value["run_index"],
    ):
        lines.append(
            (
                "| {run_index} | {block} | "
                "{label} | {goal_time} | "
                "{goal_y} | {max_y} | "
                "{rough_dy} | {work} | "
                "{work_m} | {vx} | {vx_error} | "
                "{roll} | {pitch} | {z} |"
            ).format(
                run_index=run["run_index"],
                block=run["block"],
                label=LABEL[
                    run["condition"]
                ],
                goal_time=fmt(
                    run["goal_time_sim_s"]
                ),
                goal_y=fmt(
                    run["goal_entry_abs_y"]
                ),
                max_y=fmt(
                    run["max_abs_y_pre_goal"]
                ),
                rough_dy=fmt(
                    run["rough_abs_dy"]
                ),
                work=fmt(
                    run[
                        "goal_work_abs_J_proxy"
                    ],
                    1,
                ),
                work_m=fmt(
                    run[
                        "work_abs_per_m_J_proxy"
                    ],
                    1,
                ),
                vx=fmt(
                    run[
                        "realized_vx_body_mean"
                    ],
                    4,
                ),
                vx_error=fmt(
                    run[
                        "mean_abs_vx_tracking_error"
                    ],
                    4,
                ),
                roll=fmt(
                    run["max_abs_roll_deg"]
                ),
                pitch=fmt(
                    run["max_abs_pitch_deg"]
                ),
                z=fmt(
                    run["min_base_z"]
                ),
            )
        )

    lines.extend([
        "",
        "## Condition summary",
        "",
        (
            "| condition | goal time | goal |y| | "
            "max |y| | work |τq̇| | work/m | "
            "mean body vx | vx MAE | max roll | "
            "max pitch | min z |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|"
        ),
    ])

    for row in condition_summary:
        lines.append(
            (
                "| {label} | {time} | {goal_y} | "
                "{max_y} | {work} | {work_m} | "
                "{vx} | {error} | {roll} | "
                "{pitch} | {z} |"
            ).format(
                label=row[
                    "condition_label"
                ],
                time=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "goal_time_sim_s_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "goal_time_sim_s_std"
                            ]
                        ),
                    )
                ),
                goal_y=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "goal_entry_abs_y_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "goal_entry_abs_y_std"
                            ]
                        ),
                    )
                ),
                max_y=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "max_abs_y_pre_goal_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "max_abs_y_pre_goal_std"
                            ]
                        ),
                    )
                ),
                work=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "goal_work_abs_J_proxy_mean"
                            ],
                            1,
                        ),
                        fmt(
                            row[
                                "goal_work_abs_J_proxy_std"
                            ],
                            1,
                        ),
                    )
                ),
                work_m=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "work_abs_per_m_J_proxy_mean"
                            ],
                            1,
                        ),
                        fmt(
                            row[
                                "work_abs_per_m_J_proxy_std"
                            ],
                            1,
                        ),
                    )
                ),
                vx=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "realized_vx_body_mean_mean"
                            ],
                            4,
                        ),
                        fmt(
                            row[
                                "realized_vx_body_mean_std"
                            ],
                            4,
                        ),
                    )
                ),
                error=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "mean_abs_vx_tracking_error_mean"
                            ],
                            4,
                        ),
                        fmt(
                            row[
                                "mean_abs_vx_tracking_error_std"
                            ],
                            4,
                        ),
                    )
                ),
                roll=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "max_abs_roll_deg_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "max_abs_roll_deg_std"
                            ]
                        ),
                    )
                ),
                pitch=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "max_abs_pitch_deg_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "max_abs_pitch_deg_std"
                            ]
                        ),
                    )
                ),
                z=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "min_base_z_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "min_base_z_std"
                            ]
                        ),
                    )
                ),
            )
        )

    lines.extend([
        "",
        "## Segment VX/work summary",
        "",
        (
            "| condition | context | duration | "
            "mean body vx | vx MAE | work |τq̇| | "
            "mean power | max roll | max pitch | "
            "min z |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|"
            "---:|---:|---:|"
        ),
    ])

    for row in segment_condition_summary:
        lines.append(
            (
                "| {label} | {context} | "
                "{duration} | {vx} | {error} | "
                "{work} | {power} | {roll} | "
                "{pitch} | {z} |"
            ).format(
                label=row[
                    "condition_label"
                ],
                context=row["context"],
                duration=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "duration_s_mean"
                            ]
                        ),
                        fmt(
                            row[
                                "duration_s_std"
                            ]
                        ),
                    )
                ),
                vx=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "realized_vx_body_mean_mean"
                            ],
                            4,
                        ),
                        fmt(
                            row[
                                "realized_vx_body_mean_std"
                            ],
                            4,
                        ),
                    )
                ),
                error=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "mean_abs_vx_tracking_error_mean"
                            ],
                            4,
                        ),
                        fmt(
                            row[
                                "mean_abs_vx_tracking_error_std"
                            ],
                            4,
                        ),
                    )
                ),
                work=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "work_abs_J_proxy_mean"
                            ],
                            1,
                        ),
                        fmt(
                            row[
                                "work_abs_J_proxy_std"
                            ],
                            1,
                        ),
                    )
                ),
                power=(
                    "{} ± {}".format(
                        fmt(
                            row[
                                "mean_power_abs_W_proxy_mean"
                            ],
                            2,
                        ),
                        fmt(
                            row[
                                "mean_power_abs_W_proxy_std"
                            ],
                            2,
                        ),
                    )
                ),
                roll=fmt(
                    row[
                        "max_abs_roll_deg_mean"
                    ]
                ),
                pitch=fmt(
                    row[
                        "max_abs_pitch_deg_mean"
                    ]
                ),
                z=fmt(
                    row[
                        "min_base_z_mean"
                    ]
                ),
            )
        )

    lines.extend([
        "",
        "## Paired block effects",
        "",
        (
            "Effects are left minus right. "
            "For time, lateral deviation, work, "
            "tracking error, roll, and pitch, "
            "negative is preferable. For min z, "
            "larger is generally preferable."
        ),
        "",
        (
            "| contrast | metric | mean effect | "
            "std | relative effect | signs (+/-) |"
        ),
        (
            "|---|---|---:|---:|---:|---:|"
        ),
    ])

    for row in paired_effects:
        lines.append(
            (
                "| {contrast} | {metric} | "
                "{effect} | {std} | {relative}% | "
                "{positive}/{negative} |"
            ).format(
                contrast=row["contrast"],
                metric=row["metric"],
                effect=fmt(
                    row["mean_effect"],
                    4,
                ),
                std=fmt(
                    row["std_effect"],
                    4,
                ),
                relative=fmt(
                    row[
                        "mean_relative_effect_pct"
                    ],
                    2,
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
        "## Interpretation guard",
        "",
        (
            "- n=3 per condition is balanced "
            "descriptive screening evidence, "
            "not final statistical confirmation."
        ),
        (
            "- A higher realized body velocity is "
            "not automatically preferable; mission "
            "time, lateral behavior, stability, and "
            "work must be considered together."
        ),
        (
            "- B supports terrain-specific scheduling "
            "only if it gains useful time relative to "
            "A while avoiding relevant C penalties."
        ),
        (
            "- Similar B and C outcomes would support "
            "a generic high-speed effect rather than "
            "terrain-specific selector value."
        ),
        (
            "- Large block-to-block sign reversals "
            "require additional paired repeats before "
            "action-bank promotion."
        ),
        "",
    ])

    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: {} RUN_ROOT".format(
                Path(sys.argv[0]).name
            )
        )

    root = (
        Path(sys.argv[1])
        .expanduser()
        .resolve()
    )

    manifest = root / "manifest.tsv"

    with open(
        manifest,
        newline="",
        encoding="utf-8",
    ) as stream:
        manifest_rows = list(
            csv.DictReader(
                stream,
                delimiter="\t",
            )
        )

    if len(manifest_rows) != 9:
        raise SystemExit(
            "expected 9 manifest rows, found {}".format(
                len(manifest_rows)
            )
        )

    counts = {
        condition: sum(
            row["condition"] == condition
            for row in manifest_rows
        )
        for condition in "ABC"
    }

    if counts != {
        "A": 3,
        "B": 3,
        "C": 3,
    }:
        raise SystemExit(
            "unbalanced manifest: {}".format(
                counts
            )
        )

    runs = []
    segments = []

    for manifest_row in sorted(
        manifest_rows,
        key=lambda row: int(
            row["run_index"]
        ),
    ):
        run, run_segments = analyze(
            manifest_row
        )

        runs.append(run)
        segments.extend(run_segments)

    condition_summary = summarize(runs)
    paired_effects = paired(runs)

    segment_condition_summary = (
        segment_summary(segments)
    )

    output = (
        root
        / "analysis_vx_energy_v0"
    )

    output.mkdir(exist_ok=True)

    write_csv(
        output
        / "run_metrics_vx_energy_v0.csv",
        runs,
    )

    write_csv(
        output
        / "segment_metrics_vx_energy_v0.csv",
        segments,
    )

    write_csv(
        output
        / "condition_summary_vx_energy_v0.csv",
        condition_summary,
    )

    write_csv(
        output
        / "segment_condition_summary_vx_energy_v0.csv",
        segment_condition_summary,
    )

    write_csv(
        output
        / "paired_effects_vx_energy_v0.csv",
        paired_effects,
    )

    payload = {
        "run_root": str(root),
        "run_metrics": runs,
        "segment_metrics": segments,
        "condition_summary": condition_summary,
        "segment_condition_summary": (
            segment_condition_summary
        ),
        "paired_effects": paired_effects,
    }

    (
        output
        / "analysis_vx_energy_v0.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = markdown(
        runs,
        condition_summary,
        paired_effects,
        segment_condition_summary,
    )

    (
        output
        / "summary_vx_energy_v0.md"
    ).write_text(
        summary + "\n",
        encoding="utf-8",
    )

    print(summary)

    files = [
        path
        for path in output.iterdir()
        if path.is_file()
    ]

    (
        output
        / "SHA256SUMS"
    ).write_text(
        "".join(
            "{}  {}\n".format(
                sha(path),
                path.name,
            )
            for path in sorted(files)
        ),
        encoding="utf-8",
    )

    archive = (
        root.parent
        / (
            root.name
            + "_analysis_vx_energy_v0.tar.gz"
        )
    )

    with tarfile.open(
        archive,
        "w:gz",
    ) as tar:
        tar.add(
            output,
            arcname=output.name,
        )

        tar.add(
            manifest,
            arcname="manifest.tsv",
        )

    print(
        "\nanalysis_dir={}\narchive={}"
        "\narchive_sha256={}".format(
            output,
            archive,
            sha(archive),
        )
    )


if __name__ == "__main__":
    main()
