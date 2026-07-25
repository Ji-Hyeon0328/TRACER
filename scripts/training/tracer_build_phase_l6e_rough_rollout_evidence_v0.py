#!/usr/bin/env python3

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_EVIDENCE = Path(
    "datasets/phase_l/"
    "l6d_fresh_rough_clearance_validity_window_evidence_v1.csv"
)

DEFAULT_OUTCOME = Path(
    "datasets/phase_l/"
    "l6d_fresh_rough_clearance_validity_outcome_index_v0.csv"
)

DEFAULT_OUTPUT = Path(
    "datasets/phase_l/"
    "l6e_fresh_rough_clearance_rollout_evidence_v0.csv"
)

DEFAULT_REPORT = Path(
    "reports/phase_l/"
    "l6e_fresh_rough_clearance_rollout_evidence_v0.md"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )
    parser.add_argument(
        "--outcome",
        type=Path,
        default=DEFAULT_OUTCOME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--rough-completion-x",
        type=float,
        default=5.95,
    )

    return parser.parse_args()


def to_float(value, default=math.nan):
    if value in ("", None):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    number = to_float(value)

    if math.isnan(number):
        return default

    return int(round(number))


def to_bool(value):
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def fmt(value):
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    if isinstance(value, bool):
        return "1" if value else "0"

    return value


def weighted_mean(rows, value_field, weight_field):
    numerator = 0.0
    denominator = 0.0

    for row in rows:
        value = to_float(row.get(value_field))
        weight = to_float(row.get(weight_field), 0.0)

        if math.isnan(value) or weight <= 0.0:
            continue

        numerator += value * weight
        denominator += weight

    if denominator <= 0.0:
        return math.nan

    return numerator / denominator


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()

    evidence_rows = read_csv(args.evidence)
    outcome_rows = read_csv(args.outcome)

    rough_rows = [
        row
        for row in evidence_rows
        if row.get("context") == "rough"
    ]

    outcome_by_key = {}

    for row in outcome_rows:
        key = (
            row.get("mode", ""),
            to_int(row.get("idx")),
        )

        if key in outcome_by_key:
            raise RuntimeError(
                f"Duplicate outcome row: {key}"
            )

        outcome_by_key[key] = row

    grouped = defaultdict(list)

    for row in rough_rows:
        key = (
            row.get("candidate_mode", ""),
            to_int(row.get("trial_idx")),
        )

        grouped[key].append(row)

    output_rows = []

    for key, windows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    ):
        mode, repeat_idx = key

        if key not in outcome_by_key:
            raise RuntimeError(
                f"Missing outcome for {key}"
            )

        outcome = outcome_by_key[key]

        windows = sorted(
            windows,
            key=lambda row: (
                to_float(row.get("t_start")),
                to_float(row.get("t_end")),
            ),
        )

        first = windows[0]
        last = windows[-1]

        t_start = to_float(first.get("t_start"))
        t_end = max(
            to_float(row.get("t_end"))
            for row in windows
        )

        rough_elapsed_span_s = max(
            0.0,
            t_end - t_start,
        )

        rough_observed_duration_s = sum(
            max(
                0.0,
                to_float(
                    row.get("duration_s"),
                    0.0,
                ),
            )
            for row in windows
        )

        window_rows = sum(
            to_int(row.get("window_rows"))
            for row in windows
        )

        state_rows = sum(
            to_int(row.get("state_rows"))
            for row in windows
        )

        x_start = to_float(first.get("x_start"))
        y_start = to_float(first.get("y_start"))
        x_end = to_float(last.get("x_end"))
        y_end = to_float(last.get("y_end"))

        x_candidates = []

        for row in windows:
            for field in ("x_start", "x_end"):
                value = to_float(row.get(field))

                if not math.isnan(value):
                    x_candidates.append(value)

        rough_max_x_proxy = (
            max(x_candidates)
            if x_candidates
            else math.nan
        )

        rough_progress = x_end - x_start

        rough_progress_rate = (
            rough_progress / rough_elapsed_span_s
            if rough_elapsed_span_s > 0.0
            else math.nan
        )

        mean_abs_y = weighted_mean(
            windows,
            "mean_abs_y_window",
            "state_rows",
        )

        if math.isnan(mean_abs_y):
            mean_abs_y = weighted_mean(
                windows,
                "mean_abs_y_window",
                "window_rows",
            )

        max_values = [
            to_float(row.get("max_abs_y_window"))
            for row in windows
        ]
        max_values = [
            value
            for value in max_values
            if not math.isnan(value)
        ]

        max_abs_y = (
            max(max_values)
            if max_values
            else math.nan
        )

        start_abs_y = abs(y_start)
        end_abs_y = abs(y_end)
        lateral_growth = end_abs_y - start_abs_y

        rough_completed = (
            not math.isnan(rough_max_x_proxy)
            and rough_max_x_proxy
            >= args.rough_completion_x
        )

        goal_reached = to_bool(
            outcome.get("goal_reached")
        )

        if not rough_completed:
            outcome_class = "rough_incomplete"
        elif not goal_reached:
            outcome_class = "goal_failure_after_rough"
        else:
            outcome_class = "goal_success"

        span_coverage = (
            min(
                1.0,
                rough_observed_duration_s
                / rough_elapsed_span_s,
            )
            if rough_elapsed_span_s > 0.0
            else 1.0
        )

        window_types = sorted({
            row.get("window_type", "")
            for row in windows
            if row.get("window_type", "")
        })

        evidence_roles = sorted({
            row.get("evidence_role", "")
            for row in windows
            if row.get("evidence_role", "")
        })

        output_rows.append({
            "schema_version": "phase_l6e_v0",
            "row_granularity": "rough_rollout",
            "source_phase": first.get(
                "source_phase",
                "",
            ),
            "candidate_mode": mode,
            "trial_idx": repeat_idx,
            "source_rollout_id": first.get(
                "source_rollout_id",
                "",
            ),
            "rough_window_count": len(windows),
            "rough_window_types": ";".join(
                window_types
            ),
            "rough_evidence_roles": ";".join(
                evidence_roles
            ),
            "rough_t_start": t_start,
            "rough_t_end": t_end,
            "rough_elapsed_span_s":
                rough_elapsed_span_s,
            "rough_observed_duration_s":
                rough_observed_duration_s,
            "rough_span_coverage_ratio":
                span_coverage,
            "rough_window_rows": window_rows,
            "rough_state_rows": state_rows,
            "rough_x_start": x_start,
            "rough_y_start": y_start,
            "rough_x_end": x_end,
            "rough_y_end": y_end,
            "rough_max_x_proxy": rough_max_x_proxy,
            "rough_progress": rough_progress,
            "rough_progress_rate":
                rough_progress_rate,
            "rough_mean_abs_y": mean_abs_y,
            "rough_max_abs_y": max_abs_y,
            "rough_start_abs_y": start_abs_y,
            "rough_end_abs_y": end_abs_y,
            "rough_lateral_growth":
                lateral_growth,
            "rough_completed": rough_completed,
            "goal_reached": goal_reached,
            "outcome_class": outcome_class,
            "rollout_status": outcome.get(
                "status",
                "",
            ),
            "rollout_final_x": to_float(
                outcome.get("final_x")
            ),
            "rollout_final_y": to_float(
                outcome.get("final_y")
            ),
            "rollout_max_abs_y": to_float(
                outcome.get("max_abs_y")
            ),
            "rollout_mean_abs_y": to_float(
                outcome.get("mean_abs_y")
            ),
            "mean_emp_clearance": weighted_mean(
                windows,
                "emp_clearance_mean",
                "window_rows",
            ),
            "mean_out_clearance": weighted_mean(
                windows,
                "out_clearance_mean",
                "window_rows",
            ),
            "mean_realized_clearance_delta":
                weighted_mean(
                    windows,
                    "delta_clearance_mean",
                    "window_rows",
                ),
            "manifest": outcome.get(
                "manifest",
                "",
            ),
            "d5_log_dir": outcome.get(
                "d5_log_dir",
                "",
            ),
            "j7_csv": outcome.get(
                "j7_csv",
                "",
            ),
            "run_dir": outcome.get(
                "run_dir",
                "",
            ),
        })

    outcome_keys = set(outcome_by_key)
    evidence_keys = set(grouped)

    missing_evidence = sorted(
        outcome_keys - evidence_keys
    )

    if missing_evidence:
        raise RuntimeError(
            "Outcome rows without rough evidence: "
            f"{missing_evidence}"
        )

    expected_modes = {
        "baseline",
        "rough_noop",
        "rough_clear_low05",
        "rough_clear_high05",
    }

    expected_repeats = set(range(1, 7))

    observed_modes = {
        row["candidate_mode"]
        for row in output_rows
    }

    observed_repeats = {
        int(row["trial_idx"])
        for row in output_rows
    }

    if observed_modes != expected_modes:
        raise RuntimeError(
            f"Unexpected modes: {observed_modes}"
        )

    if observed_repeats != expected_repeats:
        raise RuntimeError(
            f"Unexpected repeats: {observed_repeats}"
        )

    per_key = Counter(
        (
            row["candidate_mode"],
            int(row["trial_idx"]),
        )
        for row in output_rows
    )

    bad_keys = [
        key
        for key, count in per_key.items()
        if count != 1
    ]

    if bad_keys:
        raise RuntimeError(
            f"Non-unique rollout rows: {bad_keys}"
        )

    if len(output_rows) != 24:
        raise RuntimeError(
            f"Expected 24 rows, got "
            f"{len(output_rows)}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(output_rows[0].keys())

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in output_rows:
            writer.writerow({
                key: fmt(value)
                for key, value in row.items()
            })

    mode_counts = Counter(
        row["candidate_mode"]
        for row in output_rows
    )

    completion_counts = Counter()
    goal_counts = Counter()
    outcome_counts = Counter()

    for row in output_rows:
        mode = row["candidate_mode"]

        completion_counts[mode] += int(
            bool(row["rough_completed"])
        )
        goal_counts[mode] += int(
            bool(row["goal_reached"])
        )
        outcome_counts[
            (
                mode,
                row["outcome_class"],
            )
        ] += 1

    fragmented = [
        row
        for row in output_rows
        if int(row["rough_window_count"]) > 1
    ]

    lines = [
        "# TRACER Phase-L6E Rough Rollout Evidence v0",
        "",
        f"- source evidence: `{args.evidence}`",
        f"- source outcome: `{args.outcome}`",
        f"- input rough windows: {len(rough_rows)}",
        f"- output rollout rows: {len(output_rows)}",
        (
            "- rough completion threshold: "
            f"x >= {args.rough_completion_x:.3f}"
        ),
        (
            "- continuous rough metrics preserve "
            "each rollout as one statistical unit."
        ),
        "",
        "## Mode counts",
        "",
    ]

    for mode in sorted(mode_counts):
        lines.append(
            f"- `{mode}`: {mode_counts[mode]}"
        )

    lines += [
        "",
        "## Reliability",
        "",
        "| mode | rough completed | goal reached |",
        "|---|---:|---:|",
    ]

    for mode in sorted(mode_counts):
        n = mode_counts[mode]

        lines.append(
            f"| {mode} | "
            f"{completion_counts[mode]}/{n} | "
            f"{goal_counts[mode]}/{n} |"
        )

    lines += [
        "",
        "## Outcome classes",
        "",
        "| mode | class | count |",
        "|---|---|---:|",
    ]

    for (
        mode,
        outcome_class,
    ), count in sorted(outcome_counts.items()):
        lines.append(
            f"| {mode} | "
            f"{outcome_class} | {count} |"
        )

    lines += [
        "",
        "## Fragmented source rollouts",
        "",
    ]

    if not fragmented:
        lines.append("- none")
    else:
        lines += [
            "| mode | repeat | windows | "
            "elapsed span | observed duration | "
            "coverage | outcome |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]

        for row in fragmented:
            lines.append(
                f"| {row['candidate_mode']} | "
                f"{row['trial_idx']} | "
                f"{row['rough_window_count']} | "
                f"{row['rough_elapsed_span_s']:.3f} | "
                f"{row['rough_observed_duration_s']:.3f} | "
                f"{row['rough_span_coverage_ratio']:.3f} | "
                f"{row['outcome_class']} |"
            )

    lines += [
        "",
        "## Per-rollout rough evidence",
        "",
        "| repeat | mode | windows | elapsed | "
        "progress | rate | mean |y| | max |y| | "
        "rough complete | goal | outcome |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---|",
    ]

    for row in output_rows:
        lines.append(
            f"| {row['trial_idx']} | "
            f"{row['candidate_mode']} | "
            f"{row['rough_window_count']} | "
            f"{row['rough_elapsed_span_s']:.3f} | "
            f"{row['rough_progress']:.6f} | "
            f"{row['rough_progress_rate']:.6f} | "
            f"{row['rough_mean_abs_y']:.6f} | "
            f"{row['rough_max_abs_y']:.6f} | "
            f"{int(row['rough_completed'])} | "
            f"{int(row['goal_reached'])} | "
            f"{row['outcome_class']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "- `rough_elapsed_span_s` includes gaps "
            "between fragmented evidence windows."
        ),
        (
            "- `rough_observed_duration_s` is the sum "
            "of observed window durations."
        ),
        (
            "- Reliability outcomes include all six "
            "rollouts per mode."
        ),
        (
            "- Continuous quality comparisons must "
            "state whether they are conditional on "
            "rough completion."
        ),
        (
            "- No candidate is promoted by this "
            "aggregation step."
        ),
        "",
    ]

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        "\n".join(lines)
    )

    print("[L6E] input rough windows =", len(rough_rows))
    print("[L6E] output rollout rows =", len(output_rows))
    print("[L6E] modes =", dict(mode_counts))
    print(
        "[L6E] fragmented rollouts =",
        [
            (
                row["candidate_mode"],
                row["trial_idx"],
                row["rough_window_count"],
            )
            for row in fragmented
        ],
    )
    print("[L6E] output =", args.output)
    print("[L6E] report =", args.report)


if __name__ == "__main__":
    main()
