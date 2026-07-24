#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev


def to_float(value):
    try:
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0
    return pstdev(values)


def quantile(values, q):
    values = sorted(v for v in values if v is not None)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    w = pos - lo
    return values[lo] * (1.0 - w) + values[hi] * w


def bootstrap_mean_ci(values, n_boot, rng):
    values = [v for v in values if v is not None]

    if not values:
        return None, None

    if len(values) == 1:
        return values[0], values[0]

    boots = []

    for _ in range(n_boot):
        sample = [
            values[rng.randrange(len(values))]
            for _ in range(len(values))
        ]
        boots.append(mean(sample))

    return (
        quantile(boots, 0.025),
        quantile(boots, 0.975),
    )


def derived_metrics(row):
    x_start = to_float(row.get("x_start"))
    x_end = to_float(row.get("x_end"))

    y_start = to_float(row.get("y_start"))
    y_end = to_float(row.get("y_end"))

    duration = to_float(row.get("duration_s"))

    progress = None
    progress_rate = None

    if x_start is not None and x_end is not None:
        progress = x_end - x_start

    if (
        progress is not None
        and duration is not None
        and duration > 1e-9
    ):
        progress_rate = progress / duration

    end_abs_y = (
        abs(y_end)
        if y_end is not None
        else None
    )

    lateral_growth = None

    if y_start is not None and y_end is not None:
        lateral_growth = abs(y_end) - abs(y_start)

    return {
        "progress_rate": progress_rate,
        "mean_abs_y_window": to_float(
            row.get("mean_abs_y_window")
        ),
        "max_abs_y_window": to_float(
            row.get("max_abs_y_window")
        ),
        "end_abs_y": end_abs_y,
        "lateral_growth": lateral_growth,
    }


def match_feature_value(row, feature):
    if feature == "abs_y_start":
        value = to_float(row.get("y_start"))
        return abs(value) if value is not None else None

    return to_float(row.get(feature))


def compute_match_stats(pool, features):
    stats = {}

    for feature in features:
        values = [
            match_feature_value(row, feature)
            for row in pool
        ]

        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            continue

        mu = mean(values)
        sigma = pstdev(values) if len(values) > 1 else 0.0

        stats[feature] = {
            "mean": mu,
            "std": sigma,
        }

    return stats


def match_distance(candidate, control, features, stats):
    terms = []

    for feature in features:
        c = match_feature_value(candidate, feature)
        r = match_feature_value(control, feature)

        if c is None or r is None:
            continue

        sigma = stats.get(
            feature,
            {},
        ).get("std", 0.0)

        # Constant features do not help nearest-neighbor matching.
        if sigma < 1e-9:
            continue

        terms.append(
            ((c - r) / sigma) ** 2
        )

    if not terms:
        return 0.0

    return math.sqrt(
        sum(terms) / len(terms)
    )


def nearest_controls(
    candidate,
    pool,
    features,
    k,
):
    stats = compute_match_stats(
        pool,
        features,
    )

    ranked = []

    for control in pool:
        distance = match_distance(
            candidate,
            control,
            features,
            stats,
        )

        ranked.append(
            (distance, control)
        )

    ranked.sort(
        key=lambda pair: pair[0]
    )

    return ranked[: min(k, len(ranked))]


def signed_effect(
    candidate_value,
    control_value,
    direction,
):
    if candidate_value is None or control_value is None:
        return None

    if direction == "higher_better":
        return candidate_value - control_value

    if direction == "lower_better":
        return control_value - candidate_value

    raise ValueError(
        f"Unknown metric direction: {direction}"
    )


def build_effect_row(
    candidate,
    controls,
    analysis_type,
    metric_directions,
):
    result = {
        "analysis_type": analysis_type,
        "source_rollout_id": candidate[
            "source_rollout_id"
        ],
        "candidate_mode": candidate[
            "candidate_mode"
        ],
        "evidence_role": candidate[
            "evidence_role"
        ],
        "context": candidate["context"],
        "action_id": candidate["action_id"],
        "match_k": len(controls),
        "matched_control_ids": "|".join(
            c["source_rollout_id"]
            for _, c in controls
        ),
        "match_distance_mean": avg(
            distance
            for distance, _ in controls
        ),
        "match_distance_max": max(
            (
                distance
                for distance, _ in controls
            ),
            default=0.0,
        ),
        "duration_s": candidate.get(
            "duration_s",
            "",
        ),
        "x_start": candidate.get(
            "x_start",
            "",
        ),
        "y_start": candidate.get(
            "y_start",
            "",
        ),
        "vx_scale_mean": candidate.get(
            "vx_scale_mean",
            "",
        ),
        "clearance_delta_mean": candidate.get(
            "clearance_delta_mean",
            "",
        ),
        "rollout_decision_label": candidate.get(
            "rollout_decision_label",
            "",
        ),
    }

    candidate_metrics = derived_metrics(
        candidate
    )

    control_rows = [
        row
        for _, row in controls
    ]

    control_metrics = [
        derived_metrics(row)
        for row in control_rows
    ]

    for metric, direction in metric_directions.items():
        cand_value = candidate_metrics.get(
            metric
        )

        ref_values = [
            metrics.get(metric)
            for metrics in control_metrics
        ]

        ref_values = [
            value
            for value in ref_values
            if value is not None
        ]

        ref_mean = (
            mean(ref_values)
            if ref_values
            else None
        )

        ref_std = std(ref_values)

        effect = signed_effect(
            cand_value,
            ref_mean,
            direction,
        )

        effect_z = None

        if (
            effect is not None
            and ref_std is not None
            and ref_std > 1e-9
        ):
            effect_z = effect / ref_std

        result[f"candidate_{metric}"] = (
            cand_value
            if cand_value is not None
            else ""
        )

        result[f"control_{metric}_mean"] = (
            ref_mean
            if ref_mean is not None
            else ""
        )

        result[f"control_{metric}_std"] = (
            ref_std
            if ref_std is not None
            else ""
        )

        result[f"effect_{metric}"] = (
            effect
            if effect is not None
            else ""
        )

        result[f"effect_z_{metric}"] = (
            effect_z
            if effect_z is not None
            else ""
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "datasets/phase_l/"
            "l2_window_action_evidence_from_k6_v0.csv"
        ),
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/phase_l/"
            "l3_matched_effect_estimation_config_v0.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "datasets/phase_l/"
            "l3_matched_action_effect_estimates_v0.csv"
        ),
    )

    parser.add_argument(
        "--group-output",
        default=(
            "datasets/phase_l/"
            "l3_matched_action_effect_group_summary_v0.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "reports/phase_l/"
            "l3_matched_action_effect_summary_v0.md"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    config_path = Path(args.config)
    output_path = Path(args.output)
    group_output_path = Path(
        args.group_output
    )
    report_path = Path(args.report)

    config = json.loads(
        config_path.read_text()
    )

    with input_path.open(newline="") as f:
        rows = list(
            csv.DictReader(f)
        )

    context = config["context"]
    match_k = int(config["match_k"])

    features = config[
        "match_features"
    ]

    metric_directions = config[
        "effect_metrics"
    ]

    baseline_role = config[
        "baseline_role"
    ]

    noop_role = config[
        "noop_role"
    ]

    intervention_roles = set(
        config["intervention_roles"]
    )

    flat_rows = [
        row for row in rows
        if row.get("context") == context
    ]

    baseline = [
        row for row in flat_rows
        if row.get("evidence_role")
        == baseline_role
    ]

    noops = [
        row for row in flat_rows
        if row.get("evidence_role")
        == noop_role
    ]

    interventions = [
        row for row in flat_rows
        if row.get("evidence_role")
        in intervention_roles
    ]

    if not baseline:
        raise RuntimeError(
            "No baseline flat windows found"
        )

    empirical_equivalent_pool = (
        baseline + noops
    )

    effect_rows = []

    # --------------------------------------------------------
    # Placebo analysis:
    # no-op command windows matched only against true baseline.
    # --------------------------------------------------------
    for candidate in noops:
        controls = nearest_controls(
            candidate,
            baseline,
            features,
            match_k,
        )

        effect_rows.append(
            build_effect_row(
                candidate,
                controls,
                "placebo_noop_vs_baseline",
                metric_directions,
            )
        )

    # --------------------------------------------------------
    # Intervention analysis:
    # actual intervention windows matched against empirical
    # baseline + command-equivalent no-op windows.
    # --------------------------------------------------------
    for candidate in interventions:
        controls = nearest_controls(
            candidate,
            empirical_equivalent_pool,
            features,
            match_k,
        )

        effect_rows.append(
            build_effect_row(
                candidate,
                controls,
                "intervention_vs_empirical_equivalent",
                metric_directions,
            )
        )

    # --------------------------------------------------------
    # Placebo calibration band.
    #
    # Positive signed effect always means improvement.
    # Use absolute placebo effects to define a conservative
    # natural/no-op variability envelope.
    # --------------------------------------------------------
    placebo_rows = [
        row for row in effect_rows
        if row["analysis_type"]
        == "placebo_noop_vs_baseline"
    ]

    placebo_q = float(
        config["placebo_quantile"]
    )

    placebo_thresholds = {}

    for metric in metric_directions:
        values = [
            abs(value)
            for value in (
                to_float(
                    row.get(
                        f"effect_{metric}"
                    )
                )
                for row in placebo_rows
            )
            if value is not None
        ]

        placebo_thresholds[metric] = (
            quantile(values, placebo_q)
            if values
            else None
        )

    # Add placebo-relative row status.
    for row in effect_rows:
        if (
            row["analysis_type"]
            == "placebo_noop_vs_baseline"
        ):
            row[
                "placebo_relative_status"
            ] = "placebo_control"
            continue

        metric_status = []

        for metric in metric_directions:
            effect = to_float(
                row.get(
                    f"effect_{metric}"
                )
            )

            threshold = placebo_thresholds[
                metric
            ]

            if effect is None or threshold is None:
                continue

            if effect > threshold:
                metric_status.append("positive")
            elif effect < -threshold:
                metric_status.append("negative")
            else:
                metric_status.append("within")

        positive = metric_status.count(
            "positive"
        )

        negative = metric_status.count(
            "negative"
        )

        if positive >= 3 and negative == 0:
            status = (
                "multi_metric_positive_beyond_placebo"
            )
        elif negative >= 3 and positive == 0:
            status = (
                "multi_metric_negative_beyond_placebo"
            )
        else:
            status = "mixed_or_within_placebo"

        row[
            "placebo_relative_status"
        ] = status

    # --------------------------------------------------------
    # Group summary.
    # --------------------------------------------------------
    grouped = defaultdict(list)

    for row in effect_rows:
        grouped[
            row["candidate_mode"]
        ].append(row)

    rng = random.Random(
        int(config["random_seed"])
    )

    n_boot = int(
        config["bootstrap_samples"]
    )

    group_rows = []

    for mode, mode_rows in sorted(
        grouped.items()
    ):
        out = {
            "candidate_mode": mode,
            "n": len(mode_rows),
            "analysis_type": (
                mode_rows[0][
                    "analysis_type"
                ]
            ),
            "evidence_role": (
                mode_rows[0][
                    "evidence_role"
                ]
            ),
        }

        for metric in metric_directions:
            values = [
                to_float(
                    row.get(
                        f"effect_{metric}"
                    )
                )
                for row in mode_rows
            ]

            values = [
                value
                for value in values
                if value is not None
            ]

            effect_mean = (
                mean(values)
                if values
                else None
            )

            ci_low, ci_high = (
                bootstrap_mean_ci(
                    values,
                    n_boot,
                    rng,
                )
            )

            threshold = placebo_thresholds[
                metric
            ]

            if (
                effect_mean is None
                or threshold is None
            ):
                placebo_status = "unavailable"
            elif effect_mean > threshold:
                placebo_status = (
                    "positive_beyond_placebo"
                )
            elif effect_mean < -threshold:
                placebo_status = (
                    "negative_beyond_placebo"
                )
            else:
                placebo_status = (
                    "within_placebo_band"
                )

            out[
                f"effect_{metric}_mean"
            ] = (
                effect_mean
                if effect_mean is not None
                else ""
            )

            out[
                f"effect_{metric}_ci_low"
            ] = (
                ci_low
                if ci_low is not None
                else ""
            )

            out[
                f"effect_{metric}_ci_high"
            ] = (
                ci_high
                if ci_high is not None
                else ""
            )

            out[
                f"placebo_abs_q95_{metric}"
            ] = (
                threshold
                if threshold is not None
                else ""
            )

            out[
                f"placebo_status_{metric}"
            ] = placebo_status

        group_rows.append(out)

    # --------------------------------------------------------
    # Write detailed output.
    # --------------------------------------------------------
    detailed_fields = [
        "analysis_type",
        "source_rollout_id",
        "candidate_mode",
        "evidence_role",
        "context",
        "action_id",
        "match_k",
        "matched_control_ids",
        "match_distance_mean",
        "match_distance_max",
        "duration_s",
        "x_start",
        "y_start",
        "vx_scale_mean",
        "clearance_delta_mean",
        "rollout_decision_label",
    ]

    for metric in metric_directions:
        detailed_fields.extend([
            f"candidate_{metric}",
            f"control_{metric}_mean",
            f"control_{metric}_std",
            f"effect_{metric}",
            f"effect_z_{metric}",
        ])

    detailed_fields.append(
        "placebo_relative_status"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=detailed_fields,
        )
        writer.writeheader()
        writer.writerows(effect_rows)

    # --------------------------------------------------------
    # Write group output.
    # --------------------------------------------------------
    if group_rows:
        group_fields = list(
            group_rows[0].keys()
        )

        with group_output_path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=group_fields,
            )
            writer.writeheader()
            writer.writerows(group_rows)

    # --------------------------------------------------------
    # Markdown summary.
    # --------------------------------------------------------
    lines = [
        "# TRACER Phase-L3 Matched Action Effect Summary v0",
        "",
        f"- source: `{input_path}`",
        f"- context: `{context}`",
        f"- baseline flat windows: {len(baseline)}",
        f"- no-op control windows: {len(noops)}",
        f"- intervention windows: {len(interventions)}",
        f"- detailed effect rows: {len(effect_rows)}",
        f"- match k: {match_k}",
        "",
        "## Interpretation rule",
        "",
        "- Positive signed effect means improvement.",
        "- Progress-rate effect: candidate minus matched control.",
        "- Lateral/stability effects: matched control minus candidate.",
        "- No-op windows are matched against baseline-only windows.",
        "- Intervention windows are matched against baseline + no-op empirical-equivalent controls.",
        "- The 95th percentile absolute no-op effect defines the placebo variability band.",
        "- This is matched observational evidence, not a causal-effect claim.",
        "",
        "## Placebo variability thresholds",
        "",
    ]

    for metric, threshold in placebo_thresholds.items():
        lines.append(
            f"- `{metric}`: "
            f"{threshold if threshold is not None else 'NA'}"
        )

    lines += [
        "",
        "## Group summary",
        "",
    ]

    for row in group_rows:
        lines.append(
            f"### {row['candidate_mode']}"
        )
        lines.append("")
        lines.append(
            f"- n: {row['n']}"
        )
        lines.append(
            f"- evidence role: `{row['evidence_role']}`"
        )

        for metric in metric_directions:
            lines.append(
                f"- `{metric}` effect mean: "
                f"{row[f'effect_{metric}_mean']} "
                f"| bootstrap 95% CI "
                f"[{row[f'effect_{metric}_ci_low']}, "
                f"{row[f'effect_{metric}_ci_high']}] "
                f"| `{row[f'placebo_status_{metric}']}`"
            )

        lines.append("")

    lines += [
        "## Safety / deployment",
        "",
        "- No candidate is promoted to active deployment in L3.",
        "- Empirical/default control remains the runtime policy.",
        "- L3 outputs are intended to guide later selector supervision and broader candidate collection.",
        "",
    ]

    report_path.write_text(
        "\n".join(lines)
    )

    print(
        "[L3] baseline flat =",
        len(baseline),
    )
    print(
        "[L3] no-op controls =",
        len(noops),
    )
    print(
        "[L3] interventions =",
        len(interventions),
    )
    print(
        "[L3] effect rows =",
        len(effect_rows),
    )
    print(
        "[L3] placebo thresholds =",
        placebo_thresholds,
    )
    print(
        "[L3] detailed output =",
        output_path,
    )
    print(
        "[L3] group output =",
        group_output_path,
    )
    print(
        "[L3] report =",
        report_path,
    )


if __name__ == "__main__":
    main()
