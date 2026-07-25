#!/usr/bin/env python3

import argparse
import csv
import math
import random
import zlib
from pathlib import Path


DEFAULT_INPUT = Path(
    "datasets/phase_l/"
    "l6e_fresh_rough_clearance_rollout_evidence_v0.csv"
)

DEFAULT_OUTPUT = Path(
    "datasets/phase_l/"
    "l6f_rough_clearance_final_effect_summary_v0.csv"
)

DEFAULT_REPORT = Path(
    "reports/phase_l/"
    "l6f_rough_clearance_final_effect_summary_v0.md"
)


COMPARISONS = [
    (
        "routing_placebo",
        "rough_noop",
        "baseline",
    ),
    (
        "candidate_vs_noop",
        "rough_clear_low05",
        "rough_noop",
    ),
    (
        "candidate_vs_baseline",
        "rough_clear_low05",
        "baseline",
    ),
    (
        "candidate_vs_noop",
        "rough_clear_high05",
        "rough_noop",
    ),
    (
        "candidate_vs_baseline",
        "rough_clear_high05",
        "baseline",
    ),
]


# Positive signed effect always means candidate improvement.
QUALITY_METRICS = [
    (
        "rough_elapsed_span_s",
        False,
        "rough traversal elapsed time",
    ),
    (
        "rough_progress_rate",
        True,
        "rough progress rate",
    ),
    (
        "rough_mean_abs_y",
        False,
        "rough mean absolute lateral error",
    ),
    (
        "rough_max_abs_y",
        False,
        "rough maximum absolute lateral error",
    ),
    (
        "rough_end_abs_y",
        False,
        "rough end absolute lateral error",
    ),
    (
        "rough_lateral_growth",
        False,
        "rough lateral growth",
    ),
]


RELIABILITY_METRICS = [
    (
        "rough_completed",
        "rough completion",
    ),
    (
        "goal_reached",
        "full-course goal success",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
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
        "--bootstrap-samples",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=606,
    )

    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    return float(value)


def to_int(value):
    return int(round(float(value)))


def percentile(sorted_values, probability):
    if not sorted_values:
        return math.nan

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = probability * (
        len(sorted_values) - 1
    )

    low = int(math.floor(position))
    high = int(math.ceil(position))

    if low == high:
        return sorted_values[low]

    fraction = position - low

    return (
        sorted_values[low] * (1.0 - fraction)
        + sorted_values[high] * fraction
    )


def stable_seed(base_seed, *parts):
    key = "::".join(str(part) for part in parts)

    return (
        base_seed
        + zlib.crc32(key.encode("utf-8"))
    ) & 0xFFFFFFFF


def bootstrap_mean_ci(
    values,
    samples,
    seed,
):
    if not values:
        return math.nan, math.nan

    rng = random.Random(seed)
    n = len(values)

    means = []

    for _ in range(samples):
        sample_mean = sum(
            values[rng.randrange(n)]
            for _ in range(n)
        ) / n

        means.append(sample_mean)

    means.sort()

    return (
        percentile(means, 0.025),
        percentile(means, 0.975),
    )


def exact_two_sided_sign_p(wins, losses):
    n = wins + losses

    if n == 0:
        return 1.0

    tail = min(wins, losses)

    probability = sum(
        math.comb(n, k)
        for k in range(tail + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * probability)


def same_sign_requirement(n):
    # Preserve the earlier Phase-L rule:
    # at least a 5/6 same-sign proportion.
    return int(math.ceil((5.0 / 6.0) * n))


def quality_effect_class(
    ci_low,
    ci_high,
    positive,
    negative,
    n,
):
    required = same_sign_requirement(n)

    if (
        ci_low > 0.0
        and positive >= required
    ):
        return "repeatable_positive"

    if (
        ci_high < 0.0
        and negative >= required
    ):
        return "repeatable_negative"

    if ci_low > 0.0:
        return "positive_ci_sign_shortfall"

    if ci_high < 0.0:
        return "negative_ci_sign_shortfall"

    return "unresolved"


def reliability_direction(
    wins,
    losses,
):
    if wins > losses:
        return "candidate_directional_edge"

    if losses > wins:
        return "control_directional_edge"

    return "no_directional_edge"


def comparison_conclusion(metric_classes):
    elapsed = metric_classes.get(
        "rough_elapsed_span_s",
        "unresolved",
    )

    rate = metric_classes.get(
        "rough_progress_rate",
        "unresolved",
    )

    stability_fields = [
        "rough_mean_abs_y",
        "rough_max_abs_y",
        "rough_end_abs_y",
        "rough_lateral_growth",
    ]

    stability_positive = sum(
        metric_classes.get(field)
        == "repeatable_positive"
        for field in stability_fields
    )

    stability_negative = sum(
        metric_classes.get(field)
        == "repeatable_negative"
        for field in stability_fields
    )

    stability_negative_ci = sum(
        metric_classes.get(field)
        in {
            "repeatable_negative",
            "negative_ci_sign_shortfall",
        }
        for field in stability_fields
    )

    repeatable_progress_cost = (
        elapsed == "repeatable_negative"
        and rate == "repeatable_negative"
    )

    repeatable_progress_gain = (
        elapsed == "repeatable_positive"
        and rate == "repeatable_positive"
    )

    if (
        repeatable_progress_gain
        and stability_positive >= 2
    ):
        return "repeatable_quality_gain"

    if (
        not repeatable_progress_cost
        and stability_positive >= 2
    ):
        return (
            "stability_gain_without_"
            "repeatable_progress_cost"
        )

    if (
        repeatable_progress_cost
        and stability_positive >= 2
    ):
        return (
            "stability_gain_with_"
            "repeatable_progress_cost"
        )

    if (
        repeatable_progress_cost
        and stability_negative_ci >= 2
    ):
        return (
            "progress_cost_with_"
            "lateral_degradation_signal"
        )

    if repeatable_progress_cost:
        return "repeatable_progress_cost"

    if stability_negative >= 2:
        return "repeatable_stability_degradation"

    return "unresolved"


def main():
    args = parse_args()
    rows = read_csv(args.input)

    index = {}

    for row in rows:
        key = (
            row["candidate_mode"],
            to_int(row["trial_idx"]),
        )

        if key in index:
            raise RuntimeError(
                f"Duplicate rollout row: {key}"
            )

        index[key] = row

    expected_modes = {
        "baseline",
        "rough_noop",
        "rough_clear_low05",
        "rough_clear_high05",
    }

    expected_repeats = set(range(1, 7))

    observed_modes = {
        mode
        for mode, _ in index
    }

    observed_repeats = {
        repeat
        for _, repeat in index
    }

    if observed_modes != expected_modes:
        raise RuntimeError(
            f"Unexpected modes: {observed_modes}"
        )

    if observed_repeats != expected_repeats:
        raise RuntimeError(
            f"Unexpected repeats: "
            f"{observed_repeats}"
        )

    output_rows = []
    report_sections = []

    for comparison_type, candidate, control in COMPARISONS:
        all_repeats = sorted(expected_repeats)

        reliability_results = []

        for field, label in RELIABILITY_METRICS:
            candidate_values = [
                to_int(index[(candidate, repeat)][field])
                for repeat in all_repeats
            ]

            control_values = [
                to_int(index[(control, repeat)][field])
                for repeat in all_repeats
            ]

            wins = sum(
                candidate_value > control_value
                for candidate_value, control_value
                in zip(
                    candidate_values,
                    control_values,
                )
            )

            losses = sum(
                candidate_value < control_value
                for candidate_value, control_value
                in zip(
                    candidate_values,
                    control_values,
                )
            )

            ties = len(all_repeats) - wins - losses

            candidate_success = sum(candidate_values)
            control_success = sum(control_values)

            sign_p = exact_two_sided_sign_p(
                wins,
                losses,
            )

            direction = reliability_direction(
                wins,
                losses,
            )

            reliability_result = {
                "section": "reliability",
                "comparison_type": comparison_type,
                "candidate": candidate,
                "control": control,
                "analysis_scope": "all_six_repeats",
                "metric": field,
                "metric_label": label,
                "included_repeats": ";".join(
                    str(value)
                    for value in all_repeats
                ),
                "n_pairs": len(all_repeats),
                "mean_effect": (
                    candidate_success
                    - control_success
                ) / len(all_repeats),
                "ci_low": "",
                "ci_high": "",
                "positive_pairs": wins,
                "negative_pairs": losses,
                "ties": ties,
                "same_sign_required": "",
                "effect_class": direction,
                "sign_p": sign_p,
                "candidate_success": candidate_success,
                "control_success": control_success,
                "comparison_conclusion": "",
            }

            output_rows.append(
                reliability_result
            )
            reliability_results.append(
                reliability_result
            )

        completed_repeats = [
            repeat
            for repeat in all_repeats
            if (
                to_int(
                    index[
                        (candidate, repeat)
                    ]["rough_completed"]
                )
                == 1
                and to_int(
                    index[
                        (control, repeat)
                    ]["rough_completed"]
                )
                == 1
            )
        ]

        if not completed_repeats:
            raise RuntimeError(
                f"No jointly completed pairs for "
                f"{candidate} vs {control}"
            )

        quality_results = []
        metric_classes = {}

        for field, higher_better, label in QUALITY_METRICS:
            effects = []

            for repeat in completed_repeats:
                candidate_value = to_float(
                    index[(candidate, repeat)][field]
                )

                control_value = to_float(
                    index[(control, repeat)][field]
                )

                if higher_better:
                    effect = (
                        candidate_value
                        - control_value
                    )
                else:
                    effect = (
                        control_value
                        - candidate_value
                    )

                effects.append(effect)

            mean_effect = sum(effects) / len(effects)

            ci_low, ci_high = bootstrap_mean_ci(
                effects,
                args.bootstrap_samples,
                stable_seed(
                    args.seed,
                    comparison_type,
                    candidate,
                    control,
                    field,
                ),
            )

            positive = sum(
                value > 0.0
                for value in effects
            )

            negative = sum(
                value < 0.0
                for value in effects
            )

            ties = (
                len(effects)
                - positive
                - negative
            )

            effect_class = quality_effect_class(
                ci_low,
                ci_high,
                positive,
                negative,
                len(effects),
            )

            sign_p = exact_two_sided_sign_p(
                positive,
                negative,
            )

            metric_classes[field] = effect_class

            quality_result = {
                "section": "conditional_quality",
                "comparison_type": comparison_type,
                "candidate": candidate,
                "control": control,
                "analysis_scope": (
                    "both_rough_completed"
                ),
                "metric": field,
                "metric_label": label,
                "included_repeats": ";".join(
                    str(value)
                    for value in completed_repeats
                ),
                "n_pairs": len(effects),
                "mean_effect": mean_effect,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "positive_pairs": positive,
                "negative_pairs": negative,
                "ties": ties,
                "same_sign_required":
                    same_sign_requirement(
                        len(effects)
                    ),
                "effect_class": effect_class,
                "sign_p": sign_p,
                "candidate_success": "",
                "control_success": "",
                "comparison_conclusion": "",
            }

            output_rows.append(quality_result)
            quality_results.append(quality_result)

        conclusion = comparison_conclusion(
            metric_classes
        )

        for row in quality_results:
            row["comparison_conclusion"] = conclusion

        report_sections.append({
            "comparison_type": comparison_type,
            "candidate": candidate,
            "control": control,
            "reliability": reliability_results,
            "quality": quality_results,
            "completed_repeats": completed_repeats,
            "conclusion": conclusion,
        })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "section",
        "comparison_type",
        "candidate",
        "control",
        "analysis_scope",
        "metric",
        "metric_label",
        "included_repeats",
        "n_pairs",
        "mean_effect",
        "ci_low",
        "ci_high",
        "positive_pairs",
        "negative_pairs",
        "ties",
        "same_sign_required",
        "effect_class",
        "sign_p",
        "candidate_success",
        "control_success",
        "comparison_conclusion",
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# TRACER Phase-L6F Rough-Clearance Final Effect v0",
        "",
        f"- source: `{args.input}`",
        "- reliability uses all six paired repeats.",
        (
            "- continuous quality uses only repeats "
            "where both modes completed the rough segment."
        ),
        (
            "- positive signed effect means candidate "
            "improvement."
        ),
        (
            "- repeatable classification requires a "
            "non-crossing bootstrap CI and at least a "
            "5/6 same-sign proportion."
        ),
        (
            "- candidate-vs-noop is the primary causal "
            "comparison."
        ),
        (
            "- candidate-vs-baseline is a secondary "
            "sensitivity comparison."
        ),
        "",
    ]

    for section in report_sections:
        lines += [
            (
                f"## {section['comparison_type']}: "
                f"{section['candidate']} vs "
                f"{section['control']}"
            ),
            "",
            "### Reliability — all six repeats",
            "",
        ]

        for result in section["reliability"]:
            lines.append(
                f"- `{result['metric']}`: "
                f"candidate "
                f"{result['candidate_success']}/6, "
                f"control "
                f"{result['control_success']}/6; "
                f"wins/losses/ties "
                f"{result['positive_pairs']}/"
                f"{result['negative_pairs']}/"
                f"{result['ties']}; "
                f"`{result['effect_class']}`; "
                f"exact sign p="
                f"{result['sign_p']:.6g}"
            )

        lines += [
            "",
            (
                "### Conditional rough quality — "
                "both modes completed"
            ),
            "",
            (
                "- included repeats: "
                + ", ".join(
                    str(value)
                    for value
                    in section["completed_repeats"]
                )
            ),
            (
                "- comparison conclusion: "
                f"`{section['conclusion']}`"
            ),
            "",
        ]

        for result in section["quality"]:
            lines.append(
                f"- `{result['metric']}`: "
                f"mean {float(result['mean_effect']):.9g}; "
                f"CI [{float(result['ci_low']):.9g}, "
                f"{float(result['ci_high']):.9g}]; "
                f"+/-/tie "
                f"{result['positive_pairs']}/"
                f"{result['negative_pairs']}/"
                f"{result['ties']}; "
                f"`{result['effect_class']}`; "
                f"sign p={result['sign_p']:.6g}"
            )

        lines.append("")

    lines += [
        "## Safety and interpretation",
        "",
        (
            "- Reliability edges with only one "
            "discordant pair are descriptive, not "
            "statistical confirmation."
        ),
        (
            "- Conditional quality excludes the "
            "rough-incomplete no-op repeat, while its "
            "failure remains in reliability analysis."
        ),
        (
            "- The low-clearance repeat that completed "
            "rough slowly and later missed the goal "
            "remains in both reliability and conditional "
            "rough-quality analysis."
        ),
        (
            "- No action label is promoted unless the "
            "primary candidate-vs-noop comparison shows "
            "a repeatable useful trade-off."
        ),
        "- Empirical remains the active default.",
        "",
    ]

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        "\n".join(lines)
    )

    print("[L6F] input rows =", len(rows))
    print("[L6F] comparisons =", len(report_sections))

    for section in report_sections:
        print(
            section["comparison_type"],
            section["candidate"],
            "vs",
            section["control"],
            "=>",
            section["conclusion"],
            "completed repeats=",
            section["completed_repeats"],
        )

    print("[L6F] output =", args.output)
    print("[L6F] report =", args.report)


if __name__ == "__main__":
    main()
