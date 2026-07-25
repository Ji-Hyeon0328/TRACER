#!/usr/bin/env python3

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = {
    "progress_rate": "higher_better",
    "mean_abs_y_window": "lower_better",
    "max_abs_y_window": "lower_better",
    "end_abs_y": "lower_better",
    "lateral_growth": "lower_better",
}

CANDIDATES = [
    "flat_body_low05",
    "flat_body_high05",
]


def ff(value):
    try:
        text = str(value).strip()

        if not text or text.lower() == "nan":
            return None

        value = float(text)

        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def row_metrics(row):
    x0 = ff(row.get("x_start"))
    x1 = ff(row.get("x_end"))
    y0 = ff(row.get("y_start"))
    y1 = ff(row.get("y_end"))
    duration = ff(row.get("duration_s"))

    progress_rate = None

    if (
        x0 is not None
        and x1 is not None
        and duration is not None
        and duration > 1e-9
    ):
        progress_rate = (x1 - x0) / duration

    return {
        "progress_rate": progress_rate,
        "mean_abs_y_window": ff(
            row.get("mean_abs_y_window")
        ),
        "max_abs_y_window": ff(
            row.get("max_abs_y_window")
        ),
        "end_abs_y": (
            abs(y1)
            if y1 is not None
            else None
        ),
        "lateral_growth": (
            abs(y1) - abs(y0)
            if y0 is not None and y1 is not None
            else None
        ),
    }


def signed_effect(candidate, reference, direction):
    if candidate is None or reference is None:
        return None

    if direction == "higher_better":
        return candidate - reference

    return reference - candidate


def quantile(values, q):
    values = sorted(values)

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    weight = pos - lo

    return (
        values[lo] * (1.0 - weight)
        + values[hi] * weight
    )


def bootstrap_ci(values, rng, samples=5000):
    if not values:
        return None, None

    if len(values) == 1:
        return values[0], values[0]

    means = []

    for _ in range(samples):
        sample = [
            values[rng.randrange(len(values))]
            for _ in values
        ]
        means.append(mean(sample))

    return (
        quantile(means, 0.025),
        quantile(means, 0.975),
    )


def exact_sign_p(positive, negative):
    n = positive + negative

    if n == 0:
        return None

    extreme = max(positive, negative)

    tail = sum(
        math.comb(n, k)
        for k in range(extreme, n + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * tail)


def classify(ci_low, ci_high, positive, negative):
    if ci_low is None or ci_high is None:
        return "unavailable"

    # With only six pairs, require at least five pairs in the
    # same direction in addition to a non-crossing bootstrap CI.
    if ci_low > 0.0 and positive >= 5:
        return "repeatable_positive"

    if ci_high < 0.0 and negative >= 5:
        return "repeatable_negative"

    return "unresolved"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "datasets/phase_l/"
            "l5g_fresh_body_height_validity_flat_evidence_v1.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "datasets/phase_l/"
            "l5g_route_matched_body_height_effect_summary_v0.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "reports/phase_l/"
            "l5g_route_matched_body_height_effect_summary_v0.md"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=332,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    with input_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    by_repeat = defaultdict(dict)

    for row in rows:
        repeat_idx = int(float(row["trial_idx"]))
        mode = row["candidate_mode"]

        if mode in by_repeat[repeat_idx]:
            raise RuntimeError(
                f"duplicate row: repeat={repeat_idx}, mode={mode}"
            )

        by_repeat[repeat_idx][mode] = row

    expected = {
        "baseline",
        "flat_noop",
        *CANDIDATES,
    }

    repeats = sorted(by_repeat)

    for repeat_idx in repeats:
        actual = set(by_repeat[repeat_idx])

        if actual != expected:
            raise RuntimeError(
                f"repeat {repeat_idx}: "
                f"expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )

    comparisons = [
        {
            "name": "routing_placebo",
            "candidate": "flat_noop",
            "reference": "baseline",
        }
    ]

    for candidate in CANDIDATES:
        comparisons.extend([
            {
                "name": "candidate_vs_noop",
                "candidate": candidate,
                "reference": "flat_noop",
            },
            {
                "name": "candidate_vs_baseline",
                "candidate": candidate,
                "reference": "baseline",
            },
        ])

    rng = random.Random(args.seed)
    summary_rows = []

    for comparison in comparisons:
        candidate_mode = comparison["candidate"]
        reference_mode = comparison["reference"]

        effects = defaultdict(list)

        for repeat_idx in repeats:
            candidate = row_metrics(
                by_repeat[repeat_idx][candidate_mode]
            )

            reference = row_metrics(
                by_repeat[repeat_idx][reference_mode]
            )

            for metric, direction in METRICS.items():
                value = signed_effect(
                    candidate[metric],
                    reference[metric],
                    direction,
                )

                if value is not None:
                    effects[metric].append(value)

        row = {
            "comparison": comparison["name"],
            "candidate_mode": candidate_mode,
            "reference_mode": reference_mode,
            "n_pairs": len(repeats),
        }

        for metric in METRICS:
            values = effects[metric]
            effect_mean = mean(values)
            ci_low, ci_high = bootstrap_ci(
                values,
                rng,
            )

            positive = sum(value > 0 for value in values)
            negative = sum(value < 0 for value in values)

            row[f"effect_{metric}_mean"] = effect_mean
            row[f"effect_{metric}_ci_low"] = ci_low
            row[f"effect_{metric}_ci_high"] = ci_high
            row[f"positive_pairs_{metric}"] = positive
            row[f"negative_pairs_{metric}"] = negative
            row[f"sign_p_{metric}"] = exact_sign_p(
                positive,
                negative,
            )
            row[f"status_{metric}"] = classify(
                ci_low,
                ci_high,
                positive,
                negative,
            )

        progress_status = row[
            "status_progress_rate"
        ]

        stability_status = row[
            "status_max_abs_y_window"
        ]

        if (
            progress_status == "repeatable_negative"
            and stability_status == "repeatable_positive"
        ):
            primary_class = (
                "stability_gain_with_progress_cost"
            )
        elif (
            progress_status == "repeatable_negative"
            and stability_status
            != "repeatable_positive"
        ):
            primary_class = (
                "progress_cost_without_stability_gain"
            )
        elif (
            progress_status
            != "repeatable_negative"
            and stability_status == "repeatable_positive"
        ):
            primary_class = (
                "stability_gain_without_progress_cost"
            )
        else:
            primary_class = "unresolved"

        row["primary_class"] = primary_class
        summary_rows.append(row)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(summary_rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# TRACER Phase-L5G Route-Matched Body-Height Effect Summary v0",
        "",
        f"- source: `{input_path}`",
        f"- repeats: {repeats}",
        "- primary comparison: candidate vs same-repeat flat_noop",
        "- secondary comparison: candidate vs same-repeat baseline",
        "- routing diagnostic: flat_noop vs baseline",
        "- positive signed effect means improvement",
        "- repeatable classification requires a non-crossing bootstrap CI and at least 5/6 pairs with the same sign",
        "",
    ]

    for row in summary_rows:
        lines += [
            f"## {row['comparison']}: "
            f"{row['candidate_mode']} vs "
            f"{row['reference_mode']}",
            "",
            f"- primary class: `{row['primary_class']}`",
        ]

        for metric in METRICS:
            lines.append(
                f"- `{metric}`: mean "
                f"{row[f'effect_{metric}_mean']} | "
                f"CI [{row[f'effect_{metric}_ci_low']}, "
                f"{row[f'effect_{metric}_ci_high']}] | "
                f"`{row[f'status_{metric}']}` | "
                f"+pairs "
                f"{row[f'positive_pairs_{metric}']}/"
                f"{row['n_pairs']} | "
                f"sign p={row[f'sign_p_{metric}']}"
            )

        lines.append("")

    lines += [
        "## Safety",
        "",
        "- No route-matched body-height result automatically promotes a candidate.",
        "- Empirical/default remains active.",
        "- Candidate labels require agreement between primary route-matched and secondary baseline sensitivity analyses.",
        "",
    ]

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        "\n".join(lines)
    )

    print("[L4E] repeats =", repeats)
    print("[L4E] comparisons =", len(summary_rows))
    print("[L4E] output =", output_path)
    print("[L4E] report =", report_path)


if __name__ == "__main__":
    main()
