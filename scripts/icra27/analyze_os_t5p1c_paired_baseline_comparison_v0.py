#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FIXED_CSV = (
    ROOT
    / "results"
    / "icra27"
    / "os_t5p1b_fixed_primary_v0"
    / "episodes.csv"
)

HL_CSV = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p9b_v4_primary_heldout_v0"
    / "episodes.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t5p1c_paired_baseline_comparison_v0"
)

KEY_FIELDS = (
    "group",
    "terrain",
    "seed",
)

BETA_NAMES = (
    "balanced",
    "motion",
    "stability",
    "energy",
)

OBJECTIVES = {
    "motion":
        "mean_cost_motion",

    "stability":
        "mean_cost_stability",

    "energy":
        "mean_cost_energy",
}

ROUGH_GROUPS = {
    "rough_validation",
    "rough_test",
}

EPS = 1e-12


def read_csv(path):
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def key(row):
    return (
        row["group"],
        row["terrain"],
        int(row["seed"]),
    )


def finite(row, name):
    x = float(row[name])

    if not (
        x == x
        and abs(x) != float("inf")
    ):
        raise RuntimeError(
            f"Non-finite {name}: {x}"
        )

    return x


def as_bool(x):
    return str(x).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def sign_test_two_sided(
    wins,
    losses,
):
    n = wins + losses

    if n == 0:
        return 1.0

    k = min(
        wins,
        losses,
    )

    p = (
        2.0
        * sum(
            comb(n, i)
            for i in range(k + 1)
        )
        / (2 ** n)
    )

    return min(
        1.0,
        float(p),
    )


def summarize_pair(
    *,
    keys,
    metric,
    candidate,
    reference,
):
    deltas = []

    for k in keys:
        a = finite(
            candidate[k],
            metric,
        )

        b = finite(
            reference[k],
            metric,
        )

        deltas.append(
            a - b
        )

    wins = sum(
        x < -EPS
        for x in deltas
    )

    losses = sum(
        x > EPS
        for x in deltas
    )

    ties = (
        len(deltas)
        - wins
        - losses
    )

    candidate_mean = (
        sum(
            finite(
                candidate[k],
                metric,
            )
            for k in keys
        )
        / len(keys)
    )

    reference_mean = (
        sum(
            finite(
                reference[k],
                metric,
            )
            for k in keys
        )
        / len(keys)
    )

    mean_delta = (
        sum(deltas)
        / len(deltas)
    )

    if abs(reference_mean) > EPS:
        pct = (
            100.0
            * mean_delta
            / abs(reference_mean)
        )
    else:
        pct = None

    return {
        "n":
            len(keys),

        "wins_lower_is_better":
            int(wins),

        "losses":
            int(losses),

        "ties":
            int(ties),

        "candidate_mean":
            float(candidate_mean),

        "reference_mean":
            float(reference_mean),

        "mean_delta_candidate_minus_reference":
            float(mean_delta),

        "mean_delta_pct_reference":
            (
                None
                if pct is None
                else float(pct)
            ),

        "exact_sign_test_two_sided_p":
            sign_test_two_sided(
                wins,
                losses,
            ),
    }


def main():
    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory exists; "
            f"refusing overwrite: {OUT_DIR}"
        )

    fixed_rows = read_csv(
        FIXED_CSV
    )

    hl_rows = read_csv(
        HL_CSV
    )

    if len(fixed_rows) != 29:
        raise RuntimeError(
            "Expected 29 fixed episodes, "
            f"got {len(fixed_rows)}"
        )

    if len(hl_rows) != 116:
        raise RuntimeError(
            "Expected 116 beta episodes, "
            f"got {len(hl_rows)}"
        )

    for row in (
        fixed_rows
        + hl_rows
    ):
        if (
            row["eval_reward_mode"]
            != "tracer_cost_v4"
        ):
            raise RuntimeError(
                "Non-v4 evaluation row found."
            )

    fixed = {}

    for row in fixed_rows:
        k = key(row)

        if k in fixed:
            raise RuntimeError(
                f"Duplicate fixed context: {k}"
            )

        fixed[k] = row

    beta = {
        name: {}
        for name in BETA_NAMES
    }

    for row in hl_rows:
        name = row["beta_name"]

        if name not in beta:
            raise RuntimeError(
                f"Unexpected beta: {name}"
            )

        k = key(row)

        if k in beta[name]:
            raise RuntimeError(
                f"Duplicate beta/context: "
                f"{name}/{k}"
            )

        beta[name][k] = row

    context_keys = sorted(
        fixed.keys()
    )

    if len(context_keys) != 29:
        raise RuntimeError(
            "Expected 29 unique contexts."
        )

    for name in BETA_NAMES:
        if (
            set(beta[name])
            != set(fixed)
        ):
            raise RuntimeError(
                f"Context mismatch for beta "
                f"{name}"
            )

    scopes = {
        "all29":
            context_keys,

        "flat10": [
            k
            for k in context_keys
            if k[0] == "flat"
        ],

        "low_friction10": [
            k
            for k in context_keys
            if k[0] == "low_friction"
        ],

        "rough_validation4": [
            k
            for k in context_keys
            if k[0]
            == "rough_validation"
        ],

        "rough_test5": [
            k
            for k in context_keys
            if k[0]
            == "rough_test"
        ],

        "rough9": [
            k
            for k in context_keys
            if k[0] in ROUGH_GROUPS
        ],
    }

    expected_scope_sizes = {
        "all29": 29,
        "flat10": 10,
        "low_friction10": 10,
        "rough_validation4": 4,
        "rough_test5": 5,
        "rough9": 9,
    }

    for name, ks in scopes.items():
        if (
            len(ks)
            != expected_scope_sizes[name]
        ):
            raise RuntimeError(
                f"Bad scope size {name}: "
                f"{len(ks)}"
            )

    # --------------------------------------------------------
    # Safety / feasibility summary.
    # --------------------------------------------------------

    methods = {
        "B0_fixed":
            fixed,

        "B1_balanced":
            beta["balanced"],

        "B2_motion":
            beta["motion"],

        "B2_stability":
            beta["stability"],

        "B2_energy":
            beta["energy"],
    }

    feasibility = {}

    for name, rows in methods.items():
        feasibility[name] = {
            "episodes":
                len(rows),

            "success":
                sum(
                    as_bool(
                        row["success"]
                    )
                    for row in rows.values()
                ),

            "m4_interventions":
                sum(
                    int(
                        float(
                            row[
                                "m4_interventions"
                            ]
                        )
                    )
                    for row in rows.values()
                ),

            "m4_terminal":
                sum(
                    as_bool(
                        row["m4_terminal"]
                    )
                    for row in rows.values()
                ),
        }

    # --------------------------------------------------------
    # Per-context objective comparison.
    # --------------------------------------------------------

    context_rows = []

    for k in context_keys:
        group, terrain, seed = k

        for objective, metric in (
            OBJECTIVES.items()
        ):
            f = finite(
                fixed[k],
                metric,
            )

            b = finite(
                beta["balanced"][k],
                metric,
            )

            o = finite(
                beta[objective][k],
                metric,
            )

            context_rows.append(
                {
                    "group":
                        group,

                    "terrain":
                        terrain,

                    "seed":
                        seed,

                    "objective":
                        objective,

                    "metric":
                        metric,

                    "fixed":
                        f,

                    "balanced":
                        b,

                    "objective_beta":
                        o,

                    "balanced_minus_fixed":
                        b - f,

                    "objective_minus_balanced":
                        o - b,

                    "objective_minus_fixed":
                        o - f,

                    "q1_balanced_beats_fixed":
                        bool(
                            b < f - EPS
                        ),

                    "q2_objective_beats_balanced":
                        bool(
                            o < b - EPS
                        ),

                    "q3_objective_beats_fixed":
                        bool(
                            o < f - EPS
                        ),
                }
            )

    # --------------------------------------------------------
    # Pairwise summaries.
    # --------------------------------------------------------

    comparisons = {}

    for objective, metric in (
        OBJECTIVES.items()
    ):
        comparisons[
            objective
        ] = {}

        for scope_name, ks in (
            scopes.items()
        ):
            comparisons[
                objective
            ][
                scope_name
            ] = {
                "metric":
                    metric,

                "Q1_balanced_vs_fixed":
                    summarize_pair(
                        keys=ks,
                        metric=metric,
                        candidate=(
                            beta["balanced"]
                        ),
                        reference=fixed,
                    ),

                "Q2_objective_vs_balanced":
                    summarize_pair(
                        keys=ks,
                        metric=metric,
                        candidate=(
                            beta[objective]
                        ),
                        reference=(
                            beta["balanced"]
                        ),
                    ),

                "Q3_objective_vs_fixed":
                    summarize_pair(
                        keys=ks,
                        metric=metric,
                        candidate=(
                            beta[objective]
                        ),
                        reference=fixed,
                    ),
            }

    # --------------------------------------------------------
    # Secondary physical diagnostics on rough9.
    # These are NOT substitutes for target-cost semantics.
    # --------------------------------------------------------

    rough = scopes["rough9"]

    secondary = {
        "motion_completion_time":
            {
                "objective_vs_balanced":
                    summarize_pair(
                        keys=rough,
                        metric=(
                            "decision_time_s"
                        ),
                        candidate=(
                            beta["motion"]
                        ),
                        reference=(
                            beta["balanced"]
                        ),
                    ),

                "objective_vs_fixed":
                    summarize_pair(
                        keys=rough,
                        metric=(
                            "decision_time_s"
                        ),
                        candidate=(
                            beta["motion"]
                        ),
                        reference=fixed,
                    ),
            },

        "energy_physical_joules":
            {
                "objective_vs_balanced":
                    summarize_pair(
                        keys=rough,
                        metric="energy_abs_j",
                        candidate=(
                            beta["energy"]
                        ),
                        reference=(
                            beta["balanced"]
                        ),
                    ),

                "objective_vs_fixed":
                    summarize_pair(
                        keys=rough,
                        metric="energy_abs_j",
                        candidate=(
                            beta["energy"]
                        ),
                        reference=fixed,
                    ),
            },

        "stability_pitch_rms":
            {
                "objective_vs_balanced":
                    summarize_pair(
                        keys=rough,
                        metric="pitch_rms_rad",
                        candidate=(
                            beta["stability"]
                        ),
                        reference=(
                            beta["balanced"]
                        ),
                    ),
            },

        "stability_roll_rms":
            {
                "objective_vs_balanced":
                    summarize_pair(
                        keys=rough,
                        metric="roll_rms_rad",
                        candidate=(
                            beta["stability"]
                        ),
                        reference=(
                            beta["balanced"]
                        ),
                    ),
            },
    }

    OUT_DIR.mkdir(
        parents=True
    )

    csv_path = (
        OUT_DIR
        / "paired_objective_contexts.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                context_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            context_rows
        )

    manifest = {
        "schema":
            "icra27_os_t5p1c_paired_baseline_comparison_v0",

        "fixed_source":
            str(FIXED_CSV),

        "beta_source":
            str(HL_CSV),

        "evaluation_reward_mode":
            "tracer_cost_v4",

        "comparison_design":
            {
                "B0":
                    "TRAIN-only mean fixed command",

                "B1":
                    "frozen u30 balanced beta",

                "B2":
                    (
                        "same frozen u30 with "
                        "target objective beta"
                    ),
            },

        "primary_inferential_scope":
            "rough9",

        "statistical_note":
            (
                "Flat and low-friction seed repetitions "
                "are deterministic in the current setup "
                "and are descriptive, not independent "
                "statistical replicates."
            ),

        "feasibility":
            feasibility,

        "comparisons":
            comparisons,

        "secondary_physical_diagnostics":
            secondary,
    }

    manifest_path = (
        OUT_DIR
        / "comparison_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=" * 112)
    print(
        "ICRA27 OS-T5.1c "
        "PAIRED BASELINE COMPARISON"
    )
    print("=" * 112)

    print()
    print("FEASIBILITY")

    for name, result in (
        feasibility.items()
    ):
        print(
            f"  {name:<14} "
            f"success="
            f"{result['success']}/"
            f"{result['episodes']} "
            f"M4="
            f"{result['m4_interventions']} "
            f"M4-terminal="
            f"{result['m4_terminal']}"
        )

    print()
    print(
        "PRIMARY: HELD-OUT ROUGH 9"
    )

    for objective in (
        "motion",
        "stability",
        "energy",
    ):
        s = (
            comparisons[
                objective
            ][
                "rough9"
            ]
        )

        print()
        print(
            objective.upper()
        )

        for label in (
            "Q1_balanced_vs_fixed",
            "Q2_objective_vs_balanced",
            "Q3_objective_vs_fixed",
        ):
            r = s[label]

            print(
                f"  {label:<28} "
                f"win/loss/tie="
                f"{r['wins_lower_is_better']}/"
                f"{r['losses']}/"
                f"{r['ties']} "
                f"delta="
                f"{r['mean_delta_candidate_minus_reference']:+.8f} "
                f"pct="
                f"{r['mean_delta_pct_reference']:+.3f}% "
                f"p="
                f"{r['exact_sign_test_two_sided_p']:.6f}"
            )

    print()
    print(
        "SECONDARY PHYSICAL DIAGNOSTICS "
        "(rough9)"
    )

    for name, by_comp in (
        secondary.items()
    ):
        print()
        print(
            f"  {name}"
        )

        for comp, r in (
            by_comp.items()
        ):
            print(
                f"    {comp:<24} "
                f"win/loss/tie="
                f"{r['wins_lower_is_better']}/"
                f"{r['losses']}/"
                f"{r['ties']} "
                f"delta="
                f"{r['mean_delta_candidate_minus_reference']:+.8f}"
            )

    print()
    print("outputs:")
    print(" ", csv_path)
    print(" ", manifest_path)

    print()
    print(
        "[ICRA27] OS-T5.1c "
        "paired baseline comparison: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
