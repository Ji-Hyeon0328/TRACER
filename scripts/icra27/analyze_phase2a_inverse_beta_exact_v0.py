#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


COST_KEYS = (
    "cost_motion",
    "cost_stability",
    "cost_energy",
)

BETA_KEYS = (
    "beta_motion",
    "beta_stability",
    "beta_energy",
)

TOL = 1e-9


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def cost_vector(row):
    return np.asarray(
        [
            float(row[k])
            for k in COST_KEYS
        ],
        dtype=np.float64,
    )


def original_beta(row):
    beta = np.asarray(
        [
            float(row[k])
            for k in BETA_KEYS
        ],
        dtype=np.float64,
    )

    if not np.isclose(
        beta.sum(),
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"Invalid beta: {beta}"
        )

    return beta


def solve_max_margin(
    selected,
    feasible,
):
    ci = cost_vector(
        selected
    )

    alternatives = [
        row
        for row in feasible
        if row is not selected
    ]

    if not alternatives:
        raise RuntimeError(
            "No alternatives."
        )

    diffs = np.stack(
        [
            cost_vector(row)
            - ci
            for row in alternatives
        ],
        axis=0,
    )

    # Variables:
    #
    # x = [
    #   beta_motion,
    #   beta_stability,
    #   beta_energy,
    #   rho,
    # ]
    #
    # maximize rho
    # == minimize -rho.

    c = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            -1.0,
        ],
        dtype=np.float64,
    )

    # beta^T d_j >= rho
    #
    # -d_j^T beta + rho <= 0

    A_ub = np.zeros(
        (
            len(diffs),
            4,
        ),
        dtype=np.float64,
    )

    A_ub[:, :3] = (
        -diffs
    )

    A_ub[:, 3] = 1.0

    b_ub = np.zeros(
        len(diffs),
        dtype=np.float64,
    )

    A_eq = np.asarray(
        [
            [
                1.0,
                1.0,
                1.0,
                0.0,
            ]
        ],
        dtype=np.float64,
    )

    b_eq = np.asarray(
        [
            1.0
        ],
        dtype=np.float64,
    )

    bounds = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (None, None),
    ]

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "linprog failed: "
            f"{result.message}"
        )

    beta = np.asarray(
        result.x[:3],
        dtype=np.float64,
    )

    rho = float(
        result.x[3]
    )

    return beta, rho


def original_beta_regret(
    selected,
    feasible,
):
    beta = original_beta(
        selected
    )

    costs = np.asarray(
        [
            float(
                beta
                @ cost_vector(row)
            )
            for row in feasible
        ],
        dtype=np.float64,
    )

    selected_cost = float(
        beta
        @ cost_vector(selected)
    )

    best_cost = float(
        np.min(costs)
    )

    regret = (
        selected_cost
        - best_cost
    )

    return (
        beta,
        selected_cost,
        best_cost,
        regret,
    )


def support_class(rho):
    if rho > TOL:
        return "strict"

    if rho >= -TOL:
        return "weak"

    return "unsupported"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidates",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    src = Path(
        args.candidates
    )

    out_dir = Path(
        args.out_dir
    )

    if not src.is_file():
        raise FileNotFoundError(
            src
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with src.open(
        newline=""
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 116:
        raise RuntimeError(
            f"Expected 116 rows, got {len(rows)}"
        )

    contexts = defaultdict(
        list
    )

    for row in rows:
        contexts[
            row["context_id"]
        ].append(
            row
        )

    if len(contexts) != 29:
        raise RuntimeError(
            f"Expected 29 contexts, got {len(contexts)}"
        )

    records = []

    class_counts = Counter()
    pareto_class_counts = Counter()
    original_consistent = Counter()
    original_total = Counter()

    for context_id in sorted(
        contexts
    ):
        candidates = contexts[
            context_id
        ]

        feasible = [
            row
            for row in candidates
            if as_bool(
                row["feasible"]
            )
        ]

        for selected in candidates:
            beta_name = selected[
                "beta_name"
            ]

            feasible_flag = as_bool(
                selected["feasible"]
            )

            pareto = as_bool(
                selected[
                    "pareto_nondominated"
                ]
            )

            if not feasible_flag:
                continue

            beta_hat, rho = (
                solve_max_margin(
                    selected,
                    feasible,
                )
            )

            cls = support_class(
                rho
            )

            (
                beta_gen,
                j_selected,
                j_best,
                regret,
            ) = original_beta_regret(
                selected,
                feasible,
            )

            gen_consistent = (
                regret
                <= TOL
            )

            class_counts[
                cls
            ] += 1

            if pareto:
                pareto_class_counts[
                    cls
                ] += 1

            original_total[
                beta_name
            ] += 1

            if gen_consistent:
                original_consistent[
                    beta_name
                ] += 1

            records.append(
                {
                    "context_id":
                        context_id,

                    "group":
                        selected["group"],

                    "terrain":
                        selected["terrain"],

                    "seed":
                        int(selected["seed"]),

                    "beta_name":
                        beta_name,

                    "pareto_nondominated":
                        pareto,

                    "support_class":
                        cls,

                    "max_margin":
                        rho,

                    "beta_hat_motion":
                        float(
                            beta_hat[0]
                        ),

                    "beta_hat_stability":
                        float(
                            beta_hat[1]
                        ),

                    "beta_hat_energy":
                        float(
                            beta_hat[2]
                        ),

                    "beta_gen_motion":
                        float(
                            beta_gen[0]
                        ),

                    "beta_gen_stability":
                        float(
                            beta_gen[1]
                        ),

                    "beta_gen_energy":
                        float(
                            beta_gen[2]
                        ),

                    "original_beta_selected_cost":
                        j_selected,

                    "original_beta_best_cost":
                        j_best,

                    "original_beta_regret":
                        regret,

                    "original_beta_consistent":
                        gen_consistent,
                }
            )

    if len(records) != 116:
        raise RuntimeError(
            "Expected 116 exact-support records, "
            f"got {len(records)}"
        )

    csv_path = (
        out_dir
        / "inverse_beta_exact.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[0]
            ),
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    pareto_total = sum(
        int(
            row[
                "pareto_nondominated"
            ]
        )
        for row in records
    )

    summary = {
        "schema":
            "icra27_phase2a_inverse_beta_exact_v0",

        "source_candidates":
            str(src),

        "support_definition":
            (
                "exact LP over closed beta simplex; "
                "maximize minimum scalarized-cost margin"
            ),

        "candidate_count":
            len(records),

        "pareto_candidate_count":
            pareto_total,

        "support_class_counts":
            dict(
                class_counts
            ),

        "pareto_support_class_counts":
            dict(
                pareto_class_counts
            ),

        "original_beta_consistency":
            {
                name: {
                    "consistent":
                        int(
                            original_consistent[
                                name
                            ]
                        ),

                    "total":
                        int(
                            original_total[
                                name
                            ]
                        ),
                }
                for name in (
                    "balanced",
                    "motion",
                    "stability",
                    "energy",
                )
            },

        "important_note":
            (
                "beta_hat is a max-margin representative "
                "of a possibly non-unique inverse-preference "
                "support region. It is not assumed to be the "
                "unique true preference."
            ),
    }

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        "=" * 80
    )

    print(
        "ICRA27 PHASE-2A EXACT INVERSE-BETA SUPPORT V0"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "ALL CANDIDATES"
    )

    for cls in (
        "strict",
        "weak",
        "unsupported",
    ):
        print(
            f"{cls:<12}: "
            f"{class_counts[cls]}"
        )

    print()
    print(
        "PARETO CANDIDATES"
    )

    for cls in (
        "strict",
        "weak",
        "unsupported",
    ):
        print(
            f"{cls:<12}: "
            f"{pareto_class_counts[cls]}"
        )

    print()
    print(
        "ORIGINAL-BETA SELF-CONSISTENCY"
    )

    for name in (
        "balanced",
        "motion",
        "stability",
        "energy",
    ):
        print(
            f"{name:<10}: "
            f"{original_consistent[name]}/"
            f"{original_total[name]}"
        )

    print()
    print(
        "csv     :",
        csv_path,
    )

    print(
        "summary :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2A exact inverse-beta "
        "support: PASS"
    )


if __name__ == "__main__":
    main()
