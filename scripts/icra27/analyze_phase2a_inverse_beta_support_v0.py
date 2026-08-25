#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


COST_KEYS = (
    "cost_motion",
    "cost_stability",
    "cost_energy",
)

BETA_NAMES = (
    "balanced",
    "motion",
    "stability",
    "energy",
)


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def cost_vector(row):
    return np.asarray(
        [
            float(row[key])
            for key in COST_KEYS
        ],
        dtype=np.float64,
    )


def simplex_grid(step):
    inv = 1.0 / step
    n = int(
        round(inv)
    )

    if not np.isclose(
        float(n),
        inv,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError(
            "--grid-step must divide 1.0 exactly; "
            f"got {step}"
        )

    points = []

    for i in range(
        n + 1
    ):
        for j in range(
            n - i + 1
        ):
            k = (
                n
                - i
                - j
            )

            points.append(
                (
                    i / n,
                    j / n,
                    k / n,
                )
            )

    return np.asarray(
        points,
        dtype=np.float64,
    )


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

    ap.add_argument(
        "--grid-step",
        type=float,
        default=0.005,
    )

    ap.add_argument(
        "--support-tol",
        type=float,
        default=1e-10,
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

    if args.grid_step <= 0.0:
        raise ValueError(
            "--grid-step must be > 0"
        )

    if args.support_tol < 0.0:
        raise ValueError(
            "--support-tol must be >= 0"
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
            f"Expected 116 candidates, got {len(rows)}"
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

    betas = simplex_grid(
        args.grid_step
    )

    print(
        "simplex grid points:",
        len(betas),
    )

    records = []

    supported_count = 0
    pareto_supported_count = 0
    pareto_unsupported_count = 0

    support_by_beta_name = Counter()
    pareto_by_beta_name = Counter()
    unsupported_pareto_by_beta_name = Counter()

    for context_id in sorted(
        contexts
    ):
        candidates = contexts[
            context_id
        ]

        if len(candidates) != 4:
            raise RuntimeError(
                f"{context_id}: expected 4 candidates, "
                f"got {len(candidates)}"
            )

        feasible = [
            row
            for row in candidates
            if as_bool(
                row["feasible"]
            )
        ]

        if not feasible:
            continue

        for selected in candidates:
            selected_feasible = as_bool(
                selected["feasible"]
            )

            selected_pareto = as_bool(
                selected[
                    "pareto_nondominated"
                ]
            )

            beta_name = selected[
                "beta_name"
            ]

            if selected_pareto:
                pareto_by_beta_name[
                    beta_name
                ] += 1

            if not selected_feasible:
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

                        "feasible":
                            False,

                        "pareto_nondominated":
                            selected_pareto,

                        "linearly_supported":
                            False,

                        "support_grid_count":
                            0,

                        "support_fraction":
                            0.0,

                        "max_margin":
                            None,

                        "max_margin_beta_motion":
                            None,

                        "max_margin_beta_stability":
                            None,

                        "max_margin_beta_energy":
                            None,
                    }
                )

                continue

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
                    f"{context_id}: no alternatives"
                )

            # ------------------------------------------------
            # For each beta:
            #
            # gap_j(beta)
            #   = beta^T (C_j - C_i)
            #
            # Candidate i is optimal if every gap >= 0.
            #
            # margin(beta)
            #   = min_j gap_j(beta)
            #
            # Positive margin:
            #   strict preference for candidate i.
            #
            # Zero margin:
            #   candidate i is optimal with at least one tie.
            # ------------------------------------------------

            diffs = np.stack(
                [
                    cost_vector(other)
                    - ci
                    for other in alternatives
                ],
                axis=0,
            )

            # shape:
            #   [grid, alternatives]
            gaps = (
                betas
                @ diffs.T
            )

            margins = np.min(
                gaps,
                axis=1,
            )

            support_mask = (
                margins
                >= -args.support_tol
            )

            support_indices = np.flatnonzero(
                support_mask
            )

            support_grid_count = int(
                len(
                    support_indices
                )
            )

            support_fraction = float(
                support_grid_count
                / len(betas)
            )

            best_index = int(
                np.argmax(
                    margins
                )
            )

            max_margin = float(
                margins[
                    best_index
                ]
            )

            best_beta = betas[
                best_index
            ]

            supported = (
                support_grid_count
                > 0
            )

            if supported:
                supported_count += 1

                support_by_beta_name[
                    beta_name
                ] += 1

            if selected_pareto:
                if supported:
                    pareto_supported_count += 1
                else:
                    pareto_unsupported_count += 1

                    unsupported_pareto_by_beta_name[
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

                    "feasible":
                        True,

                    "pareto_nondominated":
                        selected_pareto,

                    "linearly_supported":
                        supported,

                    "support_grid_count":
                        support_grid_count,

                    "support_fraction":
                        support_fraction,

                    "max_margin":
                        max_margin,

                    "max_margin_beta_motion":
                        float(
                            best_beta[0]
                        ),

                    "max_margin_beta_stability":
                        float(
                            best_beta[1]
                        ),

                    "max_margin_beta_energy":
                        float(
                            best_beta[2]
                        ),
                }
            )

    if len(records) != 116:
        raise RuntimeError(
            f"Expected 116 output rows, got {len(records)}"
        )

    csv_path = (
        out_dir
        / "inverse_beta_support.csv"
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

    support_hist = Counter()

    for row in records:
        if not row[
            "feasible"
        ]:
            continue

        frac = float(
            row[
                "support_fraction"
            ]
        )

        if frac == 0.0:
            label = "0"
        elif frac < 0.01:
            label = "(0,0.01)"
        elif frac < 0.05:
            label = "[0.01,0.05)"
        elif frac < 0.10:
            label = "[0.05,0.10)"
        elif frac < 0.25:
            label = "[0.10,0.25)"
        else:
            label = "[0.25,1]"

        support_hist[
            label
        ] += 1

    summary = {
        "schema":
            "icra27_phase2a_inverse_beta_support_v0",

        "source_candidates":
            str(src),

        "grid_step":
            args.grid_step,

        "simplex_grid_points":
            int(
                len(betas)
            ),

        "support_definition":
            (
                "beta dot C_selected <= "
                "beta dot C_alternative "
                "for every feasible alternative"
            ),

        "canonical_beta_definition":
            (
                "grid beta maximizing minimum "
                "pairwise scalarized-cost margin"
            ),

        "candidate_count":
            len(records),

        "supported_candidate_count":
            supported_count,

        "pareto_candidate_count":
            sum(
                int(
                    row[
                        "pareto_nondominated"
                    ]
                )
                for row in records
            ),

        "pareto_supported_count":
            pareto_supported_count,

        "pareto_unsupported_count":
            pareto_unsupported_count,

        "support_by_original_beta_name":
            dict(
                support_by_beta_name
            ),

        "pareto_by_original_beta_name":
            dict(
                pareto_by_beta_name
            ),

        "unsupported_pareto_by_original_beta_name":
            dict(
                unsupported_pareto_by_beta_name
            ),

        "support_fraction_histogram":
            dict(
                support_hist
            ),

        "important_note":
            (
                "Original beta_name is provenance only. "
                "Inverse preference is inferred from the "
                "actual self-assessed objective vector, "
                "not from the anchor name."
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

    print()
    print(
        "=" * 80
    )

    print(
        "ICRA27 PHASE-2A INVERSE-BETA SUPPORT V0"
    )

    print(
        "=" * 80
    )

    print(
        "candidates              :",
        len(records),
    )

    print(
        "linearly supported      :",
        supported_count,
    )

    print(
        "Pareto candidates       :",
        summary[
            "pareto_candidate_count"
        ],
    )

    print(
        "Pareto + supported      :",
        pareto_supported_count,
    )

    print(
        "Pareto but unsupported  :",
        pareto_unsupported_count,
    )

    print()
    print(
        "SUPPORT BY ORIGINAL BETA NAME"
    )

    for name in BETA_NAMES:
        print(
            f"{name:<10} "
            f"{support_by_beta_name[name]}/29"
        )

    print()
    print(
        "UNSUPPORTED PARETO BY ORIGINAL BETA NAME"
    )

    for name in BETA_NAMES:
        print(
            f"{name:<10} "
            f"{unsupported_pareto_by_beta_name[name]}"
        )

    print()
    print(
        "support fraction hist   :",
        dict(
            support_hist
        ),
    )

    print()
    print(
        "csv                     :",
        csv_path,
    )

    print(
        "summary                 :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2A inverse-beta "
        "support analysis: PASS"
    )


if __name__ == "__main__":
    main()
