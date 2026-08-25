#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


TOL = 1e-9


BETA_BANK = {
    "balanced":
        np.array(
            [
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ],
            dtype=float,
        ),

    "motion":
        np.array(
            [
                0.70,
                0.15,
                0.15,
            ],
            dtype=float,
        ),

    "stability":
        np.array(
            [
                0.15,
                0.70,
                0.15,
            ],
            dtype=float,
        ),

    "energy":
        np.array(
            [
                0.15,
                0.15,
                0.70,
            ],
            dtype=float,
        ),
}


def read_csv(path):
    with path.open(
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def solve_exact_support(
    deltas,
):
    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    if (
        deltas.ndim != 2
        or deltas.shape[1] != 3
        or deltas.shape[0] < 1
    ):
        raise RuntimeError(
            f"Invalid delta matrix shape: "
            f"{deltas.shape}"
        )

    # Variables:
    #
    #   x = [beta_M, beta_S, beta_E, rho]
    #
    # maximize rho
    # == minimize -rho
    c = np.array(
        [
            0.0,
            0.0,
            0.0,
            -1.0,
        ],
        dtype=float,
    )

    # beta^T delta >= rho
    #
    # -delta^T beta + rho <= 0
    A_ub = []

    b_ub = []

    for delta in deltas:
        A_ub.append(
            [
                -float(
                    delta[0]
                ),
                -float(
                    delta[1]
                ),
                -float(
                    delta[2]
                ),
                1.0,
            ]
        )

        b_ub.append(
            0.0
        )

    A_eq = [
        [
            1.0,
            1.0,
            1.0,
            0.0,
        ]
    ]

    b_eq = [
        1.0
    ]

    bounds = [
        (
            0.0,
            1.0,
        ),
        (
            0.0,
            1.0,
        ),
        (
            0.0,
            1.0,
        ),
        (
            None,
            None,
        ),
    ]

    result = linprog(
        c=c,
        A_ub=np.asarray(
            A_ub,
            dtype=float,
        ),
        b_ub=np.asarray(
            b_ub,
            dtype=float,
        ),
        A_eq=np.asarray(
            A_eq,
            dtype=float,
        ),
        b_eq=np.asarray(
            b_eq,
            dtype=float,
        ),
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
        dtype=float,
    )

    rho = float(
        result.x[3]
    )

    if rho > TOL:
        support_class = "strict"

    elif rho >= -TOL:
        support_class = "weak"

    else:
        support_class = "unsupported"

    margins = (
        deltas
        @ beta
    )

    return {
        "beta":
            beta,

        "rho":
            rho,

        "support_class":
            support_class,

        "min_margin":
            float(
                np.min(
                    margins
                )
            ),

        "max_margin":
            float(
                np.max(
                    margins
                )
            ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preference-pairs",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    pair_path = Path(
        args.preference_pairs
    )

    out_dir = Path(
        args.out_dir
    )

    if not pair_path.is_file():
        raise FileNotFoundError(
            pair_path
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_csv(
        pair_path
    )

    if not rows:
        raise RuntimeError(
            "Preference-pair dataset is empty."
        )

    # --------------------------------------------------------
    # Group all pairwise evidence belonging to one exact
    # mission decision.
    # --------------------------------------------------------

    by_decision = defaultdict(
        list
    )

    for row in rows:
        by_decision[
            int(
                row[
                    "decision_index"
                ]
            )
        ].append(
            row
        )

    records = []

    for decision_index in sorted(
        by_decision
    ):
        group = by_decision[
            decision_index
        ]

        first = group[
            0
        ]

        # Contract: all rows in a decision share metadata.
        metadata_keys = (
            "context_id",
            "group",
            "terrain",
            "seed",
            "mission",
            "primary_metric",
            "selected_beta_name",
            "feasible_choice_count",
        )

        for row in group[
            1:
        ]:
            for key in metadata_keys:
                if row[
                    key
                ] != first[
                    key
                ]:
                    raise RuntimeError(
                        f"Decision {decision_index}: "
                        f"metadata mismatch for {key}"
                    )

        deltas = np.asarray(
            [
                [
                    float(
                        row[
                            "delta_cost_motion"
                        ]
                    ),
                    float(
                        row[
                            "delta_cost_stability"
                        ]
                    ),
                    float(
                        row[
                            "delta_cost_energy"
                        ]
                    ),
                ]
                for row in group
            ],
            dtype=float,
        )

        solved = solve_exact_support(
            deltas
        )

        source_name = first[
            "selected_beta_name"
        ]

        if source_name not in BETA_BANK:
            raise RuntimeError(
                f"Unknown source beta: "
                f"{source_name!r}"
            )

        source_beta = BETA_BANK[
            source_name
        ]

        source_margins = (
            deltas
            @ source_beta
        )

        source_consistent = bool(
            np.min(
                source_margins
            ) >= -TOL
        )

        beta = solved[
            "beta"
        ]

        record = {
            "decision_index":
                decision_index,

            "context_id":
                first[
                    "context_id"
                ],

            "group":
                first[
                    "group"
                ],

            "terrain":
                first[
                    "terrain"
                ],

            "seed":
                int(
                    first[
                        "seed"
                    ]
                ),

            "mission":
                first[
                    "mission"
                ],

            "primary_metric":
                first[
                    "primary_metric"
                ],

            "selected_beta_name":
                source_name,

            "feasible_choice_count":
                int(
                    first[
                        "feasible_choice_count"
                    ]
                ),

            "preference_pair_count":
                len(
                    group
                ),

            "support_class":
                solved[
                    "support_class"
                ],

            "max_margin_rho":
                solved[
                    "rho"
                ],

            "beta_hat_motion":
                float(
                    beta[
                        0
                    ]
                ),

            "beta_hat_stability":
                float(
                    beta[
                        1
                    ]
                ),

            "beta_hat_energy":
                float(
                    beta[
                        2
                    ]
                ),

            "min_pair_margin_at_beta_hat":
                solved[
                    "min_margin"
                ],

            "max_pair_margin_at_beta_hat":
                solved[
                    "max_margin"
                ],

            "source_beta_consistent":
                int(
                    source_consistent
                ),

            "source_beta_min_margin":
                float(
                    np.min(
                        source_margins
                    )
                ),
        }

        records.append(
            record
        )

    if len(records) != len(
        by_decision
    ):
        raise RuntimeError(
            "Decision-record count mismatch."
        )

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------

    support_counts = Counter(
        row[
            "support_class"
        ]
        for row in records
    )

    mission_support = {}

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        subset = [
            row
            for row in records
            if row[
                "mission"
            ] == mission
        ]

        counts = Counter(
            row[
                "support_class"
            ]
            for row in subset
        )

        mission_support[
            mission
        ] = {
            "count":
                len(
                    subset
                ),

            "strict":
                counts[
                    "strict"
                ],

            "weak":
                counts[
                    "weak"
                ],

            "unsupported":
                counts[
                    "unsupported"
                ],

            "expressible":
                (
                    counts[
                        "strict"
                    ]
                    + counts[
                        "weak"
                    ]
                ),

            "expressible_fraction":
                (
                    (
                        counts[
                            "strict"
                        ]
                        + counts[
                            "weak"
                        ]
                    )
                    / len(
                        subset
                    )
                    if subset
                    else None
                ),
        }

    terrain_support = {}

    for terrain in (
        "flat",
        "low_friction",
        "rough_perlin",
    ):
        subset = [
            row
            for row in records
            if row[
                "terrain"
            ] == terrain
        ]

        counts = Counter(
            row[
                "support_class"
            ]
            for row in subset
        )

        terrain_support[
            terrain
        ] = {
            "count":
                len(
                    subset
                ),

            "strict":
                counts[
                    "strict"
                ],

            "weak":
                counts[
                    "weak"
                ],

            "unsupported":
                counts[
                    "unsupported"
                ],
        }

    source_consistency = Counter(
        row[
            "selected_beta_name"
        ]
        for row in records
        if row[
            "source_beta_consistent"
        ]
    )

    source_total = Counter(
        row[
            "selected_beta_name"
        ]
        for row in records
    )

    expressible_count = (
        support_counts[
            "strict"
        ]
        + support_counts[
            "weak"
        ]
    )

    csv_path = (
        out_dir
        / "mission_local_inverse_beta_exact.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[
                    0
                ]
            ),
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    summary = {
        "schema":
            (
                "icra27_phase2b_mission_local_"
                "inverse_beta_exact_v0"
            ),

        "informative_decision_count":
            len(
                records
            ),

        "support_counts":
            dict(
                support_counts
            ),

        "expressible_count":
            expressible_count,

        "expressible_fraction":
            (
                expressible_count
                / len(
                    records
                )
            ),

        "mission_support":
            mission_support,

        "terrain_support":
            terrain_support,

        "source_beta_consistency":
            {
                name: {
                    "consistent":
                        source_consistency[
                            name
                        ],

                    "total":
                        source_total[
                            name
                        ],
                }
                for name
                in BETA_BANK
            },

        "semantics":
            (
                "Exact mission-local audit of whether "
                "the observed preference comparisons "
                "can be rationalized by the frozen "
                "linear beta cost family."
            ),

        "important_note":
            (
                "beta_hat is only a max-margin "
                "representative, not a unique "
                "ground-truth reward weight."
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
        "=" * 88
    )

    print(
        "ICRA27 PHASE-2B MISSION-LOCAL "
        "EXACT INVERSE-BETA V0"
    )

    print(
        "=" * 88
    )

    print(
        "informative decisions:",
        len(
            records
        ),
    )

    print(
        "strict               :",
        support_counts[
            "strict"
        ],
    )

    print(
        "weak                 :",
        support_counts[
            "weak"
        ],
    )

    print(
        "unsupported          :",
        support_counts[
            "unsupported"
        ],
    )

    print(
        "expressible fraction :",
        f"{expressible_count / len(records):.3f}",
    )

    print()
    print(
        "BY MISSION"
    )

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        item = mission_support[
            mission
        ]

        print(
            f"  {mission:<10} "
            f"n={item['count']:<4} "
            f"strict={item['strict']:<4} "
            f"weak={item['weak']:<4} "
            f"unsupported="
            f"{item['unsupported']:<4} "
            f"coverage="
            f"{item['expressible_fraction']:.3f}"
        )

    print()
    print(
        "SOURCE-BETA CONSISTENCY"
    )

    for name in BETA_BANK:
        print(
            f"  {name:<10}: "
            f"{source_consistency[name]}"
            f"/"
            f"{source_total[name]}"
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
        "[ICRA27] Phase-2B mission-local "
        "exact inverse-beta: PASS"
    )


if __name__ == "__main__":
    main()
