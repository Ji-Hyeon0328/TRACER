#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path


TOL = 1e-10

MISSIONS = {
    "fast": {
        "primary":
            "decision_time_s",
        "constraints": (
            "cost_stability",
            "cost_energy",
        ),
    },

    "stable": {
        "primary":
            "cost_stability",
        "constraints": (
            "decision_time_s",
            "cost_energy",
        ),
    },

    "efficient": {
        "primary":
            "cost_energy",
        "constraints": (
            "decision_time_s",
            "cost_stability",
        ),
    },
}


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def read_csv(path):
    with path.open(
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def row_key(row):
    return (
        row["context_id"],
        row["beta_name"],
    )


def unique_sorted_values(
    rows,
    metric,
):
    return sorted(
        {
            float(row[metric])
            for row in rows
        }
    )


def normalize_budget(
    value,
    values,
):
    lo = min(values)
    hi = max(values)

    if abs(
        hi - lo
    ) <= TOL:
        return 1.0

    return (
        float(value)
        - lo
    ) / (
        hi - lo
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-assessment",
        required=True,
    )

    ap.add_argument(
        "--inverse-exact",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    sa_path = Path(
        args.self_assessment
    )

    inv_path = Path(
        args.inverse_exact
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        sa_path,
        inv_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sa_rows = read_csv(
        sa_path
    )

    inv_rows = read_csv(
        inv_path
    )

    if not sa_rows:
        raise RuntimeError(
            "Self-assessment dataset is empty."
        )

    if not inv_rows:
        raise RuntimeError(
            "Inverse-beta dataset is empty."
        )

    if len(sa_rows) != len(inv_rows):
        raise RuntimeError(
            "Self-assessment/inverse row-count "
            "mismatch: "
            f"{len(sa_rows)} vs {len(inv_rows)}"
        )

    expected_beta_names = {
        "balanced",
        "motion",
        "stability",
        "energy",
    }

    actual_beta_names = {
        row["beta_name"]
        for row in sa_rows
    }

    if actual_beta_names != expected_beta_names:
        raise RuntimeError(
            "Self-assessment beta-bank mismatch: "
            f"{sorted(actual_beta_names)}"
        )

    inv_beta_names = {
        row["beta_name"]
        for row in inv_rows
    }

    if inv_beta_names != expected_beta_names:
        raise RuntimeError(
            "Inverse beta-bank mismatch: "
            f"{sorted(inv_beta_names)}"
        )

    beta_count = len(
        expected_beta_names
    )

    if len(sa_rows) % beta_count != 0:
        raise RuntimeError(
            "Self-assessment row count is not "
            "divisible by beta-bank size: "
            f"rows={len(sa_rows)}, "
            f"betas={beta_count}"
        )

    all_context_ids = {
        row["context_id"]
        for row in sa_rows
    }

    expected_context_count = len(
        all_context_ids
    )

    expected_candidate_count = (
        expected_context_count
        * beta_count
    )

    if len(sa_rows) != expected_candidate_count:
        raise RuntimeError(
            "Candidate-count contract mismatch: "
            f"expected {expected_candidate_count}, "
            f"got {len(sa_rows)}"
        )

    inv_by_key = {}

    for row in inv_rows:
        key = row_key(
            row
        )

        if key in inv_by_key:
            raise RuntimeError(
                f"Duplicate inverse row: {key}"
            )

        inv_by_key[
            key
        ] = row

    contexts = defaultdict(
        list
    )

    for row in sa_rows:
        if not (
            as_bool(
                row["feasible"]
            )
            and as_bool(
                row["pareto_nondominated"]
            )
        ):
            continue

        key = row_key(
            row
        )

        if key not in inv_by_key:
            raise RuntimeError(
                f"Missing inverse row: {key}"
            )

        contexts[
            row["context_id"]
        ].append(
            row
        )

    if len(contexts) != expected_context_count:
        missing = sorted(
            all_context_ids
            - set(contexts)
        )

        raise RuntimeError(
            "Good-trajectory context-count mismatch: "
            f"expected {expected_context_count}, "
            f"got {len(contexts)}; "
            f"missing={missing}"
        )

    records = []

    total_grid = Counter()
    no_feasible = Counter()
    ambiguous = Counter()
    resolved = Counter()
    selected_supported = Counter()
    selected_unsupported = Counter()

    selected_source = {
        name:
            Counter()
        for name in MISSIONS
    }

    for context_id in sorted(
        contexts
    ):
        good = contexts[
            context_id
        ]

        if len(good) < 2:
            raise RuntimeError(
                f"{context_id}: expected at least "
                "2 good candidates"
            )

        for mission_name, spec in (
            MISSIONS.items()
        ):
            primary = spec[
                "primary"
            ]

            c1, c2 = spec[
                "constraints"
            ]

            c1_values = (
                unique_sorted_values(
                    good,
                    c1,
                )
            )

            c2_values = (
                unique_sorted_values(
                    good,
                    c2,
                )
            )

            for b1, b2 in product(
                c1_values,
                c2_values,
            ):
                total_grid[
                    mission_name
                ] += 1

                feasible = [
                    row
                    for row in good
                    if (
                        float(
                            row[c1]
                        )
                        <= b1 + TOL
                    )
                    and (
                        float(
                            row[c2]
                        )
                        <= b2 + TOL
                    )
                ]

                if not feasible:
                    no_feasible[
                        mission_name
                    ] += 1
                    continue

                best_value = min(
                    float(
                        row[primary]
                    )
                    for row in feasible
                )

                winners = [
                    row
                    for row in feasible
                    if abs(
                        float(
                            row[primary]
                        )
                        - best_value
                    )
                    <= TOL
                ]

                if len(winners) != 1:
                    ambiguous[
                        mission_name
                    ] += 1
                    continue

                selected = winners[0]

                inv = inv_by_key[
                    row_key(
                        selected
                    )
                ]

                support_class = inv[
                    "support_class"
                ]

                expressible = (
                    support_class
                    in {
                        "strict",
                        "weak",
                    }
                )

                resolved[
                    mission_name
                ] += 1

                if expressible:
                    selected_supported[
                        mission_name
                    ] += 1
                else:
                    selected_unsupported[
                        mission_name
                    ] += 1

                selected_source[
                    mission_name
                ][
                    selected["beta_name"]
                ] += 1

                c1_norm = normalize_budget(
                    b1,
                    c1_values,
                )

                c2_norm = normalize_budget(
                    b2,
                    c2_values,
                )

                record = {
                    "context_id":
                        context_id,

                    "context_index":
                        int(
                            selected[
                                "context_index"
                            ]
                        ),

                    "group":
                        selected["group"],

                    "terrain":
                        selected["terrain"],

                    "seed":
                        int(
                            selected["seed"]
                        ),

                    "mission":
                        mission_name,

                    "primary_metric":
                        primary,

                    "constraint_1":
                        c1,

                    "constraint_1_budget":
                        float(
                            b1
                        ),

                    "constraint_1_budget_context_norm":
                        float(
                            c1_norm
                        ),

                    "constraint_2":
                        c2,

                    "constraint_2_budget":
                        float(
                            b2
                        ),

                    "constraint_2_budget_context_norm":
                        float(
                            c2_norm
                        ),

                    "feasible_good_count":
                        len(feasible),

                    "selected_source_beta_name":
                        selected[
                            "beta_name"
                        ],

                    "selected_primary_value":
                        float(
                            selected[
                                primary
                            ]
                        ),

                    "selected_cost_motion":
                        float(
                            selected[
                                "cost_motion"
                            ]
                        ),

                    "selected_cost_stability":
                        float(
                            selected[
                                "cost_stability"
                            ]
                        ),

                    "selected_cost_energy":
                        float(
                            selected[
                                "cost_energy"
                            ]
                        ),

                    "selected_decision_time_s":
                        float(
                            selected[
                                "decision_time_s"
                            ]
                        ),

                    "selected_episode_energy_abs_j":
                        float(
                            selected[
                                "episode_energy_abs_j"
                            ]
                        ),

                    "selected_support_class":
                        support_class,

                    "selected_beta_expressible":
                        expressible,

                    "beta_hat_motion":
                        float(
                            inv[
                                "beta_hat_motion"
                            ]
                        ),

                    "beta_hat_stability":
                        float(
                            inv[
                                "beta_hat_stability"
                            ]
                        ),

                    "beta_hat_energy":
                        float(
                            inv[
                                "beta_hat_energy"
                            ]
                        ),

                    "max_margin":
                        float(
                            inv[
                                "max_margin"
                            ]
                        ),
                }

                records.append(
                    record
                )

    if not records:
        raise RuntimeError(
            "No resolved mission records."
        )

    csv_path = (
        out_dir
        / "mission_decisions.csv"
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

    mode_summary = {}

    for mission_name in MISSIONS:
        n_resolved = resolved[
            mission_name
        ]

        n_supported = selected_supported[
            mission_name
        ]

        coverage = (
            n_supported
            / n_resolved
            if n_resolved
            else 0.0
        )

        mode_summary[
            mission_name
        ] = {
            "critical_budget_grid":
                int(
                    total_grid[
                        mission_name
                    ]
                ),

            "no_feasible":
                int(
                    no_feasible[
                        mission_name
                    ]
                ),

            "ambiguous":
                int(
                    ambiguous[
                        mission_name
                    ]
                ),

            "resolved":
                int(
                    n_resolved
                ),

            "selected_expressible":
                int(
                    n_supported
                ),

            "selected_unsupported":
                int(
                    selected_unsupported[
                        mission_name
                    ]
                ),

            "beta_family_coverage":
                float(
                    coverage
                ),

            "selected_source_beta_counts":
                dict(
                    selected_source[
                        mission_name
                    ]
                ),
        }

    total_resolved = sum(
        resolved.values()
    )

    total_supported = sum(
        selected_supported.values()
    )

    summary = {
        "schema":
            "icra27_phase2b_mission_oracle_v0",

        "mission_definition":
            {
                "fast":
                    (
                        "minimize decision time subject "
                        "to stability and CE budgets"
                    ),

                "stable":
                    (
                        "minimize stability cost subject "
                        "to time and CE budgets"
                    ),

                "efficient":
                    (
                        "minimize CE subject to time "
                        "and stability budgets"
                    ),
            },

        "budget_generation":
            (
                "exact critical values observed among "
                "good candidates in each paired context; "
                "no manually chosen global thresholds"
            ),

        "candidate_scope":
            (
                "all feasible Pareto-nondominated "
                "good trajectories; expressibility "
                "is checked only after mission selection"
            ),

        "context_count":
            len(contexts),

        "resolved_mission_count":
            int(
                total_resolved
            ),

        "selected_expressible_count":
            int(
                total_supported
            ),

        "overall_beta_family_coverage":
            float(
                total_supported
                / total_resolved
            ),

        "by_mission":
            mode_summary,

        "important_note":
            (
                "These are synthetic counterfactual mission "
                "decisions for controlled validation. "
                "They are not claimed to be human expert "
                "preference demonstrations."
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
        "ICRA27 PHASE-2B COUNTERFACTUAL "
        "MISSION ORACLE V0"
    )

    print(
        "=" * 88
    )

    for mission_name in MISSIONS:
        s = mode_summary[
            mission_name
        ]

        print()
        print(
            mission_name.upper()
        )

        print(
            "  critical grid       :",
            s[
                "critical_budget_grid"
            ],
        )

        print(
            "  no feasible         :",
            s[
                "no_feasible"
            ],
        )

        print(
            "  ambiguous           :",
            s[
                "ambiguous"
            ],
        )

        print(
            "  resolved            :",
            s[
                "resolved"
            ],
        )

        print(
            "  selected expressible:",
            s[
                "selected_expressible"
            ],
        )

        print(
            "  selected unsupported:",
            s[
                "selected_unsupported"
            ],
        )

        print(
            "  beta-family coverage:",
            f"{s['beta_family_coverage']:.3f}",
        )

        print(
            "  source beta counts  :",
            s[
                "selected_source_beta_counts"
            ],
        )

    print()
    print(
        "OVERALL"
    )

    print(
        "  resolved missions    :",
        total_resolved,
    )

    print(
        "  selected expressible :",
        total_supported,
    )

    print(
        "  beta-family coverage :",
        f"{total_supported / total_resolved:.3f}",
    )

    print()
    print(
        "csv                    :",
        csv_path,
    )

    print(
        "summary                :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2B counterfactual "
        "mission oracle: PASS"
    )


if __name__ == "__main__":
    main()
