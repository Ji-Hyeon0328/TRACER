#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


MISSION_PRIMARY = {
    "fast":
        "decision_time_s",

    "stable":
        "cost_stability",

    "efficient":
        "cost_energy",
}


MISSION_CONSTRAINTS = {
    "fast": {
        "cost_stability",
        "cost_energy",
    },

    "stable": {
        "decision_time_s",
        "cost_energy",
    },

    "efficient": {
        "decision_time_s",
        "cost_stability",
    },
}


COST_KEYS = (
    "cost_motion",
    "cost_stability",
    "cost_energy",
)


BETA_ORDER = (
    "balanced",
    "motion",
    "stability",
    "energy",
)


TOL = 1e-10


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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mission-decisions",
        required=True,
    )

    ap.add_argument(
        "--self-assessment",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    mission_path = Path(
        args.mission_decisions
    )

    sa_path = Path(
        args.self_assessment
    )

    out_dir = Path(
        args.out_dir
    )

    for path in (
        mission_path,
        sa_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mission_rows = read_csv(
        mission_path
    )

    sa_rows = read_csv(
        sa_path
    )

    if not mission_rows:
        raise RuntimeError(
            "Mission-decision dataset is empty."
        )

    if not sa_rows:
        raise RuntimeError(
            "Self-assessment dataset is empty."
        )

    # ========================================================
    # Exact context candidate archive.
    # ========================================================

    candidates_by_context = defaultdict(
        list
    )

    for row in sa_rows:
        candidates_by_context[
            row["context_id"]
        ].append(
            row
        )

    for context_id, candidates in (
        candidates_by_context.items()
    ):
        if len(candidates) != 4:
            raise RuntimeError(
                f"{context_id}: expected 4 candidates, "
                f"got {len(candidates)}"
            )

        beta_names = {
            row["beta_name"]
            for row in candidates
        }

        if beta_names != set(
            BETA_ORDER
        ):
            raise RuntimeError(
                f"{context_id}: beta-bank mismatch: "
                f"{sorted(beta_names)}"
            )

    # ========================================================
    # First pass:
    #
    # reconstruct the mission-local feasible choice set.
    # ========================================================

    decisions = []

    for decision_index, mission_row in enumerate(
        mission_rows
    ):
        mission = mission_row[
            "mission"
        ]

        if mission not in MISSION_PRIMARY:
            raise RuntimeError(
                f"Unknown mission: {mission!r}"
            )

        context_id = mission_row[
            "context_id"
        ]

        if context_id not in candidates_by_context:
            raise RuntimeError(
                f"Missing context: {context_id}"
            )

        constraint_names = (
            mission_row[
                "constraint_1"
            ],
            mission_row[
                "constraint_2"
            ],
        )

        if set(
            constraint_names
        ) != MISSION_CONSTRAINTS[
            mission
        ]:
            raise RuntimeError(
                f"{context_id}/{mission}: "
                "constraint contract mismatch: "
                f"{constraint_names}"
            )

        constraints = []

        for slot in (
            1,
            2,
        ):
            metric = mission_row[
                f"constraint_{slot}"
            ]

            budget = float(
                mission_row[
                    f"constraint_{slot}_budget"
                ]
            )

            constraints.append(
                (
                    metric,
                    budget,
                )
            )

        good_candidates = [
            row
            for row in candidates_by_context[
                context_id
            ]
            if (
                as_bool(
                    row["feasible"]
                )
                and as_bool(
                    row[
                        "pareto_nondominated"
                    ]
                )
            )
        ]

        if not good_candidates:
            raise RuntimeError(
                f"{context_id}: no good candidates."
            )

        feasible_choice_set = []

        for candidate in good_candidates:
            satisfies = True

            for metric, budget in constraints:
                value = float(
                    candidate[
                        metric
                    ]
                )

                if value > (
                    budget
                    + TOL
                ):
                    satisfies = False
                    break

            if satisfies:
                feasible_choice_set.append(
                    candidate
                )

        if not feasible_choice_set:
            raise RuntimeError(
                f"{context_id}/{mission}: "
                "mission row has no reconstructed "
                "feasible candidates."
            )

        selected_beta = mission_row[
            "selected_source_beta_name"
        ]

        selected_matches = [
            row
            for row in feasible_choice_set
            if row[
                "beta_name"
            ] == selected_beta
        ]

        if len(selected_matches) != 1:
            raise RuntimeError(
                f"{context_id}/{mission}: selected "
                f"{selected_beta!r} appears "
                f"{len(selected_matches)} times in "
                "feasible choice set."
            )

        selected = selected_matches[
            0
        ]

        primary_metric = MISSION_PRIMARY[
            mission
        ]

        selected_primary = float(
            selected[
                primary_metric
            ]
        )

        best_primary = min(
            float(
                row[
                    primary_metric
                ]
            )
            for row in feasible_choice_set
        )

        if selected_primary > (
            best_primary
            + TOL
        ):
            raise RuntimeError(
                f"{context_id}/{mission}: selected "
                "trajectory is not primary-optimal "
                "inside reconstructed feasible set: "
                f"selected={selected_primary}, "
                f"best={best_primary}"
            )

        tied_best = [
            row
            for row in feasible_choice_set
            if abs(
                float(
                    row[
                        primary_metric
                    ]
                )
                - best_primary
            ) <= TOL
        ]

        # Mission oracle already discarded ambiguous
        # primary ties, so this should be unique.
        if len(tied_best) != 1:
            raise RuntimeError(
                f"{context_id}/{mission}: "
                "reconstructed primary optimum is "
                f"ambiguous: {len(tied_best)}"
            )

        alternatives = [
            row
            for row in feasible_choice_set
            if row[
                "beta_name"
            ] != selected_beta
        ]

        decision = {
            "decision_index":
                decision_index,

            "context_id":
                context_id,

            "group":
                mission_row[
                    "group"
                ],

            "terrain":
                mission_row[
                    "terrain"
                ],

            "seed":
                int(
                    mission_row[
                        "seed"
                    ]
                ),

            "mission":
                mission,

            "primary_metric":
                primary_metric,

            "constraint_1":
                constraints[0][0],

            "constraint_1_budget":
                constraints[0][1],

            "constraint_2":
                constraints[1][0],

            "constraint_2_budget":
                constraints[1][1],

            "selected_beta_name":
                selected_beta,

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

            "selected_primary_value":
                selected_primary,

            "good_candidate_count":
                len(
                    good_candidates
                ),

            "feasible_choice_count":
                len(
                    feasible_choice_set
                ),

            "comparison_count":
                len(
                    alternatives
                ),

            "informative":
                int(
                    len(
                        alternatives
                    ) > 0
                ),

            "_alternatives":
                alternatives,
        }

        decisions.append(
            decision
        )

    if len(decisions) != len(
        mission_rows
    ):
        raise RuntimeError(
            "Mission decision reconstruction "
            "count mismatch."
        )

    # ========================================================
    # Equal total mass per context x mission.
    #
    # First normalize decisions within each context/mission.
    # Then comparisons within each decision.
    # ========================================================

    decisions_per_group = Counter(
        (
            row[
                "context_id"
            ],
            row[
                "mission"
            ],
        )
        for row in decisions
        if row[
            "informative"
        ]
    )

    pair_records = []

    for decision in decisions:
        if not decision[
            "informative"
        ]:
            continue

        group_key = (
            decision[
                "context_id"
            ],
            decision[
                "mission"
            ],
        )

        group_n = decisions_per_group[
            group_key
        ]

        if group_n <= 0:
            raise RuntimeError(
                f"Invalid informative group: {group_key}"
            )

        decision_weight = (
            1.0
            / group_n
        )

        alternatives = decision[
            "_alternatives"
        ]

        pair_weight = (
            decision_weight
            / len(
                alternatives
            )
        )

        for pair_index, alt in enumerate(
            alternatives
        ):
            record = {
                "decision_index":
                    decision[
                        "decision_index"
                    ],

                "pair_index":
                    pair_index,

                "context_id":
                    decision[
                        "context_id"
                    ],

                "group":
                    decision[
                        "group"
                    ],

                "terrain":
                    decision[
                        "terrain"
                    ],

                "seed":
                    decision[
                        "seed"
                    ],

                "mission":
                    decision[
                        "mission"
                    ],

                "primary_metric":
                    decision[
                        "primary_metric"
                    ],

                "constraint_1":
                    decision[
                        "constraint_1"
                    ],

                "constraint_1_budget":
                    decision[
                        "constraint_1_budget"
                    ],

                "constraint_2":
                    decision[
                        "constraint_2"
                    ],

                "constraint_2_budget":
                    decision[
                        "constraint_2_budget"
                    ],

                "selected_beta_name":
                    decision[
                        "selected_beta_name"
                    ],

                "alternative_beta_name":
                    alt[
                        "beta_name"
                    ],

                "delta_cost_motion":
                    (
                        float(
                            alt[
                                "cost_motion"
                            ]
                        )
                        - decision[
                            "selected_cost_motion"
                        ]
                    ),

                "delta_cost_stability":
                    (
                        float(
                            alt[
                                "cost_stability"
                            ]
                        )
                        - decision[
                            "selected_cost_stability"
                        ]
                    ),

                "delta_cost_energy":
                    (
                        float(
                            alt[
                                "cost_energy"
                            ]
                        )
                        - decision[
                            "selected_cost_energy"
                        ]
                    ),

                "feasible_choice_count":
                    decision[
                        "feasible_choice_count"
                    ],

                "decision_weight":
                    decision_weight,

                "pair_weight":
                    pair_weight,
            }

            pair_records.append(
                record
            )

    # ========================================================
    # Save decision summary.
    # ========================================================

    decision_records = []

    for decision in decisions:
        record = {
            key:
                value
            for key, value
            in decision.items()
            if key != "_alternatives"
        }

        if record[
            "informative"
        ]:
            key = (
                record[
                    "context_id"
                ],
                record[
                    "mission"
                ],
            )

            record[
                "decision_weight"
            ] = (
                1.0
                / decisions_per_group[
                    key
                ]
            )

        else:
            record[
                "decision_weight"
            ] = 0.0

        decision_records.append(
            record
        )

    decisions_path = (
        out_dir
        / "preference_decisions.csv"
    )

    pairs_path = (
        out_dir
        / "preference_pairs.csv"
    )

    with decisions_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                decision_records[
                    0
                ]
            ),
        )

        writer.writeheader()

        writer.writerows(
            decision_records
        )

    if not pair_records:
        raise RuntimeError(
            "No informative preference pairs."
        )

    with pairs_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                pair_records[
                    0
                ]
            ),
        )

        writer.writeheader()

        writer.writerows(
            pair_records
        )

    # ========================================================
    # Diagnostics.
    # ========================================================

    informative = [
        row
        for row in decisions
        if row[
            "informative"
        ]
    ]

    singleton = (
        len(decisions)
        - len(informative)
    )

    choice_hist = Counter(
        row[
            "feasible_choice_count"
        ]
        for row in decisions
    )

    mission_decision_counts = Counter(
        row[
            "mission"
        ]
        for row in decisions
    )

    informative_mission_counts = Counter(
        row[
            "mission"
        ]
        for row in informative
    )

    pair_mission_counts = Counter(
        row[
            "mission"
        ]
        for row in pair_records
    )

    informative_groups = {
        (
            row[
                "context_id"
            ],
            row[
                "mission"
            ],
        )
        for row in informative
    }

    expected_groups = {
        (
            context_id,
            mission,
        )
        for context_id
        in candidates_by_context
        for mission
        in MISSION_PRIMARY
    }

    missing_informative_groups = sorted(
        expected_groups
        - informative_groups
    )

    # Pair weights should sum to one per informative
    # context x mission group.
    weight_sum = defaultdict(
        float
    )

    for row in pair_records:
        weight_sum[
            (
                row[
                    "context_id"
                ],
                row[
                    "mission"
                ],
            )
        ] += float(
            row[
                "pair_weight"
            ]
        )

    bad_weights = {
        str(key):
            value
        for key, value
        in weight_sum.items()
        if abs(
            value
            - 1.0
        ) > 1e-9
    }

    if bad_weights:
        raise RuntimeError(
            "Pair-weight normalization failure: "
            f"{bad_weights}"
        )

    summary = {
        "schema":
            "icra27_phase2b_preference_evidence_v0",

        "mission_decision_count":
            len(
                decisions
            ),

        "informative_decision_count":
            len(
                informative
            ),

        "singleton_no_preference_count":
            singleton,

        "preference_pair_count":
            len(
                pair_records
            ),

        "feasible_choice_size_histogram":
            {
                str(key):
                    value
                for key, value
                in sorted(
                    choice_hist.items()
                )
            },

        "mission_decision_counts":
            dict(
                mission_decision_counts
            ),

        "informative_mission_counts":
            dict(
                informative_mission_counts
            ),

        "pair_mission_counts":
            dict(
                pair_mission_counts
            ),

        "informative_context_mission_group_count":
            len(
                informative_groups
            ),

        "expected_context_mission_group_count":
            len(
                expected_groups
            ),

        "missing_informative_context_mission_groups":
            [
                {
                    "context_id":
                        context_id,

                    "mission":
                        mission,
                }
                for context_id, mission
                in missing_informative_groups
            ],

        "preference_semantics":
            (
                "selected trajectory is preferred only "
                "over alternative good trajectories "
                "that satisfy the same mission budgets"
            ),

        "important_note":
            (
                "No source beta label or inverse-beta "
                "point estimate is used as preference "
                "ground truth."
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
        "PREFERENCE EVIDENCE V0"
    )

    print(
        "=" * 88
    )

    print(
        "mission decisions       :",
        len(
            decisions
        ),
    )

    print(
        "informative decisions   :",
        len(
            informative
        ),
    )

    print(
        "singleton / no evidence :",
        singleton,
    )

    print(
        "preference pairs        :",
        len(
            pair_records
        ),
    )

    print(
        "choice-size histogram   :",
        dict(
            sorted(
                choice_hist.items()
            )
        ),
    )

    print(
        "context x mission groups:",
        len(
            informative_groups
        ),
        "/",
        len(
            expected_groups
        ),
    )

    print()
    print(
        "INFORMATIVE DECISIONS BY MISSION"
    )

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        print(
            f"  {mission:<10}: "
            f"{informative_mission_counts[mission]}"
            f"/"
            f"{mission_decision_counts[mission]}"
        )

    print()
    print(
        "PAIR COUNTS BY MISSION"
    )

    for mission in (
        "fast",
        "stable",
        "efficient",
    ):
        print(
            f"  {mission:<10}: "
            f"{pair_mission_counts[mission]}"
        )

    print()
    print(
        "decisions:",
        decisions_path,
    )

    print(
        "pairs    :",
        pairs_path,
    )

    print(
        "summary  :",
        summary_path,
    )

    print()
    print(
        "[ICRA27] Phase-2B mission-local "
        "preference evidence: PASS"
    )


if __name__ == "__main__":
    main()
