#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BETA_ORDER = (
    "balanced",
    "motion",
    "stability",
    "energy",
)

TERRAIN_ORDER = (
    "flat",
    "low_friction",
    "rough_perlin",
)

MISSION_ORDER = (
    "fast",
    "stable",
    "efficient",
)

COST_KEYS = (
    "cost_motion",
    "cost_stability",
    "cost_energy",
)

CONSTRAINT_TO_BUDGET = {
    "decision_time_s":
        "time",

    "cost_stability":
        "stability",

    "cost_energy":
        "energy",
}

BUDGET_ORDER = (
    "time",
    "stability",
    "energy",
)


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


def normalize(
    value,
    lo,
    hi,
):
    if hi <= lo:
        raise RuntimeError(
            "Invalid normalization range: "
            f"lo={lo}, hi={hi}"
        )

    return (
        float(value)
        - lo
    ) / (
        hi - lo
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

    # --------------------------------------------------------
    # Self-assessment candidates by exact context.
    # --------------------------------------------------------

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

        if len(candidates) != 4:
            raise RuntimeError(
                f"{context_id}: expected 4 candidates, "
                f"got {len(candidates)}"
            )

    # --------------------------------------------------------
    # TRAIN-only physical-budget normalization.
    #
    # IMPORTANT:
    # use all resolved TRAIN mission queries,
    # including unsupported selections.
    #
    # No VAL/TEST information enters these scales.
    # --------------------------------------------------------

    budget_values = {
        name: []
        for name in BUDGET_ORDER
    }

    parsed_budget_by_row = {}

    for index, row in enumerate(
        mission_rows
    ):
        raw = {
            name: None
            for name in BUDGET_ORDER
        }

        active = {
            name: 0
            for name in BUDGET_ORDER
        }

        for slot in (
            1,
            2,
        ):
            constraint = row[
                f"constraint_{slot}"
            ]

            if (
                constraint
                not in CONSTRAINT_TO_BUDGET
            ):
                raise RuntimeError(
                    "Unsupported constraint metric: "
                    f"{constraint!r}"
                )

            dim = (
                CONSTRAINT_TO_BUDGET[
                    constraint
                ]
            )

            value = float(
                row[
                    f"constraint_{slot}_budget"
                ]
            )

            if active[dim]:
                raise RuntimeError(
                    f"Duplicate active budget dim "
                    f"{dim!r}"
                )

            raw[dim] = value
            active[dim] = 1

            budget_values[
                dim
            ].append(
                value
            )

        if sum(
            active.values()
        ) != 2:
            raise RuntimeError(
                "Expected exactly two active "
                f"mission budgets; got {active}"
            )

        parsed_budget_by_row[
            index
        ] = (
            raw,
            active,
        )

    normalization = {}

    for dim in BUDGET_ORDER:
        values = budget_values[
            dim
        ]

        if not values:
            raise RuntimeError(
                f"No TRAIN budget values for {dim}"
            )

        lo = min(
            values
        )

        hi = max(
            values
        )

        if hi <= lo:
            raise RuntimeError(
                f"Degenerate TRAIN scale for {dim}: "
                f"{lo} ... {hi}"
            )

        normalization[
            dim
        ] = {
            "min":
                float(lo),

            "max":
                float(hi),
        }

    # --------------------------------------------------------
    # Keep only mission choices representable by the frozen
    # linear-beta preference interface.
    # --------------------------------------------------------

    selected_rows = []

    for index, row in enumerate(
        mission_rows
    ):
        expressible = as_bool(
            row[
                "selected_beta_expressible"
            ]
        )

        if not expressible:
            continue

        if row[
            "selected_support_class"
        ] not in {
            "strict",
            "weak",
        }:
            raise RuntimeError(
                "Expressible mission row has invalid "
                "support class: "
                f"{row['selected_support_class']}"
            )

        selected_rows.append(
            (
                index,
                row,
            )
        )

    if not selected_rows:
        raise RuntimeError(
            "No beta-expressible mission rows."
        )

    # --------------------------------------------------------
    # Equal total training mass for each:
    #
    #   context x mission-family.
    #
    # This prevents a context from receiving larger total
    # weight merely because its critical-budget grid contains
    # more expressible points.
    # --------------------------------------------------------

    group_counts = Counter(
        (
            row["context_id"],
            row["mission"],
        )
        for _index, row in selected_rows
    )

    records = []

    for index, mission_row in selected_rows:
        context_id = mission_row[
            "context_id"
        ]

        mission = mission_row[
            "mission"
        ]

        terrain = mission_row[
            "terrain"
        ]

        if mission not in MISSION_ORDER:
            raise RuntimeError(
                f"Unknown mission: {mission!r}"
            )

        if terrain not in TERRAIN_ORDER:
            raise RuntimeError(
                f"Unknown terrain: {terrain!r}"
            )

        candidates = (
            candidates_by_context[
                context_id
            ]
        )

        selected_beta_name = (
            mission_row[
                "selected_source_beta_name"
            ]
        )

        selected_matches = [
            row
            for row in candidates
            if (
                row["beta_name"]
                == selected_beta_name
            )
        ]

        if len(selected_matches) != 1:
            raise RuntimeError(
                f"{context_id}: selected candidate "
                f"{selected_beta_name!r} count="
                f"{len(selected_matches)}"
            )

        selected = selected_matches[
            0
        ]

        if not (
            as_bool(
                selected["feasible"]
            )
            and as_bool(
                selected[
                    "pareto_nondominated"
                ]
            )
        ):
            raise RuntimeError(
                f"{context_id}/{selected_beta_name}: "
                "mission selected a non-good candidate"
            )

        alternatives = [
            row
            for row in candidates
            if (
                row["beta_name"]
                != selected_beta_name
            )
        ]

        if len(alternatives) != 3:
            raise RuntimeError(
                f"{context_id}: expected 3 alternatives, "
                f"got {len(alternatives)}"
            )

        alternatives = sorted(
            alternatives,
            key=lambda row:
                BETA_ORDER.index(
                    row["beta_name"]
                ),
        )

        raw_budget, active = (
            parsed_budget_by_row[
                index
            ]
        )

        budget_norm = {}

        for dim in BUDGET_ORDER:
            if not active[
                dim
            ]:
                budget_norm[
                    dim
                ] = 0.0
                continue

            scale = normalization[
                dim
            ]

            budget_norm[
                dim
            ] = normalize(
                raw_budget[
                    dim
                ],
                scale[
                    "min"
                ],
                scale[
                    "max"
                ],
            )

        terrain_one_hot = {
            name:
                int(
                    terrain == name
                )
            for name in TERRAIN_ORDER
        }

        mission_one_hot = {
            name:
                int(
                    mission == name
                )
            for name in MISSION_ORDER
        }

        group_key = (
            context_id,
            mission,
        )

        n_in_group = group_counts[
            group_key
        ]

        if n_in_group <= 0:
            raise RuntimeError(
                f"Invalid group count for {group_key}"
            )

        sample_weight = (
            1.0
            / n_in_group
        )

        record = {
            "context_id":
                context_id,

            "group":
                mission_row[
                    "group"
                ],

            "terrain":
                terrain,

            "seed":
                int(
                    mission_row[
                        "seed"
                    ]
                ),

            "mission":
                mission,

            "terrain_flat":
                terrain_one_hot[
                    "flat"
                ],

            "terrain_low_friction":
                terrain_one_hot[
                    "low_friction"
                ],

            "terrain_rough_perlin":
                terrain_one_hot[
                    "rough_perlin"
                ],

            "mission_fast":
                mission_one_hot[
                    "fast"
                ],

            "mission_stable":
                mission_one_hot[
                    "stable"
                ],

            "mission_efficient":
                mission_one_hot[
                    "efficient"
                ],

            "budget_time_raw":
                (
                    ""
                    if raw_budget[
                        "time"
                    ] is None
                    else float(
                        raw_budget[
                            "time"
                        ]
                    )
                ),

            "budget_stability_raw":
                (
                    ""
                    if raw_budget[
                        "stability"
                    ] is None
                    else float(
                        raw_budget[
                            "stability"
                        ]
                    )
                ),

            "budget_energy_raw":
                (
                    ""
                    if raw_budget[
                        "energy"
                    ] is None
                    else float(
                        raw_budget[
                            "energy"
                        ]
                    )
                ),

            "budget_time_norm":
                float(
                    budget_norm[
                        "time"
                    ]
                ),

            "budget_stability_norm":
                float(
                    budget_norm[
                        "stability"
                    ]
                ),

            "budget_energy_norm":
                float(
                    budget_norm[
                        "energy"
                    ]
                ),

            "budget_time_active":
                int(
                    active[
                        "time"
                    ]
                ),

            "budget_stability_active":
                int(
                    active[
                        "stability"
                    ]
                ),

            "budget_energy_active":
                int(
                    active[
                        "energy"
                    ]
                ),

            "selected_source_beta_name":
                selected_beta_name,

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

            "selected_support_class":
                mission_row[
                    "selected_support_class"
                ],

            "sample_weight":
                float(
                    sample_weight
                ),
        }

        for alt_index, alt in enumerate(
            alternatives
        ):
            record[
                f"alt{alt_index}_beta_name"
            ] = alt[
                "beta_name"
            ]

            for short, key in (
                (
                    "cm",
                    "cost_motion",
                ),
                (
                    "cs",
                    "cost_stability",
                ),
                (
                    "ce",
                    "cost_energy",
                ),
            ):
                record[
                    f"alt{alt_index}_d{short}"
                ] = (
                    float(
                        alt[
                            key
                        ]
                    )
                    - float(
                        selected[
                            key
                        ]
                    )
                )

        records.append(
            record
        )

    # --------------------------------------------------------
    # Contract checks.
    # --------------------------------------------------------

    if len(records) != len(
        selected_rows
    ):
        raise RuntimeError(
            "Supervision row-count mismatch."
        )

    weight_sum_by_group = defaultdict(
        float
    )

    for row in records:
        weight_sum_by_group[
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
                "sample_weight"
            ]
        )

    bad_weight_groups = {
        key:
            value
        for key, value
        in weight_sum_by_group.items()
        if abs(
            value
            - 1.0
        ) > 1e-9
    }

    if bad_weight_groups:
        raise RuntimeError(
            "Context-mission weights do not "
            "sum to one: "
            f"{bad_weight_groups}"
        )

    csv_path = (
        out_dir
        / "selector_supervision.csv"
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

    mission_counts = Counter(
        row["mission"]
        for row in records
    )

    terrain_counts = Counter(
        row["terrain"]
        for row in records
    )

    selected_beta_counts = Counter(
        row[
            "selected_source_beta_name"
        ]
        for row in records
    )

    context_mission_counts = Counter(
        (
            row["context_id"],
            row["mission"],
        )
        for row in records
    )

    zero_supervision_groups = []

    expected_contexts = sorted(
        candidates_by_context
    )

    for context_id in expected_contexts:
        for mission in MISSION_ORDER:
            if (
                context_id,
                mission,
            ) not in context_mission_counts:
                zero_supervision_groups.append(
                    (
                        context_id,
                        mission,
                    )
                )

    summary = {
        "schema":
            "icra27_phase2b_selector_supervision_v0",

        "selector_input_dim":
            12,

        "selector_input":
            (
                "terrain3 + mission3 + "
                "normalized_budget3 + "
                "budget_active_mask3"
            ),

        "training_target":
            (
                "support-region inequalities "
                "beta dot (C_alt - C_selected) >= 0"
            ),

        "mission_decision_count":
            len(
                mission_rows
            ),

        "expressible_supervision_count":
            len(
                records
            ),

        "context_count":
            len(
                candidates_by_context
            ),

        "context_mission_group_count":
            len(
                context_mission_counts
            ),

        "zero_supervision_context_mission_groups":
            [
                {
                    "context_id":
                        context_id,

                    "mission":
                        mission,
                }
                for context_id, mission
                in zero_supervision_groups
            ],

        "mission_counts":
            dict(
                mission_counts
            ),

        "terrain_counts":
            dict(
                terrain_counts
            ),

        "selected_source_beta_counts":
            dict(
                selected_beta_counts
            ),

        "budget_normalization_train_only":
            normalization,

        "weighting":
            (
                "each context x mission family "
                "has total sample weight 1"
            ),

        "important_note":
            (
                "No max-margin beta_hat point is "
                "used as a regression target."
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
        "ICRA27 PHASE-2B SELECTOR "
        "SUPERVISION DATASET V0"
    )

    print(
        "=" * 88
    )

    print(
        "mission decisions       :",
        len(
            mission_rows
        ),
    )

    print(
        "expressible supervision :",
        len(
            records
        ),
    )

    print(
        "contexts                :",
        len(
            candidates_by_context
        ),
    )

    print(
        "context x mission groups:",
        len(
            context_mission_counts
        ),
    )

    print(
        "zero-supervision groups :",
        len(
            zero_supervision_groups
        ),
    )

    print()
    print(
        "MISSION COUNTS"
    )

    for mission in MISSION_ORDER:
        print(
            f"  {mission:<10}: "
            f"{mission_counts[mission]}"
        )

    print()
    print(
        "TERRAIN COUNTS"
    )

    for terrain in TERRAIN_ORDER:
        print(
            f"  {terrain:<14}: "
            f"{terrain_counts[terrain]}"
        )

    print()
    print(
        "TRAIN-ONLY BUDGET NORMALIZATION"
    )

    for dim in BUDGET_ORDER:
        print(
            f"  {dim:<10}: "
            f"{normalization[dim]}"
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
        "[ICRA27] Phase-2B selector "
        "supervision dataset: PASS"
    )


if __name__ == "__main__":
    main()
