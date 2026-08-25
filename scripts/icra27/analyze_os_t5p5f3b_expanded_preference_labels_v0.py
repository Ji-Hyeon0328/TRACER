from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# New 35-context physical atlas
# ============================================================

ATLAS_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
)

ATLAS_CSV = (
    ATLAS_ROOT
    / "physical_beta_response_atlas.csv"
)

ATLAS_MANIFEST = (
    ATLAS_ROOT
    / "expanded_physical_atlas_manifest.json"
)


# ============================================================
# Frozen original T5.5d labels.
#
# These are used as a regression oracle:
# the new implementation MUST reproduce all 500 original
# selections and scalarization scores before it is allowed
# to generate extension labels.
# ============================================================

OLD_LABEL_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
)

OLD_LABELS = (
    OLD_LABEL_ROOT
    / "preference_to_beta_labels.csv"
)


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3b_expanded_preference_labels_v0"
)

OUT_LABELS = (
    OUT_DIR
    / "preference_to_beta_labels.csv"
)

OUT_PARETO = (
    OUT_DIR
    / "pareto_front_rows.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_label_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "expanded_preference_labels_manifest.json"
)


EXPECTED_ATLAS_ROWS = 735
EXPECTED_CONTEXTS = 35
EXPECTED_OLD_CONTEXTS = 20
EXPECTED_BETAS = 21

EXPECTED_PREFS = 25

EXPECTED_OLD_LABELS = (
    EXPECTED_OLD_CONTEXTS
    * EXPECTED_PREFS
)

EXPECTED_LABELS = (
    EXPECTED_CONTEXTS
    * EXPECTED_PREFS
)


RHO = 0.01

EPS = 1.0e-12
TOL = 1.0e-12


OBJECTIVES = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)


REQUIRED_OLD_LABEL_COLUMNS = (
    "context_id",
    "preference_name",
    "w_motion",
    "w_stability",
    "w_energy",
    "selected_beta_name",
    "scalarization_score",
)


def read_csv(
    path: Path,
):
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        raise RuntimeError(
            f"No rows: {path}"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def finite(
    value,
):
    result = float(
        value
    )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"Non-finite value: "
            f"{value!r}"
        )

    return result


def dominates(
    a: np.ndarray,
    b: np.ndarray,
) -> bool:
    # All objectives are lower-is-better.
    #
    # a dominates b iff:
    #   a <= b component-wise
    # and strictly better in at least one component.
    return bool(
        np.all(
            a
            <= b + TOL
        )
        and np.any(
            a
            < b - TOL
        )
    )


def pareto_indices(
    values: np.ndarray,
):
    front = []

    for i in range(
        len(values)
    ):
        dominated = False

        for j in range(
            len(values)
        ):
            if i == j:
                continue

            if dominates(
                values[j],
                values[i],
            ):
                dominated = True
                break

        if not dominated:
            front.append(
                i
            )

    return front


def regret_vector(
    value: np.ndarray,
    *,
    ideal: np.ndarray,
    span: np.ndarray,
):
    regret = np.zeros(
        3,
        dtype=np.float64,
    )

    for k in range(3):
        if span[k] > EPS:
            regret[k] = (
                value[k]
                - ideal[k]
            ) / span[k]

        else:
            # Frozen T5.5d zero-span semantics.
            regret[k] = 0.0

    return regret


def scalarization_score(
    regret: np.ndarray,
    w: np.ndarray,
):
    weighted = (
        regret
        * w
    )

    return float(
        np.max(
            weighted
        )
        + RHO
        * np.sum(
            weighted
        )
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    for path in (
        ATLAS_CSV,
        ATLAS_MANIFEST,
        OLD_LABELS,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Physical atlas contract
    # ========================================================

    atlas_manifest = json.loads(
        ATLAS_MANIFEST.read_text()
    )


    if atlas_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.5f3 is not FREEZE_PASS."
        )


    if bool(
        atlas_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5f3 reports heldout use."
        )


    if int(
        atlas_manifest[
            "combined_rows"
        ]
    ) != EXPECTED_ATLAS_ROWS:
        raise RuntimeError(
            "Unexpected f3 row count."
        )


    if int(
        atlas_manifest[
            "combined_contexts"
        ]
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Unexpected f3 context count."
        )


    if int(
        atlas_manifest[
            "beta_points_per_context"
        ]
    ) != EXPECTED_BETAS:
        raise RuntimeError(
            "Unexpected beta lattice size."
        )


    context_order = list(
        atlas_manifest[
            "context_order"
        ]
    )


    if len(
        context_order
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Context-order count mismatch."
        )


    atlas = read_csv(
        ATLAS_CSV
    )


    if len(
        atlas
    ) != EXPECTED_ATLAS_ROWS:
        raise RuntimeError(
            "Physical atlas row mismatch."
        )


    rows_by_context = defaultdict(
        list
    )


    for row in atlas:
        rows_by_context[
            row[
                "context_id"
            ]
        ].append(
            row
        )


    if set(
        rows_by_context
    ) != set(
        context_order
    ):
        raise RuntimeError(
            "Context set does not match "
            "f3 manifest."
        )


    # Frozen beta order comes from the first context.
    first_context = (
        context_order[
            0
        ]
    )


    beta_order = [
        row[
            "beta_name"
        ]
        for row in (
            rows_by_context[
                first_context
            ]
        )
    ]


    if (
        len(
            beta_order
        ) != EXPECTED_BETAS
        or len(
            set(
                beta_order
            )
        ) != EXPECTED_BETAS
    ):
        raise RuntimeError(
            "Expected 21 unique beta points."
        )


    beta_rank = {
        name:
            index
        for index, name in enumerate(
            beta_order
        )
    }


    for cid in context_order:
        rows = (
            rows_by_context[
                cid
            ]
        )

        if len(
            rows
        ) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{cid}: expected "
                f"{EXPECTED_BETAS} rows."
            )


        names = [
            row[
                "beta_name"
            ]
            for row in rows
        ]


        if names != beta_order:
            raise RuntimeError(
                f"{cid}: beta order differs "
                "from frozen atlas order."
            )


    # ========================================================
    # Existing T5.5d preference bank
    # ========================================================

    old_labels = read_csv(
        OLD_LABELS
    )


    if len(
        old_labels
    ) != EXPECTED_OLD_LABELS:
        raise RuntimeError(
            f"Expected 500 old labels; "
            f"got {len(old_labels)}"
        )


    if not old_labels:
        raise RuntimeError(
            "Old labels are empty."
        )


    for column in (
        REQUIRED_OLD_LABEL_COLUMNS
    ):
        if column not in old_labels[
            0
        ]:
            raise RuntimeError(
                "Missing old-label column: "
                f"{column}"
            )


    # The first frozen context defines exact preference-bank
    # order and names. We do not recreate or rename them.
    preference_rows = [
        row
        for row in old_labels
        if row[
            "context_id"
        ] == first_context
    ]


    if len(
        preference_rows
    ) != EXPECTED_PREFS:
        raise RuntimeError(
            "Expected 25 preference rows "
            "for first frozen context."
        )


    preference_bank = []


    for row in preference_rows:
        preference_bank.append(
            {
                "preference_name":
                    row[
                        "preference_name"
                    ],

                "w_motion":
                    finite(
                        row[
                            "w_motion"
                        ]
                    ),

                "w_stability":
                    finite(
                        row[
                            "w_stability"
                        ]
                    ),

                "w_energy":
                    finite(
                        row[
                            "w_energy"
                        ]
                    ),
            }
        )


    preference_names = [
        row[
            "preference_name"
        ]
        for row in preference_bank
    ]


    if len(
        set(
            preference_names
        )
    ) != EXPECTED_PREFS:
        raise RuntimeError(
            "Preference names are not unique."
        )


    preference_weight_triples = [
        (
            row[
                "w_motion"
            ],
            row[
                "w_stability"
            ],
            row[
                "w_energy"
            ],
        )
        for row in preference_bank
    ]


    if len(
        set(
            preference_weight_triples
        )
    ) != EXPECTED_PREFS:
        raise RuntimeError(
            "Preference weights are not "
            "25 unique triples."
        )


    # Verify every old context used the exact same bank.
    old_context_order = (
        context_order[
            :EXPECTED_OLD_CONTEXTS
        ]
    )


    old_labels_by_context = defaultdict(
        list
    )


    for row in old_labels:
        old_labels_by_context[
            row[
                "context_id"
            ]
        ].append(
            row
        )


    if set(
        old_labels_by_context
    ) != set(
        old_context_order
    ):
        raise RuntimeError(
            "Old T5.5d context set does not "
            "match first 20 f3 contexts."
        )


    for cid in old_context_order:
        rows = (
            old_labels_by_context[
                cid
            ]
        )


        if len(
            rows
        ) != EXPECTED_PREFS:
            raise RuntimeError(
                f"{cid}: expected 25 old labels."
            )


        actual = [
            (
                row[
                    "preference_name"
                ],
                finite(
                    row[
                        "w_motion"
                    ]
                ),
                finite(
                    row[
                        "w_stability"
                    ]
                ),
                finite(
                    row[
                        "w_energy"
                    ]
                ),
            )
            for row in rows
        ]


        expected = [
            (
                row[
                    "preference_name"
                ],
                row[
                    "w_motion"
                ],
                row[
                    "w_stability"
                ],
                row[
                    "w_energy"
                ],
            )
            for row in preference_bank
        ]


        if actual != expected:
            raise RuntimeError(
                f"{cid}: preference-bank "
                "order/content differs from "
                "first T5.5d context."
            )


    # ========================================================
    # Compute all 35 context-wise Pareto sets and labels.
    # ========================================================

    labels = []

    pareto_rows = []

    context_rows = []


    old_regression_exact = 0

    old_regression_total = 0

    max_old_score_error = 0.0


    for context_index, cid in enumerate(
        context_order
    ):
        rows = (
            rows_by_context[
                cid
            ]
        )


        values = np.asarray(
            [
                [
                    finite(
                        row[
                            column
                        ]
                    )
                    for column in OBJECTIVES
                ]
                for row in rows
            ],
            dtype=np.float64,
        )


        front_indices = (
            pareto_indices(
                values
            )
        )


        if not front_indices:
            raise RuntimeError(
                f"{cid}: empty Pareto front."
            )


        front_values = (
            values[
                front_indices
            ]
        )


        ideal = np.min(
            front_values,
            axis=0,
        )


        nadir = np.max(
            front_values,
            axis=0,
        )


        span = (
            nadir
            - ideal
        )


        regrets = np.asarray(
            [
                regret_vector(
                    value,
                    ideal=ideal,
                    span=span,
                )
                for value in values
            ],
            dtype=np.float64,
        )


        # ----------------------------------------------------
        # Save Pareto evidence.
        # ----------------------------------------------------

        for index in front_indices:
            row = rows[
                index
            ]

            pareto_rows.append(
                {
                    "context_id":
                        cid,

                    "context_index":
                        context_index,

                    "beta_name":
                        row[
                            "beta_name"
                        ],

                    "beta_motion":
                        finite(
                            row[
                                "beta_motion"
                            ]
                        ),

                    "beta_stability":
                        finite(
                            row[
                                "beta_stability"
                            ]
                        ),

                    "beta_energy":
                        finite(
                            row[
                                "beta_energy"
                            ]
                        ),

                    "lambda_motion":
                        finite(
                            row[
                                "lambda_motion"
                            ]
                        ),

                    "lambda_stability":
                        finite(
                            row[
                                "lambda_stability"
                            ]
                        ),

                    "lambda_energy":
                        finite(
                            row[
                                "lambda_energy"
                            ]
                        ),

                    "J_motion_s_per_m":
                        values[
                            index,
                            0
                        ],

                    "J_stability":
                        values[
                            index,
                            1
                        ],

                    "J_energy_j_per_m":
                        values[
                            index,
                            2
                        ],

                    "regret_motion":
                        regrets[
                            index,
                            0
                        ],

                    "regret_stability":
                        regrets[
                            index,
                            1
                        ],

                    "regret_energy":
                        regrets[
                            index,
                            2
                        ],
                }
            )


        old_lookup = {}


        if cid in (
            old_labels_by_context
        ):
            old_lookup = {
                row[
                    "preference_name"
                ]:
                    row
                for row in (
                    old_labels_by_context[
                        cid
                    ]
                )
            }


        selected_counts = Counter()


        for preference_index, pref in enumerate(
            preference_bank
        ):
            w = np.asarray(
                [
                    pref[
                        "w_motion"
                    ],
                    pref[
                        "w_stability"
                    ],
                    pref[
                        "w_energy"
                    ],
                ],
                dtype=np.float64,
            )


            candidates = []


            for beta_index in (
                front_indices
            ):
                score = (
                    scalarization_score(
                        regrets[
                            beta_index
                        ],
                        w,
                    )
                )


                candidates.append(
                    (
                        score,
                        beta_rank[
                            rows[
                                beta_index
                            ][
                                "beta_name"
                            ]
                        ],
                        beta_index,
                    )
                )


            # Deterministic frozen lattice-order tie-break.
            candidates.sort(
                key=lambda item: (
                    item[
                        0
                    ],
                    item[
                        1
                    ],
                )
            )


            selected_score = (
                candidates[
                    0
                ][
                    0
                ]
            )


            selected_index = (
                candidates[
                    0
                ][
                    2
                ]
            )


            selected = (
                rows[
                    selected_index
                ]
            )


            selected_regret = (
                regrets[
                    selected_index
                ]
            )


            selected_beta_name = (
                selected[
                    "beta_name"
                ]
            )


            selected_counts[
                selected_beta_name
            ] += 1


            # ------------------------------------------------
            # Old T5.5d exact regression guard.
            # ------------------------------------------------

            if old_lookup:
                old = old_lookup[
                    pref[
                        "preference_name"
                    ]
                ]


                old_name = (
                    old[
                        "selected_beta_name"
                    ]
                )


                old_score = finite(
                    old[
                        "scalarization_score"
                    ]
                )


                score_error = abs(
                    selected_score
                    - old_score
                )


                max_old_score_error = max(
                    max_old_score_error,
                    score_error,
                )


                old_regression_total += 1


                if (
                    selected_beta_name
                    == old_name
                    and score_error
                    <= 1.0e-10
                ):
                    old_regression_exact += 1

                else:
                    raise RuntimeError(
                        "T5.5d regression mismatch:\n"
                        f"  context={cid}\n"
                        f"  preference="
                        f"{pref['preference_name']}\n"
                        f"  expected beta={old_name}\n"
                        f"  computed beta="
                        f"{selected_beta_name}\n"
                        f"  expected score="
                        f"{old_score:.17g}\n"
                        f"  computed score="
                        f"{selected_score:.17g}\n"
                        f"  error="
                        f"{score_error:.3e}"
                    )


            labels.append(
                {
                    "context_id":
                        cid,

                    "context_index":
                        context_index,

                    "preference_name":
                        pref[
                            "preference_name"
                        ],

                    "preference_index":
                        preference_index,

                    "w_motion":
                        pref[
                            "w_motion"
                        ],

                    "w_stability":
                        pref[
                            "w_stability"
                        ],

                    "w_energy":
                        pref[
                            "w_energy"
                        ],

                    "selected_beta_name":
                        selected_beta_name,

                    "selected_beta_motion":
                        finite(
                            selected[
                                "beta_motion"
                            ]
                        ),

                    "selected_beta_stability":
                        finite(
                            selected[
                                "beta_stability"
                            ]
                        ),

                    "selected_beta_energy":
                        finite(
                            selected[
                                "beta_energy"
                            ]
                        ),

                    "selected_lambda_motion":
                        finite(
                            selected[
                                "lambda_motion"
                            ]
                        ),

                    "selected_lambda_stability":
                        finite(
                            selected[
                                "lambda_stability"
                            ]
                        ),

                    "selected_lambda_energy":
                        finite(
                            selected[
                                "lambda_energy"
                            ]
                        ),

                    "selected_J_motion_s_per_m":
                        values[
                            selected_index,
                            0
                        ],

                    "selected_J_stability":
                        values[
                            selected_index,
                            1
                        ],

                    "selected_J_energy_j_per_m":
                        values[
                            selected_index,
                            2
                        ],

                    "selected_regret_motion":
                        selected_regret[
                            0
                        ],

                    "selected_regret_stability":
                        selected_regret[
                            1
                        ],

                    "selected_regret_energy":
                        selected_regret[
                            2
                        ],

                    "scalarization_score":
                        selected_score,

                    "pareto_count":
                        len(
                            front_indices
                        ),

                    "pareto_ideal_motion":
                        ideal[
                            0
                        ],

                    "pareto_ideal_stability":
                        ideal[
                            1
                        ],

                    "pareto_ideal_energy":
                        ideal[
                            2
                        ],

                    "pareto_nadir_motion":
                        nadir[
                            0
                        ],

                    "pareto_nadir_stability":
                        nadir[
                            1
                        ],

                    "pareto_nadir_energy":
                        nadir[
                            2
                        ],

                    "pareto_span_motion":
                        span[
                            0
                        ],

                    "pareto_span_stability":
                        span[
                            1
                        ],

                    "pareto_span_energy":
                        span[
                            2
                        ],

                    "source_stage":
                        (
                            "T5.5d_frozen_regression"
                            if old_lookup
                            else
                            "T5.5f3b_extension"
                        ),
                }
            )


        context_rows.append(
            {
                "context_id":
                    cid,

                "context_index":
                    context_index,

                "source_stage":
                    (
                        "T5.5c_frozen"
                        if context_index
                        < EXPECTED_OLD_CONTEXTS
                        else
                        "T5.5f3_extension"
                    ),

                "beta_points":
                    EXPECTED_BETAS,

                "pareto_count":
                    len(
                        front_indices
                    ),

                "pareto_span_motion":
                    span[
                        0
                    ],

                "pareto_span_stability":
                    span[
                        1
                    ],

                "pareto_span_energy":
                    span[
                        2
                    ],

                "distinct_selected_betas":
                    len(
                        selected_counts
                    ),

                "selected_beta_histogram":
                    json.dumps(
                        dict(
                            sorted(
                                selected_counts.items(),
                                key=lambda item:
                                    beta_rank[
                                        item[
                                            0
                                        ]
                                    ],
                            )
                        ),
                        sort_keys=False,
                    ),
            }
        )


    # ========================================================
    # Final gates
    # ========================================================

    if len(
        labels
    ) != EXPECTED_LABELS:
        raise RuntimeError(
            f"Expected 875 labels; "
            f"got {len(labels)}"
        )


    if old_regression_total != (
        EXPECTED_OLD_LABELS
    ):
        raise RuntimeError(
            "Expected 500 old regression "
            "comparisons."
        )


    if old_regression_exact != (
        EXPECTED_OLD_LABELS
    ):
        raise RuntimeError(
            "Old T5.5d regression was not "
            "500/500 exact."
        )


    labels_per_context = Counter(
        row[
            "context_id"
        ]
        for row in labels
    )


    if any(
        count != EXPECTED_PREFS
        for count in (
            labels_per_context.values()
        )
    ):
        raise RuntimeError(
            "Every context must have "
            "exactly 25 labels."
        )


    # ========================================================
    # Write only after all regression gates pass.
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    write_csv(
        OUT_LABELS,
        labels,
    )

    write_csv(
        OUT_PARETO,
        pareto_rows,
    )

    write_csv(
        OUT_CONTEXT,
        context_rows,
    )


    pareto_counts = [
        int(
            row[
                "pareto_count"
            ]
        )
        for row in context_rows
    ]


    extension_contexts = (
        context_order[
            EXPECTED_OLD_CONTEXTS:
        ]
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5f3b_"
                "expanded_preference_labels_v0"
            ),

        "status":
            "FREEZE_PASS",

        "heldout_used":
            False,

        "original_split_modified":
            False,

        "physical_atlas_source":
            str(
                ATLAS_CSV.relative_to(
                    ROOT
                )
            ),

        "frozen_label_regression_source":
            str(
                OLD_LABELS.relative_to(
                    ROOT
                )
            ),

        "contexts":
            EXPECTED_CONTEXTS,

        "old_contexts":
            EXPECTED_OLD_CONTEXTS,

        "extension_contexts":
            len(
                extension_contexts
            ),

        "extension_context_ids":
            extension_contexts,

        "beta_points_per_context":
            EXPECTED_BETAS,

        "preferences_per_context":
            EXPECTED_PREFS,

        "labels":
            EXPECTED_LABELS,

        "preference_bank_source":
            (
                "Exact ordered 25-preference "
                "bank recovered from frozen "
                "T5.5d labels; not regenerated "
                "or renamed."
            ),

        "rho":
            RHO,

        "scalarization":
            (
                "context-local Pareto regret "
                "r_k=(J_k-ideal_k)/(nadir_k-ideal_k), "
                "zero span -> 0; "
                "S=max_k(w_k r_k) + "
                "rho * sum_k(w_k r_k)"
            ),

        "objective_direction":
            "all_lower_is_better",

        "selection_domain":
            "context-local Pareto front only",

        "tie_break":
            "frozen 21-beta atlas order",

        "old_t5p5d_regression":
            {
                "rows":
                    old_regression_total,

                "exact_selected_beta":
                    old_regression_exact,

                "max_scalarization_score_error":
                    max_old_score_error,

                "status":
                    "PASS",
            },

        "pareto_count_range":
            [
                min(
                    pareto_counts
                ),
                max(
                    pareto_counts
                ),
            ],

        "normalization_policy":
            (
                "No global objective "
                "normalization is introduced. "
                "Only context-local Pareto "
                "ideal/nadir regret is used."
            ),

        "interpretation_guard":
            (
                "TRAIN-only label expansion. "
                "The 15 extension contexts are "
                "exactly the full-21-beta feasible "
                "contexts frozen in T5.5f2r. "
                "Seeds 34, 43, 46 remain preserved "
                "as feasibility evidence and are "
                "not silently resampled."
            ),

        "next_stage":
            (
                "T5.5f4: rerun the identical "
                "nonlinear log-regret response "
                "selector LOCO on the same "
                "18 original held rough contexts, "
                "but train each fold with "
                "17 original rough + "
                "15 extension rough contexts."
            ),
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()
    print("=" * 120)

    print(
        "ICRA27 OS-T5.5f3b EXPANDED "
        "PREFERENCE LABEL FREEZE"
    )

    print("=" * 120)

    print(
        "contexts                 :",
        EXPECTED_CONTEXTS,
    )

    print(
        "preferences/context      :",
        EXPECTED_PREFS,
    )

    print(
        "labels                   :",
        len(
            labels
        ),
    )

    print(
        "Pareto rows              :",
        len(
            pareto_rows
        ),
    )

    print(
        "Pareto count range       :",
        [
            min(
                pareto_counts
            ),
            max(
                pareto_counts
            ),
        ],
    )

    print()
    print("FROZEN T5.5d REGRESSION")

    print(
        "  rows compared          :",
        old_regression_total,
    )

    print(
        "  exact selected beta    :",
        old_regression_exact,
        "/",
        EXPECTED_OLD_LABELS,
    )

    print(
        "  max score error        :",
        max_old_score_error,
    )

    print()
    print("EXTENSION CONTEXTS")

    for cid in extension_contexts:
        summary = next(
            row
            for row in context_rows
            if row[
                "context_id"
            ] == cid
        )

        print(
            f"  {cid:<16} "
            f"front="
            f"{summary['pareto_count']:<2} "
            f"selectedBetas="
            f"{summary['distinct_selected_betas']}"
        )

    print()
    print("outputs:")
    print(" ", OUT_LABELS)
    print(" ", OUT_PARETO)
    print(" ", OUT_CONTEXT)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5f3b expanded "
        "preference labels: FREEZE PASS"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
