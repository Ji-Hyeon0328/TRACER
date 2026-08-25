from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


ATLAS = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)

LABELS = (
    ROOT
    / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
    / "preference_to_beta_labels.csv"
)

V1_TRANSFER = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1b_context_identifiability_v0"
    / "nearest_context_preference_transfer.csv"
)

V2_TRANSFER = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1d_context_identifiability_v2_v0"
    / "nearest_context_preference_transfer.csv"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1g_robust_scalarization_shadow_v0"
)

OUT_FLOORS = (
    OUT_DIR
    / "span_floor_candidates.csv"
)

OUT_LABELS = (
    OUT_DIR
    / "shadow_label_assignments.csv"
)

OUT_VARIANTS = (
    OUT_DIR
    / "shadow_variant_summary.csv"
)

OUT_TRANSFER = (
    OUT_DIR
    / "shadow_transfer_summary.csv"
)

OUT_TRANSFER_ROWS = (
    OUT_DIR
    / "shadow_transfer_rows.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "shadow_scalarization_manifest.json"
)


OBJECTIVES = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

OBJECTIVE_NAMES = (
    "motion",
    "stability",
    "energy",
)


RHO = 0.01
TOL = 1.0e-10
EPS = 1.0e-15


FLOOR_QUANTILES = (
    (
        "q10",
        0.10,
    ),
    (
        "q25",
        0.25,
    ),
    (
        "q50",
        0.50,
    ),
)


EXPECTED_CONTEXTS = 20
EXPECTED_ROUGH = 18
EXPECTED_BETAS = 21
EXPECTED_PREFERENCES = 25
EXPECTED_LABELS = 500
EXPECTED_ROUGH_LABELS = 450


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite: {x!r}"
        )

    return y


def read_csv(path):
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
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
                fields.append(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def pareto(
    rows,
):
    values = np.asarray(
        [
            [
                finite(
                    row[column]
                )
                for column in OBJECTIVES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    keep = []

    for i in range(
        len(rows)
    ):
        dominated = False

        for j in range(
            len(rows)
        ):
            if i == j:
                continue

            no_worse = np.all(
                values[j]
                <= values[i]
                + TOL
            )

            strictly_better = np.any(
                values[j]
                < values[i]
                - TOL
            )

            if (
                no_worse
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            keep.append(i)

    return [
        rows[i]
        for i in keep
    ]


def summarize(
    values,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean":
            float(
                np.mean(x)
            ),

        "median":
            float(
                np.median(x)
            ),

        "p95":
            float(
                np.quantile(
                    x,
                    0.95,
                )
            ),

        "max":
            float(
                np.max(x)
            ),
    }


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        ATLAS,
        LABELS,
        V1_TRANSFER,
        V2_TRANSFER,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    atlas = read_csv(
        ATLAS
    )

    frozen_labels = read_csv(
        LABELS
    )

    if len(
        frozen_labels
    ) != EXPECTED_LABELS:
        raise RuntimeError(
            "Expected 500 frozen labels."
        )


    # ========================================================
    # Context / beta inventory
    # ========================================================

    context_order = []

    by_context = defaultdict(
        list
    )

    for row in atlas:
        cid = row[
            "context_id"
        ]

        if cid not in (
            context_order
        ):
            context_order.append(
                cid
            )

        by_context[
            cid
        ].append(
            row
        )


    if len(
        context_order
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Expected 20 contexts."
        )


    rough_contexts = [
        cid
        for cid in context_order
        if cid.startswith(
            "rough_seed_"
        )
    ]

    if len(
        rough_contexts
    ) != EXPECTED_ROUGH:
        raise RuntimeError(
            "Expected 18 rough contexts."
        )


    first_context = (
        context_order[0]
    )

    beta_order = [
        row[
            "beta_name"
        ]
        for row in (
            by_context[
                first_context
            ]
        )
    ]

    if (
        len(beta_order)
        != EXPECTED_BETAS
        or len(
            set(beta_order)
        )
        != EXPECTED_BETAS
    ):
        raise RuntimeError(
            "Expected 21 beta points."
        )


    beta_rank = {
        name:
            i
        for i, name in enumerate(
            beta_order
        )
    }


    atlas_lookup = {}

    for row in atlas:
        key = (
            row[
                "context_id"
            ],
            row[
                "beta_name"
            ],
        )

        if key in atlas_lookup:
            raise RuntimeError(
                f"Duplicate atlas pair: "
                f"{key}"
            )

        atlas_lookup[
            key
        ] = row


    # ========================================================
    # Frozen preference inventory
    # ========================================================

    frozen_lookup = {}

    preference_order = []

    for row in frozen_labels:
        key = (
            row[
                "context_id"
            ],
            row[
                "preference_name"
            ],
        )

        if key in frozen_lookup:
            raise RuntimeError(
                f"Duplicate frozen label: "
                f"{key}"
            )

        frozen_lookup[
            key
        ] = row

        if (
            row[
                "context_id"
            ]
            == first_context
        ):
            preference_order.append(
                row[
                    "preference_name"
                ]
            )


    if (
        len(preference_order)
        != EXPECTED_PREFERENCES
        or len(
            set(preference_order)
        )
        != EXPECTED_PREFERENCES
    ):
        raise RuntimeError(
            "Expected 25 preferences."
        )


    # ========================================================
    # Context-local Pareto references
    # ========================================================

    front_by_context = {}

    ideal_by_context = {}

    span_by_context = {}


    for cid in context_order:
        front = pareto(
            by_context[
                cid
            ]
        )

        front.sort(
            key=lambda row:
                beta_rank[
                    row[
                        "beta_name"
                    ]
                ]
        )

        values = np.asarray(
            [
                [
                    finite(
                        row[column]
                    )
                    for column in (
                        OBJECTIVES
                    )
                ]
                for row in front
            ],
            dtype=np.float64,
        )

        ideal = np.min(
            values,
            axis=0,
        )

        nadir = np.max(
            values,
            axis=0,
        )

        span = (
            nadir
            - ideal
        )

        front_by_context[
            cid
        ] = front

        ideal_by_context[
            cid
        ] = ideal

        span_by_context[
            cid
        ] = span


    # ========================================================
    # TRAIN-derived positive-span floor candidates
    # ========================================================

    positive_spans = [
        [],
        [],
        [],
    ]

    for cid in context_order:
        span = (
            span_by_context[
                cid
            ]
        )

        for k in range(3):
            if span[k] > TOL:
                positive_spans[
                    k
                ].append(
                    float(
                        span[k]
                    )
                )


    variants = {
        "baseline":
            np.zeros(
                3,
                dtype=np.float64,
            )
    }

    floor_rows = []


    for name, q in (
        FLOOR_QUANTILES
    ):
        floors = np.asarray(
            [
                float(
                    np.quantile(
                        positive_spans[k],
                        q,
                    )
                )
                for k in range(3)
            ],
            dtype=np.float64,
        )

        variants[
            name
        ] = floors

        floor_rows.append(
            {
                "variant":
                    name,

                "quantile":
                    q,

                "motion_floor":
                    floors[0],

                "stability_floor":
                    floors[1],

                "energy_floor":
                    floors[2],

                "motion_positive_contexts":
                    len(
                        positive_spans[0]
                    ),

                "stability_positive_contexts":
                    len(
                        positive_spans[1]
                    ),

                "energy_positive_contexts":
                    len(
                        positive_spans[2]
                    ),
            }
        )


    # ========================================================
    # Scalarization
    # ========================================================

    def score_beta(
        *,
        context_id,
        beta_name,
        w,
        floors,
    ):
        row = atlas_lookup[
            (
                context_id,
                beta_name,
            )
        ]

        j = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in OBJECTIVES
            ],
            dtype=np.float64,
        )

        ideal = (
            ideal_by_context[
                context_id
            ]
        )

        span = (
            span_by_context[
                context_id
            ]
        )

        denominator = np.maximum(
            span,
            floors,
        )

        regret = np.zeros(
            3,
            dtype=np.float64,
        )

        for k in range(3):
            if denominator[k] > TOL:
                regret[k] = (
                    j[k]
                    - ideal[k]
                ) / denominator[k]

        weighted = (
            w
            * regret
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


    # ========================================================
    # Shadow labels
    # ========================================================

    shadow_rows = []

    shadow_lookup = {}

    variant_summaries = []


    for variant_name, floors in (
        variants.items()
    ):
        margins = []

        churn = []

        frozen_sacrifice = []


        for cid in context_order:
            front = (
                front_by_context[
                    cid
                ]
            )

            for preference_name in (
                preference_order
            ):
                frozen = (
                    frozen_lookup[
                        (
                            cid,
                            preference_name,
                        )
                    ]
                )

                w = np.asarray(
                    [
                        finite(
                            frozen[
                                "w_motion"
                            ]
                        ),
                        finite(
                            frozen[
                                "w_stability"
                            ]
                        ),
                        finite(
                            frozen[
                                "w_energy"
                            ]
                        ),
                    ],
                    dtype=np.float64,
                )

                candidates = []

                for row in front:
                    beta_name = (
                        row[
                            "beta_name"
                        ]
                    )

                    score = score_beta(
                        context_id=cid,
                        beta_name=(
                            beta_name
                        ),
                        w=w,
                        floors=floors,
                    )

                    candidates.append(
                        (
                            score,
                            beta_rank[
                                beta_name
                            ],
                            beta_name,
                        )
                    )

                candidates.sort()

                best_score, _, best_beta = (
                    candidates[
                        0
                    ]
                )

                if len(
                    candidates
                ) >= 2:
                    second_score = (
                        candidates[
                            1
                        ][
                            0
                        ]
                    )

                    margin = (
                        second_score
                        - best_score
                    )
                else:
                    margin = float(
                        "inf"
                    )


                frozen_beta = (
                    frozen[
                        "selected_beta_name"
                    ]
                )

                changed = int(
                    best_beta
                    != frozen_beta
                )


                # --------------------------------------------
                # Cost of the shadow oracle under the ORIGINAL
                # frozen T5.5d scalarization.
                #
                # This prevents artificial "improvement" merely
                # caused by enlarging the new denominator.
                # --------------------------------------------

                original_shadow_score = (
                    score_beta(
                        context_id=cid,
                        beta_name=(
                            best_beta
                        ),
                        w=w,
                        floors=(
                            variants[
                                "baseline"
                            ]
                        ),
                    )
                )

                frozen_oracle_score = finite(
                    frozen[
                        "scalarization_score"
                    ]
                )

                sacrifice = (
                    original_shadow_score
                    - frozen_oracle_score
                )

                if sacrifice < -1e-8:
                    raise RuntimeError(
                        "Shadow beta beats frozen "
                        "oracle under frozen score: "
                        f"{variant_name}/"
                        f"{cid}/"
                        f"{preference_name}/"
                        f"{sacrifice}"
                    )

                sacrifice = max(
                    0.0,
                    sacrifice,
                )


                row_out = {
                    "variant":
                        variant_name,

                    "context_id":
                        cid,

                    "preference_name":
                        preference_name,

                    "preference_kind":
                        frozen[
                            "preference_kind"
                        ],

                    "w_motion":
                        finite(
                            frozen[
                                "w_motion"
                            ]
                        ),

                    "w_stability":
                        finite(
                            frozen[
                                "w_stability"
                            ]
                        ),

                    "w_energy":
                        finite(
                            frozen[
                                "w_energy"
                            ]
                        ),

                    "selected_beta_name":
                        best_beta,

                    "shadow_score":
                        best_score,

                    "best_second_margin":
                        margin,

                    "frozen_beta_name":
                        frozen_beta,

                    "changed_from_frozen":
                        changed,

                    "frozen_oracle_score":
                        frozen_oracle_score,

                    "shadow_beta_frozen_score":
                        original_shadow_score,

                    "frozen_score_sacrifice":
                        sacrifice,
                }


                shadow_rows.append(
                    row_out
                )

                shadow_lookup[
                    (
                        variant_name,
                        cid,
                        preference_name,
                    )
                ] = row_out

                if math.isfinite(
                    margin
                ):
                    margins.append(
                        margin
                    )

                churn.append(
                    changed
                )

                frozen_sacrifice.append(
                    sacrifice
                )


        if len(
            churn
        ) != EXPECTED_LABELS:
            raise RuntimeError(
                "Shadow-label count mismatch."
            )


        # Baseline must reproduce T5.5d exactly.
        if (
            variant_name
            == "baseline"
            and any(churn)
        ):
            raise RuntimeError(
                "Baseline shadow failed frozen "
                "T5.5d label regression."
            )


        margins_np = np.asarray(
            margins,
            dtype=np.float64,
        )

        sacrifice_np = np.asarray(
            frozen_sacrifice,
            dtype=np.float64,
        )


        variant_summaries.append(
            {
                "variant":
                    variant_name,

                "motion_floor":
                    floors[0],

                "stability_floor":
                    floors[1],

                "energy_floor":
                    floors[2],

                "label_churn_fraction":
                    float(
                        np.mean(
                            churn
                        )
                    ),

                "margin_median":
                    float(
                        np.median(
                            margins_np
                        )
                    ),

                "margin_p10":
                    float(
                        np.quantile(
                            margins_np,
                            0.10,
                        )
                    ),

                "margin_p05":
                    float(
                        np.quantile(
                            margins_np,
                            0.05,
                        )
                    ),

                "margin_le_0p001_fraction":
                    float(
                        np.mean(
                            margins_np
                            <= 0.001
                        )
                    ),

                "margin_le_0p01_fraction":
                    float(
                        np.mean(
                            margins_np
                            <= 0.01
                        )
                    ),

                "frozen_score_sacrifice_mean":
                    float(
                        np.mean(
                            sacrifice_np
                        )
                    ),

                "frozen_score_sacrifice_p95":
                    float(
                        np.quantile(
                            sacrifice_np,
                            0.95,
                        )
                    ),

                "frozen_score_sacrifice_max":
                    float(
                        np.max(
                            sacrifice_np
                        )
                    ),

                "frozen_score_sacrifice_le_0p01_fraction":
                    float(
                        np.mean(
                            sacrifice_np
                            <= 0.01
                        )
                    ),
            }
        )


    # ========================================================
    # Nearest-context maps from already-computed v1/v2 audits
    # ========================================================

    def nearest_map(
        path,
    ):
        rows = read_csv(
            path
        )

        mapping = {}

        for row in rows:
            test = row[
                "test_context"
            ]

            nearest = row[
                "nearest_context"
            ]

            if (
                test in mapping
                and mapping[test]
                != nearest
            ):
                raise RuntimeError(
                    "Nearest-context mapping "
                    f"not constant for {test}"
                )

            mapping[
                test
            ] = nearest

        if set(
            mapping
        ) != set(
            rough_contexts
        ):
            raise RuntimeError(
                "Nearest-map rough coverage "
                "mismatch."
            )

        return mapping


    nearest_maps = {
        "v1":
            nearest_map(
                V1_TRANSFER
            ),

        "v2":
            nearest_map(
                V2_TRANSFER
            ),
    }


    # ========================================================
    # Shadow nearest-context transfer
    # ========================================================

    transfer_rows = []

    transfer_summaries = []


    for descriptor_name, mapping in (
        nearest_maps.items()
    ):
        for variant_name, floors in (
            variants.items()
        ):
            exact = []
            excesses = []


            for test_cid in (
                rough_contexts
            ):
                source_cid = (
                    mapping[
                        test_cid
                    ]
                )

                for preference_name in (
                    preference_order
                ):
                    target = (
                        frozen_lookup[
                            (
                                test_cid,
                                preference_name,
                            )
                        ]
                    )

                    source_shadow = (
                        shadow_lookup[
                            (
                                variant_name,
                                source_cid,
                                preference_name,
                            )
                        ]
                    )

                    target_shadow = (
                        shadow_lookup[
                            (
                                variant_name,
                                test_cid,
                                preference_name,
                            )
                        ]
                    )


                    transferred_beta = (
                        source_shadow[
                            "selected_beta_name"
                        ]
                    )

                    oracle_beta = (
                        target_shadow[
                            "selected_beta_name"
                        ]
                    )

                    w = np.asarray(
                        [
                            finite(
                                target[
                                    "w_motion"
                                ]
                            ),
                            finite(
                                target[
                                    "w_stability"
                                ]
                            ),
                            finite(
                                target[
                                    "w_energy"
                                ]
                            ),
                        ],
                        dtype=np.float64,
                    )


                    transferred_score = (
                        score_beta(
                            context_id=(
                                test_cid
                            ),
                            beta_name=(
                                transferred_beta
                            ),
                            w=w,
                            floors=floors,
                        )
                    )

                    oracle_score = (
                        target_shadow[
                            "shadow_score"
                        ]
                    )

                    excess = (
                        transferred_score
                        - oracle_score
                    )

                    if excess < -1e-8:
                        raise RuntimeError(
                            "Transferred shadow beta "
                            "beats shadow oracle."
                        )

                    excess = max(
                        0.0,
                        excess,
                    )

                    is_exact = int(
                        transferred_beta
                        == oracle_beta
                    )


                    transfer_rows.append(
                        {
                            "descriptor":
                                descriptor_name,

                            "variant":
                                variant_name,

                            "test_context":
                                test_cid,

                            "nearest_context":
                                source_cid,

                            "preference_name":
                                preference_name,

                            "transferred_beta":
                                transferred_beta,

                            "shadow_oracle_beta":
                                oracle_beta,

                            "exact":
                                is_exact,

                            "shadow_oracle_score":
                                oracle_score,

                            "transferred_score":
                                transferred_score,

                            "score_excess":
                                excess,
                        }
                    )

                    exact.append(
                        is_exact
                    )

                    excesses.append(
                        excess
                    )


            if len(
                exact
            ) != EXPECTED_ROUGH_LABELS:
                raise RuntimeError(
                    "Transfer count mismatch."
                )

            x = np.asarray(
                excesses,
                dtype=np.float64,
            )


            transfer_summaries.append(
                {
                    "descriptor":
                        descriptor_name,

                    "variant":
                        variant_name,

                    "exact_accuracy":
                        float(
                            np.mean(
                                exact
                            )
                        ),

                    "score_excess_mean":
                        float(
                            np.mean(
                                x
                            )
                        ),

                    "score_excess_median":
                        float(
                            np.median(
                                x
                            )
                        ),

                    "score_excess_p95":
                        float(
                            np.quantile(
                                x,
                                0.95,
                            )
                        ),

                    "score_excess_max":
                        float(
                            np.max(
                                x
                            )
                        ),

                    "score_excess_le_0p05_fraction":
                        float(
                            np.mean(
                                x
                                <= 0.05
                            )
                        ),
                }
            )


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_FLOORS,
        floor_rows,
    )

    write_csv(
        OUT_LABELS,
        shadow_rows,
    )

    write_csv(
        OUT_VARIANTS,
        variant_summaries,
    )

    write_csv(
        OUT_TRANSFER,
        transfer_summaries,
    )

    write_csv(
        OUT_TRANSFER_ROWS,
        transfer_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1g_"
                "robust_scalarization_shadow_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "frozen_t5p5d_modified":
            False,

        "variants":
            list(
                variants.keys()
            ),

        "floor_definition":
            (
                "Per-objective quantile of positive "
                "context-local Pareto spans over the "
                "20 TRAIN physical contexts."
            ),

        "shadow_regret":
            (
                "(J_k - ideal_k) / "
                "max(context Pareto span_k, delta_k)"
            ),

        "selection_domain":
            (
                "unchanged context-local "
                "Pareto front"
            ),

        "rho":
            RHO,

        "baseline_regression":
            "PASS",

        "conditioning_metric":
            (
                "best-vs-second shadow "
                "scalarization margin"
            ),

        "preservation_metric":
            (
                "score sacrifice of the shadow-selected "
                "beta under the original frozen T5.5d "
                "scalarization relative to the frozen "
                "T5.5d oracle beta"
            ),

        "transfer_metrics":
            (
                "Existing v1 and v2 TRAIN-only nearest-"
                "context mappings are reused; no context "
                "descriptor or held-out experiment is "
                "recomputed."
            ),

        "interpretation_guard":
            (
                "Sensitivity study only. Do not adopt "
                "a robust floor or replace frozen T5.5d "
                "labels from this artifact alone."
            ),

        "next_stage":
            (
                "Inspect whether conditioning and "
                "cross-context smoothness can improve "
                "without material sacrifice under the "
                "original frozen physical preference "
                "objective."
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


    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 118)
    print(
        "ICRA27 OS-T5.5e1g ROBUST "
        "SCALARIZATION SHADOW SENSITIVITY"
    )
    print("=" * 118)

    print()
    print("SPAN FLOORS")

    for row in floor_rows:
        print(
            f"  {row['variant']:<5} "
            f"M={row['motion_floor']:.9f} "
            f"S={row['stability_floor']:.9f} "
            f"E={row['energy_floor']:.9f}"
        )

    print()
    print("SHADOW LABEL CONDITIONING")

    for row in variant_summaries:
        print(
            f"  {row['variant']:<8} "
            f"churn={row['label_churn_fraction']:.3f} "
            f"marginMed={row['margin_median']:.4f} "
            f"margin<=.01="
            f"{row['margin_le_0p01_fraction']:.3f} "
            f"frozenSacMean="
            f"{row['frozen_score_sacrifice_mean']:.4f} "
            f"frozenSacP95="
            f"{row['frozen_score_sacrifice_p95']:.4f}"
        )

    print()
    print("NEAREST-CONTEXT TRANSFER")

    for row in transfer_summaries:
        print(
            f"  {row['descriptor']:<3} "
            f"{row['variant']:<8} "
            f"exact="
            f"{row['exact_accuracy']:.3f} "
            f"meanEx="
            f"{row['score_excess_mean']:.4f} "
            f"p95="
            f"{row['score_excess_p95']:.4f} "
            f"<=.05="
            f"{row['score_excess_le_0p05_fraction']:.3f}"
        )

    print()
    print("outputs:")
    print(" ", OUT_FLOORS)
    print(" ", OUT_LABELS)
    print(" ", OUT_VARIANTS)
    print(" ", OUT_TRANSFER)
    print(" ", OUT_TRANSFER_ROWS)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1g robust "
        "scalarization shadow: COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
