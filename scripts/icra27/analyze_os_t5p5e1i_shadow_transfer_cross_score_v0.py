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

SHADOW_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1g_robust_scalarization_shadow_v0"
)

SHADOW_LABELS = (
    SHADOW_ROOT
    / "shadow_label_assignments.csv"
)

SHADOW_TRANSFER = (
    SHADOW_ROOT
    / "shadow_transfer_rows.csv"
)

SHADOW_TRANSFER_SUMMARY = (
    SHADOW_ROOT
    / "shadow_transfer_summary.csv"
)

SHADOW_MANIFEST = (
    SHADOW_ROOT
    / "shadow_scalarization_manifest.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1i_shadow_transfer_cross_score_v0"
)

OUT_ROWS = (
    OUT_DIR
    / "frozen_score_transfer_rows.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "frozen_score_transfer_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "cross_score_manifest.json"
)


OBJECTIVES = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

DESCRIPTORS = (
    "v1",
    "v2",
)

VARIANTS = (
    "baseline",
    "q10",
    "q25",
    "q50",
)

EXPECTED_CONTEXTS = 20
EXPECTED_ROUGH = 18
EXPECTED_BETAS = 21
EXPECTED_PREFERENCES = 25
EXPECTED_ROWS_PER_PAIR = 450

RHO = 0.01
TOL = 1.0e-10


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x!r}"
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


def pareto(rows):
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


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        ATLAS,
        SHADOW_LABELS,
        SHADOW_TRANSFER,
        SHADOW_TRANSFER_SUMMARY,
        SHADOW_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Provenance
    # ========================================================

    shadow_manifest = json.loads(
        SHADOW_MANIFEST.read_text()
    )

    if shadow_manifest.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5e1g is not COMPUTE_PASS."
        )

    if bool(
        shadow_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5e1g reports held-out use."
        )

    if bool(
        shadow_manifest[
            "frozen_t5p5d_modified"
        ]
    ):
        raise RuntimeError(
            "Frozen T5.5d was modified."
        )


    # ========================================================
    # Physical atlas
    # ========================================================

    atlas = read_csv(
        ATLAS
    )

    by_context = defaultdict(
        list
    )

    atlas_lookup = {}

    context_order = []


    for row in atlas:
        cid = row[
            "context_id"
        ]

        if cid not in context_order:
            context_order.append(
                cid
            )

        by_context[
            cid
        ].append(
            row
        )

        key = (
            cid,
            row[
                "beta_name"
            ],
        )

        if key in atlas_lookup:
            raise RuntimeError(
                f"Duplicate atlas pair: {key}"
            )

        atlas_lookup[
            key
        ] = row


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


    beta_order = [
        row[
            "beta_name"
        ]
        for row in (
            by_context[
                context_order[
                    0
                ]
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


    # ========================================================
    # ORIGINAL frozen T5.5d regret references
    # ========================================================

    original_reference = {}

    for cid in context_order:
        front = pareto(
            by_context[
                cid
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

        original_reference[
            cid
        ] = (
            ideal,
            nadir - ideal,
        )


    def original_score(
        *,
        context_id,
        beta_name,
        w,
    ):
        row = atlas_lookup[
            (
                context_id,
                beta_name,
            )
        ]

        values = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in (
                    OBJECTIVES
                )
            ],
            dtype=np.float64,
        )

        ideal, span = (
            original_reference[
                context_id
            ]
        )

        regret = np.zeros(
            3,
            dtype=np.float64,
        )

        for k in range(3):
            if span[k] > TOL:
                regret[k] = (
                    values[k]
                    - ideal[k]
                ) / span[k]

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
    # Shadow label lookup supplies exact w and frozen oracle
    # score for each context/preference.
    # ========================================================

    label_rows = read_csv(
        SHADOW_LABELS
    )

    label_lookup = {}


    for row in label_rows:
        key = (
            row[
                "variant"
            ],
            row[
                "context_id"
            ],
            row[
                "preference_name"
            ],
        )

        if key in label_lookup:
            raise RuntimeError(
                f"Duplicate shadow label: {key}"
            )

        label_lookup[
            key
        ] = row


    # ========================================================
    # Existing shadow transfer
    # ========================================================

    transfer_rows = read_csv(
        SHADOW_TRANSFER
    )

    expected_total = (
        len(
            DESCRIPTORS
        )
        * len(
            VARIANTS
        )
        * EXPECTED_ROWS_PER_PAIR
    )

    if len(
        transfer_rows
    ) != expected_total:
        raise RuntimeError(
            f"Transfer-row mismatch: "
            f"{len(transfer_rows)} vs "
            f"{expected_total}"
        )


    out_rows = []


    for row in transfer_rows:
        descriptor = (
            row[
                "descriptor"
            ]
        )

        variant = (
            row[
                "variant"
            ]
        )

        test_context = (
            row[
                "test_context"
            ]
        )

        preference_name = (
            row[
                "preference_name"
            ]
        )

        target = (
            label_lookup[
                (
                    variant,
                    test_context,
                    preference_name,
                )
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


        transferred_beta = (
            row[
                "transferred_beta"
            ]
        )


        frozen_transferred_score = (
            original_score(
                context_id=(
                    test_context
                ),
                beta_name=(
                    transferred_beta
                ),
                w=w,
            )
        )


        frozen_oracle_score = finite(
            target[
                "frozen_oracle_score"
            ]
        )


        frozen_excess = (
            frozen_transferred_score
            - frozen_oracle_score
        )

        if frozen_excess < -1e-8:
            raise RuntimeError(
                "Transferred beta beats frozen "
                "oracle under frozen score: "
                f"{descriptor}/{variant}/"
                f"{test_context}/"
                f"{preference_name}/"
                f"{frozen_excess}"
            )

        frozen_excess = max(
            0.0,
            frozen_excess,
        )


        out_rows.append(
            {
                "descriptor":
                    descriptor,

                "variant":
                    variant,

                "test_context":
                    test_context,

                "nearest_context":
                    row[
                        "nearest_context"
                    ],

                "preference_name":
                    preference_name,

                "transferred_beta":
                    transferred_beta,

                "shadow_oracle_beta":
                    row[
                        "shadow_oracle_beta"
                    ],

                "shadow_score_excess":
                    finite(
                        row[
                            "score_excess"
                        ]
                    ),

                "frozen_oracle_score":
                    frozen_oracle_score,

                "transferred_frozen_score":
                    frozen_transferred_score,

                "frozen_score_excess":
                    frozen_excess,
            }
        )


    # ========================================================
    # Aggregate
    # ========================================================

    summaries = []


    for descriptor in (
        DESCRIPTORS
    ):
        baseline_rows = [
            row
            for row in out_rows
            if (
                row[
                    "descriptor"
                ]
                == descriptor
                and row[
                    "variant"
                ]
                == "baseline"
            )
        ]

        baseline_mean = float(
            np.mean(
                [
                    row[
                        "frozen_score_excess"
                    ]
                    for row in baseline_rows
                ]
            )
        )


        for variant in (
            VARIANTS
        ):
            rows = [
                row
                for row in out_rows
                if (
                    row[
                        "descriptor"
                    ]
                    == descriptor
                    and row[
                        "variant"
                    ]
                    == variant
                )
            ]

            if len(
                rows
            ) != EXPECTED_ROWS_PER_PAIR:
                raise RuntimeError(
                    f"{descriptor}/{variant}: "
                    "row-count mismatch."
                )


            frozen_excess = np.asarray(
                [
                    row[
                        "frozen_score_excess"
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )

            shadow_excess = np.asarray(
                [
                    row[
                        "shadow_score_excess"
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )


            mean_frozen = float(
                np.mean(
                    frozen_excess
                )
            )


            summaries.append(
                {
                    "descriptor":
                        descriptor,

                    "variant":
                        variant,

                    "rows":
                        len(
                            rows
                        ),

                    "shadow_score_excess_mean":
                        float(
                            np.mean(
                                shadow_excess
                            )
                        ),

                    "frozen_score_excess_mean":
                        mean_frozen,

                    "frozen_score_excess_median":
                        float(
                            np.median(
                                frozen_excess
                            )
                        ),

                    "frozen_score_excess_p95":
                        float(
                            np.quantile(
                                frozen_excess,
                                0.95,
                                method="linear",
                            )
                        ),

                    "frozen_score_excess_max":
                        float(
                            np.max(
                                frozen_excess
                            )
                        ),

                    "frozen_score_excess_le_0p05_fraction":
                        float(
                            np.mean(
                                frozen_excess
                                <= 0.05
                            )
                        ),

                    "frozen_mean_improvement_vs_baseline":
                        (
                            baseline_mean
                            - mean_frozen
                        ),

                    "frozen_mean_relative_improvement_vs_baseline":
                        (
                            (
                                baseline_mean
                                - mean_frozen
                            )
                            / baseline_mean
                            if baseline_mean > TOL
                            else 0.0
                        ),
                }
            )


    # ========================================================
    # Hard regression:
    # baseline uses no floors, therefore the shadow score
    # and frozen score must be identical row-by-row.
    # ========================================================

    max_baseline_error = 0.0

    for row in out_rows:
        if row[
            "variant"
        ] != "baseline":
            continue

        error = abs(
            row[
                "shadow_score_excess"
            ]
            - row[
                "frozen_score_excess"
            ]
        )

        max_baseline_error = max(
            max_baseline_error,
            error,
        )


    if max_baseline_error > 1e-9:
        raise RuntimeError(
            "Baseline cross-score regression "
            f"failed: {max_baseline_error}"
        )


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_ROWS,
        out_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summaries,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1i_"
                "shadow_transfer_cross_score_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "frozen_t5p5d_modified":
            False,

        "frozen_t5p5e1g_modified":
            False,

        "rows":
            len(
                out_rows
            ),

        "baseline_cross_score_max_error":
            max_baseline_error,

        "evaluation":
            (
                "Each beta transferred using the "
                "baseline/q10/q25/q50 shadow oracle "
                "is re-scored under the ORIGINAL "
                "frozen T5.5d context-local "
                "Pareto-span scalarization."
            ),

        "purpose":
            (
                "Separate genuine beta-selection "
                "improvement from numerical score "
                "compression caused by robust "
                "denominator floors."
            ),

        "interpretation_guard":
            (
                "A reduction in shadow-score excess "
                "is not evidence of improved original "
                "mission preference realization unless "
                "the frozen-score excess also decreases."
            ),

        "summary":
            {
                (
                    row[
                        "descriptor"
                    ]
                    + "/"
                    + row[
                        "variant"
                    ]
                ):
                    row
                for row in summaries
            },

        "next_stage":
            (
                "If robust floors provide little "
                "improvement under the original frozen "
                "score, retain T5.5d as the reference "
                "contract and focus the next selector "
                "iteration on richer terrain-context "
                "representation."
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
        "ICRA27 OS-T5.5e1i SHADOW TRANSFER "
        "CROSS-SCORE AUDIT"
    )
    print("=" * 120)

    print(
        "baseline cross-score max error:",
        max_baseline_error,
    )

    print()
    print(
        "SHADOW SCORE -> ORIGINAL FROZEN SCORE"
    )

    for row in summaries:
        print(
            f"  {row['descriptor']:<3} "
            f"{row['variant']:<8} "
            f"shadowMean="
            f"{row['shadow_score_excess_mean']:.4f} "
            f"frozenMean="
            f"{row['frozen_score_excess_mean']:.4f} "
            f"frozenP95="
            f"{row['frozen_score_excess_p95']:.4f} "
            f"<=.05="
            f"{row['frozen_score_excess_le_0p05_fraction']:.3f} "
            f"relImprove="
            f"{100.0 * row['frozen_mean_relative_improvement_vs_baseline']:.2f}%"
        )

    print()
    print("outputs:")
    print(" ", OUT_ROWS)
    print(" ", OUT_SUMMARY)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1i "
        "shadow transfer cross-score: COMPUTE PASS"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
