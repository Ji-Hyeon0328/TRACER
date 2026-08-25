from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Frozen upstream artifacts
# ============================================================

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

SHADOW_VARIANTS = (
    SHADOW_ROOT
    / "shadow_variant_summary.csv"
)

SHADOW_MANIFEST = (
    SHADOW_ROOT
    / "shadow_scalarization_manifest.json"
)


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1h_shadow_physical_significance_v0"
)

OUT_CHANGED = (
    OUT_DIR
    / "changed_label_physical_deltas.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "physical_significance_summary.csv"
)

OUT_IQR = (
    OUT_DIR
    / "train_physical_iqr_reference.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "physical_significance_manifest.json"
)


# ============================================================
# Contract
# ============================================================

OBJECTIVES = (
    (
        "motion",
        "J_motion_s_per_m",
    ),
    (
        "stability",
        "J_stability",
    ),
    (
        "energy",
        "J_energy_j_per_m",
    ),
)

ROBUST_VARIANTS = (
    "q10",
    "q25",
    "q50",
)

EXPECTED_ATLAS_ROWS = 420
EXPECTED_CONTEXTS = 20
EXPECTED_LABELS_PER_VARIANT = 500

TOL = 1.0e-12


def finite(
    value: Any,
) -> float:
    x = float(value)

    if not math.isfinite(x):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return x


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    fields: list[str] = []

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


def quantile(
    values,
    q: float,
) -> float:
    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
            method="linear",
        )
    )


def summarize_abs(
    values,
) -> dict[str, float]:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.size == 0:
        raise RuntimeError(
            "Cannot summarize empty values."
        )

    return {
        "median":
            float(
                np.median(x)
            ),

        "p95":
            float(
                np.quantile(
                    x,
                    0.95,
                    method="linear",
                )
            ),

        "max":
            float(
                np.max(x)
            ),
    }


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        ATLAS,
        SHADOW_LABELS,
        SHADOW_VARIANTS,
        SHADOW_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Validate shadow provenance
    # ========================================================

    shadow_manifest = json.loads(
        SHADOW_MANIFEST.read_text()
    )

    if shadow_manifest.get(
        "schema"
    ) != (
        "icra27_os_t5p5e1g_"
        "robust_scalarization_shadow_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.5e1g schema."
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
            "Frozen T5.5d modification reported."
        )


    # ========================================================
    # Load expanded physical atlas
    # ========================================================

    atlas = read_csv(
        ATLAS
    )

    if len(
        atlas
    ) != EXPECTED_ATLAS_ROWS:
        raise RuntimeError(
            f"Expected 420 atlas rows; "
            f"got {len(atlas)}"
        )

    contexts = {
        row[
            "context_id"
        ]
        for row in atlas
    }

    if len(
        contexts
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected 20 contexts; "
            f"got {len(contexts)}"
        )


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
                f"Duplicate atlas pair: {key}"
            )

        atlas_lookup[
            key
        ] = row


    # ========================================================
    # Expanded TRAIN global IQR reference
    #
    # Reporting scale ONLY.
    # This does not alter the frozen scalarization contract.
    # ========================================================

    iqr_by_objective = {}

    iqr_rows = []


    for objective_name, column in (
        OBJECTIVES
    ):
        values = np.asarray(
            [
                finite(
                    row[
                        column
                    ]
                )
                for row in atlas
            ],
            dtype=np.float64,
        )

        q25 = quantile(
            values,
            0.25,
        )

        q50 = quantile(
            values,
            0.50,
        )

        q75 = quantile(
            values,
            0.75,
        )

        iqr = (
            q75
            - q25
        )

        if iqr <= TOL:
            raise RuntimeError(
                f"Degenerate global TRAIN IQR "
                f"for {objective_name}: {iqr}"
            )

        iqr_by_objective[
            objective_name
        ] = iqr

        iqr_rows.append(
            {
                "objective":
                    objective_name,

                "column":
                    column,

                "q25":
                    q25,

                "median":
                    q50,

                "q75":
                    q75,

                "iqr":
                    iqr,

                "rows":
                    len(
                        values
                    ),
            }
        )


    # ========================================================
    # Shadow assignments
    # ========================================================

    shadow_rows = read_csv(
        SHADOW_LABELS
    )

    counts = Counter(
        row[
            "variant"
        ]
        for row in shadow_rows
    )


    for variant in (
        (
            "baseline",
        )
        + ROBUST_VARIANTS
    ):
        if counts[
            variant
        ] != EXPECTED_LABELS_PER_VARIANT:
            raise RuntimeError(
                f"{variant}: expected 500 labels; "
                f"got {counts[variant]}"
            )


    # Baseline must remain unchanged.
    baseline_changed = sum(
        int(
            float(
                row[
                    "changed_from_frozen"
                ]
            )
        )
        for row in shadow_rows
        if row[
            "variant"
        ] == "baseline"
    )

    if baseline_changed != 0:
        raise RuntimeError(
            "Baseline shadow changed frozen labels."
        )


    # ========================================================
    # Cross-check variant summary churn counts
    # ========================================================

    variant_source_summary = {
        row[
            "variant"
        ]:
            row
        for row in read_csv(
            SHADOW_VARIANTS
        )
    }


    # ========================================================
    # Physical delta audit
    # ========================================================

    changed_rows = []

    summary_rows = []


    for variant in (
        ROBUST_VARIANTS
    ):
        variant_rows = [
            row
            for row in shadow_rows
            if row[
                "variant"
            ] == variant
        ]

        changed = [
            row
            for row in variant_rows
            if int(
                float(
                    row[
                        "changed_from_frozen"
                    ]
                )
            ) == 1
        ]


        expected_churn = finite(
            variant_source_summary[
                variant
            ][
                "label_churn_fraction"
            ]
        )

        actual_churn = (
            len(
                changed
            )
            / EXPECTED_LABELS_PER_VARIANT
        )

        if not math.isclose(
            actual_churn,
            expected_churn,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{variant}: churn regression "
                f"mismatch "
                f"{actual_churn} vs "
                f"{expected_churn}"
            )


        abs_delta = {
            name:
                []
            for name, _ in (
                OBJECTIVES
            )
        }

        normalized_abs_delta = {
            name:
                []
            for name, _ in (
                OBJECTIVES
            )
        }

        signed_delta = {
            name:
                []
            for name, _ in (
                OBJECTIVES
            )
        }

        normalized_l1 = []
        normalized_linf = []

        original_score_sacrifice = []


        for shadow in changed:
            cid = shadow[
                "context_id"
            ]

            frozen_beta = (
                shadow[
                    "frozen_beta_name"
                ]
            )

            shadow_beta = (
                shadow[
                    "selected_beta_name"
                ]
            )


            frozen_key = (
                cid,
                frozen_beta,
            )

            shadow_key = (
                cid,
                shadow_beta,
            )


            if frozen_key not in (
                atlas_lookup
            ):
                raise RuntimeError(
                    f"Missing frozen atlas row: "
                    f"{frozen_key}"
                )

            if shadow_key not in (
                atlas_lookup
            ):
                raise RuntimeError(
                    f"Missing shadow atlas row: "
                    f"{shadow_key}"
                )


            frozen_physical = (
                atlas_lookup[
                    frozen_key
                ]
            )

            shadow_physical = (
                atlas_lookup[
                    shadow_key
                ]
            )


            out = {
                "variant":
                    variant,

                "context_id":
                    cid,

                "preference_name":
                    shadow[
                        "preference_name"
                    ],

                "preference_kind":
                    shadow[
                        "preference_kind"
                    ],

                "frozen_beta_name":
                    frozen_beta,

                "shadow_beta_name":
                    shadow_beta,

                "frozen_score_sacrifice":
                    finite(
                        shadow[
                            "frozen_score_sacrifice"
                        ]
                    ),
            }


            normalized_components = []


            for objective_name, column in (
                OBJECTIVES
            ):
                j_frozen = finite(
                    frozen_physical[
                        column
                    ]
                )

                j_shadow = finite(
                    shadow_physical[
                        column
                    ]
                )

                delta = (
                    j_shadow
                    - j_frozen
                )

                abs_d = abs(
                    delta
                )

                normalized = (
                    abs_d
                    / iqr_by_objective[
                        objective_name
                    ]
                )


                out[
                    f"frozen_{objective_name}"
                ] = j_frozen

                out[
                    f"shadow_{objective_name}"
                ] = j_shadow

                out[
                    f"delta_{objective_name}"
                ] = delta

                out[
                    f"abs_delta_{objective_name}"
                ] = abs_d

                out[
                    f"abs_delta_{objective_name}_over_train_iqr"
                ] = normalized


                signed_delta[
                    objective_name
                ].append(
                    delta
                )

                abs_delta[
                    objective_name
                ].append(
                    abs_d
                )

                normalized_abs_delta[
                    objective_name
                ].append(
                    normalized
                )

                normalized_components.append(
                    normalized
                )


            l1 = float(
                np.mean(
                    normalized_components
                )
            )

            linf = float(
                np.max(
                    normalized_components
                )
            )

            out[
                "normalized_physical_l1_mean"
            ] = l1

            out[
                "normalized_physical_linf"
            ] = linf


            normalized_l1.append(
                l1
            )

            normalized_linf.append(
                linf
            )

            original_score_sacrifice.append(
                finite(
                    shadow[
                        "frozen_score_sacrifice"
                    ]
                )
            )

            changed_rows.append(
                out
            )


        if not changed:
            raise RuntimeError(
                f"{variant}: no changed labels."
            )


        summary = {
            "variant":
                variant,

            "labels_total":
                EXPECTED_LABELS_PER_VARIANT,

            "labels_changed":
                len(
                    changed
                ),

            "label_churn_fraction":
                actual_churn,
        }


        for objective_name, _ in (
            OBJECTIVES
        ):
            abs_summary = summarize_abs(
                abs_delta[
                    objective_name
                ]
            )

            norm_summary = summarize_abs(
                normalized_abs_delta[
                    objective_name
                ]
            )

            signed = np.asarray(
                signed_delta[
                    objective_name
                ],
                dtype=np.float64,
            )


            summary[
                f"{objective_name}_abs_delta_median"
            ] = abs_summary[
                "median"
            ]

            summary[
                f"{objective_name}_abs_delta_p95"
            ] = abs_summary[
                "p95"
            ]

            summary[
                f"{objective_name}_abs_delta_max"
            ] = abs_summary[
                "max"
            ]


            summary[
                f"{objective_name}_abs_over_iqr_median"
            ] = norm_summary[
                "median"
            ]

            summary[
                f"{objective_name}_abs_over_iqr_p95"
            ] = norm_summary[
                "p95"
            ]

            summary[
                f"{objective_name}_abs_over_iqr_max"
            ] = norm_summary[
                "max"
            ]


            summary[
                f"{objective_name}_improved_fraction"
            ] = float(
                np.mean(
                    signed
                    < -TOL
                )
            )

            summary[
                f"{objective_name}_worsened_fraction"
            ] = float(
                np.mean(
                    signed
                    > TOL
                )
            )

            summary[
                f"{objective_name}_tie_fraction"
            ] = float(
                np.mean(
                    np.abs(
                        signed
                    )
                    <= TOL
                )
            )


        l1 = np.asarray(
            normalized_l1,
            dtype=np.float64,
        )

        linf = np.asarray(
            normalized_linf,
            dtype=np.float64,
        )


        summary[
            "normalized_physical_l1_median"
        ] = float(
            np.median(
                l1
            )
        )

        summary[
            "normalized_physical_l1_p95"
        ] = float(
            np.quantile(
                l1,
                0.95,
                method="linear",
            )
        )


        summary[
            "normalized_physical_linf_median"
        ] = float(
            np.median(
                linf
            )
        )

        summary[
            "normalized_physical_linf_p95"
        ] = float(
            np.quantile(
                linf,
                0.95,
                method="linear",
            )
        )

        summary[
            "normalized_physical_linf_max"
        ] = float(
            np.max(
                linf
            )
        )


        for threshold in (
            0.01,
            0.05,
            0.10,
        ):
            key = (
                str(
                    threshold
                )
                .replace(
                    ".",
                    "p"
                )
            )

            summary[
                f"linf_le_{key}_fraction"
            ] = float(
                np.mean(
                    linf
                    <= threshold
                )
            )


        sacrifice = np.asarray(
            original_score_sacrifice,
            dtype=np.float64,
        )

        summary[
            "frozen_score_sacrifice_changed_median"
        ] = float(
            np.median(
                sacrifice
            )
        )

        summary[
            "frozen_score_sacrifice_changed_p95"
        ] = float(
            np.quantile(
                sacrifice,
                0.95,
                method="linear",
            )
        )


        # Does a large frozen score sacrifice correspond
        # to a large physical change?
        if (
            np.std(
                sacrifice
            ) > TOL
            and np.std(
                linf
            ) > TOL
        ):
            corr = float(
                np.corrcoef(
                    sacrifice,
                    linf,
                )[
                    0,
                    1
                ]
            )
        else:
            corr = float(
                "nan"
            )

        summary[
            "corr_frozen_score_sacrifice_vs_physical_linf"
        ] = corr


        summary_rows.append(
            summary
        )


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_CHANGED,
        changed_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )

    write_csv(
        OUT_IQR,
        iqr_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1h_"
                "shadow_physical_significance_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "frozen_t5p5d_modified":
            False,

        "frozen_t5p5e1g_modified":
            False,

        "atlas_rows":
            len(
                atlas
            ),

        "contexts":
            len(
                contexts
            ),

        "variants":
            list(
                ROBUST_VARIANTS
            ),

        "physical_objectives":
            {
                objective_name:
                    column
                for objective_name, column in (
                    OBJECTIVES
                )
            },

        "objective_direction":
            "all_lower_is_better",

        "delta_definition":
            (
                "J_shadow - J_frozen; "
                "positive means shadow-selected "
                "beta is worse on that physical "
                "objective, negative means better"
            ),

        "descriptive_scale":
            (
                "absolute physical delta divided by "
                "the global IQR of that objective "
                "over the frozen 420-row expanded "
                "TRAIN physical atlas"
            ),

        "descriptive_scale_guard":
            (
                "The global TRAIN IQR is reporting "
                "scale only. It does not replace or "
                "modify T5.5d/T5.5e1g scalarization, "
                "oracle labels, or selector inputs."
            ),

        "train_iqr":
            {
                row[
                    "objective"
                ]:
                    row[
                        "iqr"
                    ]
                for row in iqr_rows
            },

        "summary":
            {
                row[
                    "variant"
                ]:
                    row
                for row in summary_rows
            },

        "interpretation_guard":
            (
                "This audit measures physical "
                "significance of labels changed by "
                "the shadow span-floor variants. "
                "It does not authorize adoption of "
                "q10/q25/q50."
            ),

        "next_stage":
            (
                "If a conservative floor changes "
                "mostly physically near-equivalent "
                "Pareto points while improving "
                "cross-context transfer, define a "
                "new robust oracle contract as a "
                "separate artifact and rerun "
                "TRAIN-only selector LOCO validation."
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
    print("=" * 122)
    print(
        "ICRA27 OS-T5.5e1h SHADOW LABEL "
        "PHYSICAL SIGNIFICANCE AUDIT"
    )
    print("=" * 122)

    print()
    print("EXPANDED TRAIN GLOBAL IQR")

    for row in iqr_rows:
        print(
            f"  {row['objective']:<10} "
            f"IQR={row['iqr']:.9f} "
            f"[Q25={row['q25']:.9f}, "
            f"Q75={row['q75']:.9f}]"
        )


    print()
    print("CHANGED-LABEL PHYSICAL SIGNIFICANCE")

    for row in summary_rows:
        print()
        print(
            f"  [{row['variant']}] "
            f"changed="
            f"{row['labels_changed']}/"
            f"{row['labels_total']} "
            f"({row['label_churn_fraction']:.3f})"
        )

        for objective_name, _ in (
            OBJECTIVES
        ):
            print(
                f"    {objective_name:<10} "
                f"|dJ| med="
                f"{row[f'{objective_name}_abs_delta_median']:.9f} "
                f"p95="
                f"{row[f'{objective_name}_abs_delta_p95']:.9f} "
                f"|dJ|/IQR med="
                f"{row[f'{objective_name}_abs_over_iqr_median']:.4f} "
                f"p95="
                f"{row[f'{objective_name}_abs_over_iqr_p95']:.4f}"
            )

        print(
            "    normalized Linf "
            f"med="
            f"{row['normalized_physical_linf_median']:.4f} "
            f"p95="
            f"{row['normalized_physical_linf_p95']:.4f} "
            f"max="
            f"{row['normalized_physical_linf_max']:.4f}"
        )

        print(
            "    Linf <= 0.01/0.05/0.10 IQR : "
            f"{row['linf_le_0p01_fraction']:.3f} / "
            f"{row['linf_le_0p05_fraction']:.3f} / "
            f"{row['linf_le_0p1_fraction']:.3f}"
        )

        print(
            "    corr(frozen score sacrifice, "
            "physical Linf) = "
            f"{row['corr_frozen_score_sacrifice_vs_physical_linf']}"
        )


    print()
    print("outputs:")
    print(" ", OUT_CHANGED)
    print(" ", OUT_SUMMARY)
    print(" ", OUT_IQR)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1h "
        "shadow physical significance: COMPUTE PASS"
    )

    print("=" * 122)


if __name__ == "__main__":
    main()
