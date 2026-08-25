from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Frozen upstream artifacts
# ============================================================

CONTEXT_V2_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v2"
)

CONTEXT_V2_CSV = (
    CONTEXT_V2_ROOT
    / "oracle_context_descriptors.csv"
)

CONTEXT_V2_CONTRACT = (
    CONTEXT_V2_ROOT
    / "oracle_context_contract.json"
)


ATLAS_CSV = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)


LABEL_CSV = (
    ROOT
    / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
    / "preference_to_beta_labels.csv"
)


V1_AUDIT_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1b_context_identifiability_v0"
    / "identifiability_manifest.json"
)


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1d_context_identifiability_v2_v0"
)

OUT_NN = (
    OUT_DIR
    / "nearest_context_preference_transfer.csv"
)

OUT_PAIRWISE = (
    OUT_DIR
    / "rough_context_pairwise_similarity.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "identifiability_v2_manifest.json"
)


# ============================================================
# v2 feature contract
# ============================================================

CONTEXT_FEATURES = (
    "friction_mu",
    "height_std_m",
    "height_relief_p95_p05_m",
    "slope_rms",
    "slope_q95",
    "slope_longitudinal_rms",
    "slope_lateral_rms",
    "height_front_minus_rear_mean_m",
    "height_left_minus_right_mean_m",
    "height_front_minus_rear_std_m",
    "height_mid_minus_ends_std_m",
)

OBJECTIVE_COLUMNS = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)


EXPECTED_CONTEXT_DIM = 11
EXPECTED_ROUGH_CONTEXTS = 18
EXPECTED_BETAS = 21
EXPECTED_PREFERENCES = 25
EXPECTED_PAIR_COUNT = 153
EXPECTED_TRANSFER_ROWS = 450

RHO = 0.01
EPS = 1.0e-12
TOL = 1.0e-10


def finite(
    x: Any,
) -> float:
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x!r}"
        )

    return y


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
        raise ValueError(
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


def pearson(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x = (
        x
        - np.mean(x)
    )

    y = (
        y
        - np.mean(y)
    )

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom <= EPS:
        return float("nan")

    return float(
        np.dot(
            x,
            y,
        )
        / denom
    )


def context_id_from_descriptor(
    row,
):
    terrain = str(
        row["terrain"]
    )

    if terrain == "flat":
        return "flat"

    if terrain == "low_friction":
        return "low_friction"

    if terrain == "rough_perlin":
        seed = int(
            float(
                row["seed"]
            )
        )

        return (
            f"rough_seed_{seed}"
        )

    raise RuntimeError(
        f"Unexpected terrain: {terrain}"
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        CONTEXT_V2_CSV,
        CONTEXT_V2_CONTRACT,
        ATLAS_CSV,
        LABEL_CSV,
        V1_AUDIT_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Validate v2 context contract
    # ========================================================

    contract = json.loads(
        CONTEXT_V2_CONTRACT.read_text()
    )

    if contract.get(
        "schema"
    ) != (
        "icra27_os_t5p5a_"
        "oracle_context_descriptor_v2"
    ):
        raise RuntimeError(
            "Unexpected v2 context schema."
        )

    if contract.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "v2 descriptor is not FREEZE_PASS."
        )

    if int(
        contract[
            "context_dim"
        ]
    ) != EXPECTED_CONTEXT_DIM:
        raise RuntimeError(
            "v2 context dimension mismatch."
        )

    if list(
        contract[
            "context_feature_order"
        ]
    ) != list(
        CONTEXT_FEATURES
    ):
        raise RuntimeError(
            "v2 context feature order mismatch."
        )

    if int(
        contract[
            "fresh_process_global_hfields_unique"
        ]
    ) != EXPECTED_ROUGH_CONTEXTS:
        raise RuntimeError(
            "v2 fresh-process geometry "
            "uniqueness mismatch."
        )

    if finite(
        contract[
            "v1_regression_max_abs_error"
        ]
    ) > 1e-12:
        raise RuntimeError(
            "v2 failed frozen v1 regression."
        )


    # ========================================================
    # Load context descriptors
    # ========================================================

    context_rows = read_csv(
        CONTEXT_V2_CSV
    )

    context_lookup = {}

    rough_contexts = []

    for row in context_rows:
        cid = (
            context_id_from_descriptor(
                row
            )
        )

        if cid in context_lookup:
            raise RuntimeError(
                f"Duplicate context: {cid}"
            )

        context_lookup[
            cid
        ] = row

        if cid.startswith(
            "rough_seed_"
        ):
            rough_contexts.append(
                cid
            )


    if len(
        rough_contexts
    ) != EXPECTED_ROUGH_CONTEXTS:
        raise RuntimeError(
            f"Expected 18 rough contexts; "
            f"got {len(rough_contexts)}"
        )


    # ========================================================
    # Rough-family feature normalization
    # ========================================================

    context_matrix = np.asarray(
        [
            [
                finite(
                    context_lookup[
                        cid
                    ][feature]
                )
                for feature in (
                    CONTEXT_FEATURES
                )
            ]
            for cid in rough_contexts
        ],
        dtype=np.float64,
    )

    mean = np.mean(
        context_matrix,
        axis=0,
    )

    std = np.std(
        context_matrix,
        axis=0,
        ddof=0,
    )

    active = (
        std > EPS
    )

    active_features = [
        CONTEXT_FEATURES[i]
        for i in range(
            len(
                CONTEXT_FEATURES
            )
        )
        if active[i]
    ]

    z_context = (
        context_matrix[
            :,
            active
        ]
        - mean[
            active
        ]
    ) / std[
        active
    ]


    # friction should be the only constant rough feature.
    if len(
        active_features
    ) != 10:
        raise RuntimeError(
            "Expected 10 varying rough "
            "v2 context dimensions; got "
            f"{len(active_features)}: "
            f"{active_features}"
        )


    # ========================================================
    # Physical atlas
    # ========================================================

    atlas_rows = read_csv(
        ATLAS_CSV
    )

    atlas_lookup = {}

    beta_order = []

    first_rough = (
        rough_contexts[
            0
        ]
    )

    for row in atlas_rows:
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

        if (
            row[
                "context_id"
            ]
            == first_rough
        ):
            beta_order.append(
                row[
                    "beta_name"
                ]
            )


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


    for cid in rough_contexts:
        for beta_name in beta_order:
            if (
                cid,
                beta_name,
            ) not in atlas_lookup:
                raise RuntimeError(
                    "Missing physical atlas pair: "
                    f"{cid}/{beta_name}"
                )


    # ========================================================
    # Frozen labels
    # ========================================================

    label_rows = read_csv(
        LABEL_CSV
    )

    label_lookup = {}

    preference_order = []

    for row in label_rows:
        key = (
            row[
                "context_id"
            ],
            row[
                "preference_name"
            ],
        )

        if key in label_lookup:
            raise RuntimeError(
                f"Duplicate label: {key}"
            )

        label_lookup[
            key
        ] = row

        if (
            row[
                "context_id"
            ]
            == first_rough
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
    # Reconstruct the exact T5.5d context-local
    # Pareto ideal/nadir scalarization reference.
    # ========================================================

    regret_reference = {}

    for cid in rough_contexts:
        rows = [
            atlas_lookup[
                (
                    cid,
                    beta_name,
                )
            ]
            for beta_name in beta_order
        ]

        values = np.asarray(
            [
                [
                    finite(
                        row[column]
                    )
                    for column in (
                        OBJECTIVE_COLUMNS
                    )
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        nondominated = []

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
                nondominated.append(
                    i
                )

        if not nondominated:
            raise RuntimeError(
                f"{cid}: empty Pareto front."
            )

        front_values = (
            values[
                nondominated
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

        regret_reference[
            cid
        ] = (
            ideal,
            nadir - ideal,
        )


    def score_beta(
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
                    OBJECTIVE_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        ideal, span = (
            regret_reference[
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
    # 1-NN preference-transfer baseline using c_OS_v2
    # ========================================================

    transfer_rows = []

    nearest_context_by_test = {}

    for i, test_cid in enumerate(
        rough_contexts
    ):
        candidates = []

        for j, train_cid in enumerate(
            rough_contexts
        ):
            if i == j:
                continue

            distance = float(
                np.linalg.norm(
                    z_context[i]
                    - z_context[j]
                )
            )

            candidates.append(
                (
                    distance,
                    j,
                    train_cid,
                )
            )

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        nearest_distance, _, nearest_cid = (
            candidates[
                0
            ]
        )

        nearest_context_by_test[
            test_cid
        ] = {
            "nearest_context":
                nearest_cid,

            "distance":
                nearest_distance,
        }


        for preference_name in (
            preference_order
        ):
            source_label = (
                label_lookup[
                    (
                        nearest_cid,
                        preference_name,
                    )
                ]
            )

            target_label = (
                label_lookup[
                    (
                        test_cid,
                        preference_name,
                    )
                ]
            )

            transferred_beta = (
                source_label[
                    "selected_beta_name"
                ]
            )

            oracle_beta = (
                target_label[
                    "selected_beta_name"
                ]
            )

            w = np.asarray(
                [
                    finite(
                        target_label[
                            "w_motion"
                        ]
                    ),
                    finite(
                        target_label[
                            "w_stability"
                        ]
                    ),
                    finite(
                        target_label[
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
                )
            )

            oracle_score = finite(
                target_label[
                    "scalarization_score"
                ]
            )

            excess = (
                transferred_score
                - oracle_score
            )

            if excess < -1e-8:
                raise RuntimeError(
                    "Transferred score beats "
                    "oracle beyond tolerance: "
                    f"{test_cid}/"
                    f"{preference_name}/"
                    f"{excess}"
                )

            excess = max(
                0.0,
                excess,
            )

            transfer_rows.append(
                {
                    "test_context":
                        test_cid,

                    "nearest_context":
                        nearest_cid,

                    "context_distance":
                        nearest_distance,

                    "preference_name":
                        preference_name,

                    "transferred_beta_name":
                        transferred_beta,

                    "oracle_beta_name":
                        oracle_beta,

                    "exact":
                        int(
                            transferred_beta
                            == oracle_beta
                        ),

                    "oracle_score":
                        oracle_score,

                    "transferred_score":
                        transferred_score,

                    "score_excess":
                        excess,
                }
            )


    if len(
        transfer_rows
    ) != EXPECTED_TRANSFER_ROWS:
        raise RuntimeError(
            "Transfer-row count mismatch."
        )


    # ========================================================
    # Pairwise response-surface geometry
    #
    # Preserve the exact T5.5e1b response-distance
    # normalization semantics for apples-to-apples comparison.
    # ========================================================

    rough_values = []

    for cid in rough_contexts:
        for beta_name in beta_order:
            row = atlas_lookup[
                (
                    cid,
                    beta_name,
                )
            ]

            rough_values.append(
                [
                    finite(
                        row[column]
                    )
                    for column in (
                        OBJECTIVE_COLUMNS
                    )
                ]
            )


    rough_values = np.asarray(
        rough_values,
        dtype=np.float64,
    )

    response_mean = np.mean(
        rough_values,
        axis=0,
    )

    response_std = np.std(
        rough_values,
        axis=0,
        ddof=0,
    )

    if np.any(
        response_std <= EPS
    ):
        raise RuntimeError(
            "Degenerate response dimension."
        )


    response_vector = {}

    for cid in rough_contexts:
        values = np.asarray(
            [
                [
                    finite(
                        atlas_lookup[
                            (
                                cid,
                                beta_name,
                            )
                        ][column]
                    )
                    for column in (
                        OBJECTIVE_COLUMNS
                    )
                ]
                for beta_name in (
                    beta_order
                )
            ],
            dtype=np.float64,
        )

        values = (
            values
            - response_mean
        ) / response_std

        response_vector[
            cid
        ] = values.reshape(
            -1
        )


    pairwise_rows = []

    context_distances = []
    response_distances = []

    for i in range(
        len(
            rough_contexts
        )
    ):
        for j in range(
            i + 1,
            len(
                rough_contexts
            ),
        ):
            ci = (
                rough_contexts[i]
            )

            cj = (
                rough_contexts[j]
            )

            dc = float(
                np.linalg.norm(
                    z_context[i]
                    - z_context[j]
                )
            )

            dr = float(
                np.linalg.norm(
                    response_vector[ci]
                    - response_vector[cj]
                )
                / math.sqrt(
                    len(
                        response_vector[ci]
                    )
                )
            )

            context_distances.append(
                dc
            )

            response_distances.append(
                dr
            )

            pairwise_rows.append(
                {
                    "context_a":
                        ci,

                    "context_b":
                        cj,

                    "context_distance":
                        dc,

                    "response_distance":
                        dr,
                }
            )


    if len(
        pairwise_rows
    ) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            "Pairwise-row count mismatch."
        )


    correlation = pearson(
        context_distances,
        response_distances,
    )


    # ========================================================
    # v2 metrics
    # ========================================================

    score_excess = np.asarray(
        [
            row[
                "score_excess"
            ]
            for row in transfer_rows
        ],
        dtype=np.float64,
    )

    exact = np.asarray(
        [
            row[
                "exact"
            ]
            for row in transfer_rows
        ],
        dtype=np.float64,
    )


    v2_metrics = {
        "exact_accuracy":
            float(
                np.mean(
                    exact
                )
            ),

        "score_excess_mean":
            float(
                np.mean(
                    score_excess
                )
            ),

        "score_excess_median":
            float(
                np.median(
                    score_excess
                )
            ),

        "score_excess_p95":
            float(
                np.quantile(
                    score_excess,
                    0.95,
                )
            ),

        "score_excess_max":
            float(
                np.max(
                    score_excess
                )
            ),

        "score_excess_le_0p05_fraction":
            float(
                np.mean(
                    score_excess
                    <= 0.05
                )
            ),

        "pearson_distance_correlation":
            correlation,
    }


    # ========================================================
    # Compare against frozen v1 audit
    # ========================================================

    v1 = json.loads(
        V1_AUDIT_MANIFEST.read_text()
    )

    if v1.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "Frozen v1 identifiability audit "
            "is not COMPUTE_PASS."
        )

    v1_nn = (
        v1[
            "nearest_context_transfer"
        ]
    )

    v1_corr = finite(
        v1[
            "pairwise_context_response"
        ][
            "pearson_distance_correlation"
        ]
    )


    comparison = {
        "score_excess_mean_v1":
            finite(
                v1_nn[
                    "score_excess_mean"
                ]
            ),

        "score_excess_mean_v2":
            v2_metrics[
                "score_excess_mean"
            ],

        "score_excess_mean_change_v2_minus_v1":
            (
                v2_metrics[
                    "score_excess_mean"
                ]
                - finite(
                    v1_nn[
                        "score_excess_mean"
                    ]
                )
            ),

        "score_excess_p95_v1":
            finite(
                v1_nn[
                    "score_excess_p95"
                ]
            ),

        "score_excess_p95_v2":
            v2_metrics[
                "score_excess_p95"
            ],

        "score_excess_le_0p05_fraction_v1":
            finite(
                v1_nn[
                    "score_excess_le_0p05_fraction"
                ]
            ),

        "score_excess_le_0p05_fraction_v2":
            v2_metrics[
                "score_excess_le_0p05_fraction"
            ],

        "pearson_correlation_v1":
            v1_corr,

        "pearson_correlation_v2":
            correlation,

        "pearson_change_v2_minus_v1":
            (
                correlation
                - v1_corr
            ),
    }


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_NN,
        transfer_rows,
    )

    write_csv(
        OUT_PAIRWISE,
        pairwise_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1d_"
                "context_identifiability_v2_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "context_source":
            str(
                CONTEXT_V2_CSV.relative_to(
                    ROOT
                )
            ),

        "context_dim":
            EXPECTED_CONTEXT_DIM,

        "active_rough_context_dimensions":
            active_features,

        "active_rough_context_dim":
            len(
                active_features
            ),

        "rough_contexts":
            len(
                rough_contexts
            ),

        "beta_points":
            len(
                beta_order
            ),

        "preferences":
            len(
                preference_order
            ),

        "nearest_context_transfer":
            v2_metrics,

        "pairwise_context_response":
            {
                "pairs":
                    len(
                        pairwise_rows
                    ),

                "pearson_distance_correlation":
                    correlation,
            },

        "v1_to_v2_comparison":
            comparison,

        "heldout_used":
            False,

        "interpretation_guard":
            (
                "Only the TRAIN rough context "
                "descriptor representation changed. "
                "Frozen physical atlas, Pareto "
                "scalarization, oracle labels, and "
                "held-out policy evaluation remain "
                "unchanged."
            ),

        "next_stage":
            (
                "Compare c_OS_v2 identifiability "
                "against v1. Retrain the supervised "
                "Objective Selector only if physical "
                "context-to-response geometry improves "
                "meaningfully."
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
    print("=" * 116)
    print(
        "ICRA27 OS-T5.5e1d "
        "c_OS_v2 CONTEXT IDENTIFIABILITY AUDIT"
    )
    print("=" * 116)

    print(
        "rough contexts :",
        len(
            rough_contexts
        ),
    )

    print(
        "context dim    :",
        EXPECTED_CONTEXT_DIM,
    )

    print(
        "active rough   :",
        len(
            active_features
        ),
    )

    print()
    print("1-NN CONTEXT TRANSFER")

    for key in (
        "exact_accuracy",
        "score_excess_mean",
        "score_excess_median",
        "score_excess_p95",
        "score_excess_max",
        "score_excess_le_0p05_fraction",
    ):
        print(
            f"  {key:<36}: "
            f"{v2_metrics[key]}"
        )

    print()
    print("CONTEXT <-> RESPONSE GEOMETRY")

    print(
        "  pair count                         :",
        len(
            pairwise_rows
        ),
    )

    print(
        "  Pearson distance correlation       :",
        correlation,
    )

    print()
    print("v1 -> v2")

    print(
        "  mean score excess : "
        f"{comparison['score_excess_mean_v1']:.6f}"
        " -> "
        f"{comparison['score_excess_mean_v2']:.6f}"
    )

    print(
        "  p95 score excess  : "
        f"{comparison['score_excess_p95_v1']:.6f}"
        " -> "
        f"{comparison['score_excess_p95_v2']:.6f}"
    )

    print(
        "  <=0.05 fraction   : "
        f"{comparison['score_excess_le_0p05_fraction_v1']:.6f}"
        " -> "
        f"{comparison['score_excess_le_0p05_fraction_v2']:.6f}"
    )

    print(
        "  Pearson corr      : "
        f"{comparison['pearson_correlation_v1']:.6f}"
        " -> "
        f"{comparison['pearson_correlation_v2']:.6f}"
    )

    print()
    print("outputs:")
    print(" ", OUT_NN)
    print(" ", OUT_PAIRWISE)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1d "
        "c_OS_v2 identifiability: COMPUTE PASS"
    )

    print("=" * 116)


if __name__ == "__main__":
    main()
