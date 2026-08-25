from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


PATCH_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_height_patch_v0"
)

PATCH_CSV = (
    PATCH_ROOT
    / "oracle_height_patches.csv"
)

PATCH_CONTRACT = (
    PATCH_ROOT
    / "oracle_height_patch_contract.json"
)


PAIRWISE_V2 = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1d_context_identifiability_v2_v0"
    / "rough_context_pairwise_similarity.csv"
)

V1_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1b_context_identifiability_v0"
    / "identifiability_manifest.json"
)

V2_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1d_context_identifiability_v2_v0"
    / "identifiability_v2_manifest.json"
)


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


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1j_height_patch_identifiability_v0"
)

OUT_PAIRWISE = (
    OUT_DIR
    / "height_patch_pairwise_similarity.csv"
)

OUT_TRANSFER = (
    OUT_DIR
    / "nearest_patch_preference_transfer.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "nearest_patch_context_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "height_patch_identifiability_manifest.json"
)


OBJECTIVES = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

N_LONGITUDINAL = 8
N_LATERAL = 6
PATCH_DIM = 48

RHO = 0.01
TOL = 1e-10
EPS = 1e-15


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

    denominator = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denominator <= EPS:
        return float("nan")

    return float(
        np.dot(
            x,
            y,
        )
        / denominator
    )


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
        PATCH_CSV,
        PATCH_CONTRACT,
        PAIRWISE_V2,
        V1_MANIFEST,
        V2_MANIFEST,
        ATLAS,
        LABELS,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Patch contract
    # ========================================================

    contract = json.loads(
        PATCH_CONTRACT.read_text()
    )

    if contract.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Height patch is not FREEZE_PASS."
        )

    if list(
        contract[
            "patch_shape"
        ]
    ) != [
        N_LONGITUDINAL,
        N_LATERAL,
    ]:
        raise RuntimeError(
            "Patch shape mismatch."
        )

    if int(
        contract[
            "patch_dim"
        ]
    ) != PATCH_DIM:
        raise RuntimeError(
            "Patch dimension mismatch."
        )

    if bool(
        contract[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "Height-patch artifact used heldout."
        )


    patch_features = [
        name
        for name in (
            contract[
                "feature_order"
            ]
        )
        if name.startswith(
            "h_l"
        )
    ]


    if len(
        patch_features
    ) != PATCH_DIM:
        raise RuntimeError(
            "Expected 48 patch features."
        )


    patch_rows = read_csv(
        PATCH_CSV
    )

    rough_rows = [
        row
        for row in patch_rows
        if row[
            "context_id"
        ].startswith(
            "rough_seed_"
        )
    ]


    if len(
        rough_rows
    ) != 18:
        raise RuntimeError(
            "Expected 18 rough patches."
        )


    patch_lookup = {
        row[
            "context_id"
        ]:
            np.asarray(
                [
                    finite(
                        row[
                            feature
                        ]
                    )
                    for feature in (
                        patch_features
                    )
                ],
                dtype=np.float64,
            )
        for row in rough_rows
    }


    rough_contexts = [
        row[
            "context_id"
        ]
        for row in rough_rows
    ]


    # ========================================================
    # Reuse exact frozen response-distance definition
    # ========================================================

    response_pair_rows = read_csv(
        PAIRWISE_V2
    )

    response_lookup = {}


    for row in response_pair_rows:
        key = frozenset(
            (
                row[
                    "context_a"
                ],
                row[
                    "context_b"
                ],
            )
        )

        response_lookup[
            key
        ] = finite(
            row[
                "response_distance"
            ]
        )


    # ========================================================
    # Pairwise patch geometry
    # ========================================================

    pairwise_rows = []

    patch_distances = []
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
                rough_contexts[
                    i
                ]
            )

            cj = (
                rough_contexts[
                    j
                ]
            )


            diff = (
                patch_lookup[
                    ci
                ]
                - patch_lookup[
                    cj
                ]
            )


            # Physical RMS terrain-height difference.
            patch_distance = float(
                math.sqrt(
                    np.mean(
                        diff
                        * diff
                    )
                )
            )


            key = frozenset(
                (
                    ci,
                    cj,
                )
            )


            if key not in (
                response_lookup
            ):
                raise RuntimeError(
                    f"Missing response pair: "
                    f"{ci}/{cj}"
                )


            response_distance = (
                response_lookup[
                    key
                ]
            )


            pairwise_rows.append(
                {
                    "context_a":
                        ci,

                    "context_b":
                        cj,

                    "height_patch_rms_distance_m":
                        patch_distance,

                    "response_distance":
                        response_distance,
                }
            )


            patch_distances.append(
                patch_distance
            )

            response_distances.append(
                response_distance
            )


    if len(
        pairwise_rows
    ) != 153:
        raise RuntimeError(
            "Expected 153 rough pairs."
        )


    patch_response_corr = pearson(
        patch_distances,
        response_distances,
    )


    # ========================================================
    # Nearest patch context
    # ========================================================

    nearest = {}


    for test_cid in (
        rough_contexts
    ):
        candidates = []

        for train_cid in (
            rough_contexts
        ):
            if train_cid == test_cid:
                continue

            diff = (
                patch_lookup[
                    test_cid
                ]
                - patch_lookup[
                    train_cid
                ]
            )

            distance = float(
                math.sqrt(
                    np.mean(
                        diff
                        * diff
                    )
                )
            )

            candidates.append(
                (
                    distance,
                    train_cid,
                )
            )


        candidates.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )


        nearest[
            test_cid
        ] = (
            candidates[
                0
            ]
        )


    # ========================================================
    # Frozen atlas / labels
    # ========================================================

    atlas = read_csv(
        ATLAS
    )

    by_context = defaultdict(
        list
    )

    atlas_lookup = {}


    for row in atlas:
        cid = row[
            "context_id"
        ]

        by_context[
            cid
        ].append(
            row
        )

        atlas_lookup[
            (
                cid,
                row[
                    "beta_name"
                ],
            )
        ] = row


    labels = read_csv(
        LABELS
    )

    label_lookup = {}

    preference_order = []


    first_rough = (
        rough_contexts[
            0
        ]
    )


    for row in labels:
        key = (
            row[
                "context_id"
            ],
            row[
                "preference_name"
            ],
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


    if len(
        preference_order
    ) != 25:
        raise RuntimeError(
            "Expected 25 preferences."
        )


    # ========================================================
    # ORIGINAL frozen T5.5d score
    # ========================================================

    reference = {}


    for cid in rough_contexts:
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

        reference[
            cid
        ] = (
            ideal,
            nadir - ideal,
        )


    def score(
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
            reference[
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
    # Nearest-context preference transfer
    # ========================================================

    transfer_rows = []

    context_summary = []


    all_excess = []
    all_exact = []

    max_oracle_score_error = 0.0


    for test_cid in (
        rough_contexts
    ):
        distance, nearest_cid = (
            nearest[
                test_cid
            ]
        )

        context_excess = []
        context_exact = []


        for preference_name in (
            preference_order
        ):
            source = (
                label_lookup[
                    (
                        nearest_cid,
                        preference_name,
                    )
                ]
            )

            target = (
                label_lookup[
                    (
                        test_cid,
                        preference_name,
                    )
                ]
            )


            transferred_beta = (
                source[
                    "selected_beta_name"
                ]
            )

            oracle_beta = (
                target[
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


            oracle_score_reconstructed = (
                score(
                    context_id=(
                        test_cid
                    ),
                    beta_name=(
                        oracle_beta
                    ),
                    w=w,
                )
            )

            oracle_score_frozen = finite(
                target[
                    "scalarization_score"
                ]
            )


            oracle_error = abs(
                oracle_score_reconstructed
                - oracle_score_frozen
            )

            max_oracle_score_error = max(
                max_oracle_score_error,
                oracle_error,
            )


            if oracle_error > 1e-9:
                raise RuntimeError(
                    "Frozen oracle score "
                    "reconstruction failure."
                )


            transferred_score = (
                score(
                    context_id=(
                        test_cid
                    ),
                    beta_name=(
                        transferred_beta
                    ),
                    w=w,
                )
            )


            excess = (
                transferred_score
                - oracle_score_frozen
            )


            if excess < -1e-8:
                raise RuntimeError(
                    "Nearest-patch transfer "
                    "beats frozen oracle."
                )


            excess = max(
                0.0,
                excess,
            )

            exact = int(
                transferred_beta
                == oracle_beta
            )


            transfer_rows.append(
                {
                    "test_context":
                        test_cid,

                    "nearest_patch_context":
                        nearest_cid,

                    "height_patch_rms_distance_m":
                        distance,

                    "preference_name":
                        preference_name,

                    "transferred_beta_name":
                        transferred_beta,

                    "oracle_beta_name":
                        oracle_beta,

                    "exact":
                        exact,

                    "oracle_score":
                        oracle_score_frozen,

                    "transferred_score":
                        transferred_score,

                    "score_excess":
                        excess,
                }
            )


            context_excess.append(
                excess
            )

            context_exact.append(
                exact
            )

            all_excess.append(
                excess
            )

            all_exact.append(
                exact
            )


        context_summary.append(
            {
                "context_id":
                    test_cid,

                "nearest_patch_context":
                    nearest_cid,

                "height_patch_rms_distance_m":
                    distance,

                "exact_accuracy":
                    float(
                        np.mean(
                            context_exact
                        )
                    ),

                "score_excess_mean":
                    float(
                        np.mean(
                            context_excess
                        )
                    ),

                "score_excess_p95":
                    float(
                        np.quantile(
                            context_excess,
                            0.95,
                            method="linear",
                        )
                    ),
            }
        )


    x = np.asarray(
        all_excess,
        dtype=np.float64,
    )


    overall = {
        "exact_accuracy":
            float(
                np.mean(
                    all_exact
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
                    method="linear",
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


    # ========================================================
    # Historical comparison
    # ========================================================

    v1 = json.loads(
        V1_MANIFEST.read_text()
    )

    v2 = json.loads(
        V2_MANIFEST.read_text()
    )


    v1_corr = finite(
        v1[
            "pairwise_context_response"
        ][
            "pearson_distance_correlation"
        ]
    )

    v2_corr = finite(
        v2[
            "pairwise_context_response"
        ][
            "pearson_distance_correlation"
        ]
    )


    v1_mean = finite(
        v1[
            "nearest_context_transfer"
        ][
            "score_excess_mean"
        ]
    )

    v2_mean = finite(
        v2[
            "nearest_context_transfer"
        ][
            "score_excess_mean"
        ]
    )


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    write_csv(
        OUT_PAIRWISE,
        pairwise_rows,
    )

    write_csv(
        OUT_TRANSFER,
        transfer_rows,
    )

    write_csv(
        OUT_CONTEXT,
        context_summary,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1j_"
                "height_patch_identifiability_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "rough_contexts":
            18,

        "pair_count":
            len(
                pairwise_rows
            ),

        "preferences":
            len(
                preference_order
            ),

        "patch_shape":
            [
                N_LONGITUDINAL,
                N_LATERAL,
            ],

        "patch_distance":
            (
                "RMS cellwise difference in "
                "start-height-relative body-aligned "
                "local terrain heights; units meters."
            ),

        "response_distance":
            (
                "Reused unchanged from "
                "OS-T5.5e1d for direct "
                "v1/v2/patch comparison."
            ),

        "patch_response_pearson":
            patch_response_corr,

        "nearest_patch_transfer":
            overall,

        "oracle_score_reconstruction_max_error":
            max_oracle_score_error,

        "comparison":
            {
                "v1_context_response_pearson":
                    v1_corr,

                "v2_context_response_pearson":
                    v2_corr,

                "patch_context_response_pearson":
                    patch_response_corr,

                "v1_nearest_transfer_mean_excess":
                    v1_mean,

                "v2_nearest_transfer_mean_excess":
                    v2_mean,

                "patch_nearest_transfer_mean_excess":
                    overall[
                        "score_excess_mean"
                    ],
            },

        "interpretation_guard":
            (
                "TRAIN-only representation audit. "
                "No selector is trained and no "
                "held-out terrain outcome is used."
            ),

        "next_stage":
            (
                "If the spatial patch materially "
                "improves response geometry and/or "
                "nearest-context transfer, use the "
                "patch as the simple oracle terrain "
                "representation for the next "
                "Objective Selector LOCO experiment."
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
    print("=" * 118)
    print(
        "ICRA27 OS-T5.5e1j LOCAL HEIGHT "
        "PATCH IDENTIFIABILITY AUDIT"
    )
    print("=" * 118)

    print(
        "rough contexts:",
        len(
            rough_contexts
        ),
    )

    print(
        "patch shape   :",
        (
            N_LONGITUDINAL,
            N_LATERAL,
        ),
    )

    print()
    print("CONTEXT -> RESPONSE GEOMETRY")

    print(
        f"  v1 scalar descriptor : "
        f"{v1_corr:.6f}"
    )

    print(
        f"  v2 scalar descriptor : "
        f"{v2_corr:.6f}"
    )

    print(
        f"  raw height patch     : "
        f"{patch_response_corr:.6f}"
    )


    print()
    print("NEAREST-CONTEXT FROZEN-SCORE TRANSFER")

    print(
        f"  v1 mean excess       : "
        f"{v1_mean:.6f}"
    )

    print(
        f"  v2 mean excess       : "
        f"{v2_mean:.6f}"
    )

    print(
        f"  patch mean excess    : "
        f"{overall['score_excess_mean']:.6f}"
    )

    print(
        f"  patch median         : "
        f"{overall['score_excess_median']:.6f}"
    )

    print(
        f"  patch p95            : "
        f"{overall['score_excess_p95']:.6f}"
    )

    print(
        f"  patch exact          : "
        f"{overall['exact_accuracy']:.3f}"
    )

    print(
        f"  patch <=0.05         : "
        f"{overall['score_excess_le_0p05_fraction']:.3f}"
    )

    print()
    print(
        "oracle score reconstruction max err:",
        max_oracle_score_error,
    )

    print()
    print("outputs:")
    print(" ", OUT_PAIRWISE)
    print(" ", OUT_TRANSFER)
    print(" ", OUT_CONTEXT)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1j local height "
        "patch identifiability: COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
