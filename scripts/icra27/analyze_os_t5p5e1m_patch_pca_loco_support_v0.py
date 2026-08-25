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

PATCH_CSV = PATCH_ROOT / "oracle_height_patches.csv"
PATCH_CONTRACT = PATCH_ROOT / "oracle_height_patch_contract.json"


LOCO_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1l_selector_patch_pca_loco_v0"
)

PREDICTIONS = LOCO_ROOT / "loco_predictions.csv"
CONTEXT_SUMMARY = LOCO_ROOT / "loco_context_summary.csv"
LOCO_MANIFEST = LOCO_ROOT / "selector_patch_pca_loco_manifest.json"


ATLAS = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1m_patch_pca_loco_support_v0"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_support_physical_audit.csv"
)

OUT_PREDICTIONS = (
    OUT_DIR
    / "prediction_physical_deltas.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "support_audit_manifest.json"
)


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

PATCH_DIM = 48
EXPECTED_ROUGH = 18
EXPECTED_PREDICTIONS = 450

PCA_THRESHOLD = 0.95

EPS = 1.0e-12


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite: {x!r}"
        )

    return y


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows: {path}"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def pearson(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x = x - np.mean(x)
    y = y - np.mean(y)

    denominator = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denominator <= EPS:
        return float("nan")

    return float(
        np.dot(x, y)
        / denominator
    )


def nearest_neighbor_distances(
    z,
):
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    distances = []

    for i in range(
        len(z)
    ):
        d = np.linalg.norm(
            z
            - z[i],
            axis=1,
        )

        d[i] = np.inf

        distances.append(
            float(
                np.min(d)
            )
        )

    return np.asarray(
        distances,
        dtype=np.float64,
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    for path in (
        PATCH_CSV,
        PATCH_CONTRACT,
        PREDICTIONS,
        CONTEXT_SUMMARY,
        LOCO_MANIFEST,
        ATLAS,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Provenance
    # ========================================================

    patch_contract = json.loads(
        PATCH_CONTRACT.read_text()
    )

    if patch_contract.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Patch contract not FREEZE_PASS."
        )

    if int(
        patch_contract[
            "patch_dim"
        ]
    ) != PATCH_DIM:
        raise RuntimeError(
            "Patch dimension mismatch."
        )


    loco_manifest = json.loads(
        LOCO_MANIFEST.read_text()
    )

    if loco_manifest.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5e1l is not COMPUTE_PASS."
        )

    if bool(
        loco_manifest[
            "external_heldout_used"
        ]
    ):
        raise RuntimeError(
            "External heldout was used."
        )


    # ========================================================
    # Patch matrix
    # ========================================================

    patch_features = [
        feature
        for feature in (
            patch_contract[
                "feature_order"
            ]
        )
        if feature.startswith(
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


    patch_lookup = {
        row[
            "context_id"
        ]:
            np.asarray(
                [
                    finite(
                        row[feature]
                    )
                    for feature in (
                        patch_features
                    )
                ],
                dtype=np.float64,
            )
        for row in patch_rows
    }


    rough_contexts = [
        row[
            "context_id"
        ]
        for row in patch_rows
        if row[
            "context_id"
        ].startswith(
            "rough_seed_"
        )
    ]


    if len(
        rough_contexts
    ) != EXPECTED_ROUGH:
        raise RuntimeError(
            "Expected 18 rough contexts."
        )


    # ========================================================
    # Physical atlas + global descriptive IQR
    # ========================================================

    atlas = read_csv(
        ATLAS
    )


    atlas_lookup = {
        (
            row[
                "context_id"
            ],
            row[
                "beta_name"
            ],
        ):
            row
        for row in atlas
    }


    iqr = {}

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

        q25 = float(
            np.quantile(
                values,
                0.25,
            )
        )

        q75 = float(
            np.quantile(
                values,
                0.75,
            )
        )

        value = (
            q75
            - q25
        )

        if value <= EPS:
            raise RuntimeError(
                f"Degenerate IQR: "
                f"{objective_name}"
            )

        iqr[
            objective_name
        ] = value


    # ========================================================
    # Prediction physical deltas
    # ========================================================

    prediction_rows = read_csv(
        PREDICTIONS
    )


    if len(
        prediction_rows
    ) != EXPECTED_PREDICTIONS:
        raise RuntimeError(
            "Expected 450 LOCO predictions."
        )


    physical_prediction_rows = []

    physical_by_context = defaultdict(
        list
    )


    for row in prediction_rows:
        cid = row[
            "held_context"
        ]

        target_beta = row[
            "target_beta_name"
        ]

        predicted_beta = row[
            "projected_beta_name"
        ]


        target = atlas_lookup[
            (
                cid,
                target_beta,
            )
        ]

        predicted = atlas_lookup[
            (
                cid,
                predicted_beta,
            )
        ]


        out = {
            "held_context":
                cid,

            "preference_name":
                row[
                    "preference_name"
                ],

            "target_beta_name":
                target_beta,

            "projected_beta_name":
                predicted_beta,

            "score_excess":
                finite(
                    row[
                        "score_excess"
                    ]
                ),
        }


        normalized = []


        for objective_name, column in (
            OBJECTIVES
        ):
            target_value = finite(
                target[
                    column
                ]
            )

            predicted_value = finite(
                predicted[
                    column
                ]
            )

            delta = (
                predicted_value
                - target_value
            )

            abs_over_iqr = (
                abs(delta)
                / iqr[
                    objective_name
                ]
            )


            out[
                f"delta_{objective_name}"
            ] = delta

            out[
                f"abs_delta_{objective_name}_over_iqr"
            ] = abs_over_iqr


            normalized.append(
                abs_over_iqr
            )


        out[
            "physical_l1_mean_iqr"
        ] = float(
            np.mean(
                normalized
            )
        )

        out[
            "physical_linf_iqr"
        ] = float(
            np.max(
                normalized
            )
        )


        physical_prediction_rows.append(
            out
        )

        physical_by_context[
            cid
        ].append(
            out
        )


    # ========================================================
    # Strict fold-local PCA support diagnostics
    # ========================================================

    loco_context_rows = read_csv(
        CONTEXT_SUMMARY
    )

    loco_context_lookup = {
        row[
            "held_context"
        ]:
            row
        for row in (
            loco_context_rows
        )
    }


    context_rows = []


    for fold_index, test_cid in enumerate(
        rough_contexts,
        start=1,
    ):
        rough_train = [
            cid
            for cid in rough_contexts
            if cid != test_cid
        ]


        X_train = np.asarray(
            [
                patch_lookup[
                    cid
                ]
                for cid in rough_train
            ],
            dtype=np.float64,
        )


        x_test = patch_lookup[
            test_cid
        ]


        # ----------------------------------------------------
        # Exact fold-local PCA reconstruction
        # ----------------------------------------------------

        mean = np.mean(
            X_train,
            axis=0,
        )

        Xc = (
            X_train
            - mean
        )


        _, singular_values, Vt = (
            np.linalg.svd(
                Xc,
                full_matrices=False,
            )
        )


        energy = (
            singular_values
            ** 2
        )

        total_energy = float(
            np.sum(
                energy
            )
        )

        if total_energy <= EPS:
            raise RuntimeError(
                "Degenerate fold PCA."
            )


        cumulative = np.cumsum(
            energy
            / total_energy
        )


        rank = int(
            np.sum(
                singular_values
                > 1e-12
            )
        )


        k95 = int(
            np.searchsorted(
                cumulative,
                PCA_THRESHOLD,
                side="left",
            )
            + 1
        )

        k95 = min(
            k95,
            rank,
        )


        expected_k95 = int(
            float(
                loco_context_lookup[
                    test_cid
                ][
                    "k95"
                ]
            )
        )


        if k95 != expected_k95:
            raise RuntimeError(
                "Fold-local k95 regression "
                f"failed for {test_cid}: "
                f"{k95} vs "
                f"{expected_k95}"
            )


        basis = Vt[
            :k95
        ]


        Z_train = (
            Xc
            @ basis.T
        )


        z_test = (
            (
                x_test
                - mean
            )
            @ basis.T
        )


        # ----------------------------------------------------
        # 1) Nearest latent-distance support
        # ----------------------------------------------------

        held_distances = np.linalg.norm(
            Z_train
            - z_test[
                None,
                :
            ],
            axis=1,
        )

        held_nn = float(
            np.min(
                held_distances
            )
        )


        train_nn = (
            nearest_neighbor_distances(
                Z_train
            )
        )


        train_nn_median = float(
            np.median(
                train_nn
            )
        )

        train_nn_max = float(
            np.max(
                train_nn
            )
        )


        held_nn_over_train_median = (
            held_nn
            / max(
                train_nn_median,
                EPS,
            )
        )


        held_nn_over_train_max = (
            held_nn
            / max(
                train_nn_max,
                EPS,
            )
        )


        # ----------------------------------------------------
        # 2) PCA-whitened radius
        #
        # eigenvalue = s^2/(N-1)
        # ----------------------------------------------------

        eigenvalues = (
            singular_values[
                :k95
            ]
            ** 2
            / (
                len(
                    rough_train
                )
                - 1
            )
        )


        scale = np.sqrt(
            np.maximum(
                eigenvalues,
                EPS,
            )
        )


        train_whitened = (
            Z_train
            / scale[
                None,
                :
            ]
        )

        test_whitened = (
            z_test
            / scale
        )


        train_radius = np.sqrt(
            np.mean(
                train_whitened
                ** 2,
                axis=1,
            )
        )


        held_radius = float(
            math.sqrt(
                np.mean(
                    test_whitened
                    ** 2
                )
            )
        )


        train_radius_median = float(
            np.median(
                train_radius
            )
        )

        train_radius_max = float(
            np.max(
                train_radius
            )
        )


        # ----------------------------------------------------
        # 3) Coordinate-box extrapolation
        # ----------------------------------------------------

        z_min = np.min(
            Z_train,
            axis=0,
        )

        z_max = np.max(
            Z_train,
            axis=0,
        )


        outside = (
            (z_test < z_min)
            | (z_test > z_max)
        )


        outside_dims = int(
            np.sum(
                outside
            )
        )


        # ----------------------------------------------------
        # 4) Held patch PCA reconstruction residual
        # ----------------------------------------------------

        recon_test = (
            mean
            + z_test
            @ basis
        )


        held_residual_rms = float(
            math.sqrt(
                np.mean(
                    (
                        x_test
                        - recon_test
                    )
                    ** 2
                )
            )
        )


        train_recon = (
            mean[
                None,
                :
            ]
            + Z_train
            @ basis
        )


        train_residual_rms = np.sqrt(
            np.mean(
                (
                    X_train
                    - train_recon
                )
                ** 2,
                axis=1,
            )
        )


        train_residual_median = float(
            np.median(
                train_residual_rms
            )
        )

        train_residual_max = float(
            np.max(
                train_residual_rms
            )
        )


        # ----------------------------------------------------
        # Physical selector error for this held context
        # ----------------------------------------------------

        physical_rows = (
            physical_by_context[
                test_cid
            ]
        )


        linf = np.asarray(
            [
                row[
                    "physical_linf_iqr"
                ]
                for row in physical_rows
            ],
            dtype=np.float64,
        )


        score = np.asarray(
            [
                finite(
                    row[
                        "score_excess"
                    ]
                )
                for row in physical_rows
            ],
            dtype=np.float64,
        )


        context_rows.append(
            {
                "fold":
                    fold_index,

                "held_context":
                    test_cid,

                "pca_rank":
                    rank,

                "k95":
                    k95,

                "k95_explained_variance":
                    float(
                        cumulative[
                            k95 - 1
                        ]
                    ),

                "held_nn_latent":
                    held_nn,

                "train_nn_median":
                    train_nn_median,

                "train_nn_max":
                    train_nn_max,

                "held_nn_over_train_median":
                    held_nn_over_train_median,

                "held_nn_over_train_max":
                    held_nn_over_train_max,

                "held_whitened_rms_radius":
                    held_radius,

                "train_whitened_radius_median":
                    train_radius_median,

                "train_whitened_radius_max":
                    train_radius_max,

                "held_radius_over_train_max":
                    (
                        held_radius
                        / max(
                            train_radius_max,
                            EPS,
                        )
                    ),

                "outside_latent_box_dims":
                    outside_dims,

                "outside_latent_box_fraction":
                    (
                        outside_dims
                        / k95
                    ),

                "held_reconstruction_rms_m":
                    held_residual_rms,

                "train_reconstruction_rms_median":
                    train_residual_median,

                "train_reconstruction_rms_max":
                    train_residual_max,

                "held_reconstruction_over_train_max":
                    (
                        held_residual_rms
                        / max(
                            train_residual_max,
                            EPS,
                        )
                    ),

                "score_excess_mean":
                    float(
                        np.mean(
                            score
                        )
                    ),

                "score_excess_p95":
                    float(
                        np.quantile(
                            score,
                            0.95,
                        )
                    ),

                "physical_linf_mean_iqr":
                    float(
                        np.mean(
                            linf
                        )
                    ),

                "physical_linf_median_iqr":
                    float(
                        np.median(
                            linf
                        )
                    ),

                "physical_linf_p95_iqr":
                    float(
                        np.quantile(
                            linf,
                            0.95,
                        )
                    ),
            }
        )


    # ========================================================
    # Across-fold correlations
    # ========================================================

    score_mean = [
        row[
            "score_excess_mean"
        ]
        for row in context_rows
    ]

    physical_mean = [
        row[
            "physical_linf_mean_iqr"
        ]
        for row in context_rows
    ]


    support_metrics = (
        "held_nn_over_train_median",
        "held_nn_over_train_max",
        "held_radius_over_train_max",
        "outside_latent_box_fraction",
        "held_reconstruction_over_train_max",
    )


    correlations = {}


    for metric in (
        support_metrics
    ):
        values = [
            row[
                metric
            ]
            for row in context_rows
        ]

        correlations[
            metric
        ] = {
            "vs_score_excess_mean":
                pearson(
                    values,
                    score_mean,
                ),

            "vs_physical_linf_mean":
                pearson(
                    values,
                    physical_mean,
                ),
        }


    # ========================================================
    # Overall physical significance
    # ========================================================

    all_linf = np.asarray(
        [
            row[
                "physical_linf_iqr"
            ]
            for row in (
                physical_prediction_rows
            )
        ],
        dtype=np.float64,
    )


    all_l1 = np.asarray(
        [
            row[
                "physical_l1_mean_iqr"
            ]
            for row in (
                physical_prediction_rows
            )
        ],
        dtype=np.float64,
    )


    physical_aggregate = {
        "linf_mean":
            float(
                np.mean(
                    all_linf
                )
            ),

        "linf_median":
            float(
                np.median(
                    all_linf
                )
            ),

        "linf_p95":
            float(
                np.quantile(
                    all_linf,
                    0.95,
                )
            ),

        "linf_max":
            float(
                np.max(
                    all_linf
                )
            ),

        "l1_mean":
            float(
                np.mean(
                    all_l1
                )
            ),

        "l1_median":
            float(
                np.median(
                    all_l1
                )
            ),

        "l1_p95":
            float(
                np.quantile(
                    all_l1,
                    0.95,
                )
            ),
    }


    # ========================================================
    # 16-context diagnostic excluding the two largest
    # score-excess contexts. ANALYSIS ONLY.
    # ========================================================

    ranked = sorted(
        context_rows,
        key=lambda row:
            row[
                "score_excess_mean"
            ],
        reverse=True,
    )


    top_two = [
        ranked[
            0
        ][
            "held_context"
        ],
        ranked[
            1
        ][
            "held_context"
        ],
    ]


    remaining_predictions = [
        row
        for row in prediction_rows
        if row[
            "held_context"
        ] not in top_two
    ]


    remaining_score = np.asarray(
        [
            finite(
                row[
                    "score_excess"
                ]
            )
            for row in (
                remaining_predictions
            )
        ],
        dtype=np.float64,
    )


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    write_csv(
        OUT_CONTEXT,
        context_rows,
    )

    write_csv(
        OUT_PREDICTIONS,
        physical_prediction_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1m_"
                "patch_pca_loco_support_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "folds":
            EXPECTED_ROUGH,

        "predictions":
            EXPECTED_PREDICTIONS,

        "physical_iqr_reporting_scale":
            iqr,

        "physical_aggregate":
            physical_aggregate,

        "support_correlations":
            correlations,

        "top_two_score_failure_contexts":
            top_two,

        "remaining_16_context_diagnostic":
            {
                "rows":
                    len(
                        remaining_predictions
                    ),

                "score_excess_mean":
                    float(
                        np.mean(
                            remaining_score
                        )
                    ),

                "score_excess_median":
                    float(
                        np.median(
                            remaining_score
                        )
                    ),

                "score_excess_p95":
                    float(
                        np.quantile(
                            remaining_score,
                            0.95,
                        )
                    ),

                "score_excess_le_0p05_fraction":
                    float(
                        np.mean(
                            remaining_score
                            <= 0.05
                        )
                    ),
            },

        "interpretation_guard":
            (
                "Read-only TRAIN-context LOCO "
                "diagnostic. No selector, PCA "
                "contract, label, or held-out "
                "evaluation is modified."
            ),

        "purpose":
            (
                "Separate fold-local terrain-latent "
                "support/extrapolation failure from "
                "scalarization amplification and "
                "actual physical beta-response error."
            ),

        "next_stage":
            (
                "If catastrophic physical-error "
                "contexts are also latent-support "
                "outliers, add a simple support-aware "
                "selector/fallback or broaden TRAIN "
                "terrain support. If they are not "
                "latent outliers, revisit the "
                "terrain-to-beta mapping target rather "
                "than increasing network capacity."
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
    print("=" * 124)

    print(
        "ICRA27 OS-T5.5e1m PATCH-PCA "
        "LOCO SUPPORT + PHYSICAL ERROR AUDIT"
    )

    print("=" * 124)


    print()
    print("OVERALL PHYSICAL SELECTOR ERROR")

    for key, value in (
        physical_aggregate.items()
    ):
        print(
            f"  {key:<20}: "
            f"{value}"
        )


    print()
    print("PER-CONTEXT — SORTED BY SCORE FAILURE")

    for row in ranked:
        print(
            f"  {row['held_context']:<14} "
            f"scoreMean="
            f"{row['score_excess_mean']:.4f} "
            f"physLinf="
            f"{row['physical_linf_mean_iqr']:.4f} "
            f"nnRatio="
            f"{row['held_nn_over_train_median']:.3f} "
            f"rad/max="
            f"{row['held_radius_over_train_max']:.3f} "
            f"boxOut="
            f"{row['outside_latent_box_fraction']:.3f} "
            f"recon/max="
            f"{row['held_reconstruction_over_train_max']:.3f}"
        )


    print()
    print("SUPPORT CORRELATIONS")

    for metric, values in (
        correlations.items()
    ):
        print(
            f"  {metric:<42} "
            f"score="
            f"{values['vs_score_excess_mean']:+.4f} "
            f"physical="
            f"{values['vs_physical_linf_mean']:+.4f}"
        )


    print()
    print(
        "top two score-failure contexts:",
        top_two,
    )

    print(
        "remaining 16 score mean/p95:",
        manifest[
            "remaining_16_context_diagnostic"
        ][
            "score_excess_mean"
        ],
        "/",
        manifest[
            "remaining_16_context_diagnostic"
        ][
            "score_excess_p95"
        ],
    )


    print()
    print("outputs:")
    print(" ", OUT_CONTEXT)
    print(" ", OUT_PREDICTIONS)
    print(" ", OUT_MANIFEST)


    print()
    print(
        "[ICRA27] OS-T5.5e1m "
        "patch-PCA LOCO support audit: "
        "COMPUTE PASS"
    )

    print("=" * 124)


if __name__ == "__main__":
    main()
