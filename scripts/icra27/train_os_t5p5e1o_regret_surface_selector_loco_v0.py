from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import train_os_t5p5e1n_response_model_selector_loco_v0 as base


ROOT = Path(__file__).resolve().parents[2]


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1o_regret_surface_selector_loco_v0"
)

OUT_REGRET = (
    OUT_DIR
    / "regret_predictions.csv"
)

OUT_SELECTOR = (
    OUT_DIR
    / "selector_predictions.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "regret_surface_selector_manifest.json"
)


E1N_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1n_response_model_selector_loco_v0"
    / "response_selector_manifest.json"
)


PCA_THRESHOLD = 0.95
RIDGE_ALPHA = 1.0e-2

EXPECTED_ROUGH = 18
EXPECTED_BETA = 21
EXPECTED_PREFS = 25

TOL = 1.0e-10
EPS = 1.0e-12


def regret_from_values(
    values,
    *,
    ideal,
    span,
):
    values = np.asarray(
        values,
        dtype=np.float64,
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

    return regret


def regret_score(
    regret,
    w,
):
    weighted = (
        np.asarray(
            regret,
            dtype=np.float64,
        )
        * np.asarray(
            w,
            dtype=np.float64,
        )
    )

    return float(
        np.max(
            weighted
        )
        + base.RHO
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
        base.PATCH_CSV,
        base.PATCH_CONTRACT,
        base.ATLAS,
        base.LABELS,
        E1N_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Upstream provenance
    # ========================================================

    patch_contract = json.loads(
        base.PATCH_CONTRACT.read_text()
    )

    if patch_contract.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Height patch is not FREEZE_PASS."
        )

    if bool(
        patch_contract[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "Height patch used heldout."
        )


    e1n = json.loads(
        E1N_MANIFEST.read_text()
    )

    if e1n.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5e1n is not COMPUTE_PASS."
        )

    if bool(
        e1n[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5e1n used heldout."
        )


    # ========================================================
    # Terrain patches
    # ========================================================

    patch_features = [
        f
        for f in (
            patch_contract[
                "feature_order"
            ]
        )
        if f.startswith(
            "h_l"
        )
    ]


    if len(
        patch_features
    ) != base.PATCH_DIM:
        raise RuntimeError(
            "Expected 48 patch features."
        )


    patch_rows = base.read_csv(
        base.PATCH_CSV
    )


    patch_lookup = {}

    friction_lookup = {}

    rough_contexts = []


    for row in patch_rows:
        cid = row[
            "context_id"
        ]

        patch_lookup[
            cid
        ] = np.asarray(
            [
                base.finite(
                    row[f]
                )
                for f in (
                    patch_features
                )
            ],
            dtype=np.float64,
        )


        friction_lookup[
            cid
        ] = base.finite(
            row[
                "friction_mu"
            ]
        )


        if cid.startswith(
            "rough_seed_"
        ):
            rough_contexts.append(
                cid
            )


    if len(
        rough_contexts
    ) != EXPECTED_ROUGH:
        raise RuntimeError(
            "Expected 18 rough TRAIN contexts."
        )


    # ========================================================
    # Atlas
    # ========================================================

    atlas = base.read_csv(
        base.ATLAS
    )


    atlas_by_context = defaultdict(
        list
    )

    atlas_lookup = {}


    for row in atlas:
        cid = row[
            "context_id"
        ]

        atlas_by_context[
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
                f"Duplicate atlas row: "
                f"{key}"
            )

        atlas_lookup[
            key
        ] = row


    first_context = (
        patch_rows[
            0
        ][
            "context_id"
        ]
    )


    beta_order = [
        row[
            "beta_name"
        ]
        for row in (
            atlas_by_context[
                first_context
            ]
        )
    ]


    if (
        len(beta_order)
        != EXPECTED_BETA
        or len(
            set(beta_order)
        )
        != EXPECTED_BETA
    ):
        raise RuntimeError(
            "Expected 21 beta points."
        )


    beta_lambda = {
        name:
            base.lambda_from_beta_name(
                name
            )
        for name in (
            beta_order
        )
    }


    # ========================================================
    # Context-local frozen T5.5d references
    # ========================================================

    reference = {}


    for cid in patch_lookup:
        values = np.asarray(
            [
                [
                    base.finite(
                        row[column]
                    )
                    for column in (
                        base.OBJECTIVES
                    )
                ]
                for row in (
                    atlas_by_context[
                        cid
                    ]
                )
            ],
            dtype=np.float64,
        )


        front_indices = (
            base.pareto_indices(
                values
            )
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

        span = (
            np.max(
                front_values,
                axis=0,
            )
            - ideal
        )


        reference[
            cid
        ] = (
            ideal,
            span,
        )


    # ========================================================
    # Frozen preference labels
    # ========================================================

    labels = base.read_csv(
        base.LABELS
    )


    labels_by_context = defaultdict(
        list
    )


    for row in labels:
        labels_by_context[
            row[
                "context_id"
            ]
        ].append(
            row
        )


    for cid in patch_lookup:
        if len(
            labels_by_context[
                cid
            ]
        ) != EXPECTED_PREFS:
            raise RuntimeError(
                f"{cid}: expected "
                f"{EXPECTED_PREFS} preferences."
            )


    # ========================================================
    # Physical IQR — reporting only
    # ========================================================

    global_iqr = []


    for column in (
        base.OBJECTIVES
    ):
        x = np.asarray(
            [
                base.finite(
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
                x,
                0.25,
            )
        )

        q75 = float(
            np.quantile(
                x,
                0.75,
            )
        )


        global_iqr.append(
            q75
            - q25
        )


    global_iqr = np.asarray(
        global_iqr,
        dtype=np.float64,
    )


    # ========================================================
    # LOCO
    # ========================================================

    regret_rows = []

    selector_rows = []

    context_rows = []


    for fold, held_cid in enumerate(
        rough_contexts,
        start=1,
    ):
        print()
        print(
            "=" * 110
        )

        print(
            f"fold {fold:02d}/18 "
            f"held={held_cid}"
        )

        print(
            "=" * 110
        )


        rough_train = [
            cid
            for cid in rough_contexts
            if cid != held_cid
        ]


        train_contexts = [
            "flat",
            "low_friction",
            *rough_train,
        ]


        if len(
            rough_train
        ) != 17:
            raise RuntimeError(
                "Expected 17 rough PCA TRAIN "
                "contexts."
            )


        if len(
            train_contexts
        ) != 19:
            raise RuntimeError(
                "Expected 19 selector TRAIN "
                "contexts."
            )


        # ====================================================
        # Strict fold-local PCA
        # ====================================================

        Xrough = np.asarray(
            [
                patch_lookup[
                    cid
                ]
                for cid in (
                    rough_train
                )
            ],
            dtype=np.float64,
        )


        pca_mean = np.mean(
            Xrough,
            axis=0,
        )


        Xc = (
            Xrough
            - pca_mean
        )


        _, singular, Vt = (
            np.linalg.svd(
                Xc,
                full_matrices=False,
            )
        )


        energy = (
            singular
            ** 2
        )


        total_energy = float(
            np.sum(
                energy
            )
        )


        if total_energy <= EPS:
            raise RuntimeError(
                "Degenerate PCA."
            )


        cumulative = np.cumsum(
            energy
            / total_energy
        )


        rank = int(
            np.sum(
                singular
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


        basis = Vt[
            :k95
        ]


        def encode(
            cid,
        ):
            return (
                (
                    patch_lookup[
                        cid
                    ]
                    - pca_mean
                )
                @ basis.T
            )


        # ====================================================
        # Context normalization
        # ====================================================

        C_train_raw = np.asarray(
            [
                np.concatenate(
                    [
                        encode(
                            cid
                        ),
                        np.asarray(
                            [
                                friction_lookup[
                                    cid
                                ]
                            ],
                            dtype=np.float64,
                        ),
                    ]
                )
                for cid in (
                    train_contexts
                )
            ],
            dtype=np.float64,
        )


        c_mean = np.mean(
            C_train_raw,
            axis=0,
        )


        c_std = np.std(
            C_train_raw,
            axis=0,
        )


        c_std = np.where(
            c_std > EPS,
            c_std,
            1.0,
        )


        def context_feature(
            cid,
        ):
            raw = np.concatenate(
                [
                    encode(
                        cid
                    ),
                    np.asarray(
                        [
                            friction_lookup[
                                cid
                            ]
                        ],
                        dtype=np.float64,
                    ),
                ]
            )

            return (
                raw
                - c_mean
            ) / c_std


        # ====================================================
        # Dense normalized regret supervision
        #
        # 19 contexts x 21 beta = 399 rows.
        # ====================================================

        Phi = []

        R = []


        for cid in (
            train_contexts
        ):
            c = context_feature(
                cid
            )

            ideal, span = (
                reference[
                    cid
                ]
            )


            for beta_name in (
                beta_order
            ):
                atlas_row = (
                    atlas_lookup[
                        (
                            cid,
                            beta_name,
                        )
                    ]
                )


                values = np.asarray(
                    [
                        base.finite(
                            atlas_row[
                                column
                            ]
                        )
                        for column in (
                            base.OBJECTIVES
                        )
                    ],
                    dtype=np.float64,
                )


                regret = (
                    regret_from_values(
                        values,
                        ideal=ideal,
                        span=span,
                    )
                )


                Phi.append(
                    base.feature_map(
                        c,
                        beta_lambda[
                            beta_name
                        ],
                    )
                )


                R.append(
                    regret
                )


        Phi = np.asarray(
            Phi,
            dtype=np.float64,
        )

        R = np.asarray(
            R,
            dtype=np.float64,
        )


        if Phi.shape[0] != 399:
            raise RuntimeError(
                "Expected 399 response rows."
            )


        # ====================================================
        # Feature standardization
        # ====================================================

        phi_mean = np.mean(
            Phi[
                :,
                1:
            ],
            axis=0,
        )


        phi_std = np.std(
            Phi[
                :,
                1:
            ],
            axis=0,
        )


        phi_std = np.where(
            phi_std > EPS,
            phi_std,
            1.0,
        )


        def transform_phi(
            phi,
        ):
            result = np.asarray(
                phi,
                dtype=np.float64,
            ).copy()


            result[
                1:
            ] = (
                result[
                    1:
                ]
                - phi_mean
            ) / phi_std


            return result


        Phi_z = np.asarray(
            [
                transform_phi(
                    phi
                )
                for phi in Phi
            ],
            dtype=np.float64,
        )


        # ====================================================
        # Target standardization
        # ====================================================

        r_mean = np.mean(
            R,
            axis=0,
        )


        r_std = np.std(
            R,
            axis=0,
        )


        if np.any(
            r_std <= EPS
        ):
            raise RuntimeError(
                "Degenerate regret target."
            )


        R_z = (
            R
            - r_mean
        ) / r_std


        # ====================================================
        # Closed-form ridge
        # ====================================================

        regularizer = (
            RIDGE_ALPHA
            * np.eye(
                Phi_z.shape[
                    1
                ],
                dtype=np.float64,
            )
        )


        regularizer[
            0,
            0
        ] = 0.0


        W = np.linalg.solve(
            (
                Phi_z.T
                @ Phi_z
                + regularizer
            ),
            Phi_z.T
            @ R_z,
        )


        def predict_regret(
            cid,
            beta_name,
        ):
            phi = (
                base.feature_map(
                    context_feature(
                        cid
                    ),
                    beta_lambda[
                        beta_name
                    ],
                )
            )


            phi_z = transform_phi(
                phi
            )


            predicted_z = (
                phi_z
                @ W
            )


            return (
                predicted_z
                * r_std
                + r_mean
            )


        # ====================================================
        # TRAIN fit diagnostic
        # ====================================================

        train_pred = (
            Phi_z
            @ W
        ) * r_std + r_mean


        train_abs_error = np.abs(
            train_pred
            - R
        )


        train_regret_mae = float(
            np.mean(
                train_abs_error
            )
        )


        train_regret_p95 = float(
            np.quantile(
                train_abs_error,
                0.95,
            )
        )


        # ====================================================
        # Held 21-beta normalized response surface
        # ====================================================

        held_pred = []

        held_true = []


        held_ideal, held_span = (
            reference[
                held_cid
            ]
        )


        for beta_name in (
            beta_order
        ):
            atlas_row = (
                atlas_lookup[
                    (
                        held_cid,
                        beta_name,
                    )
                ]
            )


            true_values = np.asarray(
                [
                    base.finite(
                        atlas_row[
                            column
                        ]
                    )
                    for column in (
                        base.OBJECTIVES
                    )
                ],
                dtype=np.float64,
            )


            true_regret = (
                regret_from_values(
                    true_values,
                    ideal=held_ideal,
                    span=held_span,
                )
            )


            predicted_regret = (
                predict_regret(
                    held_cid,
                    beta_name,
                )
            )


            held_true.append(
                true_regret
            )

            held_pred.append(
                predicted_regret
            )


            regret_rows.append(
                {
                    "fold":
                        fold,

                    "held_context":
                        held_cid,

                    "k95":
                        k95,

                    "beta_name":
                        beta_name,

                    "true_regret_motion":
                        true_regret[
                            0
                        ],

                    "true_regret_stability":
                        true_regret[
                            1
                        ],

                    "true_regret_energy":
                        true_regret[
                            2
                        ],

                    "pred_regret_motion":
                        predicted_regret[
                            0
                        ],

                    "pred_regret_stability":
                        predicted_regret[
                            1
                        ],

                    "pred_regret_energy":
                        predicted_regret[
                            2
                        ],

                    "regret_abs_error_mean":
                        float(
                            np.mean(
                                np.abs(
                                    predicted_regret
                                    - true_regret
                                )
                            )
                        ),

                    "pred_any_negative":
                        int(
                            np.any(
                                predicted_regret
                                < 0.0
                            )
                        ),
                }
            )


        held_true = np.asarray(
            held_true,
            dtype=np.float64,
        )

        held_pred = np.asarray(
            held_pred,
            dtype=np.float64,
        )


        held_abs_error = np.abs(
            held_pred
            - held_true
        )


        # Predicted Pareto set in the normalized response space.
        predicted_front = (
            base.pareto_indices(
                held_pred
            )
        )


        if not predicted_front:
            raise RuntimeError(
                "Empty predicted Pareto set."
            )


        # ====================================================
        # w -> beta using predicted normalized response
        # ====================================================

        fold_score = []

        fold_phys = []

        fold_exact = []


        for label in (
            labels_by_context[
                held_cid
            ]
        ):
            w = np.asarray(
                [
                    base.finite(
                        label[
                            "w_motion"
                        ]
                    ),
                    base.finite(
                        label[
                            "w_stability"
                        ]
                    ),
                    base.finite(
                        label[
                            "w_energy"
                        ]
                    ),
                ],
                dtype=np.float64,
            )


            candidates = []


            for index in (
                predicted_front
            ):
                score_hat = (
                    regret_score(
                        held_pred[
                            index
                        ],
                        w,
                    )
                )


                candidates.append(
                    (
                        score_hat,
                        index,
                    )
                )


            candidates.sort(
                key=lambda x: (
                    x[
                        0
                    ],
                    x[
                        1
                    ],
                )
            )


            selected_index = (
                candidates[
                    0
                ][
                    1
                ]
            )


            selected_beta = (
                beta_order[
                    selected_index
                ]
            )


            oracle_beta = label[
                "selected_beta_name"
            ]


            oracle_score = base.finite(
                label[
                    "scalarization_score"
                ]
            )


            # Evaluate selected beta using TRUE
            # frozen regret of the held terrain.
            true_selected_score = (
                regret_score(
                    held_true[
                        selected_index
                    ],
                    w,
                )
            )


            score_excess = (
                true_selected_score
                - oracle_score
            )


            if score_excess < -1e-8:
                raise RuntimeError(
                    "Regret-surface selector "
                    "beats frozen oracle."
                )


            score_excess = max(
                0.0,
                score_excess,
            )


            oracle_index = (
                beta_order.index(
                    oracle_beta
                )
            )


            selected_values = np.asarray(
                [
                    base.finite(
                        atlas_lookup[
                            (
                                held_cid,
                                selected_beta,
                            )
                        ][
                            column
                        ]
                    )
                    for column in (
                        base.OBJECTIVES
                    )
                ],
                dtype=np.float64,
            )


            oracle_values = np.asarray(
                [
                    base.finite(
                        atlas_lookup[
                            (
                                held_cid,
                                oracle_beta,
                            )
                        ][
                            column
                        ]
                    )
                    for column in (
                        base.OBJECTIVES
                    )
                ],
                dtype=np.float64,
            )


            physical_linf = float(
                np.max(
                    np.abs(
                        selected_values
                        - oracle_values
                    )
                    / global_iqr
                )
            )


            exact = int(
                selected_beta
                == oracle_beta
            )


            selector_rows.append(
                {
                    "fold":
                        fold,

                    "held_context":
                        held_cid,

                    "k95":
                        k95,

                    "preference_name":
                        label[
                            "preference_name"
                        ],

                    "oracle_beta":
                        oracle_beta,

                    "selected_beta":
                        selected_beta,

                    "exact":
                        exact,

                    "oracle_score":
                        oracle_score,

                    "true_selected_score":
                        true_selected_score,

                    "score_excess":
                        score_excess,

                    "physical_linf_iqr":
                        physical_linf,
                }
            )


            fold_score.append(
                score_excess
            )

            fold_phys.append(
                physical_linf
            )

            fold_exact.append(
                exact
            )


        context_rows.append(
            {
                "fold":
                    fold,

                "held_context":
                    held_cid,

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

                "feature_dim":
                    Phi_z.shape[
                        1
                    ],

                "train_regret_mae":
                    train_regret_mae,

                "train_regret_abs_error_p95":
                    train_regret_p95,

                "held_regret_mae":
                    float(
                        np.mean(
                            held_abs_error
                        )
                    ),

                "held_regret_abs_error_p95":
                    float(
                        np.quantile(
                            held_abs_error,
                            0.95,
                        )
                    ),

                "predicted_pareto_count":
                    len(
                        predicted_front
                    ),

                "negative_prediction_fraction":
                    float(
                        np.mean(
                            held_pred
                            < 0.0
                        )
                    ),

                "selector_exact":
                    float(
                        np.mean(
                            fold_exact
                        )
                    ),

                "selector_score_excess_mean":
                    float(
                        np.mean(
                            fold_score
                        )
                    ),

                "selector_score_excess_p95":
                    float(
                        np.quantile(
                            fold_score,
                            0.95,
                        )
                    ),

                "selector_physical_linf_mean_iqr":
                    float(
                        np.mean(
                            fold_phys
                        )
                    ),

                "selector_physical_linf_p95_iqr":
                    float(
                        np.quantile(
                            fold_phys,
                            0.95,
                        )
                    ),
            }
        )


        print(
            f"  k95={k95} "
            f"trainRegMAE="
            f"{train_regret_mae:.4f} "
            f"heldRegMAE="
            f"{np.mean(held_abs_error):.4f} "
            f"selectorMeanEx="
            f"{np.mean(fold_score):.4f} "
            f"physLinf="
            f"{np.mean(fold_phys):.4f}"
        )


    # ========================================================
    # Aggregate
    # ========================================================

    all_regret_error = np.asarray(
        [
            row[
                "regret_abs_error_mean"
            ]
            for row in (
                regret_rows
            )
        ],
        dtype=np.float64,
    )


    score = np.asarray(
        [
            row[
                "score_excess"
            ]
            for row in (
                selector_rows
            )
        ],
        dtype=np.float64,
    )


    physical = np.asarray(
        [
            row[
                "physical_linf_iqr"
            ]
            for row in (
                selector_rows
            )
        ],
        dtype=np.float64,
    )


    exact = np.asarray(
        [
            row[
                "exact"
            ]
            for row in (
                selector_rows
            )
        ],
        dtype=np.float64,
    )


    aggregate = {
        "held_regret_mae":
            float(
                np.mean(
                    all_regret_error
                )
            ),

        "held_regret_mae_median":
            float(
                np.median(
                    all_regret_error
                )
            ),

        "held_regret_mae_p95":
            float(
                np.quantile(
                    all_regret_error,
                    0.95,
                )
            ),

        "selector_exact_accuracy":
            float(
                np.mean(
                    exact
                )
            ),

        "selector_score_excess_mean":
            float(
                np.mean(
                    score
                )
            ),

        "selector_score_excess_median":
            float(
                np.median(
                    score
                )
            ),

        "selector_score_excess_p95":
            float(
                np.quantile(
                    score,
                    0.95,
                )
            ),

        "selector_score_excess_max":
            float(
                np.max(
                    score
                )
            ),

        "selector_score_excess_le_0p05_fraction":
            float(
                np.mean(
                    score
                    <= 0.05
                )
            ),

        "selector_physical_linf_mean_iqr":
            float(
                np.mean(
                    physical
                )
            ),

        "selector_physical_linf_median_iqr":
            float(
                np.median(
                    physical
                )
            ),

        "selector_physical_linf_p95_iqr":
            float(
                np.quantile(
                    physical,
                    0.95,
                )
            ),
    }


    # ========================================================
    # Write
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    base.write_csv(
        OUT_REGRET,
        regret_rows,
    )

    base.write_csv(
        OUT_SELECTOR,
        selector_rows,
    )

    base.write_csv(
        OUT_CONTEXT,
        context_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1o_"
                "regret_surface_selector_loco_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "folds":
            EXPECTED_ROUGH,

        "terrain_encoder":
            (
                "strict fold-local PCA k95 "
                "from 17 rough TRAIN patches"
            ),

        "dense_supervision":
            (
                "19 TRAIN contexts x "
                "21 beta points = 399 "
                "context-local normalized "
                "regret vectors per fold"
            ),

        "target":
            (
                "Frozen T5.5d normalized "
                "response/regret "
                "(J - ideal) / Pareto span"
            ),

        "response_model":
            {
                "type":
                    "closed-form ridge",

                "alpha":
                    RIDGE_ALPHA,

                "feature_map":
                    (
                        "[1, c, lambda, "
                        "c outer lambda, "
                        "quadratic lambda]"
                    ),
            },

        "selector":
            (
                "Predict 21 normalized regret "
                "vectors and minimize the original "
                "augmented weighted Tchebycheff "
                "score for mission preference w."
            ),

        "global_iqr_reporting_only":
            global_iqr.tolist(),

        "aggregate":
            aggregate,

        "comparison_reference":
            {
                "direct_5d_e1_score_mean":
                    0.47897,

                "direct_5d_e1_score_p95":
                    1.38192,

                "response_model_e1n_score_mean":
                    float(
                        e1n[
                            "aggregate"
                        ][
                            "selector_score_excess_mean"
                        ]
                    ),

                "response_model_e1n_score_p95":
                    float(
                        e1n[
                            "aggregate"
                        ][
                            "selector_score_excess_p95"
                        ]
                    ),
            },

        "interpretation_guard":
            (
                "TRAIN rough-context LOCO only. "
                "No validation/test/hard terrain "
                "is used. Frozen T5.5d labels and "
                "scalarization remain unchanged."
            ),

        "next_stage":
            (
                "If TRAIN regret fit is good but "
                "held regret prediction remains "
                "poor, terrain representation is "
                "the remaining bottleneck. "
                "If TRAIN regret fit is poor, "
                "upgrade only the forward response "
                "surrogate before considering a "
                "richer terrain encoder."
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
        "ICRA27 OS-T5.5e1o NORMALIZED "
        "REGRET-SURFACE SELECTOR LOCO"
    )

    print("=" * 118)


    print()
    print("REGRET SURFACE")

    print(
        "  held_regret_mae          :",
        aggregate[
            "held_regret_mae"
        ],
    )

    print(
        "  held_regret_mae_median   :",
        aggregate[
            "held_regret_mae_median"
        ],
    )

    print(
        "  held_regret_mae_p95      :",
        aggregate[
            "held_regret_mae_p95"
        ],
    )


    print()
    print("MODEL-BASED SELECTOR")

    for key in (
        "selector_exact_accuracy",
        "selector_score_excess_mean",
        "selector_score_excess_median",
        "selector_score_excess_p95",
        "selector_score_excess_max",
        "selector_score_excess_le_0p05_fraction",
        "selector_physical_linf_mean_iqr",
        "selector_physical_linf_median_iqr",
        "selector_physical_linf_p95_iqr",
    ):
        print(
            f"  {key:<40}: "
            f"{aggregate[key]}"
        )


    print()
    print("outputs:")
    print(" ", OUT_REGRET)
    print(" ", OUT_SELECTOR)
    print(" ", OUT_CONTEXT)
    print(" ", OUT_MANIFEST)


    print()
    print(
        "[ICRA27] OS-T5.5e1o "
        "regret-surface selector LOCO: "
        "COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
