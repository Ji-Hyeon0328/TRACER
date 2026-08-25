from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PATCH_ROOT = (
    ROOT / "results/icra27"
    / "os_t5p5a_oracle_height_patch_v0"
)

PATCH_CSV = PATCH_ROOT / "oracle_height_patches.csv"
PATCH_CONTRACT = PATCH_ROOT / "oracle_height_patch_contract.json"

ATLAS = (
    ROOT / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)

LABELS = (
    ROOT / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
    / "preference_to_beta_labels.csv"
)

E1L_MANIFEST = (
    ROOT / "results/icra27"
    / "os_t5p5e1l_selector_patch_pca_loco_v0"
    / "selector_patch_pca_loco_manifest.json"
)

OUT_DIR = (
    ROOT / "results/icra27"
    / "os_t5p5e1n_response_model_selector_loco_v0"
)

OUT_PRED = OUT_DIR / "response_predictions.csv"
OUT_SELECT = OUT_DIR / "selector_predictions.csv"
OUT_CONTEXT = OUT_DIR / "context_summary.csv"
OUT_MANIFEST = OUT_DIR / "response_selector_manifest.json"


OBJECTIVES = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

PATCH_DIM = 48
PCA_THRESHOLD = 0.95

RIDGE_ALPHA = 1.0e-2

RHO = 0.01
TOL = 1.0e-10
EPS = 1.0e-12


def finite(x):
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(x)
    return y


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows: {path}")

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


def lambda_from_beta_name(name):
    m = re.fullmatch(
        r"lm(\d{3})_ls(\d{3})_le(\d{3})",
        name,
    )

    if m is None:
        raise ValueError(name)

    lam = np.asarray(
        [
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
        ],
        dtype=np.float64,
    ) / 100.0

    if not math.isclose(
        float(np.sum(lam)),
        1.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(name)

    return lam


def pareto_indices(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    keep = []

    for i in range(len(values)):
        dominated = False

        for j in range(len(values)):
            if i == j:
                continue

            no_worse = np.all(
                values[j]
                <= values[i] + TOL
            )

            strictly = np.any(
                values[j]
                < values[i] - TOL
            )

            if no_worse and strictly:
                dominated = True
                break

        if not dominated:
            keep.append(i)

    return keep


def preference_score(
    values,
    *,
    ideal,
    span,
    w,
):
    regret = np.zeros(
        3,
        dtype=np.float64,
    )

    for k in range(3):
        if span[k] > TOL:
            regret[k] = (
                values[k] - ideal[k]
            ) / span[k]

    weighted = w * regret

    return float(
        np.max(weighted)
        + RHO * np.sum(weighted)
    )


def feature_map(c, lam):
    c = np.asarray(
        c,
        dtype=np.float64,
    )

    lam = np.asarray(
        lam,
        dtype=np.float64,
    )

    interaction = (
        c[:, None]
        * lam[None, :]
    ).reshape(-1)

    lambda_quad = np.asarray(
        [
            lam[0] ** 2,
            lam[1] ** 2,
            lam[2] ** 2,
            lam[0] * lam[1],
            lam[0] * lam[2],
            lam[1] * lam[2],
        ],
        dtype=np.float64,
    )

    return np.concatenate(
        [
            np.asarray([1.0]),
            c,
            lam,
            interaction,
            lambda_quad,
        ]
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        PATCH_CSV,
        PATCH_CONTRACT,
        ATLAS,
        LABELS,
        E1L_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    patch_contract = json.loads(
        PATCH_CONTRACT.read_text()
    )

    if patch_contract.get("status") != "FREEZE_PASS":
        raise RuntimeError(
            "Patch contract not frozen."
        )

    if bool(
        patch_contract["heldout_used"]
    ):
        raise RuntimeError(
            "Patch artifact used heldout."
        )

    e1l = json.loads(
        E1L_MANIFEST.read_text()
    )

    if e1l.get("status") != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5e1l not COMPUTE_PASS."
        )


    # --------------------------------------------------------
    # Patch contexts
    # --------------------------------------------------------

    patch_features = [
        f
        for f in patch_contract["feature_order"]
        if f.startswith("h_l")
    ]

    if len(patch_features) != PATCH_DIM:
        raise RuntimeError(
            "Patch feature mismatch."
        )

    patch_rows = read_csv(PATCH_CSV)

    patch_lookup = {}
    friction_lookup = {}

    rough_contexts = []

    for row in patch_rows:
        cid = row["context_id"]

        patch_lookup[cid] = np.asarray(
            [
                finite(row[f])
                for f in patch_features
            ],
            dtype=np.float64,
        )

        friction_lookup[cid] = finite(
            row["friction_mu"]
        )

        if cid.startswith("rough_seed_"):
            rough_contexts.append(cid)

    if len(rough_contexts) != 18:
        raise RuntimeError(
            "Expected 18 rough contexts."
        )


    # --------------------------------------------------------
    # Physical atlas
    # --------------------------------------------------------

    atlas = read_csv(ATLAS)

    atlas_by_context = defaultdict(list)
    atlas_lookup = {}

    for row in atlas:
        cid = row["context_id"]

        atlas_by_context[cid].append(row)

        atlas_lookup[
            (
                cid,
                row["beta_name"],
            )
        ] = row

    first_context = patch_rows[0]["context_id"]

    beta_order = [
        row["beta_name"]
        for row in atlas_by_context[
            first_context
        ]
    ]

    if len(beta_order) != 21:
        raise RuntimeError(
            "Expected 21 betas."
        )

    beta_lambda = {
        name:
            lambda_from_beta_name(name)
        for name in beta_order
    }


    # Global TRAIN atlas IQR:
    # reporting only, never used for fitting/selection.
    global_iqr = []

    for column in OBJECTIVES:
        x = np.asarray(
            [
                finite(row[column])
                for row in atlas
            ],
            dtype=np.float64,
        )

        global_iqr.append(
            float(
                np.quantile(x, 0.75)
                - np.quantile(x, 0.25)
            )
        )

    global_iqr = np.asarray(
        global_iqr,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Frozen preference labels
    # --------------------------------------------------------

    labels = read_csv(LABELS)

    labels_by_context = defaultdict(list)

    for row in labels:
        labels_by_context[
            row["context_id"]
        ].append(row)


    # --------------------------------------------------------
    # True frozen T5.5d score references
    # --------------------------------------------------------

    true_reference = {}

    for cid in patch_lookup:
        values = np.asarray(
            [
                [
                    finite(row[column])
                    for column in OBJECTIVES
                ]
                for row in atlas_by_context[cid]
            ],
            dtype=np.float64,
        )

        front = pareto_indices(
            values
        )

        front_values = values[
            front
        ]

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

        true_reference[cid] = (
            ideal,
            span,
        )


    # --------------------------------------------------------
    # Outer rough-context LOCO
    # --------------------------------------------------------

    response_rows = []
    selector_rows = []
    context_rows = []


    for fold, held_cid in enumerate(
        rough_contexts,
        start=1,
    ):
        print()
        print("=" * 110)
        print(
            f"fold {fold:02d}/18 "
            f"held={held_cid}"
        )
        print("=" * 110)


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

        if len(train_contexts) != 19:
            raise RuntimeError(
                "Expected 19 train contexts."
            )


        # ====================================================
        # Strict fold-local PCA:
        # 17 rough TRAIN contexts only
        # ====================================================

        Xrough = np.asarray(
            [
                patch_lookup[cid]
                for cid in rough_train
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

        _, singular, Vt = np.linalg.svd(
            Xc,
            full_matrices=False,
        )

        explained = (
            singular ** 2
            / np.sum(
                singular ** 2
            )
        )

        cumulative = np.cumsum(
            explained
        )

        rank = int(
            np.sum(
                singular > 1e-12
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


        def encode(cid):
            return (
                patch_lookup[cid]
                - pca_mean
            ) @ basis.T


        # ====================================================
        # Context normalization:
        # [PCA latent, friction]
        # ====================================================

        C_train_raw = np.asarray(
            [
                np.concatenate(
                    [
                        encode(cid),
                        np.asarray(
                            [
                                friction_lookup[
                                    cid
                                ]
                            ]
                        ),
                    ]
                )
                for cid in train_contexts
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


        def context_feature(cid):
            raw = np.concatenate(
                [
                    encode(cid),
                    np.asarray(
                        [
                            friction_lookup[cid]
                        ]
                    ),
                ]
            )

            return (
                raw - c_mean
            ) / c_std


        # ====================================================
        # Dense response-supervision matrix
        # 19 contexts × 21 betas = 399 rows
        # ====================================================

        Phi = []
        Y = []


        for cid in train_contexts:
            c = context_feature(
                cid
            )

            for beta_name in beta_order:
                row = atlas_lookup[
                    (
                        cid,
                        beta_name,
                    )
                ]

                Phi.append(
                    feature_map(
                        c,
                        beta_lambda[
                            beta_name
                        ],
                    )
                )

                Y.append(
                    [
                        finite(
                            row[column]
                        )
                        for column in OBJECTIVES
                    ]
                )


        Phi = np.asarray(
            Phi,
            dtype=np.float64,
        )

        Y = np.asarray(
            Y,
            dtype=np.float64,
        )


        if Phi.shape[0] != 399:
            raise RuntimeError(
                "Expected 399 response samples."
            )


        # ====================================================
        # Feature standardization except intercept
        # ====================================================

        phi_mean = np.mean(
            Phi[:, 1:],
            axis=0,
        )

        phi_std = np.std(
            Phi[:, 1:],
            axis=0,
        )

        phi_std = np.where(
            phi_std > EPS,
            phi_std,
            1.0,
        )


        def transform_phi(phi):
            out = np.asarray(
                phi,
                dtype=np.float64,
            ).copy()

            out[1:] = (
                out[1:]
                - phi_mean
            ) / phi_std

            return out


        Phi_z = np.asarray(
            [
                transform_phi(phi)
                for phi in Phi
            ],
            dtype=np.float64,
        )


        # Target normalization on training responses only.
        y_mean = np.mean(
            Y,
            axis=0,
        )

        y_std = np.std(
            Y,
            axis=0,
        )

        if np.any(
            y_std <= EPS
        ):
            raise RuntimeError(
                "Degenerate response target."
            )

        Y_z = (
            Y - y_mean
        ) / y_std


        # ====================================================
        # Closed-form ridge
        # ====================================================

        regularizer = (
            RIDGE_ALPHA
            * np.eye(
                Phi_z.shape[1],
                dtype=np.float64,
            )
        )

        # Do not regularize intercept.
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
            @ Y_z,
        )


        def predict_response(
            cid,
            beta_name,
        ):
            phi = feature_map(
                context_feature(cid),
                beta_lambda[
                    beta_name
                ],
            )

            phi = transform_phi(
                phi
            )

            yz = (
                phi @ W
            )

            return (
                yz * y_std
                + y_mean
            )


        # ====================================================
        # TRAIN response-fit diagnostic
        # ====================================================

        train_pred = (
            Phi_z @ W
        ) * y_std + y_mean

        train_abs_over_iqr = (
            np.abs(
                train_pred - Y
            )
            / global_iqr[
                None,
                :
            ]
        )

        train_response_linf = np.max(
            train_abs_over_iqr,
            axis=1,
        )


        # ====================================================
        # Held-context 21-beta response surface
        # ====================================================

        held_true = []

        held_pred = []


        for beta_name in beta_order:
            true_row = atlas_lookup[
                (
                    held_cid,
                    beta_name,
                )
            ]

            true_values = np.asarray(
                [
                    finite(
                        true_row[column]
                    )
                    for column in OBJECTIVES
                ],
                dtype=np.float64,
            )

            predicted_values = (
                predict_response(
                    held_cid,
                    beta_name,
                )
            )


            held_true.append(
                true_values
            )

            held_pred.append(
                predicted_values
            )


            abs_over_iqr = (
                np.abs(
                    predicted_values
                    - true_values
                )
                / global_iqr
            )


            response_rows.append(
                {
                    "fold":
                        fold,

                    "held_context":
                        held_cid,

                    "k95":
                        k95,

                    "beta_name":
                        beta_name,

                    "true_motion":
                        true_values[0],

                    "true_stability":
                        true_values[1],

                    "true_energy":
                        true_values[2],

                    "pred_motion":
                        predicted_values[0],

                    "pred_stability":
                        predicted_values[1],

                    "pred_energy":
                        predicted_values[2],

                    "motion_abs_over_iqr":
                        abs_over_iqr[0],

                    "stability_abs_over_iqr":
                        abs_over_iqr[1],

                    "energy_abs_over_iqr":
                        abs_over_iqr[2],

                    "physical_linf_iqr":
                        float(
                            np.max(
                                abs_over_iqr
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


        response_linf = np.max(
            (
                np.abs(
                    held_pred
                    - held_true
                )
                / global_iqr[
                    None,
                    :
                ]
            ),
            axis=1,
        )


        # ====================================================
        # Build predicted Pareto response model
        # ====================================================

        predicted_front = (
            pareto_indices(
                held_pred
            )
        )

        if not predicted_front:
            raise RuntimeError(
                "Empty predicted Pareto front."
            )

        predicted_front_values = (
            held_pred[
                predicted_front
            ]
        )

        predicted_ideal = np.min(
            predicted_front_values,
            axis=0,
        )

        predicted_span = (
            np.max(
                predicted_front_values,
                axis=0,
            )
            - predicted_ideal
        )


        # ====================================================
        # Mission preference -> beta selection
        # ====================================================

        selector_excess = []
        selector_physical_linf = []
        selector_exact = []


        true_ideal, true_span = (
            true_reference[
                held_cid
            ]
        )


        for label in labels_by_context[
            held_cid
        ]:
            w = np.asarray(
                [
                    finite(
                        label[
                            "w_motion"
                        ]
                    ),
                    finite(
                        label[
                            "w_stability"
                        ]
                    ),
                    finite(
                        label[
                            "w_energy"
                        ]
                    ),
                ],
                dtype=np.float64,
            )


            predicted_candidates = []


            for index in predicted_front:
                score_hat = (
                    preference_score(
                        held_pred[
                            index
                        ],
                        ideal=(
                            predicted_ideal
                        ),
                        span=(
                            predicted_span
                        ),
                        w=w,
                    )
                )

                predicted_candidates.append(
                    (
                        score_hat,
                        index,
                    )
                )


            predicted_candidates.sort(
                key=lambda x: (
                    x[0],
                    x[1],
                )
            )


            selected_index = (
                predicted_candidates[
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

            oracle_score = finite(
                label[
                    "scalarization_score"
                ]
            )


            true_selected_values = (
                held_true[
                    selected_index
                ]
            )


            true_selected_score = (
                preference_score(
                    true_selected_values,
                    ideal=true_ideal,
                    span=true_span,
                    w=w,
                )
            )


            excess = (
                true_selected_score
                - oracle_score
            )

            if excess < -1e-8:
                raise RuntimeError(
                    "Response selector beats "
                    "frozen oracle."
                )

            excess = max(
                0.0,
                excess,
            )


            oracle_index = (
                beta_order.index(
                    oracle_beta
                )
            )

            physical_delta = (
                np.abs(
                    held_true[
                        selected_index
                    ]
                    - held_true[
                        oracle_index
                    ]
                )
                / global_iqr
            )


            physical_linf = float(
                np.max(
                    physical_delta
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

                    "w_motion":
                        w[0],

                    "w_stability":
                        w[1],

                    "w_energy":
                        w[2],

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
                        excess,

                    "physical_linf_iqr":
                        physical_linf,
                }
            )


            selector_excess.append(
                excess
            )

            selector_physical_linf.append(
                physical_linf
            )

            selector_exact.append(
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

                "ridge_alpha":
                    RIDGE_ALPHA,

                "feature_dim":
                    Phi_z.shape[
                        1
                    ],

                "train_response_linf_mean_iqr":
                    float(
                        np.mean(
                            train_response_linf
                        )
                    ),

                "held_response_linf_mean_iqr":
                    float(
                        np.mean(
                            response_linf
                        )
                    ),

                "held_response_linf_p95_iqr":
                    float(
                        np.quantile(
                            response_linf,
                            0.95,
                        )
                    ),

                "predicted_pareto_count":
                    len(
                        predicted_front
                    ),

                "selector_exact":
                    float(
                        np.mean(
                            selector_exact
                        )
                    ),

                "selector_score_excess_mean":
                    float(
                        np.mean(
                            selector_excess
                        )
                    ),

                "selector_score_excess_p95":
                    float(
                        np.quantile(
                            selector_excess,
                            0.95,
                        )
                    ),

                "selector_physical_linf_mean_iqr":
                    float(
                        np.mean(
                            selector_physical_linf
                        )
                    ),

                "selector_physical_linf_p95_iqr":
                    float(
                        np.quantile(
                            selector_physical_linf,
                            0.95,
                        )
                    ),
            }
        )


        print(
            f"  k95={k95} "
            f"responseLinf="
            f"{np.mean(response_linf):.4f} "
            f"selectorMeanEx="
            f"{np.mean(selector_excess):.4f} "
            f"physLinf="
            f"{np.mean(selector_physical_linf):.4f}"
        )


    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    all_response_linf = np.asarray(
        [
            row[
                "physical_linf_iqr"
            ]
            for row in response_rows
        ],
        dtype=np.float64,
    )

    all_score = np.asarray(
        [
            row[
                "score_excess"
            ]
            for row in selector_rows
        ],
        dtype=np.float64,
    )

    all_phys = np.asarray(
        [
            row[
                "physical_linf_iqr"
            ]
            for row in selector_rows
        ],
        dtype=np.float64,
    )

    all_exact = np.asarray(
        [
            row[
                "exact"
            ]
            for row in selector_rows
        ],
        dtype=np.float64,
    )


    aggregate = {
        "response_linf_mean_iqr":
            float(
                np.mean(
                    all_response_linf
                )
            ),

        "response_linf_median_iqr":
            float(
                np.median(
                    all_response_linf
                )
            ),

        "response_linf_p95_iqr":
            float(
                np.quantile(
                    all_response_linf,
                    0.95,
                )
            ),

        "selector_exact_accuracy":
            float(
                np.mean(
                    all_exact
                )
            ),

        "selector_score_excess_mean":
            float(
                np.mean(
                    all_score
                )
            ),

        "selector_score_excess_median":
            float(
                np.median(
                    all_score
                )
            ),

        "selector_score_excess_p95":
            float(
                np.quantile(
                    all_score,
                    0.95,
                )
            ),

        "selector_score_excess_max":
            float(
                np.max(
                    all_score
                )
            ),

        "selector_score_excess_le_0p05_fraction":
            float(
                np.mean(
                    all_score
                    <= 0.05
                )
            ),

        "selector_physical_linf_mean_iqr":
            float(
                np.mean(
                    all_phys
                )
            ),

        "selector_physical_linf_median_iqr":
            float(
                np.median(
                    all_phys
                )
            ),

        "selector_physical_linf_p95_iqr":
            float(
                np.quantile(
                    all_phys,
                    0.95,
                )
            ),
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_PRED,
        response_rows,
    )

    write_csv(
        OUT_SELECT,
        selector_rows,
    )

    write_csv(
        OUT_CONTEXT,
        context_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1n_"
                "response_model_selector_loco_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "folds":
            18,

        "terrain_encoder":
            (
                "strict fold-local PCA k95 "
                "from 17 rough TRAIN patches"
            ),

        "response_model":
            {
                "type":
                    "closed-form ridge",

                "alpha":
                    RIDGE_ALPHA,

                "supervision_per_fold":
                    (
                        "19 TRAIN contexts x "
                        "21 beta points = 399 "
                        "physical response rows"
                    ),

                "feature_map":
                    (
                        "[1, c, lambda, "
                        "c outer lambda, "
                        "quadratic lambda]"
                    ),

                "targets":
                    list(
                        OBJECTIVES
                    ),
            },

        "selector":
            (
                "Predict all 21 beta responses, "
                "construct predicted Pareto "
                "reference, then minimize the "
                "same augmented weighted "
                "Tchebycheff objective for w."
            ),

        "global_iqr_reporting_only":
            global_iqr.tolist(),

        "aggregate":
            aggregate,

        "interpretation_guard":
            (
                "TRAIN rough-context LOCO only. "
                "No validation/test/hard terrain "
                "is used. The global IQR is "
                "reporting scale only."
            ),

        "future_interface":
            (
                "Terrain PCA latent may later be "
                "replaced by CART/world-model "
                "latent without changing the "
                "response-model selector logic."
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
        "ICRA27 OS-T5.5e1n RESPONSE-MODEL "
        "OBJECTIVE SELECTOR LOCO"
    )
    print("=" * 118)

    print()
    print("HELD RESPONSE MODEL")

    print(
        "  Linf mean IQR    :",
        aggregate[
            "response_linf_mean_iqr"
        ],
    )

    print(
        "  Linf median IQR  :",
        aggregate[
            "response_linf_median_iqr"
        ],
    )

    print(
        "  Linf p95 IQR     :",
        aggregate[
            "response_linf_p95_iqr"
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
    print(" ", OUT_PRED)
    print(" ", OUT_SELECT)
    print(" ", OUT_CONTEXT)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1n "
        "response-model selector LOCO: "
        "COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
