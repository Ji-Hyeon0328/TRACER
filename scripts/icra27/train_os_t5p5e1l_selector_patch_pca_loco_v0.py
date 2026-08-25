from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Frozen inputs
# ============================================================

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

PCA_AUDIT_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1k_height_patch_pca_v0"
    / "pca_representation_manifest.json"
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


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1l_selector_patch_pca_loco_v0"
)

OUT_PREDICTIONS = (
    OUT_DIR
    / "loco_predictions.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "loco_context_summary.csv"
)

OUT_PCA = (
    OUT_DIR
    / "fold_pca_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "selector_patch_pca_loco_manifest.json"
)


# ============================================================
# Fixed experimental contract
# ============================================================

PATCH_DIM = 48

EXPECTED_CONTEXTS = 20
EXPECTED_ROUGH = 18
EXPECTED_PREFS = 25
EXPECTED_TRAIN_CONTEXTS = 19
EXPECTED_TRAIN_ROWS = 475
EXPECTED_TEST_ROWS = 25
EXPECTED_CV_ROWS = 450

HIDDEN = 64
EPOCHS = 2000

LR = 3.0e-3
WEIGHT_DECAY = 1.0e-4

SEED = 27551

PCA_VARIANCE_THRESHOLD = 0.95

RHO = 0.01
TOL = 1.0e-10
EPS = 1.0e-12


OBJECTIVES = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)

CANONICAL_PREFERENCES = {
    "balanced",
    "motion",
    "stability",
    "energy",
}


# ============================================================
# Helpers
# ============================================================

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


def lambda_from_beta_name(
    beta_name: str,
) -> np.ndarray:
    match = re.fullmatch(
        r"lm(\d{3})_ls(\d{3})_le(\d{3})",
        beta_name,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse beta name: "
            f"{beta_name}"
        )

    lam = np.asarray(
        [
            int(
                match.group(1)
            ),
            int(
                match.group(2)
            ),
            int(
                match.group(3)
            ),
        ],
        dtype=np.float64,
    ) / 100.0

    if not math.isclose(
        float(
            np.sum(lam)
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Invalid lambda simplex point: "
            f"{beta_name} -> {lam}"
        )

    return lam


def beta_from_lambda(
    lam: np.ndarray,
) -> np.ndarray:
    return (
        0.15
        + 0.55
        * np.asarray(
            lam,
            dtype=np.float64,
        )
    )


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


class SelectorMLP(
    nn.Module
):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                HIDDEN,
            ),
            nn.SiLU(),
            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),
            nn.SiLU(),
            nn.Linear(
                HIDDEN,
                3,
            ),
        )

    def forward(
        self,
        x,
    ):
        logits = self.net(
            x
        )

        return torch.softmax(
            logits,
            dim=-1,
        )


def summary_metrics(
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


# ============================================================
# Main
# ============================================================

def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    for path in (
        PATCH_CSV,
        PATCH_CONTRACT,
        PCA_AUDIT_MANIFEST,
        ATLAS,
        LABELS,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )


    # ========================================================
    # Validate representation provenance
    # ========================================================

    patch_contract = json.loads(
        PATCH_CONTRACT.read_text()
    )

    if patch_contract.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Height patch is not FREEZE_PASS."
        )

    if int(
        patch_contract[
            "patch_dim"
        ]
    ) != PATCH_DIM:
        raise RuntimeError(
            "Patch dimension mismatch."
        )

    if bool(
        patch_contract[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "Patch artifact reports heldout use."
        )


    pca_audit = json.loads(
        PCA_AUDIT_MANIFEST.read_text()
    )

    if pca_audit.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5e1k PCA audit is not "
            "COMPUTE_PASS."
        )

    if bool(
        pca_audit[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5e1k reports heldout use."
        )


    # ========================================================
    # Terrain patches
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
            "Expected 48 height-patch features."
        )


    patch_rows = read_csv(
        PATCH_CSV
    )

    if len(
        patch_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected 20 patch contexts; "
            f"got {len(patch_rows)}"
        )


    patch_lookup = {}

    friction_lookup = {}


    for row in patch_rows:
        cid = row[
            "context_id"
        ]

        if cid in patch_lookup:
            raise RuntimeError(
                f"Duplicate patch context: "
                f"{cid}"
            )

        patch_lookup[
            cid
        ] = np.asarray(
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

        friction_lookup[
            cid
        ] = finite(
            row[
                "friction_mu"
            ]
        )


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
            "Expected 18 rough TRAIN contexts."
        )


    for required in (
        "flat",
        "low_friction",
    ):
        if required not in patch_lookup:
            raise RuntimeError(
                f"Missing context: {required}"
            )


    # ========================================================
    # Frozen labels
    # ========================================================

    labels = read_csv(
        LABELS
    )

    labels_by_context = defaultdict(
        list
    )

    label_lookup = {}


    for row in labels:
        cid = row[
            "context_id"
        ]

        labels_by_context[
            cid
        ].append(
            row
        )

        key = (
            cid,
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


    for cid in patch_lookup:
        if len(
            labels_by_context[
                cid
            ]
        ) != EXPECTED_PREFS:
            raise RuntimeError(
                f"{cid}: expected 25 labels; "
                f"got "
                f"{len(labels_by_context[cid])}"
            )


    # ========================================================
    # Physical atlas and original frozen T5.5d score
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

        key = (
            cid,
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
            by_context[
                first_context
            ]
        )
    ]


    if (
        len(beta_order)
        != 21
        or len(
            set(beta_order)
        )
        != 21
    ):
        raise RuntimeError(
            "Expected 21 beta lattice points."
        )


    beta_lambdas = np.asarray(
        [
            lambda_from_beta_name(
                beta_name
            )
            for beta_name in (
                beta_order
            )
        ],
        dtype=np.float64,
    )


    score_reference = {}


    for cid in patch_lookup:
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

        span = (
            np.max(
                values,
                axis=0,
            )
            - ideal
        )

        score_reference[
            cid
        ] = (
            ideal,
            span,
        )


    def score_beta(
        *,
        context_id: str,
        beta_name: str,
        w: np.ndarray,
    ) -> float:
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
            score_reference[
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


    def project_lambda(
        lam: np.ndarray,
    ):
        distances = np.sum(
            (
                beta_lambdas
                - lam[
                    None,
                    :
                ]
            )
            ** 2,
            axis=1,
        )

        index = int(
            np.argmin(
                distances
            )
        )

        return (
            beta_order[
                index
            ],
            beta_lambdas[
                index
            ],
            float(
                math.sqrt(
                    distances[
                        index
                    ]
                )
            ),
        )


    # ========================================================
    # LOCO
    # ========================================================

    prediction_rows = []

    context_summary_rows = []

    fold_pca_rows = []


    train_lambda_mae_all = []

    train_exact_all = []


    for fold_index, test_cid in enumerate(
        rough_contexts,
        start=1,
    ):
        print()
        print(
            "=" * 110
        )

        print(
            f"LOCO fold "
            f"{fold_index:02d}/"
            f"{len(rough_contexts)} "
            f"held rough context: "
            f"{test_cid}"
        )

        print(
            "=" * 110
        )


        rough_train = [
            cid
            for cid in rough_contexts
            if cid != test_cid
        ]


        if len(
            rough_train
        ) != 17:
            raise RuntimeError(
                "Expected 17 rough PCA-TRAIN "
                "contexts."
            )


        train_contexts = [
            "flat",
            "low_friction",
            *rough_train,
        ]


        if len(
            train_contexts
        ) != EXPECTED_TRAIN_CONTEXTS:
            raise RuntimeError(
                "Expected 19 selector TRAIN "
                "contexts."
            )


        # ----------------------------------------------------
        # STRICT FOLD-LOCAL PCA
        #
        # Fit on 17 rough TRAIN patches only.
        # Held rough patch does not participate in:
        # - PCA mean
        # - PCA basis
        # - PCA rank
        # - k95 selection
        # ----------------------------------------------------

        X_rough_train = np.asarray(
            [
                patch_lookup[
                    cid
                ]
                for cid in rough_train
            ],
            dtype=np.float64,
        )


        pca_mean = np.mean(
            X_rough_train,
            axis=0,
        )

        Xc = (
            X_rough_train
            - pca_mean
        )


        _, singular_values, Vt = (
            np.linalg.svd(
                Xc,
                full_matrices=False,
            )
        )


        total_energy = float(
            np.sum(
                singular_values
                ** 2
            )
        )


        if total_energy <= EPS:
            raise RuntimeError(
                "Degenerate fold PCA."
            )


        explained = (
            singular_values
            ** 2
            / total_energy
        )

        cumulative = np.cumsum(
            explained
        )


        pca_rank = int(
            np.sum(
                singular_values
                > 1e-12
            )
        )


        k95 = int(
            np.searchsorted(
                cumulative,
                PCA_VARIANCE_THRESHOLD,
                side="left",
            )
            + 1
        )


        k95 = min(
            k95,
            pca_rank,
        )


        if not (
            1
            <= k95
            <= pca_rank
        ):
            raise RuntimeError(
                "Invalid fold-local k95."
            )


        basis = Vt[
            :k95
        ]


        def encode_patch(
            cid: str,
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


        # ----------------------------------------------------
        # Selector-context normalization
        #
        # c = [z_PCA, friction]
        # Stats use 19 selector TRAIN contexts only.
        # w remains unnormalized as in T5.5e1.
        # ----------------------------------------------------

        train_context_raw = np.asarray(
            [
                np.concatenate(
                    [
                        encode_patch(
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
                for cid in train_contexts
            ],
            dtype=np.float64,
        )


        context_mean = np.mean(
            train_context_raw,
            axis=0,
        )

        context_std = np.std(
            train_context_raw,
            axis=0,
            ddof=0,
        )


        safe_std = np.where(
            context_std
            > EPS,
            context_std,
            1.0,
        )


        def selector_context(
            cid: str,
        ):
            raw = np.concatenate(
                [
                    encode_patch(
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
                - context_mean
            ) / safe_std


        # ----------------------------------------------------
        # Build supervised datasets
        # ----------------------------------------------------

        train_x = []

        train_y = []


        for cid in train_contexts:
            c = selector_context(
                cid
            )

            for row in (
                labels_by_context[
                    cid
                ]
            ):
                w = np.asarray(
                    [
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
                    ],
                    dtype=np.float64,
                )

                target_lambda = (
                    lambda_from_beta_name(
                        row[
                            "selected_beta_name"
                        ]
                    )
                )

                train_x.append(
                    np.concatenate(
                        [
                            c,
                            w,
                        ]
                    )
                )

                train_y.append(
                    target_lambda
                )


        if len(
            train_x
        ) != EXPECTED_TRAIN_ROWS:
            raise RuntimeError(
                f"Expected 475 training rows; "
                f"got {len(train_x)}"
            )


        train_x_np = np.asarray(
            train_x,
            dtype=np.float32,
        )

        train_y_np = np.asarray(
            train_y,
            dtype=np.float32,
        )


        # ----------------------------------------------------
        # Train exact same-size MLP family as T5.5e1
        # ----------------------------------------------------

        np.random.seed(
            SEED
        )

        torch.manual_seed(
            SEED
        )


        model = SelectorMLP(
            input_dim=(
                train_x_np.shape[
                    1
                ]
            )
        )


        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=(
                WEIGHT_DECAY
            ),
        )


        x_tensor = torch.from_numpy(
            train_x_np
        )

        y_tensor = torch.from_numpy(
            train_y_np
        )


        model.train()


        final_loss = None


        for _ in range(
            EPOCHS
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                x_tensor
            )

            loss = torch.mean(
                (
                    pred
                    - y_tensor
                )
                ** 2
            )

            loss.backward()

            optimizer.step()

            final_loss = float(
                loss.detach().cpu()
            )


        # ----------------------------------------------------
        # TRAIN-fit diagnostic only
        # ----------------------------------------------------

        model.eval()


        with torch.no_grad():
            train_pred = (
                model(
                    x_tensor
                )
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )


        train_lambda_mae = float(
            np.mean(
                np.abs(
                    train_pred
                    - train_y_np
                )
            )
        )


        train_projected = [
            project_lambda(
                lam
            )[
                0
            ]
            for lam in train_pred
        ]


        train_target_names = []

        for cid in train_contexts:
            for row in (
                labels_by_context[
                    cid
                ]
            ):
                train_target_names.append(
                    row[
                        "selected_beta_name"
                    ]
                )


        train_exact = float(
            np.mean(
                [
                    int(
                        pred_name
                        == target_name
                    )
                    for (
                        pred_name,
                        target_name,
                    ) in zip(
                        train_projected,
                        train_target_names,
                    )
                ]
            )
        )


        train_lambda_mae_all.append(
            train_lambda_mae
        )

        train_exact_all.append(
            train_exact
        )


        # ----------------------------------------------------
        # Held rough-context test
        # ----------------------------------------------------

        test_c = selector_context(
            test_cid
        )


        fold_rows = []

        fold_excess = []

        fold_lambda_mae = []

        fold_beta_mae = []

        fold_exact = []


        for row in (
            labels_by_context[
                test_cid
            ]
        ):
            preference_name = row[
                "preference_name"
            ]


            w = np.asarray(
                [
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
                ],
                dtype=np.float64,
            )


            model_input = np.concatenate(
                [
                    test_c,
                    w,
                ]
            ).astype(
                np.float32
            )


            with torch.no_grad():
                pred_lambda = (
                    model(
                        torch.from_numpy(
                            model_input[
                                None,
                                :
                            ]
                        )
                    )[
                        0
                    ]
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64
                    )
                )


            target_beta_name = row[
                "selected_beta_name"
            ]

            target_lambda = (
                lambda_from_beta_name(
                    target_beta_name
                )
            )


            pred_beta = beta_from_lambda(
                pred_lambda
            )

            target_beta = beta_from_lambda(
                target_lambda
            )


            projected_name, (
                projected_lambda
            ), projection_distance = (
                project_lambda(
                    pred_lambda
                )
            )


            projected_score = (
                score_beta(
                    context_id=(
                        test_cid
                    ),
                    beta_name=(
                        projected_name
                    ),
                    w=w,
                )
            )


            oracle_score = finite(
                row[
                    "scalarization_score"
                ]
            )


            score_excess = (
                projected_score
                - oracle_score
            )


            if score_excess < -1e-8:
                raise RuntimeError(
                    "Projected selector beta "
                    "beats frozen oracle."
                )


            score_excess = max(
                0.0,
                score_excess,
            )


            lambda_mae = float(
                np.mean(
                    np.abs(
                        pred_lambda
                        - target_lambda
                    )
                )
            )


            beta_mae = float(
                np.mean(
                    np.abs(
                        pred_beta
                        - target_beta
                    )
                )
            )


            exact = int(
                projected_name
                == target_beta_name
            )


            pred_row = {
                "fold":
                    fold_index,

                "held_context":
                    test_cid,

                "fold_pca_rank":
                    pca_rank,

                "fold_k95":
                    k95,

                "fold_k95_explained_variance":
                    float(
                        cumulative[
                            k95
                            - 1
                        ]
                    ),

                "selector_context_dim":
                    (
                        k95
                        + 1
                    ),

                "selector_input_dim":
                    (
                        k95
                        + 4
                    ),

                "preference_name":
                    preference_name,

                "w_motion":
                    w[0],

                "w_stability":
                    w[1],

                "w_energy":
                    w[2],

                "target_beta_name":
                    target_beta_name,

                "target_lambda_motion":
                    target_lambda[
                        0
                    ],

                "target_lambda_stability":
                    target_lambda[
                        1
                    ],

                "target_lambda_energy":
                    target_lambda[
                        2
                    ],

                "pred_lambda_motion":
                    pred_lambda[
                        0
                    ],

                "pred_lambda_stability":
                    pred_lambda[
                        1
                    ],

                "pred_lambda_energy":
                    pred_lambda[
                        2
                    ],

                "pred_beta_motion":
                    pred_beta[
                        0
                    ],

                "pred_beta_stability":
                    pred_beta[
                        1
                    ],

                "pred_beta_energy":
                    pred_beta[
                        2
                    ],

                "projected_beta_name":
                    projected_name,

                "projection_lambda_distance":
                    projection_distance,

                "lambda_mae":
                    lambda_mae,

                "beta_mae":
                    beta_mae,

                "projected_exact":
                    exact,

                "oracle_score":
                    oracle_score,

                "projected_score":
                    projected_score,

                "score_excess":
                    score_excess,

                "train_final_loss":
                    final_loss,

                "train_lambda_mae":
                    train_lambda_mae,

                "train_projected_exact":
                    train_exact,
            }


            prediction_rows.append(
                pred_row
            )

            fold_rows.append(
                pred_row
            )

            fold_excess.append(
                score_excess
            )

            fold_lambda_mae.append(
                lambda_mae
            )

            fold_beta_mae.append(
                beta_mae
            )

            fold_exact.append(
                exact
            )


        if len(
            fold_rows
        ) != EXPECTED_TEST_ROWS:
            raise RuntimeError(
                "Expected 25 held-context rows."
            )


        fold_excess_np = np.asarray(
            fold_excess,
            dtype=np.float64,
        )


        context_summary_rows.append(
            {
                "fold":
                    fold_index,

                "held_context":
                    test_cid,

                "pca_rank":
                    pca_rank,

                "k95":
                    k95,

                "k95_explained_variance":
                    float(
                        cumulative[
                            k95
                            - 1
                        ]
                    ),

                "train_final_loss":
                    final_loss,

                "train_lambda_mae":
                    train_lambda_mae,

                "train_projected_exact":
                    train_exact,

                "test_lambda_mae":
                    float(
                        np.mean(
                            fold_lambda_mae
                        )
                    ),

                "test_beta_mae":
                    float(
                        np.mean(
                            fold_beta_mae
                        )
                    ),

                "test_projected_exact":
                    float(
                        np.mean(
                            fold_exact
                        )
                    ),

                "test_score_excess_mean":
                    float(
                        np.mean(
                            fold_excess_np
                        )
                    ),

                "test_score_excess_p95":
                    float(
                        np.quantile(
                            fold_excess_np,
                            0.95,
                        )
                    ),

                "test_score_excess_max":
                    float(
                        np.max(
                            fold_excess_np
                        )
                    ),

                "test_score_excess_le_0p05_fraction":
                    float(
                        np.mean(
                            fold_excess_np
                            <= 0.05
                        )
                    ),
            }
        )


        fold_pca_rows.append(
            {
                "fold":
                    fold_index,

                "held_context":
                    test_cid,

                "rough_pca_train_contexts":
                    len(
                        rough_train
                    ),

                "pca_rank":
                    pca_rank,

                "k95":
                    k95,

                "k95_explained_variance":
                    float(
                        cumulative[
                            k95
                            - 1
                        ]
                    ),

                "selector_context_dim":
                    (
                        k95
                        + 1
                    ),

                "selector_input_dim":
                    (
                        k95
                        + 4
                    ),
            }
        )


        print(
            f"  PCA rank/k95      : "
            f"{pca_rank}/{k95}"
        )

        print(
            f"  k95 variance      : "
            f"{cumulative[k95 - 1]:.6f}"
        )

        print(
            f"  train exact       : "
            f"{train_exact:.3f}"
        )

        print(
            f"  test exact        : "
            f"{np.mean(fold_exact):.3f}"
        )

        print(
            f"  test mean excess  : "
            f"{np.mean(fold_excess_np):.6f}"
        )


    # ========================================================
    # Aggregate metrics
    # ========================================================

    if len(
        prediction_rows
    ) != EXPECTED_CV_ROWS:
        raise RuntimeError(
            f"Expected 450 CV rows; "
            f"got {len(prediction_rows)}"
        )


    lambda_mae = np.asarray(
        [
            row[
                "lambda_mae"
            ]
            for row in prediction_rows
        ],
        dtype=np.float64,
    )

    beta_mae = np.asarray(
        [
            row[
                "beta_mae"
            ]
            for row in prediction_rows
        ],
        dtype=np.float64,
    )

    exact = np.asarray(
        [
            row[
                "projected_exact"
            ]
            for row in prediction_rows
        ],
        dtype=np.float64,
    )

    score_excess = np.asarray(
        [
            row[
                "score_excess"
            ]
            for row in prediction_rows
        ],
        dtype=np.float64,
    )


    canonical_rows = [
        row
        for row in prediction_rows
        if row[
            "preference_name"
        ] in CANONICAL_PREFERENCES
    ]


    if len(
        canonical_rows
    ) != (
        EXPECTED_ROUGH
        * 4
    ):
        raise RuntimeError(
            "Canonical row-count mismatch."
        )


    canonical_exact = float(
        np.mean(
            [
                row[
                    "projected_exact"
                ]
                for row in canonical_rows
            ]
        )
    )


    k95_counts = Counter(
        row[
            "k95"
        ]
        for row in (
            fold_pca_rows
        )
    )


    aggregate = {
        "lambda_mae_mean":
            float(
                np.mean(
                    lambda_mae
                )
            ),

        "beta_mae_mean":
            float(
                np.mean(
                    beta_mae
                )
            ),

        "projected_exact_accuracy":
            float(
                np.mean(
                    exact
                )
            ),

        "canonical_exact_accuracy":
            canonical_exact,

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

        "score_excess_le_0p01_fraction":
            float(
                np.mean(
                    score_excess
                    <= 0.01
                )
            ),

        "score_excess_le_0p05_fraction":
            float(
                np.mean(
                    score_excess
                    <= 0.05
                )
            ),

        "train_lambda_mae_mean":
            float(
                np.mean(
                    train_lambda_mae_all
                )
            ),

        "train_projected_exact_mean":
            float(
                np.mean(
                    train_exact_all
                )
            ),
    }


    # ========================================================
    # Write outputs
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    write_csv(
        OUT_PREDICTIONS,
        prediction_rows,
    )

    write_csv(
        OUT_CONTEXT,
        context_summary_rows,
    )

    write_csv(
        OUT_PCA,
        fold_pca_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1l_"
                "selector_patch_pca_loco_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "external_heldout_used":
            False,

        "evaluation":
            (
                "18-fold leave-one-rough-TRAIN-"
                "context-out selector validation."
            ),

        "folds":
            EXPECTED_ROUGH,

        "cv_test_rows":
            EXPECTED_CV_ROWS,

        "train_contexts_per_fold":
            EXPECTED_TRAIN_CONTEXTS,

        "train_rows_per_fold":
            EXPECTED_TRAIN_ROWS,

        "test_rows_per_fold":
            EXPECTED_TEST_ROWS,

        "terrain_representation":
            {
                "source":
                    (
                        "8x6 start-height-relative "
                        "body-aligned oracle "
                        "height patch"
                    ),

                "raw_dim":
                    PATCH_DIM,

                "encoder":
                    "fold-local PCA",

                "pca_fit_contexts_per_fold":
                    17,

                "pca_fit_context_type":
                    (
                        "rough TRAIN contexts "
                        "excluding held rough context"
                    ),

                "pca_threshold":
                    PCA_VARIANCE_THRESHOLD,

                "k_selection":
                    (
                        "smallest fold-local PCA "
                        "dimension with >=95% "
                        "explained rough-patch "
                        "variance"
                    ),

                "whitening":
                    False,

                "held_context_used_for_pca":
                    False,

                "friction_appended_after_pca":
                    True,

                "k95_distribution":
                    {
                        str(k):
                            int(count)
                        for k, count in sorted(
                            k95_counts.items()
                        )
                    },
            },

        "model":
            {
                "family":
                    "MLP",

                "hidden_layers":
                    [
                        HIDDEN,
                        HIDDEN,
                    ],

                "activation":
                    "SiLU",

                "output":
                    (
                        "3 logits -> softmax "
                        "lambda"
                    ),

                "target":
                    (
                        "frozen oracle selected "
                        "lambda"
                    ),

                "loss":
                    "MSE(lambda)",

                "epochs":
                    EPOCHS,

                "optimizer":
                    "AdamW",

                "learning_rate":
                    LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "seed":
                    SEED,
            },

        "selector_semantics":
            (
                "[fold-local terrain latent z_T, "
                "friction mu, physical mission "
                "preference w] -> lambda -> "
                "beta = 0.15 + 0.55 lambda"
            ),

        "physical_evaluation":
            (
                "Continuous predicted lambda is "
                "projected to the nearest frozen "
                "21-point beta lattice, then "
                "evaluated using the original "
                "frozen T5.5d context-local "
                "Pareto scalarization."
            ),

        "aggregate":
            aggregate,

        "interpretation_guard":
            (
                "TRAIN-context LOCO only. "
                "No validation/test/hard rough "
                "seed is used. This is not final "
                "selector fitting."
            ),

        "future_interface":
            (
                "The fold-local PCA terrain latent "
                "is a simple encoder baseline and "
                "may later be replaced by CART, a "
                "world model, or another learned "
                "terrain representation without "
                "changing w->beta Objective "
                "Selector semantics."
            ),

        "next_stage":
            (
                "Compare against the original "
                "T5.5e1 5D-selector LOCO. "
                "Proceed to final TRAIN fit and "
                "external held-out validation only "
                "if cross-context physical score "
                "generalization improves "
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
    print("=" * 118)

    print(
        "ICRA27 OS-T5.5e1l PATCH-PCA "
        "OBJECTIVE SELECTOR LOCO"
    )

    print("=" * 118)


    print(
        "folds                         :",
        EXPECTED_ROUGH,
    )

    print(
        "CV test rows                  :",
        EXPECTED_CV_ROWS,
    )

    print(
        "fold-local k95 distribution   :",
        dict(
            sorted(
                k95_counts.items()
            )
        ),
    )


    print()
    print("TRAIN-FIT DIAGNOSTIC")

    print(
        "  lambda_mae_mean              :",
        aggregate[
            "train_lambda_mae_mean"
        ],
    )

    print(
        "  projected_exact_mean         :",
        aggregate[
            "train_projected_exact_mean"
        ],
    )


    print()
    print("ROUGH-LOCO PRIMARY")

    for key in (
        "lambda_mae_mean",
        "beta_mae_mean",
        "projected_exact_accuracy",
        "canonical_exact_accuracy",
        "score_excess_mean",
        "score_excess_median",
        "score_excess_p95",
        "score_excess_max",
        "score_excess_le_0p01_fraction",
        "score_excess_le_0p05_fraction",
    ):
        print(
            f"  {key:<32}: "
            f"{aggregate[key]}"
        )


    print()
    print("outputs:")
    print(" ", OUT_PREDICTIONS)
    print(" ", OUT_CONTEXT)
    print(" ", OUT_PCA)
    print(" ", OUT_MANIFEST)


    print()
    print(
        "[ICRA27] OS-T5.5e1l "
        "patch-PCA selector LOCO: "
        "COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
