from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_os_t5p5e1n_response_model_selector_loco_v0 as base


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

EXT_PATCH_CSV = (
    ROOT
    / "results/icra27"
    / "os_t5p5f1_selector_train_extension_height_patches_v0"
    / "extension_height_patches.csv"
)

EXT_PATCH_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5f1_selector_train_extension_height_patches_v0"
    / "extension_height_patch_manifest.json"
)

ATLAS = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
    / "physical_beta_response_atlas.csv"
)

ATLAS_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3_expanded_physical_atlas_v0"
    / "expanded_physical_atlas_manifest.json"
)

LABELS = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3b_expanded_preference_labels_v0"
    / "preference_to_beta_labels.csv"
)

LABEL_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5f3b_expanded_preference_labels_v0"
    / "expanded_preference_labels_manifest.json"
)

BASELINE_E1P_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1p_nonlinear_log_regret_selector_loco_v0"
    / "nonlinear_log_regret_selector_manifest.json"
)


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5f4b_fixed_k6_coverage_log_regret_selector_loco_v0"
)

OUT_REGRET = (
    OUT_DIR
    / "log_regret_predictions.csv"
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
    / "nonlinear_log_regret_selector_manifest.json"
)


# ============================================================
# Frozen experiment choices
# ============================================================

PATCH_DIM = 48
EXPECTED_ROUGH = 18
EXPECTED_BETA = 21
EXPECTED_PREFS = 25

PCA_THRESHOLD = 0.95

HIDDEN = 64
EPOCHS = 2000

LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-4

SEED = 27552

TOL = 1.0e-10
EPS = 1.0e-12


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


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite: {x!r}"
        )

    return y


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

    # Numerical guard only.
    regret = np.maximum(
        regret,
        0.0,
    )

    return regret


def regret_score(
    regret,
    w,
):
    regret = np.asarray(
        regret,
        dtype=np.float64,
    )

    w = np.asarray(
        w,
        dtype=np.float64,
    )

    weighted = (
        w * regret
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


class RegretMLP(
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
        return self.net(
            x
        )


def decode_log_regret(
    prediction_z,
    *,
    target_mean,
    target_std,
):
    prediction_z = np.asarray(
        prediction_z,
        dtype=np.float64,
    )

    log_prediction = (
        prediction_z
        * target_std
        + target_mean
    )

    negative_fraction = float(
        np.mean(
            log_prediction
            < 0.0
        )
    )

    # True log1p(regret) is non-negative.
    # Lower clipping is therefore a semantic
    # constraint, not a learned calibration.
    #
    # Upper=50 is only a numerical overflow guard;
    # exp(50)-1 is effectively "very bad" for the
    # selector and does not impose a useful
    # physical scale.
    log_prediction_clipped = np.clip(
        log_prediction,
        0.0,
        50.0,
    )

    regret = np.expm1(
        log_prediction_clipped
    )

    return (
        regret,
        negative_fraction,
        log_prediction,
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
        EXT_PATCH_CSV,
        EXT_PATCH_MANIFEST,
        ATLAS,
        ATLAS_MANIFEST,
        LABELS,
        LABEL_MANIFEST,
        BASELINE_E1P_MANIFEST,
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

    if bool(
        patch_contract[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "Patch artifact reports heldout use."
        )

    if int(
        patch_contract[
            "patch_dim"
        ]
    ) != PATCH_DIM:
        raise RuntimeError(
            "Patch dimension mismatch."
        )


    baseline_e1p = json.loads(
        BASELINE_E1P_MANIFEST.read_text()
    )

    if baseline_e1p.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "Baseline T5.5e1p is not COMPUTE_PASS."
        )

    if bool(
        baseline_e1p[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "Baseline T5.5e1p reports heldout use."
        )


    # ========================================================
    # Height patches
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


    old_patch_rows = read_csv(
        PATCH_CSV
    )

    extension_patch_rows_all = read_csv(
        EXT_PATCH_CSV
    )


    extension_patch_manifest = json.loads(
        EXT_PATCH_MANIFEST.read_text()
    )

    if extension_patch_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "Extension height patches are not "
            "FREEZE_PASS."
        )

    if bool(
        extension_patch_manifest[
            "external_heldout_used"
        ]
    ):
        raise RuntimeError(
            "Extension patch artifact reports "
            "heldout use."
        )


    atlas_manifest = json.loads(
        ATLAS_MANIFEST.read_text()
    )

    if atlas_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.5f3 physical atlas is not "
            "FREEZE_PASS."
        )

    if bool(
        atlas_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5f3 reports heldout use."
        )


    label_manifest = json.loads(
        LABEL_MANIFEST.read_text()
    )

    if label_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.5f3b labels are not "
            "FREEZE_PASS."
        )

    if bool(
        label_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5f3b reports heldout use."
        )

    regression = label_manifest[
        "old_t5p5d_regression"
    ]

    if (
        int(
            regression[
                "rows"
            ]
        ) != 500
        or int(
            regression[
                "exact_selected_beta"
            ]
        ) != 500
        or float(
            regression[
                "max_scalarization_score_error"
            ]
        ) != 0.0
    ):
        raise RuntimeError(
            "T5.5f3b did not exactly reproduce "
            "frozen T5.5d."
        )


    eligible_seeds = [
        int(seed)
        for seed in atlas_manifest[
            "full_grid_extension_seeds"
        ]
    ]

    if len(
        eligible_seeds
    ) != 15:
        raise RuntimeError(
            "Expected exactly 15 full-grid "
            "extension rough contexts."
        )


    extension_contexts = [
        f"rough_seed_{seed}"
        for seed in eligible_seeds
    ]


    extension_rows_by_context = {
        row[
            "context_id"
        ]:
            row
        for row in extension_patch_rows_all
    }


    if len(
        extension_rows_by_context
    ) != 18:
        raise RuntimeError(
            "Expected 18 frozen extension "
            "height-patch rows."
        )


    extension_patch_rows = []

    for cid in extension_contexts:
        if cid not in extension_rows_by_context:
            raise RuntimeError(
                f"Missing extension patch: {cid}"
            )

        extension_patch_rows.append(
            extension_rows_by_context[
                cid
            ]
        )


    patch_rows = (
        old_patch_rows
        + extension_patch_rows
    )


    patch_lookup = {}

    friction_lookup = {}

    rough_contexts = []


    for row in old_patch_rows:
        cid = row[
            "context_id"
        ]

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
            "Expected exactly 18 original "
            "rough LOCO evaluation contexts."
        )


    if set(
        rough_contexts
    ) & set(
        extension_contexts
    ):
        raise RuntimeError(
            "Original and extension rough "
            "contexts overlap."
        )


    for row in patch_rows:
        cid = row[
            "context_id"
        ]

        if cid in patch_lookup:
            raise RuntimeError(
                f"Duplicate patch context: {cid}"
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


    if len(
        patch_lookup
    ) != 35:
        raise RuntimeError(
            f"Expected 35 representation "
            f"contexts; got {len(patch_lookup)}"
        )


    # ========================================================
    # Physical atlas
    # ========================================================

    atlas = read_csv(
        ATLAS
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
                f"Duplicate atlas row: {key}"
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
        len(
            beta_order
        )
        != EXPECTED_BETA
        or len(
            set(
                beta_order
            )
        )
        != EXPECTED_BETA
    ):
        raise RuntimeError(
            "Expected 21 beta points."
        )


    beta_lambda = {
        beta_name:
            base.lambda_from_beta_name(
                beta_name
            )
        for beta_name in (
            beta_order
        )
    }


    # ========================================================
    # Frozen context-local regret references
    # ========================================================

    reference = {}


    for cid in patch_lookup:
        values = np.asarray(
            [
                [
                    finite(
                        row[
                            column
                        ]
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
    # Preference labels
    # ========================================================

    labels = read_csv(
        LABELS
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
                f"{cid}: expected 25 "
                f"preference labels."
            )


    # ========================================================
    # Frozen e1p global physical IQR
    #
    # Reporting only.
    #
    # IMPORTANT:
    # Keep the ORIGINAL e1p scale so physical Linf is directly
    # comparable between 17-context e1p and 32-context f4.
    # ========================================================

    global_iqr = np.asarray(
        baseline_e1p[
            "global_iqr_reporting_only"
        ],
        dtype=np.float64,
    )


    if (
        global_iqr.shape != (3,)
        or np.any(
            global_iqr <= 0.0
        )
    ):
        raise RuntimeError(
            "Invalid frozen e1p physical IQR."
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
            "=" * 112
        )

        print(
            f"fold {fold:02d}/18 "
            f"held={held_cid}"
        )

        print(
            "=" * 112
        )


        old_rough_train = [
            cid
            for cid in rough_contexts
            if cid != held_cid
        ]


        if len(
            old_rough_train
        ) != 17:
            raise RuntimeError(
                "Expected 17 original rough "
                "TRAIN contexts."
            )


        rough_train = [
            *old_rough_train,
            *extension_contexts,
        ]


        if len(
            rough_train
        ) != 32:
            raise RuntimeError(
                "Expected 32 rough PCA/TRAIN "
                "contexts: 17 original + "
                "15 extension."
            )


        if held_cid in rough_train:
            raise RuntimeError(
                "Held original rough context "
                "leaked into TRAIN."
            )


        train_contexts = [
            "flat",
            "low_friction",
            *rough_train,
        ]


        if len(
            train_contexts
        ) != 34:
            raise RuntimeError(
                "Expected 34 selector TRAIN "
                "contexts: flat + LF + "
                "32 rough."
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
                "Degenerate fold PCA."
            )


        cumulative = np.cumsum(
            energy
            / total_energy
        )


        rank = int(
            np.sum(
                singular
                > 1.0e-12
            )
        )


        k95 = 6  # T5.5f4b fixed-k coverage control


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
            ddof=0,
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
        # Dense supervision
        #
        # input  = [context, lambda]
        # target = standardized log1p(regret)
        # ====================================================

        train_x = []

        train_regret = []


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
                        finite(
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


                train_x.append(
                    np.concatenate(
                        [
                            c,
                            beta_lambda[
                                beta_name
                            ],
                        ]
                    )
                )


                train_regret.append(
                    regret
                )


        train_x = np.asarray(
            train_x,
            dtype=np.float64,
        )

        train_regret = np.asarray(
            train_regret,
            dtype=np.float64,
        )


        if train_x.shape[0] != 714:
            raise RuntimeError(
                "Expected 714 response rows: "
                "34 contexts x 21 beta."
            )


        # ----------------------------------------------------
        # Input normalization
        # ----------------------------------------------------

        x_mean = np.mean(
            train_x,
            axis=0,
        )

        x_std = np.std(
            train_x,
            axis=0,
            ddof=0,
        )


        x_std = np.where(
            x_std > EPS,
            x_std,
            1.0,
        )


        train_x_z = (
            train_x
            - x_mean
        ) / x_std


        # ----------------------------------------------------
        # log1p target conditioning
        # ----------------------------------------------------

        train_log_target = np.log1p(
            train_regret
        )


        target_mean = np.mean(
            train_log_target,
            axis=0,
        )


        target_std = np.std(
            train_log_target,
            axis=0,
            ddof=0,
        )


        if np.any(
            target_std <= EPS
        ):
            raise RuntimeError(
                "Degenerate log-regret target."
            )


        train_y_z = (
            train_log_target
            - target_mean
        ) / target_std


        # ====================================================
        # Small nonlinear surrogate
        # ====================================================

        np.random.seed(
            SEED
        )

        torch.manual_seed(
            SEED
        )


        model = RegretMLP(
            input_dim=(
                train_x_z.shape[
                    1
                ]
            )
        )


        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=(
                    LEARNING_RATE
                ),
                weight_decay=(
                    WEIGHT_DECAY
                ),
            )
        )


        x_tensor = torch.from_numpy(
            train_x_z.astype(
                np.float32
            )
        )


        y_tensor = torch.from_numpy(
            train_y_z.astype(
                np.float32
            )
        )


        final_loss = None


        model.train()


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


        # ====================================================
        # TRAIN fit diagnostic
        # ====================================================

        model.eval()


        with torch.no_grad():
            train_pred_z = (
                model(
                    x_tensor
                )
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )


        (
            train_pred_regret,
            train_negative_fraction,
            train_pred_log,
        ) = decode_log_regret(
            train_pred_z,
            target_mean=target_mean,
            target_std=target_std,
        )


        train_log_mae = float(
            np.mean(
                np.abs(
                    train_pred_log
                    - train_log_target
                )
            )
        )


        train_regret_mae = float(
            np.mean(
                np.abs(
                    train_pred_regret
                    - train_regret
                )
            )
        )


        train_regret_p95 = float(
            np.quantile(
                np.abs(
                    train_pred_regret
                    - train_regret
                ),
                0.95,
            )
        )


        # ====================================================
        # Held terrain — 21 beta surface
        # ====================================================

        held_true = []

        held_pred = []

        held_log_true = []

        held_log_pred = []


        held_ideal, held_span = (
            reference[
                held_cid
            ]
        )


        held_negative_count = 0


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
                    finite(
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


            true_log = np.log1p(
                true_regret
            )


            model_input = np.concatenate(
                [
                    context_feature(
                        held_cid
                    ),

                    beta_lambda[
                        beta_name
                    ],
                ]
            )


            model_input_z = (
                model_input
                - x_mean
            ) / x_std


            with torch.no_grad():
                pred_z = (
                    model(
                        torch.from_numpy(
                            model_input_z[
                                None,
                                :
                            ].astype(
                                np.float32
                            )
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


            (
                predicted_regret,
                negative_fraction,
                predicted_log,
            ) = decode_log_regret(
                pred_z,
                target_mean=target_mean,
                target_std=target_std,
            )


            held_negative_count += int(
                np.sum(
                    predicted_log
                    < 0.0
                )
            )


            held_true.append(
                true_regret
            )

            held_pred.append(
                predicted_regret
            )

            held_log_true.append(
                true_log
            )

            held_log_pred.append(
                predicted_log
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

                    "true_log_motion":
                        true_log[
                            0
                        ],

                    "true_log_stability":
                        true_log[
                            1
                        ],

                    "true_log_energy":
                        true_log[
                            2
                        ],

                    "pred_log_motion":
                        predicted_log[
                            0
                        ],

                    "pred_log_stability":
                        predicted_log[
                            1
                        ],

                    "pred_log_energy":
                        predicted_log[
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

                    "log_abs_error_mean":
                        float(
                            np.mean(
                                np.abs(
                                    predicted_log
                                    - true_log
                                )
                            )
                        ),

                    "preclip_negative_log_fraction":
                        negative_fraction,
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

        held_log_true = np.asarray(
            held_log_true,
            dtype=np.float64,
        )

        held_log_pred = np.asarray(
            held_log_pred,
            dtype=np.float64,
        )


        held_regret_error = np.abs(
            held_pred
            - held_true
        )


        held_log_error = np.abs(
            held_log_pred
            - held_log_true
        )


        # ====================================================
        # Predicted Pareto geometry
        # ====================================================

        predicted_front = (
            base.pareto_indices(
                held_pred
            )
        )


        if not predicted_front:
            raise RuntimeError(
                "Empty predicted Pareto front."
            )


        # ====================================================
        # Mission preference -> beta
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


            candidates = []


            for index in (
                predicted_front
            ):
                predicted_score = (
                    regret_score(
                        held_pred[
                            index
                        ],
                        w,
                    )
                )


                candidates.append(
                    (
                        predicted_score,
                        index,
                    )
                )


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


            oracle_beta = (
                label[
                    "selected_beta_name"
                ]
            )


            oracle_score = finite(
                label[
                    "scalarization_score"
                ]
            )


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


            if score_excess < -1.0e-8:
                raise RuntimeError(
                    "Nonlinear regret selector "
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
                    finite(
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
                    finite(
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

                "model_input_dim":
                    int(
                        train_x_z.shape[
                            1
                        ]
                    ),

                "train_final_loss":
                    final_loss,

                "train_log_mae":
                    train_log_mae,

                "train_regret_mae":
                    train_regret_mae,

                "train_regret_abs_error_p95":
                    train_regret_p95,

                "train_preclip_negative_log_fraction":
                    train_negative_fraction,

                "held_log_mae":
                    float(
                        np.mean(
                            held_log_error
                        )
                    ),

                "held_log_abs_error_p95":
                    float(
                        np.quantile(
                            held_log_error,
                            0.95,
                        )
                    ),

                "held_regret_mae":
                    float(
                        np.mean(
                            held_regret_error
                        )
                    ),

                "held_regret_abs_error_p95":
                    float(
                        np.quantile(
                            held_regret_error,
                            0.95,
                        )
                    ),

                "held_preclip_negative_log_fraction":
                    (
                        held_negative_count
                        / (
                            EXPECTED_BETA
                            * 3
                        )
                    ),

                "predicted_pareto_count":
                    len(
                        predicted_front
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


        if k95 != 6:
            raise RuntimeError(
                f"T5.5f4b fixed-k contract violated: "
                f"k95={k95}"
            )


        print(
            f"  k95={k95} "
            f"trainLogMAE="
            f"{train_log_mae:.4f} "
            f"trainRegMAE="
            f"{train_regret_mae:.4f} "
            f"heldLogMAE="
            f"{np.mean(held_log_error):.4f} "
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


    all_log_error = np.asarray(
        [
            row[
                "log_abs_error_mean"
            ]
            for row in (
                regret_rows
            )
        ],
        dtype=np.float64,
    )


    all_score = np.asarray(
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


    all_physical = np.asarray(
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


    all_exact = np.asarray(
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


    train_log_mae_mean = float(
        np.mean(
            [
                row[
                    "train_log_mae"
                ]
                for row in (
                    context_rows
                )
            ]
        )
    )


    train_regret_mae_mean = float(
        np.mean(
            [
                row[
                    "train_regret_mae"
                ]
                for row in (
                    context_rows
                )
            ]
        )
    )


    aggregate = {
        "train_log_mae_mean":
            train_log_mae_mean,

        "train_regret_mae_mean":
            train_regret_mae_mean,

        "held_log_mae":
            float(
                np.mean(
                    all_log_error
                )
            ),

        "held_log_mae_median":
            float(
                np.median(
                    all_log_error
                )
            ),

        "held_log_mae_p95":
            float(
                np.quantile(
                    all_log_error,
                    0.95,
                )
            ),

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
                    all_physical
                )
            ),

        "selector_physical_linf_median_iqr":
            float(
                np.median(
                    all_physical
                )
            ),

        "selector_physical_linf_p95_iqr":
            float(
                np.quantile(
                    all_physical,
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


    write_csv(
        OUT_REGRET,
        regret_rows,
    )

    write_csv(
        OUT_SELECTOR,
        selector_rows,
    )

    write_csv(
        OUT_CONTEXT,
        context_rows,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5e1p_"
                "nonlinear_log_regret_"
                "selector_loco_v0"
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
                "from 32 rough TRAIN patches (17 original + 15 extension)"
            ),

        "supervision":
            (
                "19 TRAIN contexts x "
                "21 beta points = 399 "
                "dense regret-surface samples "
                "per fold"
            ),

        "model":
            {
                "type":
                    "MLP",

                "input":
                    (
                        "[fold-local PCA terrain "
                        "latent, friction mu, "
                        "beta simplex lambda]"
                    ),

                "hidden_layers":
                    [
                        HIDDEN,
                        HIDDEN,
                    ],

                "activation":
                    "SiLU",

                "epochs":
                    EPOCHS,

                "optimizer":
                    "AdamW",

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "seed":
                    SEED,
            },

        "target":
            (
                "standardized log1p("
                "frozen T5.5d context-local "
                "normalized regret)"
            ),

        "target_transform_guard":
            (
                "log1p is used only to condition "
                "the regression loss. Predictions "
                "are inverse-transformed back to "
                "the original frozen regret space "
                "before Pareto filtering, mission "
                "scalarization, and beta selection."
            ),

        "selector":
            (
                "Predict all 21 original-space "
                "regret vectors, construct the "
                "predicted Pareto front, and "
                "minimize the original augmented "
                "weighted Tchebycheff score for w."
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

                "baseline_e1p_score_mean":
                    finite(
                        baseline_e1p[
                            "aggregate"
                        ][
                            "selector_score_excess_mean"
                        ]
                    ),

                "baseline_e1p_score_p95":
                    finite(
                        baseline_e1p[
                            "aggregate"
                        ][
                            "selector_score_excess_p95"
                        ]
                    ),

                "baseline_e1p_held_regret_mae":
                    finite(
                        baseline_e1p[
                            "aggregate"
                        ][
                            "held_regret_mae"
                        ]
                    ),
            },

        "interpretation_guard":
            (
                "TRAIN rough-context LOCO only. "
                "No validation/test/hard terrain "
                "is used. Frozen T5.5d labels, "
                "physical atlas, and scalarization "
                "are unchanged."
            ),

        "decision_rule":
            (
                "If nonlinear log-regret training "
                "fit becomes strong but LOCO "
                "generalization remains poor, the "
                "remaining bottleneck is terrain "
                "representation / context-to-"
                "response generalization. If "
                "training fit remains poor, do not "
                "change terrain encoding yet."
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
        "ICRA27 OS-T5.5f4 NONLINEAR "
        "LOG-REGRET SURFACE SELECTOR LOCO"
    )

    print("=" * 122)


    print()
    print("SURROGATE FIT")

    print(
        "  train_log_mae_mean       :",
        aggregate[
            "train_log_mae_mean"
        ],
    )

    print(
        "  train_regret_mae_mean    :",
        aggregate[
            "train_regret_mae_mean"
        ],
    )

    print(
        "  held_log_mae             :",
        aggregate[
            "held_log_mae"
        ],
    )

    print(
        "  held_regret_mae          :",
        aggregate[
            "held_regret_mae"
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
        "[ICRA27] OS-T5.5f4 "
        "nonlinear log-regret selector "
        "LOCO: COMPUTE PASS"
    )

    print("=" * 122)


if __name__ == "__main__":
    main()
