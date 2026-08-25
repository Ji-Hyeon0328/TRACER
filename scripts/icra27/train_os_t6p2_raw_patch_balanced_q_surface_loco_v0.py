from __future__ import annotations

import csv
import json
import math
import re
import runpy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Reuse already validated path contracts from T5.5f4.
# This avoids guessing old height-patch filenames.
# ============================================================

F4_SOURCE = (
    ROOT
    / "scripts/icra27"
    / "train_os_t5p5f4_coverage_expanded_log_regret_selector_loco_v0.py"
)

if not F4_SOURCE.exists():
    raise FileNotFoundError(
        F4_SOURCE
    )

contract = runpy.run_path(
    str(F4_SOURCE),
    run_name="icra27_t6p2_contract",
)

PATCH_CSV = Path(
    contract["PATCH_CSV"]
)

EXT_PATCH_CSV = Path(
    contract["EXT_PATCH_CSV"]
)

ATLAS = Path(
    contract["ATLAS"]
)

ATLAS_MANIFEST = Path(
    contract["ATLAS_MANIFEST"]
)

BASELINE_E1P_MANIFEST = Path(
    contract["BASELINE_E1P_MANIFEST"]
)


QUALITY_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p1_balanced_physical_quality_surface_v0"
)

QUALITY_SURFACE = (
    QUALITY_DIR
    / "balanced_beta_quality_surface.csv"
)

QUALITY_MANIFEST = (
    QUALITY_DIR
    / "balanced_quality_surface_manifest.json"
)


OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p2_raw_patch_balanced_q_surface_loco_v0"
)

OUT_PRED = (
    OUT_DIR
    / "q_surface_predictions.csv"
)

OUT_CONTEXT = (
    OUT_DIR
    / "context_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "raw_patch_q_surface_selector_manifest.json"
)


EXPECTED_OLD_ROUGH = 18
EXPECTED_EXT_ROUGH = 15
EXPECTED_BETA = 21

HIDDEN = 64
EPOCHS = 2000
LR = 3.0e-3
WEIGHT_DECAY = 1.0e-4
SEED = 27621

EPS = 1.0e-12


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def finite(x):
    y = float(x)

    if not math.isfinite(y):
        raise ValueError(
            f"Non-finite value: {x}"
        )

    return y


def infer_patch_features(
    old_rows,
    ext_rows,
):
    old_fields = list(
        old_rows[0].keys()
    )

    ext_fields = set(
        ext_rows[0].keys()
    )

    common = [
        x
        for x in old_fields
        if x in ext_fields
    ]

    # Frozen T5.5 height-patch schema:
    # 8 longitudinal x 6 lateral cells.
    #
    # h_l00_r00_m ... h_l07_r05_m
    semantic = [
        x
        for x in common
        if re.fullmatch(
            r"h_l\d{2}_r\d{2}_m",
            x,
        )
    ]

    if len(semantic) == 48:
        def patch_key(name):
            match = re.fullmatch(
                r"h_l(\d{2})_r(\d{2})_m",
                name,
            )

            if match is None:
                raise RuntimeError(
                    f"Invalid patch field: {name}"
                )

            return (
                int(match.group(1)),
                int(match.group(2)),
            )

        semantic = sorted(
            semantic,
            key=patch_key,
        )

        expected = [
            f"h_l{l:02d}_r{r:02d}_m"
            for l in range(8)
            for r in range(6)
        ]

        if semantic != expected:
            raise RuntimeError(
                "48 patch fields found, but frozen "
                "8x6 ordering/schema does not match."
            )

        return semantic

    excluded = {
        "context_id",
        "friction_mu",
    }

    numeric = []

    for name in common:
        if name in excluded:
            continue

        values = []

        ok = True

        for row in (
            old_rows
            + ext_rows
        ):
            try:
                values.append(
                    finite(
                        row[name]
                    )
                )
            except Exception:
                ok = False
                break

        if ok:
            numeric.append(
                name
            )

    # Remove obvious metadata if necessary.
    metadata_tokens = (
        "seed",
        "hash",
        "min",
        "max",
        "mean",
        "std",
        "range",
        "resolution",
        "size",
        "count",
        "rows",
        "cols",
    )

    filtered = [
        x
        for x in numeric
        if not any(
            token in x.lower()
            for token in metadata_tokens
        )
    ]

    if len(filtered) == 48:
        return filtered

    raise RuntimeError(
        "Could not uniquely infer the 48 height-patch "
        "features.\n"
        f"semantic candidates ({len(semantic)}): "
        f"{semantic}\n"
        f"numeric filtered ({len(filtered)}): "
        f"{filtered}"
    )


def parse_lambda(
    beta_name,
):
    match = re.fullmatch(
        r"lm(\d+)_ls(\d+)_le(\d+)",
        beta_name,
    )

    if match is None:
        raise RuntimeError(
            f"Cannot parse beta lattice name: "
            f"{beta_name}"
        )

    lam = np.asarray(
        [
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        ],
        dtype=np.float64,
    ) / 100.0

    if abs(
        float(
            np.sum(lam)
        )
        - 1.0
    ) > 1.0e-9:
        raise RuntimeError(
            f"Invalid simplex beta name: "
            f"{beta_name}"
        )

    return lam


class QNet(nn.Module):
    def __init__(
        self,
        input_dim,
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
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


def nondestructive_standardize(
    train_x,
    test_x,
):
    mean = np.mean(
        train_x,
        axis=0,
    )

    std = np.std(
        train_x,
        axis=0,
    )

    std = np.where(
        std > 1.0e-8,
        std,
        1.0,
    )

    return (
        (train_x - mean) / std,
        (test_x - mean) / std,
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        PATCH_CSV,
        EXT_PATCH_CSV,
        ATLAS,
        ATLAS_MANIFEST,
        BASELINE_E1P_MANIFEST,
        QUALITY_SURFACE,
        QUALITY_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    q_manifest = json.loads(
        QUALITY_MANIFEST.read_text()
    )

    if q_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.1 quality surface is not "
            "FREEZE_PASS."
        )

    if bool(
        q_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.1 reports heldout use."
        )

    atlas_manifest = json.loads(
        ATLAS_MANIFEST.read_text()
    )

    eligible_seeds = [
        int(x)
        for x in atlas_manifest[
            "full_grid_extension_seeds"
        ]
    ]

    if len(
        eligible_seeds
    ) != EXPECTED_EXT_ROUGH:
        raise RuntimeError(
            "Expected 15 full-grid extension "
            "rough contexts."
        )

    extension_contexts = [
        f"rough_seed_{seed}"
        for seed in eligible_seeds
    ]


    # ========================================================
    # Terrain representation
    # ========================================================

    old_patch_rows = read_csv(
        PATCH_CSV
    )

    ext_patch_rows_all = read_csv(
        EXT_PATCH_CSV
    )

    old_rough_rows = [
        row
        for row in old_patch_rows
        if row[
            "context_id"
        ].startswith(
            "rough_seed_"
        )
    ]

    if len(
        old_rough_rows
    ) != EXPECTED_OLD_ROUGH:
        raise RuntimeError(
            f"Expected 18 original rough patches; "
            f"got {len(old_rough_rows)}."
        )

    ext_by_context = {
        row[
            "context_id"
        ]:
            row
        for row in ext_patch_rows_all
    }

    ext_rows = [
        ext_by_context[
            cid
        ]
        for cid in extension_contexts
    ]

    patch_features = infer_patch_features(
        old_rough_rows,
        ext_rows,
    )

    print(
        "height-patch features:",
        len(patch_features),
    )

    terrain_rows = (
        old_rough_rows
        + ext_rows
    )

    terrain_lookup = {}

    for row in terrain_rows:
        cid = row[
            "context_id"
        ]

        patch = np.asarray(
            [
                finite(
                    row[name]
                )
                for name in patch_features
            ],
            dtype=np.float64,
        )

        if patch.shape != (48,):
            raise RuntimeError(
                f"{cid}: patch dimension "
                f"{patch.shape}"
            )

        mu = finite(
            row[
                "friction_mu"
            ]
        )

        terrain_lookup[
            cid
        ] = np.concatenate(
            [
                np.asarray(
                    [mu],
                    dtype=np.float64,
                ),
                patch,
            ]
        )

    if len(
        terrain_lookup
    ) != 33:
        raise RuntimeError(
            f"Expected 33 rough terrain contexts; "
            f"got {len(terrain_lookup)}."
        )


    # ========================================================
    # Q surface
    # ========================================================

    q_rows = read_csv(
        QUALITY_SURFACE
    )

    q_lookup = {}

    beta_order = None

    for row in q_rows:
        cid = row[
            "context_id"
        ]

        if cid not in terrain_lookup:
            continue

        q_lookup.setdefault(
            cid,
            [],
        ).append(
            row
        )

    for cid in terrain_lookup:
        if cid not in q_lookup:
            raise RuntimeError(
                f"Missing T6.1 Q surface: "
                f"{cid}"
            )

        local = sorted(
            q_lookup[cid],
            key=lambda x:
                int(
                    x[
                        "beta_index"
                    ]
                ),
        )

        if len(
            local
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 Q rows."
            )

        names = [
            x[
                "beta_name"
            ]
            for x in local
        ]

        if beta_order is None:
            beta_order = names

        elif names != beta_order:
            raise RuntimeError(
                f"{cid}: beta order mismatch."
            )

        q_lookup[
            cid
        ] = local

    beta_lambda = np.stack(
        [
            parse_lambda(
                name
            )
            for name in beta_order
        ],
        axis=0,
    )


    original_contexts = [
        row[
            "context_id"
        ]
        for row in old_rough_rows
    ]

    prediction_rows = []

    context_rows = []


    baseline_manifest = json.loads(
        BASELINE_E1P_MANIFEST.read_text()
    )

    physical_iqr = np.asarray(
        baseline_manifest[
            "global_iqr_reporting_only"
        ],
        dtype=np.float64,
    )


    # Physical J lookup for selected-beta audit.
    atlas_rows = read_csv(
        ATLAS
    )

    physical_lookup = {}

    for row in atlas_rows:
        cid = row[
            "context_id"
        ]

        beta = row[
            "beta_name"
        ]

        physical_lookup[
            (
                cid,
                beta,
            )
        ] = np.asarray(
            [
                finite(
                    row[
                        "J_motion_s_per_m"
                    ]
                ),
                finite(
                    row[
                        "J_stability"
                    ]
                ),
                finite(
                    row[
                        "J_energy_j_per_m"
                    ]
                ),
            ],
            dtype=np.float64,
        )


    for fold_idx, held_cid in enumerate(
        original_contexts
    ):
        print()
        print("=" * 112)
        print(
            f"fold {fold_idx + 1:02d}/18 "
            f"held={held_cid}"
        )
        print("=" * 112)

        old_train = [
            cid
            for cid in original_contexts
            if cid != held_cid
        ]

        train_contexts = (
            old_train
            + extension_contexts
        )

        if len(
            train_contexts
        ) != 32:
            raise RuntimeError(
                "Expected 32 rough TRAIN contexts."
            )

        train_x = []

        train_y = []

        for cid in train_contexts:
            cvec = terrain_lookup[
                cid
            ]

            for beta_idx in range(
                EXPECTED_BETA
            ):
                train_x.append(
                    np.concatenate(
                        [
                            cvec,
                            beta_lambda[
                                beta_idx
                            ],
                        ]
                    )
                )

                q = finite(
                    q_lookup[
                        cid
                    ][
                        beta_idx
                    ][
                        "quality_Q"
                    ]
                )

                if q < -EPS:
                    raise RuntimeError(
                        f"Negative Q: {q}"
                    )

                train_y.append(
                    math.log1p(
                        max(
                            0.0,
                            q,
                        )
                    )
                )

        held_x = np.stack(
            [
                np.concatenate(
                    [
                        terrain_lookup[
                            held_cid
                        ],
                        beta_lambda[
                            beta_idx
                        ],
                    ]
                )
                for beta_idx in range(
                    EXPECTED_BETA
                )
            ],
            axis=0,
        )

        held_true_q = np.asarray(
            [
                finite(
                    q_lookup[
                        held_cid
                    ][
                        beta_idx
                    ][
                        "quality_Q"
                    ]
                )
                for beta_idx in range(
                    EXPECTED_BETA
                )
            ],
            dtype=np.float64,
        )

        train_x = np.asarray(
            train_x,
            dtype=np.float64,
        )

        train_y = np.asarray(
            train_y,
            dtype=np.float64,
        ).reshape(
            -1,
            1,
        )

        if train_x.shape[0] != 672:
            raise RuntimeError(
                "Expected 32 x 21 = 672 "
                "TRAIN Q samples."
            )

        (
            train_x,
            held_x,
        ) = nondestructive_standardize(
            train_x,
            held_x,
        )

        if not isinstance(
            train_x,
            np.ndarray,
        ):
            raise RuntimeError(
                f"train_x must be ndarray; "
                f"got {type(train_x)}"
            )

        if not isinstance(
            held_x,
            np.ndarray,
        ):
            raise RuntimeError(
                f"held_x must be ndarray; "
                f"got {type(held_x)}"
            )

        if train_x.shape[0] != 672:
            raise RuntimeError(
                f"Expected standardized TRAIN shape "
                f"(672,D); got {train_x.shape}"
            )

        if held_x.shape[0] != 21:
            raise RuntimeError(
                f"Expected standardized HELD shape "
                f"(21,D); got {held_x.shape}"
            )


        y_mean = np.mean(
            train_y,
            axis=0,
        )

        y_std = np.std(
            train_y,
            axis=0,
        )

        y_std = np.where(
            y_std > 1.0e-8,
            y_std,
            1.0,
        )

        train_y_std = (
            train_y
            - y_mean
        ) / y_std


        torch.manual_seed(
            SEED + fold_idx
        )

        np.random.seed(
            SEED + fold_idx
        )

        model = QNet(
            train_x.shape[1]
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        loss_fn = nn.MSELoss()

        X = torch.as_tensor(
            train_x,
            dtype=torch.float32,
        )

        Y = torch.as_tensor(
            train_y_std,
            dtype=torch.float32,
        )

        model.train()

        for _ in range(
            EPOCHS
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                X
            )

            loss = loss_fn(
                pred,
                Y,
            )

            loss.backward()

            optimizer.step()


        model.eval()

        with torch.no_grad():
            train_pred_std = (
                model(
                    X
                )
                .cpu()
                .numpy()
            )

            held_pred_std = (
                model(
                    torch.as_tensor(
                        held_x,
                        dtype=torch.float32,
                    )
                )
                .cpu()
                .numpy()
                .reshape(-1)
            )


        train_pred_log = (
            train_pred_std
            * y_std
            + y_mean
        ).reshape(-1)

        train_true_log = (
            train_y
        ).reshape(-1)

        train_log_mae = float(
            np.mean(
                np.abs(
                    train_pred_log
                    - train_true_log
                )
            )
        )


        held_pred_log = (
            held_pred_std
            * float(
                y_std[0]
            )
            + float(
                y_mean[0]
            )
        )

        held_pred_q = np.maximum(
            0.0,
            np.expm1(
                held_pred_log
            ),
        )

        if not np.all(
            np.isfinite(
                held_pred_q
            )
        ):
            raise RuntimeError(
                "Non-finite held Q prediction."
            )


        oracle_idx = int(
            np.argmin(
                held_true_q
            )
        )

        pred_idx = int(
            np.argmin(
                held_pred_q
            )
        )


        # No-context baseline:
        # choose beta with minimum average Q over TRAIN terrains.
        train_context_q = np.stack(
            [
                np.asarray(
                    [
                        finite(
                            q_lookup[
                                cid
                            ][
                                beta_idx
                            ][
                                "quality_Q"
                            ]
                        )
                        for beta_idx in range(
                            EXPECTED_BETA
                        )
                    ],
                    dtype=np.float64,
                )
                for cid in train_contexts
            ],
            axis=0,
        )

        baseline_idx = int(
            np.argmin(
                np.mean(
                    train_context_q,
                    axis=0,
                )
            )
        )


        oracle_q = float(
            held_true_q[
                oracle_idx
            ]
        )

        pred_selected_q = float(
            held_true_q[
                pred_idx
            ]
        )

        baseline_selected_q = float(
            held_true_q[
                baseline_idx
            ]
        )

        q_excess = (
            pred_selected_q
            - oracle_q
        )

        baseline_q_excess = (
            baseline_selected_q
            - oracle_q
        )


        true_order = np.argsort(
            held_true_q,
            kind="stable",
        )

        true_rank = int(
            np.where(
                true_order
                == pred_idx
            )[0][0]
        ) + 1


        oracle_beta = beta_order[
            oracle_idx
        ]

        pred_beta = beta_order[
            pred_idx
        ]

        baseline_beta = beta_order[
            baseline_idx
        ]


        J_oracle = physical_lookup[
            (
                held_cid,
                oracle_beta,
            )
        ]

        J_pred = physical_lookup[
            (
                held_cid,
                pred_beta,
            )
        ]

        physical_linf = float(
            np.max(
                np.abs(
                    J_pred
                    - J_oracle
                )
                / physical_iqr
            )
        )


        q_mae = float(
            np.mean(
                np.abs(
                    held_pred_q
                    - held_true_q
                )
            )
        )


        print(
            f"  trainLogMAE={train_log_mae:.4f} "
            f"heldQMAE={q_mae:.4f} "
            f"oracle={oracle_beta} "
            f"pred={pred_beta} "
            f"rank={true_rank} "
            f"Qex={q_excess:.4f} "
            f"baseEx={baseline_q_excess:.4f}"
        )


        for beta_idx in range(
            EXPECTED_BETA
        ):
            prediction_rows.append(
                {
                    "fold":
                        fold_idx,

                    "held_context":
                        held_cid,

                    "beta_index":
                        beta_idx,

                    "beta_name":
                        beta_order[
                            beta_idx
                        ],

                    "true_Q":
                        held_true_q[
                            beta_idx
                        ],

                    "pred_Q":
                        held_pred_q[
                            beta_idx
                        ],

                    "abs_Q_error":
                        abs(
                            held_pred_q[
                                beta_idx
                            ]
                            - held_true_q[
                                beta_idx
                            ]
                        ),

                    "is_oracle_best":
                        int(
                            beta_idx
                            == oracle_idx
                        ),

                    "is_pred_selected":
                        int(
                            beta_idx
                            == pred_idx
                        ),
                }
            )


        context_rows.append(
            {
                "fold":
                    fold_idx,

                "held_context":
                    held_cid,

                "train_contexts":
                    len(
                        train_contexts
                    ),

                "train_dense_rows":
                    train_x.shape[0],

                "train_log_mae":
                    train_log_mae,

                "held_q_mae":
                    q_mae,

                "oracle_beta":
                    oracle_beta,

                "pred_beta":
                    pred_beta,

                "baseline_beta":
                    baseline_beta,

                "pred_true_rank":
                    true_rank,

                "selector_exact":
                    int(
                        pred_idx
                        == oracle_idx
                    ),

                "selector_Q_excess":
                    q_excess,

                "baseline_Q_excess":
                    baseline_q_excess,

                "selector_physical_linf_iqr":
                    physical_linf,
            }
        )


    # ========================================================
    # Aggregate
    # ========================================================

    exact = np.asarray(
        [
            row[
                "selector_exact"
            ]
            for row in context_rows
        ],
        dtype=np.float64,
    )

    ranks = np.asarray(
        [
            row[
                "pred_true_rank"
            ]
            for row in context_rows
        ],
        dtype=np.float64,
    )

    q_excess = np.asarray(
        [
            row[
                "selector_Q_excess"
            ]
            for row in context_rows
        ],
        dtype=np.float64,
    )

    baseline_excess = np.asarray(
        [
            row[
                "baseline_Q_excess"
            ]
            for row in context_rows
        ],
        dtype=np.float64,
    )

    held_q_mae = np.asarray(
        [
            row[
                "held_q_mae"
            ]
            for row in context_rows
        ],
        dtype=np.float64,
    )

    physical = np.asarray(
        [
            row[
                "selector_physical_linf_iqr"
            ]
            for row in context_rows
        ],
        dtype=np.float64,
    )

    train_log = np.asarray(
        [
            row[
                "train_log_mae"
            ]
            for row in context_rows
        ],
        dtype=np.float64,
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    with OUT_PRED.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                prediction_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            prediction_rows
        )

    with OUT_CONTEXT.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                context_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            context_rows
        )


    aggregate = {
        "train_log_mae_mean":
            float(
                np.mean(
                    train_log
                )
            ),

        "held_Q_mae_mean":
            float(
                np.mean(
                    held_q_mae
                )
            ),

        "selector_exact_accuracy":
            float(
                np.mean(
                    exact
                )
            ),

        "selector_true_rank_mean":
            float(
                np.mean(
                    ranks
                )
            ),

        "selector_true_rank_median":
            float(
                np.median(
                    ranks
                )
            ),

        "selector_Q_excess_mean":
            float(
                np.mean(
                    q_excess
                )
            ),

        "selector_Q_excess_median":
            float(
                np.median(
                    q_excess
                )
            ),

        "selector_Q_excess_p95":
            float(
                np.quantile(
                    q_excess,
                    0.95,
                )
            ),

        "no_context_baseline_Q_excess_mean":
            float(
                np.mean(
                    baseline_excess
                )
            ),

        "selector_beats_no_context_fraction":
            float(
                np.mean(
                    q_excess
                    < baseline_excess
                    - EPS
                )
            ),

        "selector_physical_linf_mean_iqr":
            float(
                np.mean(
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


    output_manifest = {
        "schema":
            "icra27_os_t6p2_raw_patch_balanced_q_surface_loco_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "evaluation":
            (
                "LOCO over the same 18 original rough "
                "TRAIN terrains only"
            ),

        "train_per_fold":
            (
                "17 remaining original rough + "
                "15 frozen extension rough = 32"
            ),

        "dense_supervision_per_fold":
            672,

        "terrain_input":
            (
                "friction_mu + 48D raw oracle local "
                "relative height patch"
            ),

        "candidate_input":
            (
                "3D beta simplex lambda"
            ),

        "model":
            {
                "type":
                    "shared candidate Q regressor",

                "hidden":
                    [
                        64,
                        64,
                    ],

                "activation":
                    "SiLU",

                "epochs":
                    EPOCHS,

                "lr":
                    LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "target":
                    "standardized log1p(Q)",
            },

        "runtime_interface":
            (
                "evaluate all 21 beta candidates -> "
                "Q-vector -> argmin beta"
            ),

        "direct_beta_regression":
            False,

        "user_preference_or_irl":
            False,

        "aggregate":
            aggregate,
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            output_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()
    print("=" * 118)
    print(
        "ICRA27 OS-T6.2 RAW-PATCH "
        "BALANCED Q-SURFACE LOCO"
    )
    print("=" * 118)

    for key, value in aggregate.items():
        print(
            f"  {key:<42}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.2 raw-patch balanced "
        "Q-surface selector: COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
