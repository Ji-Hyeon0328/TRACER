from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]

T66_PATH = (
    ROOT
    / "scripts/icra27"
    / "analyze_os_t6p6a_semantic_geometry_identifiability_v0.py"
)

CONTEXT_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
    / "augmented_causal_context.csv"
)

CONTEXT_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
    / "augmented_causal_context_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p9b_augmented_context_physical_response_loco_v0"
)

OUT_FOLDS = (
    OUT_DIR
    / "physical_response_fold_results.csv"
)

OUT_PREDICTIONS = (
    OUT_DIR
    / "physical_response_candidate_predictions.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "physical_response_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "physical_response_manifest.json"
)


REPRESENTATIONS = (
    "semantic_only",
    "semantic_plus_probe",
    "semantic_plus_prop_probe",
)

ENSEMBLE_SEEDS = (
    27027,
    27028,
    27029,
)

HIDDEN = (
    64,
    64,
)

EPOCHS = 1200
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4

ETA = np.asarray(
    [
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ],
    dtype=np.float64,
)

TCHEBY_RHO = 0.01

EXPECTED_BETA = 21

BETA_RE = re.compile(
    r"^lm(\d+)_ls(\d+)_le(\d+)$"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def read_csv(path):
    with path.open(
        "r",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def beta_vector(beta_name):
    match = BETA_RE.match(
        beta_name
    )

    if match is None:
        raise RuntimeError(
            f"Unexpected beta name: "
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
        float(np.sum(lam))
        - 1.0
    ) > 1.0e-12:
        raise RuntimeError(
            f"{beta_name}: lambda simplex "
            f"sum={np.sum(lam)}"
        )

    beta = (
        0.15
        + 0.55
        * lam
    )

    return beta


def pareto_mask_lower_better(J):
    """
    Return the nondominated mask for lower-is-better
    [J_M, J_S, J_E].

    The frozen OS scalarization normalizes physical
    regret using the context-local Pareto ideal/nadir,
    not the min/max over all 21 candidates.
    """
    J = np.asarray(
        J,
        dtype=np.float64,
    )

    if (
        J.ndim != 2
        or J.shape[1] != 3
    ):
        raise RuntimeError(
            f"Unexpected J shape: {J.shape}"
        )

    n = J.shape[0]

    keep = np.ones(
        n,
        dtype=bool,
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # j dominates i iff j is no worse in
            # every objective and strictly better
            # in at least one objective.
            if (
                np.all(
                    J[j] <= J[i]
                )
                and np.any(
                    J[j] < J[i]
                )
            ):
                keep[i] = False
                break

    if not np.any(
        keep
    ):
        raise RuntimeError(
            "Pareto front is empty."
        )

    return keep


def tcheby_quality(J):
    J = np.asarray(
        J,
        dtype=np.float64,
    )

    if (
        J.ndim != 2
        or J.shape[1] != 3
    ):
        raise RuntimeError(
            f"Unexpected J shape: {J.shape}"
        )

    pareto = pareto_mask_lower_better(
        J
    )

    pareto_J = J[
        pareto
    ]

    # Frozen context-local Pareto reference.
    ideal = np.min(
        pareto_J,
        axis=0,
    )

    nadir = np.max(
        pareto_J,
        axis=0,
    )

    span = nadir - ideal

    regret = np.zeros_like(
        J
    )

    active = span > 0.0

    regret[
        :,
        active
    ] = (
        J[
            :,
            active
        ]
        - ideal[
            active
        ]
    ) / span[
        active
    ]

    # Do NOT clip regret to [0,1].
    # A dominated candidate may lie beyond the
    # Pareto nadir and therefore legitimately have
    # regret > 1 under the frozen definition.

    weighted = (
        regret
        * ETA[
            None,
            :
        ]
    )

    Q = (
        np.max(
            weighted,
            axis=1,
        )
        + TCHEBY_RHO
        * np.sum(
            weighted,
            axis=1,
        )
    )

    return (
        Q,
        regret,
    )


def rank_of(index, values):
    order = np.argsort(
        values,
        kind="stable",
    )

    return (
        int(
            np.where(
                order == index
            )[0][0]
        )
        + 1
    )


class ResponseMLP(nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                HIDDEN[0],
            ),
            nn.SiLU(),
            nn.Linear(
                HIDDEN[0],
                HIDDEN[1],
            ),
            nn.SiLU(),
            nn.Linear(
                HIDDEN[1],
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


def set_seed(seed):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.use_deterministic_algorithms(
        True
    )


def train_one(
    X_train,
    Y_train,
    X_held,
    *,
    seed,
):
    set_seed(
        seed
    )

    device = torch.device(
        "cpu"
    )

    torch.set_num_threads(
        1
    )

    x_mean = np.mean(
        X_train,
        axis=0,
    )

    x_std = np.std(
        X_train,
        axis=0,
    )

    x_std = np.where(
        x_std > 1.0e-8,
        x_std,
        1.0,
    )

    y_mean = np.mean(
        Y_train,
        axis=0,
    )

    y_std = np.std(
        Y_train,
        axis=0,
    )

    y_std = np.where(
        y_std > 1.0e-8,
        y_std,
        1.0,
    )

    Xn = (
        X_train
        - x_mean
    ) / x_std

    Yn = (
        Y_train
        - y_mean
    ) / y_std

    Xhn = (
        X_held
        - x_mean
    ) / x_std


    x = torch.tensor(
        Xn,
        dtype=torch.float32,
        device=device,
    )

    y = torch.tensor(
        Yn,
        dtype=torch.float32,
        device=device,
    )

    xh = torch.tensor(
        Xhn,
        dtype=torch.float32,
        device=device,
    )


    model = ResponseMLP(
        X_train.shape[1]
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()


    model.train()

    final_loss = None

    for _epoch in range(
        EPOCHS
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            x
        )

        loss = loss_fn(
            pred,
            y,
        )

        loss.backward()

        optimizer.step()

        final_loss = float(
            loss.detach().cpu()
        )


    model.eval()

    with torch.no_grad():
        pred_train_n = (
            model(
                x
            )
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        pred_held_n = (
            model(
                xh
            )
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


    pred_train = (
        pred_train_n
        * y_std
        + y_mean
    )

    pred_held = (
        pred_held_n
        * y_std
        + y_mean
    )


    return {
        "pred_train":
            pred_train,

        "pred_held":
            pred_held,

        "final_standardized_mse":
            final_loss,
    }


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )


    for path in (
        T66_PATH,
        CONTEXT_CSV,
        CONTEXT_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    context_manifest = json.loads(
        CONTEXT_MANIFEST.read_text()
    )

    if context_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.9a context is not "
            "FREEZE_PASS."
        )

    if bool(
        context_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.9a reports heldout use."
        )

    if abs(
        float(
            context_manifest[
                "selected_probe_time_s"
            ]
        )
        - 0.4
    ) > 1.0e-12:
        raise RuntimeError(
            "Expected frozen 0.4 s "
            "probe horizon."
        )


    context_rows = read_csv(
        CONTEXT_CSV
    )

    if len(
        context_rows
    ) != 33:
        raise RuntimeError(
            f"Expected 33 context rows, "
            f"got {len(context_rows)}"
        )


    context_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in context_rows
    }

    if len(
        context_lookup
    ) != 33:
        raise RuntimeError(
            "Duplicate context IDs."
        )


    original_contexts = [
        row[
            "context_id"
        ]
        for row in context_rows
        if row[
            "split_group"
        ]
        == "original_rough_train"
    ]

    extension_contexts = [
        row[
            "context_id"
        ]
        for row in context_rows
        if row[
            "split_group"
        ]
        == "extension_rough_train"
    ]


    if len(
        original_contexts
    ) != 18:
        raise RuntimeError(
            "Expected 18 original contexts."
        )

    if len(
        extension_contexts
    ) != 15:
        raise RuntimeError(
            "Expected 15 extension contexts."
        )


    semantic_cols = [
        f"semantic_{i:03d}"
        for i in range(
            74
        )
    ]

    prop_cols = [
        "prop_vx_mps",
        "prop_vy_mps",
        "prop_yaw_rate_rps",
        "prop_base_z_m",
        "prop_roll_rad",
        "prop_pitch_rad",
    ]

    probe_cols = [
        "probe_probe_progress_m",
        "probe_delta_vx_mps",
        "probe_delta_vy_mps",
        "probe_delta_yaw_rate_rps",
        "probe_delta_base_z_m",
        "probe_delta_roll_rad",
        "probe_delta_pitch_rad",
    ]


    representation_cols = {
        "semantic_only":
            semantic_cols,

        "semantic_plus_probe":
            (
                semantic_cols
                + probe_cols
            ),

        "semantic_plus_prop_probe":
            (
                semantic_cols
                + prop_cols
                + probe_cols
            ),
    }


    expected_dims = {
        "semantic_only":
            74,

        "semantic_plus_probe":
            81,

        "semantic_plus_prop_probe":
            87,
    }


    for name in REPRESENTATIONS:
        cols = representation_cols[
            name
        ]

        if len(
            cols
        ) != expected_dims[
            name
        ]:
            raise RuntimeError(
                f"{name}: context dimension "
                "mismatch."
            )

        for col in cols:
            if col not in context_rows[
                0
            ]:
                raise RuntimeError(
                    f"Missing context column: "
                    f"{col}"
                )


    # ========================================================
    # Frozen physical atlas and frozen Q surface.
    # ========================================================

    t66 = load_module(
        T66_PATH,
        "os_t6p9b_t66",
    )

    physical_rows = t66.read_csv(
        t66.PHYSICAL_CSV
    )

    q_rows = t66.read_csv(
        t66.Q_CSV
    )


    all_contexts = set(
        original_contexts
        + extension_contexts
    )


    physical_by_context = {}

    for row in physical_rows:
        cid = row[
            "context_id"
        ]

        if cid not in all_contexts:
            continue

        physical_by_context.setdefault(
            cid,
            [],
        ).append(
            row
        )


    q_by_context = {}

    for row in q_rows:
        cid = row[
            "context_id"
        ]

        if cid not in all_contexts:
            continue

        q_by_context.setdefault(
            cid,
            [],
        ).append(
            row
        )


    beta_names = None

    J_lookup = {}
    frozen_Q_lookup = {}


    for cid in (
        original_contexts
        + extension_contexts
    ):

        p_rows = sorted(
            physical_by_context[
                cid
            ],
            key=lambda row:
                row[
                    "beta_name"
                ],
        )

        q_context_rows = sorted(
            q_by_context[
                cid
            ],
            key=lambda row:
                row[
                    "beta_name"
                ],
        )


        if len(
            p_rows
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 "
                "physical rows."
            )

        if len(
            q_context_rows
        ) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 "
                "Q rows."
            )


        names_p = [
            row[
                "beta_name"
            ]
            for row in p_rows
        ]

        names_q = [
            row[
                "beta_name"
            ]
            for row in q_context_rows
        ]


        if names_p != names_q:
            raise RuntimeError(
                f"{cid}: physical/Q "
                "beta order mismatch."
            )


        if beta_names is None:
            beta_names = names_p

        elif names_p != beta_names:
            raise RuntimeError(
                f"{cid}: global beta "
                "order mismatch."
            )


        J = np.asarray(
            [
                [
                    float(
                        row[
                            "J_motion_s_per_m"
                        ]
                    ),

                    float(
                        row[
                            "J_stability"
                        ]
                    ),

                    float(
                        row[
                            "J_energy_j_per_m"
                        ]
                    ),
                ]
                for row in p_rows
            ],
            dtype=np.float64,
        )


        if not np.all(
            np.isfinite(
                J
            )
        ):
            raise RuntimeError(
                f"{cid}: non-finite J."
            )


        frozen_Q = np.asarray(
            [
                float(
                    row[
                        "quality_Q"
                    ]
                )
                for row in q_context_rows
            ],
            dtype=np.float64,
        )


        computed_Q, _ = (
            tcheby_quality(
                J
            )
        )


        max_diff = float(
            np.max(
                np.abs(
                    computed_Q
                    - frozen_Q
                )
            )
        )


        if max_diff > 1.0e-10:
            raise RuntimeError(
                f"{cid}: frozen Q "
                "regression failed; "
                f"max diff={max_diff}"
            )


        J_lookup[
            cid
        ] = J

        frozen_Q_lookup[
            cid
        ] = frozen_Q


    beta_vectors = np.stack(
        [
            beta_vector(
                name
            )
            for name in beta_names
        ],
        axis=0,
    )


    if beta_vectors.shape != (
        EXPECTED_BETA,
        3,
    ):
        raise RuntimeError(
            "Beta matrix shape mismatch."
        )


    # ========================================================
    # Build context representations.
    # ========================================================

    context_vectors = {}

    for representation in REPRESENTATIONS:

        context_vectors[
            representation
        ] = {}


        for cid in (
            original_contexts
            + extension_contexts
        ):

            row = context_lookup[
                cid
            ]

            vector = np.asarray(
                [
                    float(
                        row[
                            col
                        ]
                    )
                    for col in (
                        representation_cols[
                            representation
                        ]
                    )
                ],
                dtype=np.float64,
            )


            if vector.shape != (
                expected_dims[
                    representation
                ],
            ):
                raise RuntimeError(
                    f"{representation}/"
                    f"{cid}: vector shape "
                    f"{vector.shape}"
                )


            context_vectors[
                representation
            ][
                cid
            ] = vector


    # ========================================================
    # Strict LOCO.
    # ========================================================

    fold_rows = []
    prediction_rows = []


    for representation in REPRESENTATIONS:

        print()
        print(
            "=" * 122
        )

        print(
            f"REPRESENTATION: "
            f"{representation} "
            f"context_dim="
            f"{expected_dims[representation]}"
        )

        print(
            "=" * 122
        )


        for fold_index, held in enumerate(
            original_contexts
        ):

            train_contexts = (
                [
                    cid
                    for cid in original_contexts
                    if cid != held
                ]
                + extension_contexts
            )


            if len(
                train_contexts
            ) != 32:
                raise RuntimeError(
                    "Expected 32 outer TRAIN "
                    "contexts."
                )


            X_train_parts = []
            Y_train_parts = []


            for cid in train_contexts:

                c = context_vectors[
                    representation
                ][
                    cid
                ]


                X_train_parts.append(
                    np.concatenate(
                        (
                            np.repeat(
                                c[
                                    None,
                                    :
                                ],
                                EXPECTED_BETA,
                                axis=0,
                            ),

                            beta_vectors,
                        ),
                        axis=1,
                    )
                )


                Y_train_parts.append(
                    J_lookup[
                        cid
                    ]
                )


            X_train = np.concatenate(
                X_train_parts,
                axis=0,
            )

            Y_train = np.concatenate(
                Y_train_parts,
                axis=0,
            )


            c_held = context_vectors[
                representation
            ][
                held
            ]


            X_held = np.concatenate(
                (
                    np.repeat(
                        c_held[
                            None,
                            :
                        ],
                        EXPECTED_BETA,
                        axis=0,
                    ),

                    beta_vectors,
                ),
                axis=1,
            )


            ensemble_pred = []
            ensemble_train_pred = []
            final_losses = []


            for ensemble_seed in (
                ENSEMBLE_SEEDS
            ):

                result = train_one(
                    X_train,
                    Y_train,
                    X_held,
                    seed=(
                        ensemble_seed
                        + 1000
                        * fold_index
                    ),
                )

                ensemble_pred.append(
                    result[
                        "pred_held"
                    ]
                )

                ensemble_train_pred.append(
                    result[
                        "pred_train"
                    ]
                )

                final_losses.append(
                    result[
                        "final_standardized_mse"
                    ]
                )


            pred_J = np.mean(
                np.stack(
                    ensemble_pred,
                    axis=0,
                ),
                axis=0,
            )

            pred_train_J = np.mean(
                np.stack(
                    ensemble_train_pred,
                    axis=0,
                ),
                axis=0,
            )


            true_J = J_lookup[
                held
            ]


            train_mae = np.mean(
                np.abs(
                    pred_train_J
                    - Y_train
                ),
                axis=0,
            )

            held_mae = np.mean(
                np.abs(
                    pred_J
                    - true_J
                ),
                axis=0,
            )


            rho_M = t66.spearman(
                pred_J[
                    :,
                    0
                ],
                true_J[
                    :,
                    0
                ],
            )

            rho_S = t66.spearman(
                pred_J[
                    :,
                    1
                ],
                true_J[
                    :,
                    1
                ],
            )

            rho_E = t66.spearman(
                pred_J[
                    :,
                    2
                ],
                true_J[
                    :,
                    2
                ],
            )


            pred_Q, _ = tcheby_quality(
                pred_J
            )

            true_Q = frozen_Q_lookup[
                held
            ]


            rho_Q = t66.spearman(
                pred_Q,
                true_Q,
            )


            selected_index = int(
                np.argmin(
                    pred_Q
                )
            )

            oracle_index = int(
                np.argmin(
                    true_Q
                )
            )


            true_rank = rank_of(
                selected_index,
                true_Q,
            )


            Q_excess = float(
                true_Q[
                    selected_index
                ]
                - true_Q[
                    oracle_index
                ]
            )


            # No-context baseline:
            # mean frozen true Q over outer TRAIN.
            mean_train_Q = np.mean(
                np.stack(
                    [
                        frozen_Q_lookup[
                            cid
                        ]
                        for cid in train_contexts
                    ],
                    axis=0,
                ),
                axis=0,
            )


            baseline_index = int(
                np.argmin(
                    mean_train_Q
                )
            )


            baseline_Q_excess = float(
                true_Q[
                    baseline_index
                ]
                - true_Q[
                    oracle_index
                ]
            )


            beats_baseline = int(
                Q_excess
                < baseline_Q_excess
                - 1.0e-12
            )


            fold_row = {
                "representation":
                    representation,

                "held_context":
                    held,

                "context_dim":
                    expected_dims[
                        representation
                    ],

                "model_input_dim":
                    X_train.shape[
                        1
                    ],

                "train_samples":
                    X_train.shape[
                        0
                    ],

                "ensemble_size":
                    len(
                        ENSEMBLE_SEEDS
                    ),

                "final_standardized_mse_mean":
                    float(
                        np.mean(
                            final_losses
                        )
                    ),

                "train_mae_J_motion":
                    float(
                        train_mae[
                            0
                        ]
                    ),

                "train_mae_J_stability":
                    float(
                        train_mae[
                            1
                        ]
                    ),

                "train_mae_J_energy":
                    float(
                        train_mae[
                            2
                        ]
                    ),

                "held_mae_J_motion":
                    float(
                        held_mae[
                            0
                        ]
                    ),

                "held_mae_J_stability":
                    float(
                        held_mae[
                            1
                        ]
                    ),

                "held_mae_J_energy":
                    float(
                        held_mae[
                            2
                        ]
                    ),

                "rho_J_motion":
                    float(
                        rho_M
                    ),

                "rho_J_stability":
                    float(
                        rho_S
                    ),

                "rho_J_energy":
                    float(
                        rho_E
                    ),

                "rho_Q":
                    float(
                        rho_Q
                    ),

                "oracle_beta":
                    beta_names[
                        oracle_index
                    ],

                "selected_beta":
                    beta_names[
                        selected_index
                    ],

                "baseline_beta":
                    beta_names[
                        baseline_index
                    ],

                "true_rank":
                    int(
                        true_rank
                    ),

                "exact":
                    int(
                        true_rank
                        == 1
                    ),

                "top3":
                    int(
                        true_rank
                        <= 3
                    ),

                "top5":
                    int(
                        true_rank
                        <= 5
                    ),

                "Q_excess":
                    Q_excess,

                "baseline_Q_excess":
                    baseline_Q_excess,

                "beats_baseline":
                    beats_baseline,
            }


            fold_rows.append(
                fold_row
            )


            for beta_index, beta_name in enumerate(
                beta_names
            ):

                prediction_rows.append(
                    {
                        "representation":
                            representation,

                        "held_context":
                            held,

                        "beta_index":
                            beta_index,

                        "beta_name":
                            beta_name,

                        "beta_motion":
                            float(
                                beta_vectors[
                                    beta_index,
                                    0
                                ]
                            ),

                        "beta_stability":
                            float(
                                beta_vectors[
                                    beta_index,
                                    1
                                ]
                            ),

                        "beta_energy":
                            float(
                                beta_vectors[
                                    beta_index,
                                    2
                                ]
                            ),

                        "true_J_motion":
                            float(
                                true_J[
                                    beta_index,
                                    0
                                ]
                            ),

                        "pred_J_motion":
                            float(
                                pred_J[
                                    beta_index,
                                    0
                                ]
                            ),

                        "true_J_stability":
                            float(
                                true_J[
                                    beta_index,
                                    1
                                ]
                            ),

                        "pred_J_stability":
                            float(
                                pred_J[
                                    beta_index,
                                    1
                                ]
                            ),

                        "true_J_energy":
                            float(
                                true_J[
                                    beta_index,
                                    2
                                ]
                            ),

                        "pred_J_energy":
                            float(
                                pred_J[
                                    beta_index,
                                    2
                                ]
                            ),

                        "true_Q_balanced":
                            float(
                                true_Q[
                                    beta_index
                                ]
                            ),

                        "pred_Q_balanced":
                            float(
                                pred_Q[
                                    beta_index
                                ]
                            ),

                        "is_oracle":
                            int(
                                beta_index
                                == oracle_index
                            ),

                        "is_selected":
                            int(
                                beta_index
                                == selected_index
                            ),

                        "is_baseline":
                            int(
                                beta_index
                                == baseline_index
                            ),
                    }
                )


            print(
                f"{held:<16} "
                f"rhoM={rho_M:+.3f} "
                f"rhoS={rho_S:+.3f} "
                f"rhoE={rho_E:+.3f} "
                f"rhoQ={rho_Q:+.3f} "
                f"rank={true_rank:>2} "
                f"Qex={Q_excess:.4f} "
                f"base={baseline_Q_excess:.4f}"
            )


    # ========================================================
    # Aggregate summaries.
    # ========================================================

    summary_rows = []


    for representation in REPRESENTATIONS:

        rows = [
            row
            for row in fold_rows
            if row[
                "representation"
            ]
            == representation
        ]


        def values(key):
            return np.asarray(
                [
                    row[
                        key
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )


        q_excess = values(
            "Q_excess"
        )

        summary = {
            "representation":
                representation,

            "context_dim":
                expected_dims[
                    representation
                ],

            "held_mae_J_motion_mean":
                float(
                    np.mean(
                        values(
                            "held_mae_J_motion"
                        )
                    )
                ),

            "held_mae_J_stability_mean":
                float(
                    np.mean(
                        values(
                            "held_mae_J_stability"
                        )
                    )
                ),

            "held_mae_J_energy_mean":
                float(
                    np.mean(
                        values(
                            "held_mae_J_energy"
                        )
                    )
                ),

            "rho_J_motion_mean":
                float(
                    np.mean(
                        values(
                            "rho_J_motion"
                        )
                    )
                ),

            "rho_J_stability_mean":
                float(
                    np.mean(
                        values(
                            "rho_J_stability"
                        )
                    )
                ),

            "rho_J_energy_mean":
                float(
                    np.mean(
                        values(
                            "rho_J_energy"
                        )
                    )
                ),

            "rho_Q_mean":
                float(
                    np.mean(
                        values(
                            "rho_Q"
                        )
                    )
                ),

            "Q_excess_mean":
                float(
                    np.mean(
                        q_excess
                    )
                ),

            "Q_excess_median":
                float(
                    np.median(
                        q_excess
                    )
                ),

            "Q_excess_p95":
                float(
                    np.percentile(
                        q_excess,
                        95,
                    )
                ),

            "Q_excess_max":
                float(
                    np.max(
                        q_excess
                    )
                ),

            "baseline_Q_excess_mean":
                float(
                    np.mean(
                        values(
                            "baseline_Q_excess"
                        )
                    )
                ),

            "true_rank_mean":
                float(
                    np.mean(
                        values(
                            "true_rank"
                        )
                    )
                ),

            "true_rank_median":
                float(
                    np.median(
                        values(
                            "true_rank"
                        )
                    )
                ),

            "exact_fraction":
                float(
                    np.mean(
                        values(
                            "exact"
                        )
                    )
                ),

            "top3_fraction":
                float(
                    np.mean(
                        values(
                            "top3"
                        )
                    )
                ),

            "top5_fraction":
                float(
                    np.mean(
                        values(
                            "top5"
                        )
                    )
                ),

            "beats_baseline_fraction":
                float(
                    np.mean(
                        values(
                            "beats_baseline"
                        )
                    )
                ),
        }


        summary_rows.append(
            summary
        )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_FOLDS,
        fold_rows,
    )

    write_csv(
        OUT_PREDICTIONS,
        prediction_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p9b_augmented_context_physical_response_loco_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "objective":
            (
                "Learn a candidate-conditioned "
                "physical response surrogate "
                "f(context,beta)->"
                "[J_motion,J_stability,J_energy]."
            ),

        "representations":
            list(
                REPRESENTATIONS
            ),

        "probe_horizon_s":
            0.4,

        "beta_candidates":
            EXPECTED_BETA,

        "beta_conditioning":
            (
                "Physical reward weights reconstructed "
                "from the frozen 21-point lambda lattice."
            ),

        "model":
            {
                "type":
                    "MLP",

                "hidden":
                    list(
                        HIDDEN
                    ),

                "activation":
                    "SiLU",

                "outputs":
                    [
                        "J_motion",
                        "J_stability",
                        "J_energy",
                    ],

                "epochs":
                    EPOCHS,

                "optimizer":
                    "AdamW",

                "lr":
                    LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "ensemble_seeds":
                    list(
                        ENSEMBLE_SEEDS
                    ),

                "device":
                    "CPU",

                "target_standardization":
                    (
                        "outer-TRAIN-only per-objective"
                    ),

                "input_standardization":
                    (
                        "outer-TRAIN-only per-dimension"
                    ),
            },

        "evaluation":
            (
                "Strict outer LOCO over 18 original "
                "rough TRAIN contexts. Each fold trains "
                "on remaining 17 original + 15 extension "
                "TRAIN contexts. No validation/test/hard "
                "context is used."
            ),

        "balanced_quality":
            {
                "eta":
                    ETA.tolist(),

                "rho":
                    TCHEBY_RHO,

                "selection":
                    (
                        "argmin predicted augmented "
                        "Tchebycheff quality over the "
                        "21 beta candidates."
                    ),

                "final_decision_metric":
                    (
                        "true held-context Q excess of "
                        "the beta selected from predicted "
                        "physical response."
                    ),
            },

        "important_semantics":
            (
                "The response surrogate is eta-agnostic. "
                "Once J(context,beta) is predicted, "
                "different physical preference vectors "
                "eta can be applied without retraining "
                "the surrogate."
            ),

        "summary":
            summary_rows,
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
    print(
        "=" * 138
    )

    print(
        "ICRA27 OS-T6.9b AUGMENTED-CONTEXT "
        "PHYSICAL RESPONSE SURROGATE"
    )

    print(
        "=" * 138
    )


    print(
        " representation                | "
        "rhoM   rhoS   rhoE   rhoQ | "
        "Qmean   Qmed   Qp95 | "
        "rank  top3  top5  beatB"
    )

    print(
        "-" * 138
    )


    for row in summary_rows:

        print(
            f" {row['representation']:<29} | "
            f"{row['rho_J_motion_mean']:>5.3f} "
            f"{row['rho_J_stability_mean']:>5.3f} "
            f"{row['rho_J_energy_mean']:>5.3f} "
            f"{row['rho_Q_mean']:>5.3f} | "
            f"{row['Q_excess_mean']:>6.3f} "
            f"{row['Q_excess_median']:>6.3f} "
            f"{row['Q_excess_p95']:>6.3f} | "
            f"{row['true_rank_mean']:>5.2f} "
            f"{row['top3_fraction']:>5.3f} "
            f"{row['top5_fraction']:>5.3f} "
            f"{row['beats_baseline_fraction']:>5.3f}"
        )


    print()
    print(
        "[ICRA27] OS-T6.9b physical "
        "response surrogate: COMPUTE PASS"
    )

    print(
        "=" * 138
    )


if __name__ == "__main__":
    main()
