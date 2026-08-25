from __future__ import annotations

import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]

T69C_PATH = (
    ROOT
    / "scripts/icra27"
    / "train_os_t6p9c_augmented_context_regret_surface_loco_v0.py"
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

SHARED_SUMMARY_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p10c_eta_specific_baseline_audit_v0"
    / "eta_specific_summary.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p11a_factorized_regret_surface_loco_v0"
)

OUT_FOLDS = (
    OUT_DIR
    / "factorized_regret_fold_results.csv"
)

OUT_PRED = (
    OUT_DIR
    / "factorized_regret_candidate_predictions.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "factorized_regret_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "factorized_regret_manifest.json"
)


REPRESENTATIONS = (
    "semantic_only",
    "semantic_plus_probe",
    "semantic_plus_prop_probe",
)

EXPECTED_DIMS = {
    "semantic_only": 74,
    "semantic_plus_probe": 81,
    "semantic_plus_prop_probe": 87,
}

# Parameter-matched against the shared 64x64, 3-output MLP.
#
# Shared params:
#   (I+1)*64 + (64+1)*64 + (64+1)*3
#
# Three independent scalar 28x28 MLPs:
#   3 * [(I+1)*28 + (28+1)*28 + (28+1)]
#
# Across I = context_dim + 3 beta dimensions,
# this gives approximately 97-100% of shared parameter count.
HIDDEN = (
    28,
    28,
)

SHARED_HIDDEN = (
    64,
    64,
)

ENSEMBLE_SEEDS = (
    27027,
    27028,
    27029,
)

EPOCHS = 1200
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4

EXPECTED_BETA = 21

TCHEBY_RHO = 0.01
EPS = 1.0e-12

ETAS = {
    "balanced":
        np.asarray(
            [1/3, 1/3, 1/3],
            dtype=np.float64,
        ),

    "motion_biased":
        np.asarray(
            [0.70, 0.15, 0.15],
            dtype=np.float64,
        ),

    "stability_biased":
        np.asarray(
            [0.15, 0.70, 0.15],
            dtype=np.float64,
        ),

    "energy_biased":
        np.asarray(
            [0.15, 0.15, 0.70],
            dtype=np.float64,
        ),
}


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

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True
    )


class ScalarRegretMLP(nn.Module):
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
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


def count_shared_params(input_dim):
    return (
        (input_dim + 1)
        * SHARED_HIDDEN[0]
        +
        (SHARED_HIDDEN[0] + 1)
        * SHARED_HIDDEN[1]
        +
        (SHARED_HIDDEN[1] + 1)
        * 3
    )


def count_factorized_params(input_dim):
    one = (
        (input_dim + 1)
        * HIDDEN[0]
        +
        (HIDDEN[0] + 1)
        * HIDDEN[1]
        +
        (HIDDEN[1] + 1)
    )

    return 3 * one


def q_from_regret(
    regret,
    eta,
):
    weighted = (
        regret
        * eta[
            None,
            :
        ]
    )

    return (
        np.max(
            weighted,
            axis=1,
        )
        +
        TCHEBY_RHO
        * np.sum(
            weighted,
            axis=1,
        )
    )


def rank_of(
    index,
    values,
):
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


def train_scalar(
    X_train,
    y_train,
    X_held,
    *,
    seed,
):
    set_seed(seed)

    torch.set_num_threads(1)

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

    y_mean = float(
        np.mean(y_train)
    )

    y_std = float(
        np.std(y_train)
    )

    if y_std <= 1.0e-8:
        y_std = 1.0


    Xn = (
        X_train
        - x_mean
    ) / x_std

    Xhn = (
        X_held
        - x_mean
    ) / x_std

    yn = (
        y_train
        - y_mean
    ) / y_std


    x = torch.tensor(
        Xn,
        dtype=torch.float32,
    )

    xh = torch.tensor(
        Xhn,
        dtype=torch.float32,
    )

    y = torch.tensor(
        yn[
            :,
            None
        ],
        dtype=torch.float32,
    )


    model = ScalarRegretMLP(
        X_train.shape[1]
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

        pred = model(x)

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
            model(x)
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )

        pred_held_n = (
            model(xh)
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )


    pred_train_log = (
        pred_train_n
        * y_std
        + y_mean
    )

    pred_held_log = (
        pred_held_n
        * y_std
        + y_mean
    )


    return {
        "pred_train_log":
            pred_train_log,

        "pred_held_log":
            pred_held_log,

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
        T69C_PATH,
        CONTEXT_CSV,
        CONTEXT_MANIFEST,
        SHARED_SUMMARY_CSV,
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


    t69c = load_module(
        T69C_PATH,
        "os_t6p11a_t69c",
    )

    t66 = t69c.load_module(
        t69c.T66_PATH,
        "os_t6p11a_t66",
    )


    # ========================================================
    # Frozen context representations.
    # ========================================================

    context_rows = read_csv(
        CONTEXT_CSV
    )

    if len(context_rows) != 33:
        raise RuntimeError(
            "Expected 33 context rows."
        )


    context_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in context_rows
    }


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


    if len(original_contexts) != 18:
        raise RuntimeError(
            "Expected 18 original contexts."
        )

    if len(extension_contexts) != 15:
        raise RuntimeError(
            "Expected 15 extension contexts."
        )


    all_contexts = (
        original_contexts
        + extension_contexts
    )


    semantic_cols = [
        f"semantic_{i:03d}"
        for i in range(74)
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


    context_vectors = {}

    for rep in REPRESENTATIONS:

        cols = representation_cols[
            rep
        ]

        if len(cols) != EXPECTED_DIMS[
            rep
        ]:
            raise RuntimeError(
                f"{rep}: dimension mismatch."
            )


        context_vectors[
            rep
        ] = {}


        for cid in all_contexts:

            row = context_lookup[
                cid
            ]

            vector = np.asarray(
                [
                    float(
                        row[col]
                    )
                    for col in cols
                ],
                dtype=np.float64,
            )

            if vector.shape != (
                EXPECTED_DIMS[
                    rep
                ],
            ):
                raise RuntimeError(
                    f"{rep}/{cid}: "
                    "shape mismatch."
                )


            context_vectors[
                rep
            ][
                cid
            ] = vector


    # ========================================================
    # Frozen physical atlas -> physical regret surfaces.
    # ========================================================

    physical_rows = t66.read_csv(
        t66.PHYSICAL_CSV
    )

    grouped = {}

    for row in physical_rows:

        cid = row[
            "context_id"
        ]

        if cid in set(
            all_contexts
        ):
            grouped.setdefault(
                cid,
                [],
            ).append(row)


    regret_lookup = {}
    beta_names = None
    beta_vectors = None


    for cid in all_contexts:

        rows = sorted(
            grouped[
                cid
            ],
            key=lambda row:
                row[
                    "beta_name"
                ],
        )

        if len(rows) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 beta rows."
            )


        names = [
            row[
                "beta_name"
            ]
            for row in rows
        ]


        if beta_names is None:

            beta_names = names

            beta_vectors = np.stack(
                [
                    t69c.beta_vector(
                        name
                    )
                    for name in names
                ],
                axis=0,
            )

        elif names != beta_names:

            raise RuntimeError(
                f"{cid}: beta order mismatch."
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
                for row in rows
            ],
            dtype=np.float64,
        )


        _, regret = (
            t69c.tcheby_quality(
                J
            )
        )

        regret_lookup[
            cid
        ] = regret


    # ========================================================
    # Frozen shared-head comparison.
    # ========================================================

    shared_summary_rows = read_csv(
        SHARED_SUMMARY_CSV
    )

    shared_summary = {
        (
            row[
                "representation"
            ],
            row[
                "eta_name"
            ],
        ):
            row
        for row in shared_summary_rows
    }


    # ========================================================
    # Parameter-matching audit.
    # ========================================================

    parameter_audit = {}

    for rep in REPRESENTATIONS:

        input_dim = (
            EXPECTED_DIMS[
                rep
            ]
            + 3
        )

        shared_params = (
            count_shared_params(
                input_dim
            )
        )

        factor_params = (
            count_factorized_params(
                input_dim
            )
        )

        ratio = (
            factor_params
            / shared_params
        )

        parameter_audit[
            rep
        ] = {
            "input_dim":
                input_dim,

            "shared_64x64_params":
                shared_params,

            "factorized_3x_28x28_params":
                factor_params,

            "factorized_over_shared":
                ratio,
        }


    # ========================================================
    # Strict outer LOCO.
    # ========================================================

    fold_rows = []
    prediction_rows = []


    for rep in REPRESENTATIONS:

        print()
        print("=" * 126)

        print(
            f"REPRESENTATION: {rep} "
            f"context_dim={EXPECTED_DIMS[rep]} "
            f"param_ratio="
            f"{parameter_audit[rep]['factorized_over_shared']:.3f}"
        )

        print("=" * 126)


        for fold_index, held in enumerate(
            original_contexts
        ):

            outer_train = (
                [
                    cid
                    for cid
                    in original_contexts
                    if cid != held
                ]
                + extension_contexts
            )

            if len(outer_train) != 32:
                raise RuntimeError(
                    "Expected 32 outer TRAIN "
                    "contexts."
                )


            X_parts = []
            R_parts = []


            for cid in outer_train:

                c = context_vectors[
                    rep
                ][
                    cid
                ]


                X_parts.append(
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


                R_parts.append(
                    regret_lookup[
                        cid
                    ]
                )


            X_train = np.concatenate(
                X_parts,
                axis=0,
            )

            R_train = np.concatenate(
                R_parts,
                axis=0,
            )


            Y_train_log = np.log1p(
                R_train
            )


            c_held = context_vectors[
                rep
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


            true_regret = regret_lookup[
                held
            ]


            pred_log_columns = []
            final_losses = []


            for objective_index in range(3):

                ensemble = []
                objective_losses = []


                for ensemble_seed in (
                    ENSEMBLE_SEEDS
                ):

                    seed = (
                        ensemble_seed
                        + 1000
                        * fold_index
                        + 100000
                        * objective_index
                    )


                    result = train_scalar(
                        X_train,
                        Y_train_log[
                            :,
                            objective_index
                        ],
                        X_held,
                        seed=seed,
                    )


                    ensemble.append(
                        result[
                            "pred_held_log"
                        ]
                    )

                    objective_losses.append(
                        result[
                            "final_standardized_mse"
                        ]
                    )


                pred_log_columns.append(
                    np.mean(
                        np.stack(
                            ensemble,
                            axis=0,
                        ),
                        axis=0,
                    )
                )

                final_losses.append(
                    float(
                        np.mean(
                            objective_losses
                        )
                    )
                )


            pred_log_regret = np.stack(
                pred_log_columns,
                axis=1,
            )


            pred_regret = np.maximum(
                np.expm1(
                    pred_log_regret
                ),
                0.0,
            )


            rho_M = t66.spearman(
                pred_regret[
                    :,
                    0
                ],
                true_regret[
                    :,
                    0
                ],
            )

            rho_S = t66.spearman(
                pred_regret[
                    :,
                    1
                ],
                true_regret[
                    :,
                    1
                ],
            )

            rho_E = t66.spearman(
                pred_regret[
                    :,
                    2
                ],
                true_regret[
                    :,
                    2
                ],
            )


            # --------------------------------------------
            # Eta-specific no-context baseline.
            # --------------------------------------------

            eta_metrics = {}


            for eta_name, eta in ETAS.items():

                true_Q = q_from_regret(
                    true_regret,
                    eta,
                )

                pred_Q = q_from_regret(
                    pred_regret,
                    eta,
                )


                oracle_idx = int(
                    np.argmin(
                        true_Q
                    )
                )

                selected_idx = int(
                    np.argmin(
                        pred_Q
                    )
                )


                mean_train_Q = np.mean(
                    np.stack(
                        [
                            q_from_regret(
                                regret_lookup[
                                    cid
                                ],
                                eta,
                            )
                            for cid
                            in outer_train
                        ],
                        axis=0,
                    ),
                    axis=0,
                )


                baseline_idx = int(
                    np.argmin(
                        mean_train_Q
                    )
                )


                oracle_Q = float(
                    true_Q[
                        oracle_idx
                    ]
                )

                learned_excess = float(
                    true_Q[
                        selected_idx
                    ]
                    - oracle_Q
                )

                baseline_excess = float(
                    true_Q[
                        baseline_idx
                    ]
                    - oracle_Q
                )


                selected_rank = rank_of(
                    selected_idx,
                    true_Q,
                )


                eta_metrics[
                    eta_name
                ] = {
                    "oracle_idx":
                        oracle_idx,

                    "selected_idx":
                        selected_idx,

                    "baseline_idx":
                        baseline_idx,

                    "learned_Q_excess":
                        learned_excess,

                    "baseline_Q_excess":
                        baseline_excess,

                    "rank":
                        selected_rank,

                    "beats_baseline":
                        int(
                            learned_excess
                            <
                            baseline_excess
                            - EPS
                        ),

                    "ties_baseline":
                        int(
                            abs(
                                learned_excess
                                - baseline_excess
                            )
                            <= EPS
                        ),

                    "worse_baseline":
                        int(
                            learned_excess
                            >
                            baseline_excess
                            + EPS
                        ),
                }


            balanced = eta_metrics[
                "balanced"
            ]


            fold_row = {
                "representation":
                    rep,

                "held_context":
                    held,

                "context_dim":
                    EXPECTED_DIMS[
                        rep
                    ],

                "model_input_dim":
                    X_train.shape[
                        1
                    ],

                "shared_param_count":
                    parameter_audit[
                        rep
                    ][
                        "shared_64x64_params"
                    ],

                "factorized_param_count":
                    parameter_audit[
                        rep
                    ][
                        "factorized_3x_28x28_params"
                    ],

                "factorized_over_shared":
                    parameter_audit[
                        rep
                    ][
                        "factorized_over_shared"
                    ],

                "rho_regret_motion":
                    float(rho_M),

                "rho_regret_stability":
                    float(rho_S),

                "rho_regret_energy":
                    float(rho_E),

                "final_mse_motion":
                    final_losses[0],

                "final_mse_stability":
                    final_losses[1],

                "final_mse_energy":
                    final_losses[2],
            }


            for eta_name in ETAS:

                metric = eta_metrics[
                    eta_name
                ]

                prefix = eta_name

                fold_row[
                    f"{prefix}_oracle_beta"
                ] = beta_names[
                    metric[
                        "oracle_idx"
                    ]
                ]

                fold_row[
                    f"{prefix}_selected_beta"
                ] = beta_names[
                    metric[
                        "selected_idx"
                    ]
                ]

                fold_row[
                    f"{prefix}_baseline_beta"
                ] = beta_names[
                    metric[
                        "baseline_idx"
                    ]
                ]

                fold_row[
                    f"{prefix}_Q_excess"
                ] = metric[
                    "learned_Q_excess"
                ]

                fold_row[
                    f"{prefix}_baseline_Q_excess"
                ] = metric[
                    "baseline_Q_excess"
                ]

                fold_row[
                    f"{prefix}_rank"
                ] = metric[
                    "rank"
                ]

                fold_row[
                    f"{prefix}_beats_baseline"
                ] = metric[
                    "beats_baseline"
                ]

                fold_row[
                    f"{prefix}_ties_baseline"
                ] = metric[
                    "ties_baseline"
                ]

                fold_row[
                    f"{prefix}_worse_baseline"
                ] = metric[
                    "worse_baseline"
                ]


            fold_rows.append(
                fold_row
            )


            for beta_index, beta_name in enumerate(
                beta_names
            ):

                prediction_rows.append(
                    {
                        "representation":
                            rep,

                        "held_context":
                            held,

                        "beta_index":
                            beta_index,

                        "beta_name":
                            beta_name,

                        "true_regret_motion":
                            float(
                                true_regret[
                                    beta_index,
                                    0
                                ]
                            ),

                        "pred_regret_motion":
                            float(
                                pred_regret[
                                    beta_index,
                                    0
                                ]
                            ),

                        "true_regret_stability":
                            float(
                                true_regret[
                                    beta_index,
                                    1
                                ]
                            ),

                        "pred_regret_stability":
                            float(
                                pred_regret[
                                    beta_index,
                                    1
                                ]
                            ),

                        "true_regret_energy":
                            float(
                                true_regret[
                                    beta_index,
                                    2
                                ]
                            ),

                        "pred_regret_energy":
                            float(
                                pred_regret[
                                    beta_index,
                                    2
                                ]
                            ),
                    }
                )


            print(
                f"{held:<16} "
                f"rhoM={rho_M:+.3f} "
                f"rhoS={rho_S:+.3f} "
                f"rhoE={rho_E:+.3f} "
                f"balancedQ="
                f"{balanced['learned_Q_excess']:.3f}"
            )


    # ========================================================
    # Aggregate + compare against frozen shared head.
    # ========================================================

    summary_rows = []


    for rep in REPRESENTATIONS:

        rep_rows = [
            row
            for row in fold_rows
            if row[
                "representation"
            ]
            == rep
        ]


        rho_M = np.asarray(
            [
                row[
                    "rho_regret_motion"
                ]
                for row in rep_rows
            ]
        )

        rho_S = np.asarray(
            [
                row[
                    "rho_regret_stability"
                ]
                for row in rep_rows
            ]
        )

        rho_E = np.asarray(
            [
                row[
                    "rho_regret_energy"
                ]
                for row in rep_rows
            ]
        )


        for eta_name in ETAS:

            learned = np.asarray(
                [
                    row[
                        f"{eta_name}_Q_excess"
                    ]
                    for row in rep_rows
                ],
                dtype=np.float64,
            )

            baseline = np.asarray(
                [
                    row[
                        f"{eta_name}_baseline_Q_excess"
                    ]
                    for row in rep_rows
                ],
                dtype=np.float64,
            )

            ranks = np.asarray(
                [
                    row[
                        f"{eta_name}_rank"
                    ]
                    for row in rep_rows
                ],
                dtype=np.float64,
            )


            beat = int(
                sum(
                    row[
                        f"{eta_name}_beats_baseline"
                    ]
                    for row in rep_rows
                )
            )

            tie = int(
                sum(
                    row[
                        f"{eta_name}_ties_baseline"
                    ]
                    for row in rep_rows
                )
            )

            worse = int(
                sum(
                    row[
                        f"{eta_name}_worse_baseline"
                    ]
                    for row in rep_rows
                )
            )


            shared = shared_summary[
                (
                    rep,
                    eta_name,
                )
            ]


            shared_qmean = float(
                shared[
                    "learned_Q_excess_mean"
                ]
            )

            shared_base = float(
                shared[
                    "baseline_Q_excess_mean"
                ]
            )


            summary_rows.append(
                {
                    "representation":
                        rep,

                    "eta_name":
                        eta_name,

                    "context_dim":
                        EXPECTED_DIMS[
                            rep
                        ],

                    "shared_param_count":
                        parameter_audit[
                            rep
                        ][
                            "shared_64x64_params"
                        ],

                    "factorized_param_count":
                        parameter_audit[
                            rep
                        ][
                            "factorized_3x_28x28_params"
                        ],

                    "factorized_over_shared":
                        parameter_audit[
                            rep
                        ][
                            "factorized_over_shared"
                        ],

                    "rho_regret_motion_mean":
                        float(
                            np.mean(
                                rho_M
                            )
                        ),

                    "rho_regret_stability_mean":
                        float(
                            np.mean(
                                rho_S
                            )
                        ),

                    "rho_regret_energy_mean":
                        float(
                            np.mean(
                                rho_E
                            )
                        ),

                    "factorized_Q_excess_mean":
                        float(
                            np.mean(
                                learned
                            )
                        ),

                    "factorized_Q_excess_median":
                        float(
                            np.median(
                                learned
                            )
                        ),

                    "factorized_Q_excess_p95":
                        float(
                            np.percentile(
                                learned,
                                95,
                            )
                        ),

                    "baseline_Q_excess_mean":
                        float(
                            np.mean(
                                baseline
                            )
                        ),

                    "shared_Q_excess_mean":
                        shared_qmean,

                    "factorized_minus_shared_Qmean":
                        float(
                            np.mean(
                                learned
                            )
                            - shared_qmean
                        ),

                    "factorized_minus_baseline_Qmean":
                        float(
                            np.mean(
                                learned
                            )
                            - np.mean(
                                baseline
                            )
                        ),

                    "shared_baseline_regression_diff":
                        float(
                            np.mean(
                                baseline
                            )
                            - shared_base
                        ),

                    "beat_tie_worse_baseline":
                        f"{beat}/{tie}/{worse}",

                    "rank_mean":
                        float(
                            np.mean(
                                ranks
                            )
                        ),

                    "top3_fraction":
                        float(
                            np.mean(
                                ranks <= 3
                            )
                        ),

                    "top5_fraction":
                        float(
                            np.mean(
                                ranks <= 5
                            )
                        ),
                }
            )


    # Baseline must reproduce T6.10c exactly.
    max_baseline_diff = max(
        abs(
            row[
                "shared_baseline_regression_diff"
            ]
        )
        for row in summary_rows
    )

    if max_baseline_diff > 1.0e-12:
        raise RuntimeError(
            "Eta-specific baseline regression "
            f"failed: max diff={max_baseline_diff}"
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
        OUT_PRED,
        prediction_rows,
    )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p11a_factorized_regret_surface_loco_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "purpose":
            (
                "Test whether objective factorization "
                "improves strict-LOCO regret-surface "
                "generalization relative to the shared "
                "three-output surrogate."
            ),

        "factorization":
            (
                "Three independent scalar MLPs predict "
                "log1p physical regret for motion, "
                "stability, and energy."
            ),

        "parameter_matching":
            {
                "shared_hidden":
                    list(
                        SHARED_HIDDEN
                    ),

                "factorized_hidden_each":
                    list(
                        HIDDEN
                    ),

                "audit":
                    parameter_audit,

                "interpretation":
                    (
                        "The total factorized parameter "
                        "count is approximately matched "
                        "to the frozen shared 64x64 "
                        "three-output model."
                    ),
            },

        "training":
            {
                "epochs":
                    EPOCHS,

                "lr":
                    LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "ensemble_seeds":
                    list(
                        ENSEMBLE_SEEDS
                    ),

                "target":
                    "log1p(context-local physical regret)",

                "outer_split":
                    (
                        "18 original rough contexts "
                        "strict LOCO; each fold trains "
                        "on remaining17 original + "
                        "15 extension TRAIN contexts."
                    ),
            },

        "eta_profiles":
            {
                key:
                    value.tolist()
                for key, value
                in ETAS.items()
            },

        "baseline_regression_max_abs_diff":
            max_baseline_diff,

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
    print("=" * 150)

    print(
        "ICRA27 OS-T6.11a PARAMETER-MATCHED "
        "OBJECTIVE-FACTORIZED REGRET SURROGATE"
    )

    print("=" * 150)

    print(
        " representation              eta              "
        "| factor/shared/base Qmean | "
        "dShared dBase | beat/tie/worse | "
        "rhoM rhoS rhoE | top5"
    )

    print("-" * 150)


    for row in summary_rows:

        print(
            f" {row['representation']:<27} "
            f"{row['eta_name']:<17} | "
            f"{row['factorized_Q_excess_mean']:.3f}/"
            f"{row['shared_Q_excess_mean']:.3f}/"
            f"{row['baseline_Q_excess_mean']:.3f} | "
            f"{row['factorized_minus_shared_Qmean']:+.3f} "
            f"{row['factorized_minus_baseline_Qmean']:+.3f} | "
            f"{row['beat_tie_worse_baseline']:>11} | "
            f"{row['rho_regret_motion_mean']:+.3f} "
            f"{row['rho_regret_stability_mean']:+.3f} "
            f"{row['rho_regret_energy_mean']:+.3f} | "
            f"{row['top5_fraction']:.3f}"
        )


    print()
    print(
        "eta-specific baseline regression max diff:",
        max_baseline_diff,
    )

    print()
    print(
        "[ICRA27] OS-T6.11a factorized "
        "regret surrogate: COMPUTE PASS"
    )

    print("=" * 150)


if __name__ == "__main__":
    main()
