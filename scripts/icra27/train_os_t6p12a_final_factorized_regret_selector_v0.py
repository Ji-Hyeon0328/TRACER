from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

T611A_PATH = (
    ROOT
    / "scripts/icra27"
    / "train_os_t6p11a_factorized_regret_surface_loco_v0.py"
)

FREEZE_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p11b_selector_candidate_freeze_v0"
    / "selector_candidate_freeze_manifest.json"
)

CONTEXT_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p9a_augmented_causal_context_v0"
    / "augmented_causal_context.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12a_final_factorized_regret_selector_v0"
)

OUT_MANIFEST = (
    OUT_DIR
    / "final_selector_manifest.json"
)

OUT_FIT = (
    OUT_DIR
    / "train_fit_summary.csv"
)

OUT_BETA = (
    OUT_DIR
    / "beta_candidates.csv"
)


OBJECTIVE_NAMES = (
    "motion",
    "stability",
    "energy",
)

EXPECTED_CONTEXT_DIM = 87
EXPECTED_BETA = 21
EXPECTED_CONTEXTS = 33


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

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def q_from_regret(
    regret,
    eta,
    rho,
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
        rho
        * np.sum(
            weighted,
            axis=1,
        )
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        T611A_PATH,
        FREEZE_MANIFEST,
        CONTEXT_CSV,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Load frozen architecture.
    # ========================================================

    freeze = json.loads(
        FREEZE_MANIFEST.read_text()
    )

    if freeze.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.11b candidate is not "
            "FREEZE_PASS."
        )

    if bool(
        freeze.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "T6.11b reports heldout use."
        )

    if freeze[
        "selected_representation"
    ] != "semantic_plus_prop_probe":
        raise RuntimeError(
            "Unexpected frozen representation."
        )

    if int(
        freeze[
            "context_dim"
        ]
    ) != EXPECTED_CONTEXT_DIM:
        raise RuntimeError(
            "Unexpected frozen context dimension."
        )

    if freeze[
        "selected_surrogate"
    ] != (
        "three independent scalar regret MLPs"
    ):
        raise RuntimeError(
            "Unexpected frozen surrogate."
        )

    if list(
        freeze[
            "hidden_each"
        ]
    ) != [
        28,
        28,
    ]:
        raise RuntimeError(
            "Unexpected frozen hidden width."
        )


    t611 = load_module(
        T611A_PATH,
        "os_t6p12a_t611",
    )

    t69c = t611.load_module(
        t611.T69C_PATH,
        "os_t6p12a_t69c",
    )

    t66 = t69c.load_module(
        t69c.T66_PATH,
        "os_t6p12a_t66",
    )


    # Guard against accidental architecture drift.
    if tuple(
        t611.HIDDEN
    ) != (
        28,
        28,
    ):
        raise RuntimeError(
            "T6.11a hidden architecture drift."
        )

    ensemble_seeds = tuple(
        int(x)
        for x in t611.ENSEMBLE_SEEDS
    )

    if ensemble_seeds != (
        27027,
        27028,
        27029,
    ):
        raise RuntimeError(
            "Unexpected ensemble seeds."
        )


    # ========================================================
    # Frozen 87D causal TRAIN contexts.
    # ========================================================

    context_rows = read_csv(
        CONTEXT_CSV
    )

    if len(
        context_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTEXTS} "
            f"contexts, got "
            f"{len(context_rows)}"
        )


    original = [
        row[
            "context_id"
        ]
        for row in context_rows
        if row[
            "split_group"
        ]
        == "original_rough_train"
    ]

    extension = [
        row[
            "context_id"
        ]
        for row in context_rows
        if row[
            "split_group"
        ]
        == "extension_rough_train"
    ]


    if len(original) != 18:
        raise RuntimeError(
            "Expected 18 original TRAIN contexts."
        )

    if len(extension) != 15:
        raise RuntimeError(
            "Expected 15 extension TRAIN contexts."
        )


    context_ids = (
        original
        + extension
    )

    if len(
        set(
            context_ids
        )
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "TRAIN context IDs are not unique."
        )


    context_lookup = {
        row[
            "context_id"
        ]:
            row
        for row in context_rows
    }


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

    context_cols = (
        semantic_cols
        + prop_cols
        + probe_cols
    )


    if len(
        context_cols
    ) != EXPECTED_CONTEXT_DIM:
        raise RuntimeError(
            "87D context column count mismatch."
        )


    context_vectors = {}

    for cid in context_ids:

        row = context_lookup[
            cid
        ]

        vector = np.asarray(
            [
                float(
                    row[col]
                )
                for col in context_cols
            ],
            dtype=np.float64,
        )

        if vector.shape != (
            EXPECTED_CONTEXT_DIM,
        ):
            raise RuntimeError(
                f"{cid}: context shape mismatch."
            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):
            raise RuntimeError(
                f"{cid}: non-finite context."
            )

        context_vectors[
            cid
        ] = vector


    # ========================================================
    # Frozen physical regret labels.
    # ========================================================

    physical_rows = t66.read_csv(
        t66.PHYSICAL_CSV
    )

    grouped = {}

    context_set = set(
        context_ids
    )

    for row in physical_rows:

        cid = row[
            "context_id"
        ]

        if cid in context_set:
            grouped.setdefault(
                cid,
                [],
            ).append(row)


    beta_names = None
    beta_vectors = None
    regret_lookup = {}


    for cid in context_ids:

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
                f"{cid}: expected "
                f"{EXPECTED_BETA} beta rows."
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
                f"{cid}: beta ordering mismatch."
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
    # Final all-TRAIN design matrix.
    # ========================================================

    X_parts = []
    R_parts = []
    sample_context = []
    sample_beta_index = []


    for cid in context_ids:

        c = context_vectors[
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

        sample_context.extend(
            [
                cid
            ]
            * EXPECTED_BETA
        )

        sample_beta_index.extend(
            range(
                EXPECTED_BETA
            )
        )


    X = np.concatenate(
        X_parts,
        axis=0,
    )

    R = np.concatenate(
        R_parts,
        axis=0,
    )

    Y_log = np.log1p(
        R
    )


    expected_samples = (
        EXPECTED_CONTEXTS
        * EXPECTED_BETA
    )

    if X.shape != (
        expected_samples,
        EXPECTED_CONTEXT_DIM + 3,
    ):
        raise RuntimeError(
            f"Unexpected X shape: {X.shape}"
        )

    if R.shape != (
        expected_samples,
        3,
    ):
        raise RuntimeError(
            f"Unexpected regret shape: {R.shape}"
        )


    # Input normalization is identical for all objectives.
    x_mean = np.mean(
        X,
        axis=0,
    )

    x_std = np.std(
        X,
        axis=0,
    )

    x_std = np.where(
        x_std > 1.0e-8,
        x_std,
        1.0,
    )

    Xn = (
        X
        - x_mean
    ) / x_std

    x_tensor = torch.tensor(
        Xn,
        dtype=torch.float32,
    )


    # ========================================================
    # Train frozen objective-factorized ensemble.
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoint_records = []

    ensemble_log_predictions = []


    for objective_index, objective_name in enumerate(
        OBJECTIVE_NAMES
    ):

        y_log = Y_log[
            :,
            objective_index
        ]

        y_mean = float(
            np.mean(
                y_log
            )
        )

        y_std = float(
            np.std(
                y_log
            )
        )

        if y_std <= 1.0e-8:
            y_std = 1.0


        yn = (
            y_log
            - y_mean
        ) / y_std

        y_tensor = torch.tensor(
            yn[
                :,
                None
            ],
            dtype=torch.float32,
        )


        objective_predictions = []


        for base_seed in ensemble_seeds:

            seed = (
                base_seed
                + 100000
                * objective_index
            )

            set_seed(
                seed
            )

            torch.set_num_threads(
                1
            )


            model = t611.ScalarRegretMLP(
                X.shape[
                    1
                ]
            )


            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=t611.LR,
                weight_decay=t611.WEIGHT_DECAY,
            )

            loss_fn = torch.nn.MSELoss()


            model.train()

            final_loss = None


            for _epoch in range(
                t611.EPOCHS
            ):

                optimizer.zero_grad(
                    set_to_none=True
                )

                pred = model(
                    x_tensor
                )

                loss = loss_fn(
                    pred,
                    y_tensor,
                )

                loss.backward()
                optimizer.step()

                final_loss = float(
                    loss.detach().cpu()
                )


            model.eval()

            with torch.no_grad():

                pred_n = (
                    model(
                        x_tensor
                    )
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(
                        np.float64
                    )
                )


            pred_log = (
                pred_n
                * y_std
                + y_mean
            )

            objective_predictions.append(
                pred_log
            )


            ckpt_name = (
                f"{objective_name}_"
                f"seed_{base_seed}.pt"
            )

            ckpt_path = (
                OUT_DIR
                / ckpt_name
            )


            torch.save(
                {
                    "schema":
                        "icra27_os_t6p12a_scalar_regret_checkpoint_v0",

                    "objective_name":
                        objective_name,

                    "objective_index":
                        objective_index,

                    "ensemble_base_seed":
                        base_seed,

                    "actual_training_seed":
                        seed,

                    "context_dim":
                        EXPECTED_CONTEXT_DIM,

                    "beta_dim":
                        3,

                    "model_input_dim":
                        int(
                            X.shape[
                                1
                            ]
                        ),

                    "hidden":
                        list(
                            t611.HIDDEN
                        ),

                    "epochs":
                        int(
                            t611.EPOCHS
                        ),

                    "lr":
                        float(
                            t611.LR
                        ),

                    "weight_decay":
                        float(
                            t611.WEIGHT_DECAY
                        ),

                    "x_mean":
                        torch.tensor(
                            x_mean,
                            dtype=torch.float64,
                        ),

                    "x_std":
                        torch.tensor(
                            x_std,
                            dtype=torch.float64,
                        ),

                    "y_log_mean":
                        y_mean,

                    "y_log_std":
                        y_std,

                    "context_columns":
                        context_cols,

                    "beta_names":
                        beta_names,

                    "beta_vectors":
                        torch.tensor(
                            beta_vectors,
                            dtype=torch.float64,
                        ),

                    "model_state_dict":
                        model.state_dict(),

                    "final_standardized_mse":
                        final_loss,

                    "ensemble_semantics":
                        (
                            "Average predictions in "
                            "log1p-regret space across "
                            "the three ensemble members, "
                            "then apply expm1 and clamp "
                            "at zero."
                        ),
                },
                ckpt_path,
            )


            checkpoint_records.append(
                {
                    "objective":
                        objective_name,

                    "base_seed":
                        base_seed,

                    "actual_training_seed":
                        seed,

                    "file":
                        ckpt_name,

                    "sha256":
                        sha256(
                            ckpt_path
                        ),

                    "final_standardized_mse":
                        final_loss,
                }
            )


        ensemble_log_predictions.append(
            np.mean(
                np.stack(
                    objective_predictions,
                    axis=0,
                ),
                axis=0,
            )
        )


    pred_log = np.stack(
        ensemble_log_predictions,
        axis=1,
    )

    pred_regret = np.maximum(
        np.expm1(
            pred_log
        ),
        0.0,
    )


    # ========================================================
    # TRAIN-fit sanity only. NOT a generalization result.
    # ========================================================

    fit_rows = []


    for context_index, cid in enumerate(
        context_ids
    ):

        lo = (
            context_index
            * EXPECTED_BETA
        )

        hi = (
            lo
            + EXPECTED_BETA
        )

        true_r = R[
            lo:hi
        ]

        pred_r = pred_regret[
            lo:hi
        ]


        rho_M = t66.spearman(
            pred_r[
                :,
                0
            ],
            true_r[
                :,
                0
            ],
        )

        rho_S = t66.spearman(
            pred_r[
                :,
                1
            ],
            true_r[
                :,
                1
            ],
        )

        rho_E = t66.spearman(
            pred_r[
                :,
                2
            ],
            true_r[
                :,
                2
            ],
        )


        eta = np.asarray(
            [
                1/3,
                1/3,
                1/3,
            ],
            dtype=np.float64,
        )

        true_Q = q_from_regret(
            true_r,
            eta,
            t611.TCHEBY_RHO,
        )

        pred_Q = q_from_regret(
            pred_r,
            eta,
            t611.TCHEBY_RHO,
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


        fit_rows.append(
            {
                "context_id":
                    cid,

                "rho_regret_motion":
                    float(
                        rho_M
                    ),

                "rho_regret_stability":
                    float(
                        rho_S
                    ),

                "rho_regret_energy":
                    float(
                        rho_E
                    ),

                "balanced_oracle_beta":
                    beta_names[
                        oracle_idx
                    ],

                "balanced_selected_beta":
                    beta_names[
                        selected_idx
                    ],

                "balanced_Q_excess":
                    float(
                        true_Q[
                            selected_idx
                        ]
                        - true_Q[
                            oracle_idx
                        ]
                    ),

                "balanced_exact":
                    int(
                        selected_idx
                        == oracle_idx
                    ),
            }
        )


    write_csv(
        OUT_FIT,
        fit_rows,
    )


    beta_rows = []

    for i, name in enumerate(
        beta_names
    ):

        beta_rows.append(
            {
                "beta_index":
                    i,

                "beta_name":
                    name,

                "beta_motion":
                    float(
                        beta_vectors[
                            i,
                            0
                        ]
                    ),

                "beta_stability":
                    float(
                        beta_vectors[
                            i,
                            1
                        ]
                    ),

                "beta_energy":
                    float(
                        beta_vectors[
                            i,
                            2
                        ]
                    ),
            }
        )


    write_csv(
        OUT_BETA,
        beta_rows,
    )


    train_q = np.asarray(
        [
            row[
                "balanced_Q_excess"
            ]
            for row in fit_rows
        ],
        dtype=np.float64,
    )


    manifest = {
        "schema":
            "icra27_os_t6p12a_final_factorized_regret_selector_v0",

        "status":
            "FINAL_TRAIN_PASS",

        "heldout_used":
            False,

        "source_candidate_freeze":
            str(
                FREEZE_MANIFEST.relative_to(
                    ROOT
                )
            ),

        "training_contexts":
            context_ids,

        "training_context_count":
            len(
                context_ids
            ),

        "training_samples":
            int(
                X.shape[
                    0
                ]
            ),

        "representation":
            "semantic_plus_prop_probe",

        "context_dim":
            EXPECTED_CONTEXT_DIM,

        "beta_dim":
            3,

        "model_input_dim":
            int(
                X.shape[
                    1
                ]
            ),

        "surrogate":
            "three independent scalar regret MLP ensembles",

        "hidden_each":
            list(
                t611.HIDDEN
            ),

        "ensemble_base_seeds":
            list(
                ensemble_seeds
            ),

        "objective_training_seed_offsets":
            {
                "motion":
                    0,

                "stability":
                    100000,

                "energy":
                    200000,
            },

        "target":
            "log1p(context-local physical regret)",

        "probe_horizon_s":
            0.4,

        "primary_eta":
            [
                1/3,
                1/3,
                1/3,
            ],

        "tcheby_rho":
            float(
                t611.TCHEBY_RHO
            ),

        "ensemble_inference":
            (
                "For each objective, average the three "
                "network predictions in log1p-regret "
                "space. Apply expm1, clamp predicted "
                "regret at zero, then compute the "
                "augmented-Tchebycheff score."
            ),

        "checkpoints":
            checkpoint_records,

        "train_fit_sanity_only":
            {
                "balanced_Q_excess_mean":
                    float(
                        np.mean(
                            train_q
                        )
                    ),

                "balanced_Q_excess_median":
                    float(
                        np.median(
                            train_q
                        )
                    ),

                "balanced_exact_fraction":
                    float(
                        np.mean(
                            [
                                row[
                                    "balanced_exact"
                                ]
                                for row in fit_rows
                            ]
                        )
                    ),

                "warning":
                    (
                        "These are in-sample TRAIN-fit "
                        "diagnostics only and must not "
                        "be reported as generalization "
                        "performance."
                    ),
            },

        "untouched_evaluation_contexts":
            freeze[
                "untouched_evaluation_contexts"
            ],

        "scientific_guard":
            (
                "This model is trained only after "
                "T6.11b architecture freeze. No model "
                "or preprocessing choice may be changed "
                "using val/test/hard outcomes."
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
        "ICRA27 OS-T6.12a FINAL "
        "FACTORIZED REGRET SELECTOR"
    )

    print("=" * 118)

    print(
        "TRAIN contexts             :",
        len(
            context_ids
        ),
    )

    print(
        "TRAIN samples              :",
        X.shape[
            0
        ],
    )

    print(
        "context / input dim        :",
        EXPECTED_CONTEXT_DIM,
        "/",
        X.shape[
            1
        ],
    )

    print(
        "hidden each                :",
        t611.HIDDEN,
    )

    print(
        "ensemble seeds             :",
        ensemble_seeds,
    )

    print(
        "checkpoint count           :",
        len(
            checkpoint_records
        ),
    )

    print(
        "TRAIN-fit balanced Qmean   :",
        float(
            np.mean(
                train_q
            )
        ),
    )

    print(
        "TRAIN-fit balanced Qmedian :",
        float(
            np.median(
                train_q
            )
        ),
    )

    print(
        "TRAIN-fit exact fraction   :",
        float(
            np.mean(
                [
                    row[
                        "balanced_exact"
                    ]
                    for row in fit_rows
                ]
            )
        ),
    )

    print()
    print(
        "NOTE: TRAIN-fit metrics above are "
        "sanity checks only."
    )

    print()
    print(
        "[ICRA27] OS-T6.12a final selector: "
        "FINAL TRAIN PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
