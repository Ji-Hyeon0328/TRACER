from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

T611A_PATH = (
    ROOT
    / "scripts/icra27"
    / "train_os_t6p11a_factorized_regret_surface_loco_v0.py"
)

PROTOCOL_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p12b_untouched_eval_protocol_v0"
    / "untouched_eval_protocol_manifest.json"
)

FINAL_SELECTOR_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12a_final_factorized_regret_selector_v0"
)

FINAL_SELECTOR_MANIFEST = (
    FINAL_SELECTOR_DIR
    / "final_selector_manifest.json"
)

HELDOUT_CONTEXT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12c_heldout_augmented_causal_context_v0"
)

HELDOUT_CONTEXT_CSV = (
    HELDOUT_CONTEXT_DIR
    / "heldout_augmented_causal_context.csv"
)

HELDOUT_CONTEXT_MANIFEST = (
    HELDOUT_CONTEXT_DIR
    / "heldout_augmented_causal_context_manifest.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12d_heldout_selector_precommit_v0"
)

OUT_SURFACE = (
    OUT_DIR
    / "precommitted_regret_surface.csv"
)

OUT_SELECTIONS = (
    OUT_DIR
    / "precommitted_eta_selections.csv"
)

OUT_BASELINES = (
    OUT_DIR
    / "train_only_eta_baselines.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "heldout_selector_precommit_manifest.json"
)


EXPECTED_CONTEXTS = 12
EXPECTED_BETA = 21
EXPECTED_CONTEXT_DIM = 87
EXPECTED_INPUT_DIM = 90
EXPECTED_CHECKPOINTS = 9

TCHEBY_RHO = 0.01

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

    fields = []
    seen = set()

    for row in rows:
        for key in row.keys():
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


def torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


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


def second_best_gap(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="stable",
    )

    if len(order) < 2:
        raise RuntimeError(
            "Need at least two candidates."
        )

    return float(
        values[
            order[1]
        ]
        - values[
            order[0]
        ]
    )


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        T611A_PATH,
        PROTOCOL_MANIFEST,
        FINAL_SELECTOR_MANIFEST,
        HELDOUT_CONTEXT_CSV,
        HELDOUT_CONTEXT_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # Frozen source guards.
    # ========================================================

    protocol = json.loads(
        PROTOCOL_MANIFEST.read_text()
    )

    selector = json.loads(
        FINAL_SELECTOR_MANIFEST.read_text()
    )

    heldout = json.loads(
        HELDOUT_CONTEXT_MANIFEST.read_text()
    )


    if protocol.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12b protocol is not FREEZE_PASS."
        )

    if selector.get(
        "status"
    ) != "FINAL_TRAIN_PASS":
        raise RuntimeError(
            "T6.12a selector is not "
            "FINAL_TRAIN_PASS."
        )

    if heldout.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12c heldout context is not "
            "FREEZE_PASS."
        )


    if bool(
        protocol.get(
            "heldout_outcomes_used",
            True,
        )
    ):
        raise RuntimeError(
            "Protocol reports heldout outcome use."
        )

    if bool(
        selector.get(
            "heldout_used",
            True,
        )
    ):
        raise RuntimeError(
            "Selector reports heldout use."
        )

    if bool(
        heldout.get(
            "heldout_outcomes_used",
            True,
        )
    ):
        raise RuntimeError(
            "T6.12c reports heldout outcome use."
        )


    if selector[
        "representation"
    ] != "semantic_plus_prop_probe":
        raise RuntimeError(
            "Frozen representation drift."
        )

    if int(
        selector[
            "context_dim"
        ]
    ) != EXPECTED_CONTEXT_DIM:
        raise RuntimeError(
            "Frozen context dimension drift."
        )

    if list(
        selector[
            "hidden_each"
        ]
    ) != [
        28,
        28,
    ]:
        raise RuntimeError(
            "Frozen architecture drift."
        )

    if abs(
        float(
            selector[
                "probe_horizon_s"
            ]
        )
        - 0.4
    ) > 1.0e-12:
        raise RuntimeError(
            "Frozen probe horizon drift."
        )

    if abs(
        float(
            selector[
                "tcheby_rho"
            ]
        )
        - TCHEBY_RHO
    ) > 1.0e-12:
        raise RuntimeError(
            "Tchebycheff rho drift."
        )


    checkpoints = selector[
        "checkpoints"
    ]

    if len(
        checkpoints
    ) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHECKPOINTS} "
            f"checkpoints."
        )


    # ========================================================
    # Load exact model implementation.
    # ========================================================

    t611 = load_module(
        T611A_PATH,
        "os_t6p12d_t611",
    )

    t69c = t611.load_module(
        t611.T69C_PATH,
        "os_t6p12d_t69c",
    )

    t66 = t69c.load_module(
        t69c.T66_PATH,
        "os_t6p12d_t66",
    )


    if tuple(
        t611.HIDDEN
    ) != (
        28,
        28,
    ):
        raise RuntimeError(
            "ScalarRegretMLP architecture drift."
        )


    # ========================================================
    # Verify every checkpoint hash before inference.
    # ========================================================

    checkpoint_payloads = {
        "motion": [],
        "stability": [],
        "energy": [],
    }

    hash_pass_count = 0


    for record in checkpoints:

        path = (
            FINAL_SELECTOR_DIR
            / record[
                "file"
            ]
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )


        actual_hash = sha256(
            path
        )

        expected_hash = record[
            "sha256"
        ]


        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Checkpoint hash drift: {path}"
            )


        hash_pass_count += 1


        payload = torch_load(
            path
        )

        objective = str(
            payload[
                "objective_name"
            ]
        )

        if objective not in (
            checkpoint_payloads
        ):
            raise RuntimeError(
                f"Unexpected objective: {objective}"
            )


        if int(
            payload[
                "context_dim"
            ]
        ) != EXPECTED_CONTEXT_DIM:
            raise RuntimeError(
                f"{path}: context dimension drift."
            )

        if int(
            payload[
                "model_input_dim"
            ]
        ) != EXPECTED_INPUT_DIM:
            raise RuntimeError(
                f"{path}: input dimension drift."
            )

        if list(
            payload[
                "hidden"
            ]
        ) != [
            28,
            28,
        ]:
            raise RuntimeError(
                f"{path}: hidden architecture drift."
            )


        checkpoint_payloads[
            objective
        ].append(
            (
                record,
                payload,
                path,
            )
        )


    for objective, items in (
        checkpoint_payloads.items()
    ):
        if len(items) != 3:
            raise RuntimeError(
                f"{objective}: expected 3 "
                f"ensemble checkpoints."
            )


    # ========================================================
    # Canonical feature / beta contract from checkpoints.
    # ========================================================

    reference_payload = (
        checkpoint_payloads[
            "motion"
        ][
            0
        ][
            1
        ]
    )


    context_columns = list(
        reference_payload[
            "context_columns"
        ]
    )

    beta_names = list(
        reference_payload[
            "beta_names"
        ]
    )

    beta_vectors = np.asarray(
        reference_payload[
            "beta_vectors"
        ],
        dtype=np.float64,
    )


    if len(
        context_columns
    ) != EXPECTED_CONTEXT_DIM:
        raise RuntimeError(
            "Checkpoint context-column count drift."
        )

    if len(
        beta_names
    ) != EXPECTED_BETA:
        raise RuntimeError(
            "Checkpoint beta candidate count drift."
        )

    if beta_vectors.shape != (
        EXPECTED_BETA,
        3,
    ):
        raise RuntimeError(
            "Checkpoint beta-vector shape drift."
        )


    for objective, items in (
        checkpoint_payloads.items()
    ):
        for _record, payload, path in items:

            if list(
                payload[
                    "context_columns"
                ]
            ) != context_columns:
                raise RuntimeError(
                    f"{path}: context-column drift."
                )

            if list(
                payload[
                    "beta_names"
                ]
            ) != beta_names:
                raise RuntimeError(
                    f"{path}: beta-name drift."
                )


            candidate_vectors = np.asarray(
                payload[
                    "beta_vectors"
                ],
                dtype=np.float64,
            )

            if not np.array_equal(
                candidate_vectors,
                beta_vectors,
            ):
                raise RuntimeError(
                    f"{path}: beta-vector drift."
                )


    # ========================================================
    # Load untouched causal contexts.
    # ========================================================

    context_rows = read_csv(
        HELDOUT_CONTEXT_CSV
    )

    if len(
        context_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTEXTS} "
            f"heldout contexts."
        )


    context_lookup = {}

    for row in context_rows:

        cid = row[
            "context_id"
        ]

        if cid in context_lookup:
            raise RuntimeError(
                f"Duplicate heldout context: {cid}"
            )

        context_lookup[
            cid
        ] = row


    protocol_order = []

    for split in (
        "val",
        "test",
        "hard",
    ):
        for seed in protocol[
            "evaluation_splits"
        ][
            split
        ]:
            protocol_order.append(
                (
                    f"rough_seed_{int(seed)}",
                    split,
                    int(seed),
                )
            )


    if len(
        protocol_order
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Protocol evaluation count drift."
        )


    # ========================================================
    # Frozen model inference.
    #
    # Average each objective's 3 ensemble members in
    # log1p-regret space, then expm1 and clamp at zero.
    # ========================================================

    torch.set_num_threads(
        1
    )

    torch.use_deterministic_algorithms(
        True
    )


    surface_rows = []
    selection_rows = []

    predicted_surfaces = {}


    for context_id, split, seed in (
        protocol_order
    ):

        if context_id not in context_lookup:
            raise RuntimeError(
                f"Missing heldout context: "
                f"{context_id}"
            )


        row = context_lookup[
            context_id
        ]

        if row[
            "split_group"
        ] != split:
            raise RuntimeError(
                f"{context_id}: split mismatch."
            )

        if int(
            row[
                "seed"
            ]
        ) != seed:
            raise RuntimeError(
                f"{context_id}: seed mismatch."
            )


        context_vector = np.asarray(
            [
                float(
                    row[
                        col
                    ]
                )
                for col in context_columns
            ],
            dtype=np.float64,
        )


        if context_vector.shape != (
            EXPECTED_CONTEXT_DIM,
        ):
            raise RuntimeError(
                f"{context_id}: 87D shape drift."
            )

        if not np.all(
            np.isfinite(
                context_vector
            )
        ):
            raise RuntimeError(
                f"{context_id}: non-finite context."
            )


        X = np.concatenate(
            (
                np.repeat(
                    context_vector[
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


        if X.shape != (
            EXPECTED_BETA,
            EXPECTED_INPUT_DIM,
        ):
            raise RuntimeError(
                f"{context_id}: model input "
                f"shape drift {X.shape}"
            )


        objective_log_predictions = []


        for objective in (
            "motion",
            "stability",
            "energy",
        ):

            member_predictions = []


            for (
                _record,
                payload,
                _path,
            ) in checkpoint_payloads[
                objective
            ]:

                x_mean = np.asarray(
                    payload[
                        "x_mean"
                    ],
                    dtype=np.float64,
                )

                x_std = np.asarray(
                    payload[
                        "x_std"
                    ],
                    dtype=np.float64,
                )


                if x_mean.shape != (
                    EXPECTED_INPUT_DIM,
                ):
                    raise RuntimeError(
                        "x_mean shape drift."
                    )

                if x_std.shape != (
                    EXPECTED_INPUT_DIM,
                ):
                    raise RuntimeError(
                        "x_std shape drift."
                    )


                Xn = (
                    X
                    - x_mean
                ) / x_std


                model = (
                    t611.ScalarRegretMLP(
                        EXPECTED_INPUT_DIM
                    )
                )

                model.load_state_dict(
                    payload[
                        "model_state_dict"
                    ]
                )

                model.eval()


                with torch.no_grad():

                    pred_n = (
                        model(
                            torch.tensor(
                                Xn,
                                dtype=torch.float32,
                            )
                        )
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .astype(
                            np.float64
                        )
                    )


                y_mean = float(
                    payload[
                        "y_log_mean"
                    ]
                )

                y_std = float(
                    payload[
                        "y_log_std"
                    ]
                )


                pred_log = (
                    pred_n
                    * y_std
                    + y_mean
                )


                member_predictions.append(
                    pred_log
                )


            objective_log_predictions.append(
                np.mean(
                    np.stack(
                        member_predictions,
                        axis=0,
                    ),
                    axis=0,
                )
            )


        pred_log_regret = np.stack(
            objective_log_predictions,
            axis=1,
        )


        pred_regret = np.maximum(
            np.expm1(
                pred_log_regret
            ),
            0.0,
        )


        if pred_regret.shape != (
            EXPECTED_BETA,
            3,
        ):
            raise RuntimeError(
                f"{context_id}: predicted regret "
                f"shape drift."
            )

        if not np.all(
            np.isfinite(
                pred_regret
            )
        ):
            raise RuntimeError(
                f"{context_id}: non-finite "
                "predicted regret."
            )


        predicted_surfaces[
            context_id
        ] = pred_regret


        eta_q = {
            eta_name:
                q_from_regret(
                    pred_regret,
                    eta,
                )
            for eta_name, eta
            in ETAS.items()
        }


        for beta_index, beta_name in enumerate(
            beta_names
        ):

            surface_row = {
                "context_id":
                    context_id,

                "seed":
                    seed,

                "split_group":
                    split,

                "context_sha256":
                    row[
                        "semantic_plus_prop_probe_sha256"
                    ],

                "probe_safety_state_diagnostic":
                    row[
                        "probe_safety_state"
                    ],

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

                "pred_regret_motion":
                    float(
                        pred_regret[
                            beta_index,
                            0
                        ]
                    ),

                "pred_regret_stability":
                    float(
                        pred_regret[
                            beta_index,
                            1
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


            for eta_name, q_values in (
                eta_q.items()
            ):
                surface_row[
                    f"pred_Q_{eta_name}"
                ] = float(
                    q_values[
                        beta_index
                    ]
                )


            surface_rows.append(
                surface_row
            )


        for eta_name, eta in (
            ETAS.items()
        ):

            q_values = eta_q[
                eta_name
            ]

            selected_index = int(
                np.argmin(
                    q_values
                )
            )


            selection_rows.append(
                {
                    "context_id":
                        context_id,

                    "seed":
                        seed,

                    "split_group":
                        split,

                    "context_sha256":
                        row[
                            "semantic_plus_prop_probe_sha256"
                        ],

                    "probe_safety_state_diagnostic":
                        row[
                            "probe_safety_state"
                        ],

                    "eta_name":
                        eta_name,

                    "eta_motion":
                        float(
                            eta[
                                0
                            ]
                        ),

                    "eta_stability":
                        float(
                            eta[
                                1
                            ]
                        ),

                    "eta_energy":
                        float(
                            eta[
                                2
                            ]
                        ),

                    "selected_beta_index":
                        selected_index,

                    "selected_beta_name":
                        beta_names[
                            selected_index
                        ],

                    "selected_beta_motion":
                        float(
                            beta_vectors[
                                selected_index,
                                0
                            ]
                        ),

                    "selected_beta_stability":
                        float(
                            beta_vectors[
                                selected_index,
                                1
                            ]
                        ),

                    "selected_beta_energy":
                        float(
                            beta_vectors[
                                selected_index,
                                2
                            ]
                        ),

                    "predicted_Q_selected":
                        float(
                            q_values[
                                selected_index
                            ]
                        ),

                    # Raw diagnostic only.
                    # No threshold is introduced.
                    "predicted_Q_best_second_gap":
                        second_best_gap(
                            q_values
                        ),
                }
            )


    if len(
        surface_rows
    ) != (
        EXPECTED_CONTEXTS
        * EXPECTED_BETA
    ):
        raise RuntimeError(
            "Predicted surface row count drift."
        )

    if len(
        selection_rows
    ) != (
        EXPECTED_CONTEXTS
        * len(
            ETAS
        )
    ):
        raise RuntimeError(
            "Precommitted selection count drift."
        )


    # ========================================================
    # TRAIN-only eta-specific no-context baselines.
    #
    # These are also committed BEFORE any heldout physical
    # outcome is generated.
    # ========================================================

    train_context_ids = list(
        selector[
            "training_contexts"
        ]
    )

    if len(
        train_context_ids
    ) != 33:
        raise RuntimeError(
            "Expected 33 final TRAIN contexts."
        )


    physical_rows = t66.read_csv(
        t66.PHYSICAL_CSV
    )

    train_set = set(
        train_context_ids
    )

    grouped_train = {}


    for physical_row in physical_rows:

        cid = physical_row[
            "context_id"
        ]

        if cid in train_set:
            grouped_train.setdefault(
                cid,
                [],
            ).append(
                physical_row
            )


    train_regret = {}


    for cid in train_context_ids:

        if cid not in grouped_train:
            raise RuntimeError(
                f"Missing TRAIN physical atlas "
                f"context: {cid}"
            )


        rows = sorted(
            grouped_train[
                cid
            ],
            key=lambda r:
                r[
                    "beta_name"
                ],
        )


        if len(rows) != EXPECTED_BETA:
            raise RuntimeError(
                f"{cid}: expected 21 TRAIN "
                "physical rows."
            )


        names = [
            r[
                "beta_name"
            ]
            for r in rows
        ]


        if names != beta_names:
            raise RuntimeError(
                f"{cid}: TRAIN beta-order drift."
            )


        J = np.asarray(
            [
                [
                    float(
                        r[
                            "J_motion_s_per_m"
                        ]
                    ),

                    float(
                        r[
                            "J_stability"
                        ]
                    ),

                    float(
                        r[
                            "J_energy_j_per_m"
                        ]
                    ),
                ]
                for r in rows
            ],
            dtype=np.float64,
        )


        _balanced_Q, regret = (
            t69c.tcheby_quality(
                J
            )
        )


        train_regret[
            cid
        ] = regret


    baseline_rows = []


    for eta_name, eta in (
        ETAS.items()
    ):

        train_q_stack = np.stack(
            [
                q_from_regret(
                    train_regret[
                        cid
                    ],
                    eta,
                )
                for cid
                in train_context_ids
            ],
            axis=0,
        )


        mean_q = np.mean(
            train_q_stack,
            axis=0,
        )

        baseline_index = int(
            np.argmin(
                mean_q
            )
        )


        baseline_rows.append(
            {
                "eta_name":
                    eta_name,

                "eta_motion":
                    float(
                        eta[
                            0
                        ]
                    ),

                "eta_stability":
                    float(
                        eta[
                            1
                        ]
                    ),

                "eta_energy":
                    float(
                        eta[
                            2
                        ]
                    ),

                "train_context_count":
                    len(
                        train_context_ids
                    ),

                "baseline_beta_index":
                    baseline_index,

                "baseline_beta_name":
                    beta_names[
                        baseline_index
                    ],

                "baseline_beta_motion":
                    float(
                        beta_vectors[
                            baseline_index,
                            0
                        ]
                    ),

                "baseline_beta_stability":
                    float(
                        beta_vectors[
                            baseline_index,
                            1
                        ]
                    ),

                "baseline_beta_energy":
                    float(
                        beta_vectors[
                            baseline_index,
                            2
                        ]
                    ),

                "train_mean_Q":
                    float(
                        mean_q[
                            baseline_index
                        ]
                    ),

                "train_Q_best_second_gap":
                    second_best_gap(
                        mean_q
                    ),
            }
        )


    # ========================================================
    # Persist precommit artifacts.
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_SURFACE,
        surface_rows,
    )

    write_csv(
        OUT_SELECTIONS,
        selection_rows,
    )

    write_csv(
        OUT_BASELINES,
        baseline_rows,
    )


    artifact_hashes = {
        "precommitted_regret_surface.csv":
            sha256(
                OUT_SURFACE
            ),

        "precommitted_eta_selections.csv":
            sha256(
                OUT_SELECTIONS
            ),

        "train_only_eta_baselines.csv":
            sha256(
                OUT_BASELINES
            ),
    }


    manifest = {
        "schema":
            "icra27_os_t6p12d_heldout_selector_precommit_v0",

        "status":
            "FREEZE_PASS",

        "role":
            "HELDOUT_SELECTOR_PRECOMMIT",

        "heldout_physical_outcomes_used":
            False,

        "contexts":
            EXPECTED_CONTEXTS,

        "beta_candidates":
            EXPECTED_BETA,

        "predicted_surface_rows":
            len(
                surface_rows
            ),

        "eta_profiles":
            {
                name:
                    eta.tolist()
                for name, eta
                in ETAS.items()
            },

        "precommitted_selections":
            len(
                selection_rows
            ),

        "train_only_baselines":
            len(
                baseline_rows
            ),

        "checkpoint_hashes_verified":
            {
                "pass":
                    (
                        hash_pass_count
                        == EXPECTED_CHECKPOINTS
                    ),

                "verified":
                    hash_pass_count,

                "expected":
                    EXPECTED_CHECKPOINTS,
            },

        "selector_contract": {
            "representation":
                "semantic_plus_prop_probe",

            "context_dim":
                EXPECTED_CONTEXT_DIM,

            "beta_dim":
                3,

            "input_dim":
                EXPECTED_INPUT_DIM,

            "factorized_objectives":
                [
                    "motion",
                    "stability",
                    "energy",
                ],

            "ensemble_members_per_objective":
                3,

            "ensemble_semantics":
                (
                    "Average network predictions in "
                    "log1p-regret space per objective, "
                    "then expm1 and clamp at zero."
                ),

            "tcheby_rho":
                TCHEBY_RHO,
        },

        "source_hashes": {
            "heldout_context_csv":
                sha256(
                    HELDOUT_CONTEXT_CSV
                ),

            "final_selector_manifest":
                sha256(
                    FINAL_SELECTOR_MANIFEST
                ),

            "protocol_manifest":
                sha256(
                    PROTOCOL_MANIFEST
                ),
        },

        "artifact_hashes":
            artifact_hashes,

        "diagnostic_only":
            (
                "probe_safety_state and predicted "
                "best-second Q gaps are recorded but "
                "do not alter selection or inclusion."
            ),

        "scientific_guard":
            (
                "All selector predictions, eta-specific "
                "beta selections, and TRAIN-only "
                "no-context baseline beta selections "
                "are frozen before any heldout 21-beta "
                "physical outcome atlas is generated."
            ),

        "next_step":
            (
                "Generate the frozen heldout 12x21 "
                "physical atlas without modifying any "
                "precommitted selector output."
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
    # Human-readable precommit summary.
    # ========================================================

    by_context = {}

    for row in selection_rows:
        by_context.setdefault(
            row[
                "context_id"
            ],
            {},
        )[
            row[
                "eta_name"
            ]
        ] = row


    print()
    print("=" * 142)
    print(
        "ICRA27 OS-T6.12d HELDOUT "
        "SELECTOR PRECOMMIT"
    )
    print("=" * 142)

    print(
        " context           split | "
        "balanced              motion                "
        "stability             energy                "
        "| balanced gap"
    )

    print("-" * 142)


    for context_id, split, _seed in (
        protocol_order
    ):

        selected = by_context[
            context_id
        ]

        print(
            f" {context_id:<17} {split:<4} | "
            f"{selected['balanced']['selected_beta_name']:<21} "
            f"{selected['motion_biased']['selected_beta_name']:<21} "
            f"{selected['stability_biased']['selected_beta_name']:<21} "
            f"{selected['energy_biased']['selected_beta_name']:<21} "
            f"| "
            f"{float(selected['balanced']['predicted_Q_best_second_gap']):.6g}"
        )


    print()
    print("TRAIN-only no-context baselines:")

    for row in baseline_rows:
        print(
            f"  {row['eta_name']:<18} "
            f"{row['baseline_beta_name']:<22} "
            f"meanQ={float(row['train_mean_Q']):.6f} "
            f"gap={float(row['train_Q_best_second_gap']):.6g}"
        )


    print()
    print(
        "checkpoint hashes verified     :",
        f"{hash_pass_count}/{EXPECTED_CHECKPOINTS}",
    )

    print(
        "predicted surface rows         :",
        len(
            surface_rows
        ),
    )

    print(
        "precommitted eta selections    :",
        len(
            selection_rows
        ),
    )

    print(
        "heldout physical outcomes used :",
        False,
    )

    print()
    print(
        "[ICRA27] OS-T6.12d heldout selector "
        "precommit: FREEZE PASS"
    )

    print("=" * 142)


if __name__ == "__main__":
    main()
