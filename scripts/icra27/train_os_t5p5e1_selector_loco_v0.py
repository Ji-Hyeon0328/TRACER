from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Frozen supervision
# ============================================================

LABEL_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5d_expanded_preference_labels_v0"
)

LABEL_CSV = (
    LABEL_ROOT
    / "preference_to_beta_labels.csv"
)

LABEL_MANIFEST = (
    LABEL_ROOT
    / "preference_label_manifest.json"
)

PARETO_CSV = (
    LABEL_ROOT
    / "pareto_front.csv"
)


ATLAS_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
)

ATLAS_CSV = (
    ATLAS_ROOT
    / "physical_beta_response_atlas.csv"
)


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5e1_selector_loco_v0"
)

OUT_PREDICTIONS = (
    OUT_DIR
    / "loco_predictions.csv"
)

OUT_CONTEXT_SUMMARY = (
    OUT_DIR
    / "loco_context_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "selector_loco_manifest.json"
)


# ============================================================
# Frozen selector contract
# ============================================================

CONTEXT_COLUMNS = (
    "context_friction_mu",
    "context_height_std_m",
    "context_height_relief_p95_p05_m",
    "context_slope_rms",
    "context_slope_q95",
)

PREFERENCE_COLUMNS = (
    "w_motion",
    "w_stability",
    "w_energy",
)

TARGET_LAMBDA_COLUMNS = (
    "selected_lambda_motion",
    "selected_lambda_stability",
    "selected_lambda_energy",
)

OBJECTIVE_COLUMNS = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)


EXPECTED_CONTEXTS = 20
EXPECTED_ROUGH_CONTEXTS = 18
EXPECTED_PREFERENCES = 25
EXPECTED_LABELS = 500
EXPECTED_BETAS = 21
EXPECTED_CV_ROWS = (
    EXPECTED_ROUGH_CONTEXTS
    * EXPECTED_PREFERENCES
)


BETA_FLOOR = 0.15
BETA_SPAN = 0.55

RHO = 0.01

MODEL_SEED = 27551

HIDDEN_DIM = 64
EPOCHS = 2000
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-4

EPS = 1.0e-12
TOL = 1.0e-10


# ============================================================
# Utilities
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
) -> list[dict[str, str]]:
    with path.open(
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows for {path}"
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


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb",
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True
    )


# ============================================================
# Model
# ============================================================

class ObjectiveSelector(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                8,
                HIDDEN_DIM,
            ),
            nn.SiLU(),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM,
            ),
            nn.SiLU(),

            nn.Linear(
                HIDDEN_DIM,
                3,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        logits = self.net(
            x
        )

        lam = torch.softmax(
            logits,
            dim=-1,
        )

        beta = (
            BETA_FLOOR
            + BETA_SPAN
            * lam
        )

        return (
            lam,
            beta,
        )


# ============================================================
# Context normalization
# ============================================================

def fit_context_scaler(
    rows: list[dict[str, str]],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    # Each context occurs 25 times. Use one physical
    # context vector per context so preference multiplicity
    # never changes the scaler weighting.

    first_by_context = {}

    for row in rows:
        cid = row[
            "context_id"
        ]

        first_by_context.setdefault(
            cid,
            row,
        )

    values = np.asarray(
        [
            [
                finite(
                    row[column]
                )
                for column in (
                    CONTEXT_COLUMNS
                )
            ]
            for row in (
                first_by_context.values()
            )
        ],
        dtype=np.float64,
    )

    mean = np.mean(
        values,
        axis=0,
    )

    std = np.std(
        values,
        axis=0,
        ddof=0,
    )

    if np.any(
        std <= EPS
    ):
        raise RuntimeError(
            "Degenerate context scaler: "
            f"std={std.tolist()}"
        )

    return (
        mean,
        std,
    )


def make_xy(
    rows: list[dict[str, str]],
    *,
    context_mean: np.ndarray,
    context_std: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    xs = []
    ys = []

    for row in rows:
        context = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in (
                    CONTEXT_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        context = (
            context
            - context_mean
        ) / context_std

        preference = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in (
                    PREFERENCE_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        target = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in (
                    TARGET_LAMBDA_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        xs.append(
            np.concatenate(
                [
                    context,
                    preference,
                ]
            )
        )

        ys.append(
            target
        )

    return (
        np.asarray(
            xs,
            dtype=np.float32,
        ),
        np.asarray(
            ys,
            dtype=np.float32,
        ),
    )


# ============================================================
# Training
# ============================================================

def train_selector(
    train_rows: list[dict[str, str]],
    *,
    context_mean: np.ndarray,
    context_std: np.ndarray,
) -> tuple[
    ObjectiveSelector,
    float,
]:
    seed_everything(
        MODEL_SEED
    )

    x_np, y_np = make_xy(
        train_rows,
        context_mean=context_mean,
        context_std=context_std,
    )

    x = torch.from_numpy(
        x_np
    )

    y = torch.from_numpy(
        y_np
    )

    model = ObjectiveSelector()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    final_loss = float(
        "nan"
    )

    for _ in range(
        EPOCHS
    ):
        model.train()

        predicted_lambda, _ = model(
            x
        )

        loss = torch.mean(
            (
                predicted_lambda
                - y
            )
            ** 2
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite selector loss."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        final_loss = float(
            loss.detach().item()
        )

    return (
        model,
        final_loss,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        LABEL_CSV,
        LABEL_MANIFEST,
        PARETO_CSV,
        ATLAS_CSV,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    # --------------------------------------------------------
    # Frozen supervision provenance
    # --------------------------------------------------------

    label_manifest = json.loads(
        LABEL_MANIFEST.read_text()
    )

    if label_manifest.get(
        "schema"
    ) != (
        "icra27_os_t5p5d_"
        "expanded_preference_labels_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.5d schema."
        )

    if label_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.5d is not FREEZE_PASS."
        )

    if bool(
        label_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5d reports held-out use."
        )


    labels = read_csv(
        LABEL_CSV
    )

    atlas_rows = read_csv(
        ATLAS_CSV
    )

    pareto_rows = read_csv(
        PARETO_CSV
    )


    if len(
        labels
    ) != EXPECTED_LABELS:
        raise RuntimeError(
            f"Expected 500 labels; "
            f"got {len(labels)}"
        )


    # --------------------------------------------------------
    # Input / target sanity
    # --------------------------------------------------------

    contexts = []

    for row in labels:
        cid = row[
            "context_id"
        ]

        if cid not in contexts:
            contexts.append(
                cid
            )

        w_sum = sum(
            finite(
                row[column]
            )
            for column in (
                PREFERENCE_COLUMNS
            )
        )

        if not math.isclose(
            w_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                f"{cid}: w does not sum to 1."
            )

        lam = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in (
                    TARGET_LAMBDA_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        if not math.isclose(
            float(
                np.sum(lam)
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                f"{cid}: lambda target "
                "does not sum to 1."
            )

        expected_beta = (
            BETA_FLOOR
            + BETA_SPAN
            * lam
        )

        actual_beta = np.asarray(
            [
                finite(
                    row[
                        "selected_beta_motion"
                    ]
                ),
                finite(
                    row[
                        "selected_beta_stability"
                    ]
                ),
                finite(
                    row[
                        "selected_beta_energy"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        if np.max(
            np.abs(
                expected_beta
                - actual_beta
            )
        ) > 1e-10:
            raise RuntimeError(
                f"{cid}: beta/lambda "
                "target contract mismatch."
            )


    if len(
        contexts
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Expected 20 contexts; "
            f"got {len(contexts)}"
        )


    rough_contexts = [
        cid
        for cid in contexts
        if cid.startswith(
            "rough_seed_"
        )
    ]

    if len(
        rough_contexts
    ) != (
        EXPECTED_ROUGH_CONTEXTS
    ):
        raise RuntimeError(
            "Expected 18 rough contexts; "
            f"got {len(rough_contexts)}"
        )


    nonrough = [
        cid
        for cid in contexts
        if not cid.startswith(
            "rough_seed_"
        )
    ]

    if set(
        nonrough
    ) != {
        "flat",
        "low_friction",
    }:
        raise RuntimeError(
            f"Unexpected non-rough "
            f"contexts: {nonrough}"
        )


    preference_counts = Counter(
        row[
            "context_id"
        ]
        for row in labels
    )

    if any(
        preference_counts[
            cid
        ]
        != EXPECTED_PREFERENCES
        for cid in contexts
    ):
        raise RuntimeError(
            "Each context must have "
            "exactly 25 preferences."
        )


    # --------------------------------------------------------
    # Frozen beta lattice
    # --------------------------------------------------------

    beta_names = []

    lambda_by_beta = {}

    first_context = (
        contexts[0]
    )

    for row in atlas_rows:
        if row[
            "context_id"
        ] != first_context:
            continue

        name = row[
            "beta_name"
        ]

        beta_names.append(
            name
        )

        lambda_by_beta[
            name
        ] = np.asarray(
            [
                finite(
                    row[
                        "lambda_motion"
                    ]
                ),
                finite(
                    row[
                        "lambda_stability"
                    ]
                ),
                finite(
                    row[
                        "lambda_energy"
                    ]
                ),
            ],
            dtype=np.float64,
        )


    if (
        len(beta_names)
        != EXPECTED_BETAS
        or len(
            set(beta_names)
        )
        != EXPECTED_BETAS
    ):
        raise RuntimeError(
            "Expected 21 unique beta "
            "lattice points."
        )


    lattice = np.asarray(
        [
            lambda_by_beta[
                name
            ]
            for name in beta_names
        ],
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Physical atlas lookup
    # --------------------------------------------------------

    atlas_lookup = {}

    for row in atlas_rows:
        key = (
            row[
                "context_id"
            ],
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


    pareto_by_context = {
        cid:
            []
        for cid in contexts
    }

    for row in pareto_rows:
        cid = row[
            "context_id"
        ]

        pareto_by_context[
            cid
        ].append(
            row
        )


    # --------------------------------------------------------
    # Verify local scalarization calculation against frozen
    # T5.5d oracle score before training anything.
    # --------------------------------------------------------

    regret_reference = {}

    for cid in contexts:
        front = (
            pareto_by_context[
                cid
            ]
        )

        if not front:
            raise RuntimeError(
                f"{cid}: empty Pareto front."
            )

        values = np.asarray(
            [
                [
                    finite(
                        row[column]
                    )
                    for column in (
                        OBJECTIVE_COLUMNS
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

        nadir = np.max(
            values,
            axis=0,
        )

        span = (
            nadir
            - ideal
        )

        regret_reference[
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
                    OBJECTIVE_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        ideal, span = (
            regret_reference[
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


    max_oracle_score_error = 0.0

    for row in labels:
        w = np.asarray(
            [
                finite(
                    row[column]
                )
                for column in (
                    PREFERENCE_COLUMNS
                )
            ],
            dtype=np.float64,
        )

        reconstructed = score_beta(
            context_id=row[
                "context_id"
            ],
            beta_name=row[
                "selected_beta_name"
            ],
            w=w,
        )

        frozen = finite(
            row[
                "scalarization_score"
            ]
        )

        error = abs(
            reconstructed
            - frozen
        )

        max_oracle_score_error = max(
            max_oracle_score_error,
            error,
        )

        if error > 1e-9:
            raise RuntimeError(
                "Frozen scalarization "
                "reconstruction mismatch: "
                f"{error}"
            )


    # ========================================================
    # Leave-one-rough-context-out validation
    # ========================================================

    predictions = []
    context_summaries = []


    for fold_index, test_context in enumerate(
        rough_contexts
    ):
        train_rows = [
            row
            for row in labels
            if row[
                "context_id"
            ] != test_context
        ]

        test_rows = [
            row
            for row in labels
            if row[
                "context_id"
            ] == test_context
        ]


        if len(
            test_rows
        ) != EXPECTED_PREFERENCES:
            raise RuntimeError(
                f"{test_context}: expected "
                "25 held-out preference rows."
            )


        context_mean, context_std = (
            fit_context_scaler(
                train_rows
            )
        )


        model, final_train_loss = (
            train_selector(
                train_rows,
                context_mean=context_mean,
                context_std=context_std,
            )
        )


        x_test, y_test = make_xy(
            test_rows,
            context_mean=context_mean,
            context_std=context_std,
        )


        model.eval()

        with torch.no_grad():
            predicted_lambda_t, (
                predicted_beta_t
            ) = model(
                torch.from_numpy(
                    x_test
                )
            )


        predicted_lambda = (
            predicted_lambda_t
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        predicted_beta = (
            predicted_beta_t
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        exact_count = 0
        canonical_total = 0
        canonical_exact = 0

        lambda_maes = []
        beta_maes = []
        score_excesses = []


        for index, row in enumerate(
            test_rows
        ):
            target_lambda = (
                y_test[
                    index
                ].astype(
                    np.float64
                )
            )

            target_beta = (
                BETA_FLOOR
                + BETA_SPAN
                * target_lambda
            )

            pred_lambda = (
                predicted_lambda[
                    index
                ]
            )

            pred_beta = (
                predicted_beta[
                    index
                ]
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


            nearest_index = int(
                np.argmin(
                    np.sum(
                        (
                            lattice
                            - pred_lambda[
                                None,
                                :
                            ]
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )

            projected_beta_name = (
                beta_names[
                    nearest_index
                ]
            )


            exact = int(
                projected_beta_name
                == row[
                    "selected_beta_name"
                ]
            )

            exact_count += exact


            is_canonical = (
                row[
                    "preference_kind"
                ]
                == "canonical"
            )

            if is_canonical:
                canonical_total += 1
                canonical_exact += exact


            w = np.asarray(
                [
                    finite(
                        row[column]
                    )
                    for column in (
                        PREFERENCE_COLUMNS
                    )
                ],
                dtype=np.float64,
            )


            projected_score = score_beta(
                context_id=test_context,
                beta_name=(
                    projected_beta_name
                ),
                w=w,
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
                    "Projected selector score "
                    "appears better than frozen "
                    "oracle beyond tolerance: "
                    f"{score_excess}"
                )

            score_excess = max(
                0.0,
                score_excess,
            )


            lambda_maes.append(
                lambda_mae
            )

            beta_maes.append(
                beta_mae
            )

            score_excesses.append(
                score_excess
            )


            predictions.append(
                {
                    "fold":
                        fold_index,

                    "test_context":
                        test_context,

                    "preference_name":
                        row[
                            "preference_name"
                        ],

                    "preference_kind":
                        row[
                            "preference_kind"
                        ],

                    "w_motion":
                        finite(
                            row[
                                "w_motion"
                            ]
                        ),

                    "w_stability":
                        finite(
                            row[
                                "w_stability"
                            ]
                        ),

                    "w_energy":
                        finite(
                            row[
                                "w_energy"
                            ]
                        ),

                    "target_beta_name":
                        row[
                            "selected_beta_name"
                        ],

                    "target_lambda_motion":
                        float(
                            target_lambda[0]
                        ),

                    "target_lambda_stability":
                        float(
                            target_lambda[1]
                        ),

                    "target_lambda_energy":
                        float(
                            target_lambda[2]
                        ),

                    "pred_lambda_motion":
                        float(
                            pred_lambda[0]
                        ),

                    "pred_lambda_stability":
                        float(
                            pred_lambda[1]
                        ),

                    "pred_lambda_energy":
                        float(
                            pred_lambda[2]
                        ),

                    "pred_beta_motion":
                        float(
                            pred_beta[0]
                        ),

                    "pred_beta_stability":
                        float(
                            pred_beta[1]
                        ),

                    "pred_beta_energy":
                        float(
                            pred_beta[2]
                        ),

                    "projected_beta_name":
                        projected_beta_name,

                    "projected_exact":
                        exact,

                    "lambda_mae":
                        lambda_mae,

                    "beta_mae":
                        beta_mae,

                    "oracle_score":
                        oracle_score,

                    "projected_score":
                        projected_score,

                    "score_excess":
                        score_excess,

                    "train_loss":
                        final_train_loss,
                }
            )


        context_summaries.append(
            {
                "fold":
                    fold_index,

                "test_context":
                    test_context,

                "train_contexts":
                    len(
                        set(
                            row[
                                "context_id"
                            ]
                            for row in train_rows
                        )
                    ),

                "train_rows":
                    len(
                        train_rows
                    ),

                "test_rows":
                    len(
                        test_rows
                    ),

                "final_train_loss":
                    final_train_loss,

                "lambda_mae_mean":
                    float(
                        np.mean(
                            lambda_maes
                        )
                    ),

                "beta_mae_mean":
                    float(
                        np.mean(
                            beta_maes
                        )
                    ),

                "projected_exact_accuracy":
                    (
                        exact_count
                        / len(
                            test_rows
                        )
                    ),

                "canonical_exact_accuracy":
                    (
                        canonical_exact
                        / canonical_total
                    ),

                "score_excess_mean":
                    float(
                        np.mean(
                            score_excesses
                        )
                    ),

                "score_excess_p95":
                    float(
                        np.quantile(
                            score_excesses,
                            0.95,
                        )
                    ),

                "score_excess_max":
                    float(
                        np.max(
                            score_excesses
                        )
                    ),

                "score_excess_le_0p01_fraction":
                    float(
                        np.mean(
                            np.asarray(
                                score_excesses
                            )
                            <= 0.01
                        )
                    ),

                "score_excess_le_0p05_fraction":
                    float(
                        np.mean(
                            np.asarray(
                                score_excesses
                            )
                            <= 0.05
                        )
                    ),
            }
        )


    # ========================================================
    # Aggregate CV diagnostics
    # ========================================================

    if len(
        predictions
    ) != EXPECTED_CV_ROWS:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_CV_ROWS} LOCO rows; "
            f"got {len(predictions)}"
        )


    tested_contexts = Counter(
        row[
            "test_context"
        ]
        for row in predictions
    )

    if set(
        tested_contexts
    ) != set(
        rough_contexts
    ):
        raise RuntimeError(
            "LOCO rough-context coverage "
            "mismatch."
        )

    if any(
        count != EXPECTED_PREFERENCES
        for count in (
            tested_contexts.values()
        )
    ):
        raise RuntimeError(
            "Each rough context must appear "
            "exactly once as a 25-row test fold."
        )


    overall_lambda_mae = float(
        np.mean(
            [
                row[
                    "lambda_mae"
                ]
                for row in predictions
            ]
        )
    )

    overall_beta_mae = float(
        np.mean(
            [
                row[
                    "beta_mae"
                ]
                for row in predictions
            ]
        )
    )

    overall_exact_accuracy = float(
        np.mean(
            [
                row[
                    "projected_exact"
                ]
                for row in predictions
            ]
        )
    )


    canonical_predictions = [
        row
        for row in predictions
        if row[
            "preference_kind"
        ] == "canonical"
    ]

    canonical_accuracy = float(
        np.mean(
            [
                row[
                    "projected_exact"
                ]
                for row in (
                    canonical_predictions
                )
            ]
        )
    )


    score_excess = np.asarray(
        [
            row[
                "score_excess"
            ]
            for row in predictions
        ],
        dtype=np.float64,
    )


    overall = {
        "lambda_mae_mean":
            overall_lambda_mae,

        "beta_mae_mean":
            overall_beta_mae,

        "projected_exact_accuracy":
            overall_exact_accuracy,

        "canonical_exact_accuracy":
            canonical_accuracy,

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
    }


    # ========================================================
    # Write only after all folds complete
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_PREDICTIONS,
        predictions,
    )

    write_csv(
        OUT_CONTEXT_SUMMARY,
        context_summaries,
    )


    manifest_out = {
        "schema":
            (
                "icra27_os_t5p5e1_"
                "selector_loco_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "source_labels":
            str(
                LABEL_CSV.relative_to(
                    ROOT
                )
            ),

        "source_labels_sha256":
            sha256_file(
                LABEL_CSV
            ),

        "source_atlas":
            str(
                ATLAS_CSV.relative_to(
                    ROOT
                )
            ),

        "source_atlas_sha256":
            sha256_file(
                ATLAS_CSV
            ),

        "heldout_used":
            False,

        "validation_protocol":
            (
                "leave-one-rough-TRAIN-context-out; "
                "flat and low_friction remain in "
                "training because each has only one "
                "unique deterministic physical context"
            ),

        "contexts_total":
            len(
                contexts
            ),

        "rough_contexts_tested":
            len(
                rough_contexts
            ),

        "cv_rows":
            len(
                predictions
            ),

        "input_dim":
            8,

        "context_dim":
            5,

        "preference_dim":
            3,

        "output_dim":
            3,

        "architecture":
            (
                "8 -> 64 SiLU -> 64 SiLU "
                "-> 3 logits -> softmax lambda"
            ),

        "beta_parameterization":
            (
                "beta = 0.15*1 + 0.55*lambda"
            ),

        "training_target":
            "oracle selected lambda",

        "loss":
            "mean squared error in lambda space",

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "model_seed":
            MODEL_SEED,

        "context_normalization":
            (
                "fold-local z-score fitted from "
                "unique TRAIN contexts only; "
                "w is not normalized"
            ),

        "physical_validation_projection":
            (
                "continuous predicted lambda is "
                "projected to the nearest frozen "
                "21-point beta lattice only for "
                "atlas-grounded validation metrics"
            ),

        "primary_validation_metric":
            (
                "augmented-Tchebycheff scalarization "
                "score excess after nearest-grid "
                "projection"
            ),

        "secondary_metrics":
            [
                "continuous lambda MAE",
                "continuous beta MAE",
                "nearest-grid exact beta accuracy",
                "canonical exact beta accuracy",
            ],

        "oracle_score_reconstruction_max_error":
            max_oracle_score_error,

        "overall":
            overall,

        "interpretation_guard":
            (
                "Exact beta-name accuracy is a "
                "secondary metric because discrete "
                "oracle labels can jump between "
                "near-equivalent Pareto points. "
                "Physical mission-score excess is "
                "the primary selector metric."
            ),

        "next_stage":
            (
                "Review TRAIN-only cross-context "
                "generalization. If acceptable, "
                "OS-T5.5e2 fits the final selector "
                "on all 500 TRAIN oracle labels; "
                "held-out contexts remain untouched."
            ),
    }


    OUT_MANIFEST.write_text(
        json.dumps(
            manifest_out,
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
        "ICRA27 OS-T5.5e1 OBJECTIVE SELECTOR "
        "ROUGH-CONTEXT LOCO VALIDATION"
    )
    print("=" * 118)

    print(
        "selector input : "
        "[c_OS(5), w(3)]"
    )

    print(
        "selector output: "
        "softmax lambda -> beta=0.15+0.55lambda"
    )

    print(
        "rough folds    :",
        len(
            rough_contexts
        ),
    )

    print(
        "CV rows        :",
        len(
            predictions
        ),
    )

    print()
    print("PER-CONTEXT")

    for row in (
        context_summaries
    ):
        print(
            f"  {row['test_context']:<20} "
            f"lamMAE="
            f"{row['lambda_mae_mean']:.4f} "
            f"betaMAE="
            f"{row['beta_mae_mean']:.4f} "
            f"exact="
            f"{row['projected_exact_accuracy']:.3f} "
            f"canon="
            f"{row['canonical_exact_accuracy']:.3f} "
            f"scoreEx="
            f"{row['score_excess_mean']:.4f} "
            f"p95="
            f"{row['score_excess_p95']:.4f}"
        )

    print()
    print("OVERALL")

    for key, value in (
        overall.items()
    ):
        print(
            f"  {key:<36}: "
            f"{value}"
        )

    print()
    print(
        "oracle score reconstruction max err:",
        max_oracle_score_error,
    )

    print()
    print("outputs:")
    print(" ", OUT_PREDICTIONS)
    print(" ", OUT_CONTEXT_SUMMARY)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5e1 "
        "selector LOCO validation: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
