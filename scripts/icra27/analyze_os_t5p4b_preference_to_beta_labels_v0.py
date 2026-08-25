from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PARETO_FRONT = (
    ROOT
    / "results/icra27"
    / "os_t5p4a_context_pareto_audit_v0"
    / "pareto_front.csv"
)

PARETO_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t5p4a_context_pareto_audit_v0"
    / "pareto_manifest.json"
)

GLOBAL_NORMALIZATION = (
    ROOT
    / "results/icra27"
    / "os_t5p3b_physical_atlas_normalization_v0"
    / "normalization_stats.json"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p4b_preference_to_beta_labels_v0"
)

LABELS_CSV = (
    OUT_DIR
    / "preferred_beta_labels.csv"
)

CONTEXT_SCALE_CSV = (
    OUT_DIR
    / "context_pareto_regret_scales.csv"
)

W_BANK_JSON = (
    OUT_DIR
    / "mission_preference_bank.json"
)

MANIFEST = (
    OUT_DIR
    / "preference_selection_manifest.json"
)

EXPECTED_CONTEXTS = 9
EXPECTED_BETAS = 21

GRID_DENOMINATOR = 5
AUGMENTATION_RHO = 0.01

SPAN_EPS = 1.0e-12
TIE_TOL = 1.0e-12


J_COLUMNS = (
    "J_motion_s_per_m",
    "J_stability",
    "J_energy_j_per_m",
)


def finite(
    value: Any,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return value


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing empty CSV: {path}"
        )

    fields: list[str] = []

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


def validate_w(
    w,
):
    arr = np.asarray(
        w,
        dtype=np.float64,
    )

    if arr.shape != (3,):
        raise RuntimeError(
            f"w must be 3D, got {arr.shape}"
        )

    if np.any(
        arr < -1e-12
    ):
        raise RuntimeError(
            f"Negative w: {arr}"
        )

    if not math.isclose(
        float(
            np.sum(arr)
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"w does not sum to one: {arr}"
        )

    return arr


def build_w_bank():
    bank = []

    # --------------------------------------------------------
    # Full external mission-preference simplex.
    #
    # Unlike beta, w is not constrained by the PPO training
    # support. It is an external preference variable.
    #
    # Integer simplex:
    #   i_M + i_S + i_E = 5
    #   w = i / 5
    # --------------------------------------------------------

    for i_m in range(
        GRID_DENOMINATOR,
        -1,
        -1,
    ):
        remaining = (
            GRID_DENOMINATOR
            - i_m
        )

        for i_s in range(
            remaining,
            -1,
            -1,
        ):
            i_e = (
                GRID_DENOMINATOR
                - i_m
                - i_s
            )

            w = validate_w(
                (
                    i_m
                    / GRID_DENOMINATOR,
                    i_s
                    / GRID_DENOMINATOR,
                    i_e
                    / GRID_DENOMINATOR,
                )
            )

            bank.append(
                {
                    "name":
                        (
                            f"wg_m{i_m * 20:03d}_"
                            f"s{i_s * 20:03d}_"
                            f"e{i_e * 20:03d}"
                        ),

                    "kind":
                        "simplex_grid",

                    "w":
                        w,
                }
            )

    if len(bank) != 21:
        raise RuntimeError(
            f"Expected 21 grid preferences, "
            f"got {len(bank)}"
        )

    # --------------------------------------------------------
    # Canonical mission preferences retained for clear
    # reporting and continuity with earlier objective-anchor
    # experiments.
    #
    # These numbers do NOT imply w == beta.
    # --------------------------------------------------------

    canonical = (
        (
            "w_balanced",
            (
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ),
        ),
        (
            "w_motion",
            (
                0.70,
                0.15,
                0.15,
            ),
        ),
        (
            "w_stability",
            (
                0.15,
                0.70,
                0.15,
            ),
        ),
        (
            "w_energy",
            (
                0.15,
                0.15,
                0.70,
            ),
        ),
    )

    for name, w_raw in canonical:
        bank.append(
            {
                "name":
                    name,

                "kind":
                    "canonical",

                "w":
                    validate_w(
                        w_raw
                    ),
            }
        )

    names = [
        item["name"]
        for item in bank
    ]

    if len(
        set(names)
    ) != len(names):
        raise RuntimeError(
            "Duplicate w names."
        )

    return bank


def compute_regrets(
    rows,
):
    values = np.asarray(
        [
            [
                finite(row[key])
                for key in J_COLUMNS
            ]
            for row in rows
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

    regrets = np.zeros_like(
        values
    )

    active = (
        span > SPAN_EPS
    )

    regrets[
        :,
        active
    ] = (
        values[
            :,
            active
        ]
        - ideal[
            active
        ]
    ) / span[
        active
    ]

    # Numerical contract.
    if np.any(
        regrets < -1e-10
    ) or np.any(
        regrets > 1.0 + 1e-10
    ):
        raise RuntimeError(
            "Regret outside [0,1]."
        )

    regrets = np.clip(
        regrets,
        0.0,
        1.0,
    )

    return (
        values,
        ideal,
        nadir,
        span,
        regrets,
        active,
    )


def select_beta(
    rows,
    regrets,
    w,
):
    w = validate_w(
        w
    )

    weighted = (
        regrets
        * w[None, :]
    )

    max_term = np.max(
        weighted,
        axis=1,
    )

    weighted_sum = np.sum(
        weighted,
        axis=1,
    )

    score = (
        max_term
        + AUGMENTATION_RHO
        * weighted_sum
    )

    minimum = float(
        np.min(score)
    )

    candidate_indices = [
        i
        for i, value
        in enumerate(score)
        if abs(
            float(value)
            - minimum
        ) <= TIE_TOL
    ]

    # Deterministic tie-breaking:
    #
    # 1) augmented score
    # 2) weighted sum
    # 3) unweighted total physical regret
    # 4) beta_name
    #
    # The latter two only resolve numerical/exact ties.
    index = min(
        candidate_indices,
        key=lambda i: (
            float(score[i]),
            float(
                weighted_sum[i]
            ),
            float(
                np.sum(
                    regrets[i]
                )
            ),
            str(
                rows[i][
                    "beta_name"
                ]
            ),
        ),
    )

    return {
        "index":
            index,

        "score":
            float(
                score[index]
            ),

        "max_term":
            float(
                max_term[index]
            ),

        "weighted_sum_term":
            float(
                weighted_sum[index]
            ),

        "regret":
            regrets[index],
    }


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    for path in (
        PARETO_FRONT,
        PARETO_MANIFEST,
        GLOBAL_NORMALIZATION,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    pareto_meta = json.loads(
        PARETO_MANIFEST.read_text()
    )

    if pareto_meta.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.4a Pareto audit not PASS."
        )

    if not bool(
        pareto_meta.get(
            "raw_normalized_fronts_match",
            False,
        )
    ):
        raise RuntimeError(
            "T5.4a raw/normalized Pareto "
            "front contract not satisfied."
        )

    global_norm = json.loads(
        GLOBAL_NORMALIZATION.read_text()
    )

    if global_norm.get(
        "status"
    ) != "FROZEN":
        raise RuntimeError(
            "T5.3b global normalization "
            "is not frozen."
        )

    if bool(
        global_norm.get(
            "heldout_used",
            True,
        )
    ):
        raise RuntimeError(
            "Held-out leakage detected."
        )

    with PARETO_FRONT.open(
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    contexts = sorted({
        (
            row[
                "pareto_context_group"
            ],
            int(
                float(
                    row[
                        "pareto_context_seed"
                    ]
                )
            ),
        )
        for row in rows
    })

    if len(contexts) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTEXTS} "
            f"contexts, got {len(contexts)}"
        )

    w_bank = build_w_bank()

    labels = []
    scale_rows = []

    context_selection_maps = {}

    for group, seed in contexts:
        front = [
            row
            for row in rows
            if (
                row[
                    "pareto_context_group"
                ] == group
                and int(
                    float(
                        row[
                            "pareto_context_seed"
                        ]
                    )
                ) == seed
            )
        ]

        front.sort(
            key=lambda row: (
                str(
                    row[
                        "beta_name"
                    ]
                )
            )
        )

        (
            values,
            ideal,
            nadir,
            span,
            regrets,
            active,
        ) = compute_regrets(
            front
        )

        scale_rows.append(
            {
                "group":
                    group,

                "seed":
                    seed,

                "pareto_count":
                    len(front),

                "J_motion_ideal":
                    ideal[0],

                "J_motion_nadir":
                    nadir[0],

                "J_motion_span":
                    span[0],

                "J_stability_ideal":
                    ideal[1],

                "J_stability_nadir":
                    nadir[1],

                "J_stability_span":
                    span[1],

                "J_energy_ideal":
                    ideal[2],

                "J_energy_nadir":
                    nadir[2],

                "J_energy_span":
                    span[2],

                "motion_regret_active":
                    int(
                        active[0]
                    ),

                "stability_regret_active":
                    int(
                        active[1]
                    ),

                "energy_regret_active":
                    int(
                        active[2]
                    ),
            }
        )

        selection_map = {}

        for preference in w_bank:
            w = preference["w"]

            result = select_beta(
                front,
                regrets,
                w,
            )

            i = result[
                "index"
            ]

            selected = front[i]

            r = result[
                "regret"
            ]

            label = {
                "group":
                    group,

                "seed":
                    seed,

                "w_name":
                    preference[
                        "name"
                    ],

                "w_kind":
                    preference[
                        "kind"
                    ],

                "w_motion":
                    float(
                        w[0]
                    ),

                "w_stability":
                    float(
                        w[1]
                    ),

                "w_energy":
                    float(
                        w[2]
                    ),

                "selected_beta_name":
                    selected[
                        "beta_name"
                    ],

                "beta_motion":
                    finite(
                        selected[
                            "beta_motion"
                        ]
                    ),

                "beta_stability":
                    finite(
                        selected[
                            "beta_stability"
                        ]
                    ),

                "beta_energy":
                    finite(
                        selected[
                            "beta_energy"
                        ]
                    ),

                "lambda_motion":
                    finite(
                        selected[
                            "lambda_motion"
                        ]
                    ),

                "lambda_stability":
                    finite(
                        selected[
                            "lambda_stability"
                        ]
                    ),

                "lambda_energy":
                    finite(
                        selected[
                            "lambda_energy"
                        ]
                    ),

                "J_motion":
                    finite(
                        selected[
                            "J_motion_s_per_m"
                        ]
                    ),

                "J_stability":
                    finite(
                        selected[
                            "J_stability"
                        ]
                    ),

                "J_energy":
                    finite(
                        selected[
                            "J_energy_j_per_m"
                        ]
                    ),

                "regret_motion":
                    float(
                        r[0]
                    ),

                "regret_stability":
                    float(
                        r[1]
                    ),

                "regret_energy":
                    float(
                        r[2]
                    ),

                "selection_score":
                    result[
                        "score"
                    ],

                "selection_max_term":
                    result[
                        "max_term"
                    ],

                "selection_weighted_sum_term":
                    result[
                        "weighted_sum_term"
                    ],

                "pareto_count":
                    len(front),
            }

            labels.append(
                label
            )

            selection_map[
                preference[
                    "name"
                ]
            ] = selected[
                "beta_name"
            ]

        context_selection_maps[
            (
                group,
                seed,
            )
        ] = selection_map

    expected_labels = (
        len(contexts)
        * len(w_bank)
    )

    if len(labels) != expected_labels:
        raise RuntimeError(
            "Preference-label count mismatch."
        )

    # --------------------------------------------------------
    # Deterministic flat/LF duplicated seed slots must map
    # every w to exactly the same beta.
    # --------------------------------------------------------

    deterministic_checks = {}

    for group in (
        "flat",
        "low_friction",
    ):
        maps = [
            context_selection_maps[
                (
                    g,
                    seed,
                )
            ]
            for g, seed in contexts
            if g == group
        ]

        first = maps[0]

        identical = all(
            item == first
            for item in maps[1:]
        )

        deterministic_checks[
            group
        ] = {
            "selection_maps_identical":
                bool(
                    identical
                ),

            "context_slots":
                len(maps),
        }

        if not identical:
            raise RuntimeError(
                f"{group}: deterministic "
                "preference mapping mismatch."
            )

    # --------------------------------------------------------
    # Selection diversity by context.
    # --------------------------------------------------------

    diversity_rows = {}

    for group, seed in contexts:
        context_labels = [
            row
            for row in labels
            if (
                row["group"] == group
                and int(
                    row["seed"]
                ) == seed
            )
        ]

        selected = sorted({
            row[
                "selected_beta_name"
            ]
            for row in context_labels
        })

        diversity_rows[
            f"{group}/{seed}"
        ] = {
            "unique_selected_betas":
                len(
                    selected
                ),

            "selected_beta_names":
                selected,
        }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        LABELS_CSV,
        labels,
    )

    write_csv(
        CONTEXT_SCALE_CSV,
        scale_rows,
    )

    w_payload = {
        "schema":
            "icra27_os_t5p4b_mission_preference_bank_v0",

        "semantics":
            (
                "External mission preference over "
                "relative attainable physical regret "
                "in motion, stability, and energy."
            ),

        "components":
            [
                "w_M",
                "w_S",
                "w_E",
            ],

        "simplex":
            "w_k >= 0; sum(w)=1",

        "preferences":
            [
                {
                    "name":
                        item["name"],

                    "kind":
                        item["kind"],

                    "w":
                        [
                            float(x)
                            for x in item[
                                "w"
                            ]
                        ],
                }
                for item in w_bank
            ],
    }

    W_BANK_JSON.write_text(
        json.dumps(
            w_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    manifest = {
        "schema":
            "icra27_os_t5p4b_preference_to_beta_labels_v0",

        "status":
            "FREEZE_PASS",

        "source_pareto_front":
            str(
                PARETO_FRONT.relative_to(
                    ROOT
                )
            ),

        "source_pareto_sha256":
            hashlib.sha256(
                PARETO_FRONT.read_bytes()
            ).hexdigest(),

        "global_normalization_preserved":
            str(
                GLOBAL_NORMALIZATION.relative_to(
                    ROOT
                )
            ),

        "global_normalization_role":
            (
                "Frozen cross-context reporting "
                "standardization; not replaced "
                "or modified by T5.4b."
            ),

        "selection_normalization":
            {
                "type":
                    "context_pareto_ideal_nadir_regret",

                "formula":
                    (
                        "(J_k - ideal_k(c)) / "
                        "(nadir_k(c) - ideal_k(c))"
                    ),

                "zero_span_rule":
                    "regret_k = 0",

                "scope":
                    "TRAIN-only Pareto front",

                "heldout_used":
                    False,
            },

        "scalarization":
            {
                "type":
                    "augmented_weighted_tchebycheff",

                "formula":
                    (
                        "max_k(w_k * r_k) + "
                        "rho * sum_k(w_k * r_k)"
                    ),

                "rho":
                    AUGMENTATION_RHO,

                "direction":
                    "lower_is_better",
            },

        "w_semantics":
            (
                "w specifies relative mission "
                "priority over attainable physical "
                "regret. It is distinct from beta, "
                "which is the internal planner "
                "conditioning variable."
            ),

        "w_count":
            len(
                w_bank
            ),

        "context_slots":
            len(
                contexts
            ),

        "labels":
            len(
                labels
            ),

        "deterministic_checks":
            deterministic_checks,

        "selection_diversity":
            diversity_rows,

        "next_stage":
            (
                "OS-T5.5 supervised Objective "
                "Selector learning: (c,w) -> beta*"
            ),
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=" * 108)
    print(
        "ICRA27 OS-T5.4b "
        "PREFERENCE -> PREFERRED BETA LABELS"
    )
    print("=" * 108)

    print()
    print(
        "contexts       :",
        len(contexts),
    )
    print(
        "w probes       :",
        len(w_bank),
    )
    print(
        "oracle labels  :",
        len(labels),
    )
    print(
        "rho            :",
        AUGMENTATION_RHO,
    )

    print()
    print("SELECTION DIVERSITY")

    for key, item in (
        diversity_rows.items()
    ):
        print(
            f"  {key:<22} "
            f"unique beta="
            f"{item['unique_selected_betas']}"
        )

    print()
    print("CANONICAL w -> beta")

    canonical_names = (
        "w_balanced",
        "w_motion",
        "w_stability",
        "w_energy",
    )

    for group, seed in contexts:
        mapping = (
            context_selection_maps[
                (
                    group,
                    seed,
                )
            ]
        )

        print(
            f"  {group:<14} "
            f"seed={seed:<6} "
            + " ".join(
                (
                    f"{name}="
                    f"{mapping[name]}"
                )
                for name
                in canonical_names
            )
        )

    print()
    print(
        "flat deterministic mapping:",
        deterministic_checks[
            "flat"
        ],
    )

    print(
        "LF deterministic mapping  :",
        deterministic_checks[
            "low_friction"
        ],
    )

    print()
    print("outputs:")
    print(" ", LABELS_CSV)
    print(" ", CONTEXT_SCALE_CSV)
    print(" ", W_BANK_JSON)
    print(" ", MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.4b "
        "preference selection contract: FREEZE PASS"
    )


if __name__ == "__main__":
    main()
