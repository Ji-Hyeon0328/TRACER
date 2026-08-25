from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

T55F2_PATH = (
    ROOT
    / "scripts/icra27"
    / "evaluate_os_t5p5f2_selector_train_extension_atlas_v0.py"
)

T55F3_PATH = (
    ROOT
    / "scripts/icra27"
    / "analyze_os_t5p5f3_expanded_physical_atlas_v0.py"
)

PROTOCOL_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p12b_untouched_eval_protocol_v0"
    / "untouched_eval_protocol_manifest.json"
)

PRECOMMIT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12d_heldout_selector_precommit_v0"
)

PRECOMMIT_MANIFEST = (
    PRECOMMIT_DIR
    / "heldout_selector_precommit_manifest.json"
)

PRECOMMIT_SURFACE = (
    PRECOMMIT_DIR
    / "precommitted_regret_surface.csv"
)

DEFAULT_OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p12e_heldout_physical_atlas_v0"
)

EXPECTED_REWARD_MODE = "tracer_cost_v4"

EXPECTED_CONTEXTS = 12
EXPECTED_BETAS = 21
EXPECTED_EPISODES = (
    EXPECTED_CONTEXTS
    * EXPECTED_BETAS
)

EXPECTED_SETTLING_STEPS = 5
EXPECTED_POLICY_HORIZON = 50

VAL = (1, 21, 16, 14)
TEST = (27, 2, 3, 19, 26)
HARD = (9, 23, 24)

ALL_SEEDS = (
    VAL
    + TEST
    + HARD
)

TOL = 1.0e-10


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
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        name
    ] = module

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


def write_csv_atomic(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
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

    os.replace(
        tmp,
        path,
    )


def write_json_atomic(
    path,
    payload,
):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    os.replace(
        tmp,
        path,
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def as_int(x):
    return int(
        float(x)
    )


def finite(x):
    y = float(x)

    if not math.isfinite(
        y
    ):
        raise ValueError(
            f"Non-finite: {x!r}"
        )

    return y


def maybe_finite(x):
    try:
        y = float(x)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        y
    ):
        return None

    return y


def split_name(seed):
    if seed in VAL:
        return "val"

    if seed in TEST:
        return "test"

    if seed in HARD:
        return "hard"

    raise RuntimeError(
        f"Unknown heldout seed: {seed}"
    )


def context_id_for_seed(seed):
    return (
        f"rough_seed_{int(seed)}"
    )


def compact_feasible(row):
    progress = maybe_finite(
        row.get(
            "progress_m"
        )
    )

    return (
        str(
            row.get(
                "status",
                "",
            )
        )
        == "success"

        and as_int(
            row.get(
                "m4_interventions",
                0,
            )
            or 0
        )
        == 0

        and progress is not None
        and progress > 0.0
    )


def infeasible_reason(row):
    reasons = []

    if str(
        row.get(
            "status",
            "",
        )
    ) != "success":
        reasons.append(
            "status_not_success"
        )

    if as_int(
        row.get(
            "m4_interventions",
            0,
        )
        or 0
    ) != 0:
        reasons.append(
            "m4_intervention"
        )

    progress = maybe_finite(
        row.get(
            "progress_m"
        )
    )

    if (
        progress is None
        or progress <= 0.0
    ):
        reasons.append(
            "nonpositive_progress"
        )

    if not reasons:
        return ""

    return "+".join(
        reasons
    )


def verify_beta_against_precommit(
    beta_points,
):
    """
    Verify candidate SUPPORT, not local index ordering.

    T6.12a/T6.12d inherited a lexicographic beta_name
    ordering from the final selector training artifact.

    T5.5f2 build_beta_lattice() uses its historical
    barycentric enumeration order.

    These orderings need not match.  The frozen beta
    identity is therefore beta_name + beta vector.
    beta_index is local bookkeeping only.
    """

    rows = read_csv(
        PRECOMMIT_SURFACE
    )

    if len(rows) != (
        EXPECTED_CONTEXTS
        * EXPECTED_BETAS
    ):
        raise RuntimeError(
            "Precommit surface row-count drift: "
            f"{len(rows)}"
        )


    canonical = {}

    for point in beta_points:

        name = point[
            "name"
        ]

        if name in canonical:
            raise RuntimeError(
                f"Duplicate canonical beta name: "
                f"{name}"
            )

        canonical[
            name
        ] = tuple(
            float(x)
            for x in point[
                "beta"
            ]
        )


    if len(canonical) != EXPECTED_BETAS:
        raise RuntimeError(
            "Canonical beta support is not 21."
        )


    by_context = {}

    for row in rows:

        cid = row[
            "context_id"
        ]

        by_context.setdefault(
            cid,
            [],
        ).append(
            row
        )


    if len(by_context) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Precommit context-count drift: "
            f"{len(by_context)}"
        )


    for cid, context_rows in (
        by_context.items()
    ):

        if len(context_rows) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{cid}: expected 21 "
                "precommit beta rows; got "
                f"{len(context_rows)}"
            )


        names = [
            row[
                "beta_name"
            ]
            for row in context_rows
        ]


        if len(set(names)) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{cid}: duplicate beta names "
                "in precommit surface."
            )


        if set(names) != set(
            canonical
        ):
            missing = sorted(
                set(canonical)
                - set(names)
            )

            extra = sorted(
                set(names)
                - set(canonical)
            )

            raise RuntimeError(
                f"{cid}: beta support drift.\n"
                f"  missing={missing}\n"
                f"  extra={extra}"
            )


        # Local precommit indices must still form
        # a valid 0..20 permutation, but their order
        # is NOT required to equal the T5.5f2 order.
        local_indices = sorted(
            int(
                row[
                    "beta_index"
                ]
            )
            for row in context_rows
        )

        if local_indices != list(
            range(
                EXPECTED_BETAS
            )
        ):
            raise RuntimeError(
                f"{cid}: invalid local "
                "precommit beta-index set."
            )


        for row in context_rows:

            name = row[
                "beta_name"
            ]

            expected = canonical[
                name
            ]

            observed = (
                finite(
                    row[
                        "beta_motion"
                    ]
                ),

                finite(
                    row[
                        "beta_stability"
                    ]
                ),

                finite(
                    row[
                        "beta_energy"
                    ]
                ),
            )


            if any(
                abs(
                    expected[k]
                    - observed[k]
                )
                > 1.0e-12
                for k in range(3)
            ):
                raise RuntimeError(
                    f"{cid}/{name}: "
                    "beta-vector drift.\n"
                    f"  canonical={expected}\n"
                    f"  precommit={observed}"
                )


    print(
        "beta support regression       :",
        (
            "PASS "
            "(21 names/vectors; "
            "local index order ignored)"
        ),
    )

def extract_physical(
    *,
    compact,
    raw_path,
    f3,
    roll_unsafe,
    pitch_unsafe,
    slip_deadband,
):
    seed = as_int(
        compact[
            "seed"
        ]
    )

    context_id = (
        compact[
            "context_id"
        ]
    )

    feasible = compact_feasible(
        compact
    )

    base = {
        "context_id":
            context_id,

        "seed":
            seed,

        "split_group":
            compact[
                "split_group"
            ],

        "terrain":
            compact.get(
                "terrain",
                "rough_perlin",
            ),

        "beta_index":
            as_int(
                compact[
                    "beta_index"
                ]
            ),

        "beta_name":
            compact[
                "beta_name"
            ],

        "beta_motion":
            finite(
                compact[
                    "beta_motion"
                ]
            ),

        "beta_stability":
            finite(
                compact[
                    "beta_stability"
                ]
            ),

        "beta_energy":
            finite(
                compact[
                    "beta_energy"
                ]
            ),

        "lambda_motion":
            finite(
                compact[
                    "lambda_motion"
                ]
            ),

        "lambda_stability":
            finite(
                compact[
                    "lambda_stability"
                ]
            ),

        "lambda_energy":
            finite(
                compact[
                    "lambda_energy"
                ]
            ),

        "status":
            compact[
                "status"
            ],

        "success":
            int(
                str(
                    compact[
                        "status"
                    ]
                )
                == "success"
            ),

        "m4_interventions":
            as_int(
                compact.get(
                    "m4_interventions",
                    0,
                )
                or 0
            ),

        "feasible":
            int(
                feasible
            ),

        "infeasible_reason":
            (
                ""
                if feasible
                else infeasible_reason(
                    compact
                )
            ),

        "raw_log_relpath":
            compact[
                "raw_log_relpath"
            ],

        "source_stage":
            "T6.12e_heldout",
    }


    if not feasible:
        base.update(
            {
                "policy_steps":
                    compact.get(
                        "policy_steps",
                        "",
                    ),

                "settling_steps":
                    "",

                "decision_time_s":
                    compact.get(
                        "decision_time_s",
                        "",
                    ),

                "progress_m":
                    compact.get(
                        "progress_m",
                        "",
                    ),

                "energy_abs_j":
                    compact.get(
                        "energy_abs_j",
                        "",
                    ),

                "roll_rms_rad":
                    "",

                "pitch_rms_rad":
                    "",

                "max_abs_roll_rad":
                    "",

                "max_abs_pitch_rad":
                    "",

                "established_slip_fraction_ge_0p002":
                    "",

                "established_slip_fraction_ge_0p05":
                    "",

                "established_slip_fraction_ge_0p10":
                    "",

                "J_motion_s_per_m":
                    "",

                "J_stability_attitude":
                    "",

                "J_stability_slip":
                    "",

                "J_stability":
                    "",

                "J_stability_dominant":
                    "",

                "J_energy_j_per_m":
                    "",
            }
        )

        return base


    steps = f3.phys.load_steps(
        raw_path
    )

    policy_steps = as_int(
        compact[
            "policy_steps"
        ]
    )

    settling_steps = (
        len(
            steps
        )
        - policy_steps
    )

    if settling_steps != (
        EXPECTED_SETTLING_STEPS
    ):
        raise RuntimeError(
            f"{raw_path}: settling="
            f"{settling_steps}, expected="
            f"{EXPECTED_SETTLING_STEPS}"
        )

    boundary = steps[
        settling_steps - 1
    ]

    policy = steps[
        settling_steps:
    ]

    if len(
        policy
    ) != policy_steps:
        raise RuntimeError(
            f"{raw_path}: policy-window "
            "slicing mismatch."
        )

    if not policy:
        raise RuntimeError(
            f"{raw_path}: empty policy window."
        )

    final = policy[
        -1
    ]


    # --------------------------------------------------------
    # Frozen policy-window decision time.
    # --------------------------------------------------------

    decision_time = sum(
        finite(
            x[
                "decision_dt_s"
            ]
        )
        for x in policy
    )

    compact_time = finite(
        compact[
            "decision_time_s"
        ]
    )

    if abs(
        decision_time
        - compact_time
    ) > 1.0e-7:
        raise RuntimeError(
            f"{raw_path}: decision-time "
            "reconstruction mismatch."
        )


    # --------------------------------------------------------
    # Frozen progress semantics:
    # boundary goal distance - final goal distance.
    # --------------------------------------------------------

    progress_m = (
        finite(
            boundary[
                "goal_distance"
            ]
        )
        - finite(
            final[
                "goal_distance"
            ]
        )
    )

    compact_progress = finite(
        compact[
            "progress_m"
        ]
    )

    if abs(
        progress_m
        - compact_progress
    ) > 1.0e-7:
        raise RuntimeError(
            f"{raw_path}: progress "
            "reconstruction mismatch."
        )

    progress_integrated = sum(
        finite(
            x[
                "reward_components"
            ][
                "progress_rate_mps"
            ]
        )
        * finite(
            x[
                "decision_dt_s"
            ]
        )
        for x in policy
    )

    if abs(
        progress_m
        - progress_integrated
    ) > 1.0e-7:
        raise RuntimeError(
            f"{raw_path}: integrated "
            "progress mismatch."
        )

    if progress_m <= 0.0:
        raise RuntimeError(
            f"{raw_path}: feasible row "
            "has nonpositive progress."
        )


    # --------------------------------------------------------
    # Frozen mechanical absolute work.
    # --------------------------------------------------------

    energy_abs_j = sum(
        finite(
            x[
                "reward_components"
            ][
                "energy_abs_j"
            ]
        )
        for x in policy
    )

    compact_energy = finite(
        compact[
            "energy_abs_j"
        ]
    )

    if abs(
        energy_abs_j
        - compact_energy
    ) > 1.0e-6:
        raise RuntimeError(
            f"{raw_path}: energy "
            "reconstruction mismatch."
        )


    # --------------------------------------------------------
    # Frozen physical attitude reconstruction.
    # --------------------------------------------------------

    abs_roll = []
    abs_pitch = []

    for x in policy:
        rc = x[
            "reward_components"
        ]

        abs_roll.append(
            finite(
                rc[
                    "roll_fraction_of_unsafe"
                ]
            )
            * finite(
                rc[
                    "m4_roll_unsafe_rad"
                ]
            )
        )

        abs_pitch.append(
            finite(
                rc[
                    "pitch_fraction_of_unsafe"
                ]
            )
            * finite(
                rc[
                    "m4_pitch_unsafe_rad"
                ]
            )
        )


    roll_rms = f3.phys.rms(
        abs_roll
    )

    pitch_rms = f3.phys.rms(
        abs_pitch
    )

    max_abs_roll = max(
        abs_roll
    )

    max_abs_pitch = max(
        abs_pitch
    )


    # --------------------------------------------------------
    # Frozen established-contact slip histogram difference.
    # --------------------------------------------------------

    boundary_slip = (
        boundary[
            "eval_stance_slip"
        ]
    )

    final_slip = (
        final[
            "eval_stance_slip"
        ]
    )


    slip_fraction = (
        f3.phys.policy_tail_fraction(
            boundary_slip=(
                boundary_slip
            ),
            final_slip=(
                final_slip
            ),
            threshold=(
                slip_deadband
            ),
        )
    )

    slip_tail_005 = (
        f3.phys.policy_tail_fraction(
            boundary_slip=(
                boundary_slip
            ),
            final_slip=(
                final_slip
            ),
            threshold=0.05,
        )
    )

    slip_tail_010 = (
        f3.phys.policy_tail_fraction(
            boundary_slip=(
                boundary_slip
            ),
            final_slip=(
                final_slip
            ),
            threshold=0.10,
        )
    )


    # --------------------------------------------------------
    # Frozen T5.2b objectives.
    # --------------------------------------------------------

    j_motion = (
        decision_time
        / progress_m
    )

    j_stability_attitude = max(
        roll_rms
        / roll_unsafe,

        pitch_rms
        / pitch_unsafe,
    )

    j_stability_slip = (
        slip_fraction
    )

    j_stability = max(
        j_stability_attitude,
        j_stability_slip,
    )

    j_energy = (
        energy_abs_j
        / progress_m
    )


    if (
        j_stability_attitude
        > j_stability_slip
        + TOL
    ):
        dominant = "attitude"

    elif (
        j_stability_slip
        > j_stability_attitude
        + TOL
    ):
        dominant = "slip"

    else:
        dominant = "tie"


    base.update(
        {
            "policy_steps":
                policy_steps,

            "settling_steps":
                settling_steps,

            "decision_time_s":
                decision_time,

            "progress_m":
                progress_m,

            "energy_abs_j":
                energy_abs_j,

            "roll_rms_rad":
                roll_rms,

            "pitch_rms_rad":
                pitch_rms,

            "max_abs_roll_rad":
                max_abs_roll,

            "max_abs_pitch_rad":
                max_abs_pitch,

            "established_slip_fraction_ge_0p002":
                slip_fraction,

            "established_slip_fraction_ge_0p05":
                slip_tail_005,

            "established_slip_fraction_ge_0p10":
                slip_tail_010,

            "J_motion_s_per_m":
                j_motion,

            "J_stability_attitude":
                j_stability_attitude,

            "J_stability_slip":
                j_stability_slip,

            "J_stability":
                j_stability,

            "J_stability_dominant":
                dominant,

            "J_energy_j_per_m":
                j_energy,
        }
    )

    return base


def next_attempt_dir(
    block_dir,
):
    existing = sorted(
        block_dir.glob(
            "attempt_*"
        )
    )

    used = []

    for path in existing:
        try:
            used.append(
                int(
                    path.name.split(
                        "_"
                    )[
                        -1
                    ]
                )
            )
        except ValueError:
            continue

    index = (
        max(
            used,
            default=0,
        )
        + 1
    )

    return (
        block_dir
        / f"attempt_{index:03d}"
    )


def load_complete_block(
    *,
    block_dir,
    beta_index,
    beta_name,
):
    marker = (
        block_dir
        / "block_complete.json"
    )

    if not marker.exists():
        return None


    payload = json.loads(
        marker.read_text()
    )

    if payload.get(
        "status"
    ) != "BLOCK_COMPLETE":
        raise RuntimeError(
            f"{marker}: invalid block status."
        )

    if int(
        payload[
            "beta_index"
        ]
    ) != beta_index:
        raise RuntimeError(
            f"{marker}: beta-index drift."
        )

    if payload[
        "beta_name"
    ] != beta_name:
        raise RuntimeError(
            f"{marker}: beta-name drift."
        )


    compact_path = (
        ROOT
        / payload[
            "compact_csv"
        ]
    )

    physical_path = (
        ROOT
        / payload[
            "physical_csv"
        ]
    )


    if not compact_path.exists():
        raise FileNotFoundError(
            compact_path
        )

    if not physical_path.exists():
        raise FileNotFoundError(
            physical_path
        )


    if sha256(
        compact_path
    ) != payload[
        "compact_sha256"
    ]:
        raise RuntimeError(
            f"{compact_path}: hash drift."
        )

    if sha256(
        physical_path
    ) != payload[
        "physical_sha256"
    ]:
        raise RuntimeError(
            f"{physical_path}: hash drift."
        )


    compact_rows = read_csv(
        compact_path
    )

    physical_rows = read_csv(
        physical_path
    )


    if len(
        compact_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"{block_dir}: compact row "
            "count drift."
        )

    if len(
        physical_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            f"{block_dir}: physical row "
            "count drift."
        )


    seeds = [
        as_int(
            row[
                "seed"
            ]
        )
        for row in compact_rows
    ]

    if seeds != list(
        ALL_SEEDS
    ):
        raise RuntimeError(
            f"{block_dir}: heldout seed "
            "order drift."
        )


    return (
        compact_rows,
        physical_rows,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--base-port",
        type=int,
        default=62010,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = (
        args.out_dir.resolve()
    )


    # ========================================================
    # Load frozen source implementations.
    # ========================================================

    for path in (
        T55F2_PATH,
        T55F3_PATH,
        PROTOCOL_MANIFEST,
        PRECOMMIT_MANIFEST,
        PRECOMMIT_SURFACE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    f2 = load_module(
        T55F2_PATH,
        "os_t6p12e_t55f2",
    )

    f3 = load_module(
        T55F3_PATH,
        "os_t6p12e_t55f3",
    )


    beta_points = (
        f2.build_beta_lattice()
    )

    if len(
        beta_points
    ) != EXPECTED_BETAS:
        raise RuntimeError(
            "Expected exactly 21 beta points."
        )


    verify_beta_against_precommit(
        beta_points
    )


    protocol = json.loads(
        PROTOCOL_MANIFEST.read_text()
    )

    precommit = json.loads(
        PRECOMMIT_MANIFEST.read_text()
    )


    if protocol.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12b protocol is not FREEZE_PASS."
        )

    if precommit.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.12d precommit is not FREEZE_PASS."
        )

    if bool(
        precommit.get(
            "heldout_physical_outcomes_used",
            True,
        )
    ):
        raise RuntimeError(
            "T6.12d says heldout outcomes "
            "were already used."
        )


    expected_splits = protocol[
        "evaluation_splits"
    ]

    if expected_splits[
        "val"
    ] != list(VAL):
        raise RuntimeError(
            "VAL split drift."
        )

    if expected_splits[
        "test"
    ] != list(TEST):
        raise RuntimeError(
            "TEST split drift."
        )

    if expected_splits[
        "hard"
    ] != list(HARD):
        raise RuntimeError(
            "HARD split drift."
        )


    # Verify all frozen T6.12d artifacts before revealing outcomes.
    for name, expected_hash in (
        precommit[
            "artifact_hashes"
        ].items()
    ):
        path = (
            PRECOMMIT_DIR
            / name
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        if sha256(
            path
        ) != expected_hash:
            raise RuntimeError(
                f"Precommit artifact hash drift: "
                f"{path}"
            )


    checkpoint = (
        f2.DEFAULT_CHECKPOINT.resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    if checkpoint.name != (
        "checkpoint_update_0030.pt"
    ):
        raise RuntimeError(
            "T6.12e must use frozen u30 "
            "checkpoint."
        )


    # ========================================================
    # Frozen physical contract.
    # ========================================================

    contract = json.loads(
        f3.CONTRACT.read_text()
    )

    if contract.get(
        "schema"
    ) != (
        "icra27_os_t5p2b_"
        "physical_preference_contract_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.2b physical "
            "contract schema."
        )


    stability_contract = (
        contract[
            "primary_objectives"
        ][
            "stability"
        ]
    )


    roll_unsafe = finite(
        stability_contract[
            "attitude_component"
        ][
            "roll_unsafe_rad"
        ]
    )

    pitch_unsafe = finite(
        stability_contract[
            "attitude_component"
        ][
            "pitch_unsafe_rad"
        ]
    )

    slip_deadband = finite(
        stability_contract[
            "slip_component"
        ][
            "threshold_mps"
        ]
    )


    if not math.isclose(
        slip_deadband,
        0.002,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Frozen slip threshold drift."
        )


    print("=" * 126)
    print(
        "ICRA27 OS-T6.12e HELDOUT "
        "PHYSICAL ATLAS PLAN"
    )
    print("=" * 126)

    print(
        "checkpoint          :",
        checkpoint,
    )

    print(
        "contexts            :",
        EXPECTED_CONTEXTS,
    )

    print(
        "beta points         :",
        EXPECTED_BETAS,
    )

    print(
        "episodes            :",
        EXPECTED_EPISODES,
    )

    print(
        "VAL / TEST / HARD   :",
        "4 / 5 / 3",
    )

    print(
        "feasible            :",
        "success AND no M4 AND progress>0",
    )

    print(
        "infeasible scoring  :",
        "true Q = +inf; NO selector reselection",
    )

    print(
        "zero-feasible ctx   :",
        "report separately; oracle undefined",
    )

    print(
        "precommit hash      :",
        sha256(
            PRECOMMIT_MANIFEST
        ),
    )

    print("=" * 126)


    if args.dry_run:
        print()
        print(
            "[ICRA27] OS-T6.12e atlas plan: "
            "DRY-RUN PASS"
        )
        return


    # ========================================================
    # Output namespace / resume guard.
    # ========================================================

    plan_path = (
        out_dir
        / "atlas_plan.json"
    )

    final_compact = (
        out_dir
        / "episodes.csv"
    )

    partial_compact = (
        out_dir
        / "episodes_partial.csv"
    )

    final_physical = (
        out_dir
        / "heldout_physical_beta_response_atlas.csv"
    )

    partial_physical = (
        out_dir
        / "physical_atlas_partial.csv"
    )

    manifest_path = (
        out_dir
        / "heldout_physical_atlas_manifest.json"
    )


    plan = {
        "schema":
            "icra27_os_t6p12e_heldout_physical_atlas_plan_v0",

        "selector_precommit":
            str(
                PRECOMMIT_MANIFEST.relative_to(
                    ROOT
                )
            ),

        "selector_precommit_sha256":
            sha256(
                PRECOMMIT_MANIFEST
            ),

        "selector_outputs_frozen_before_outcomes":
            True,

        "checkpoint":
            str(
                checkpoint.relative_to(
                    ROOT
                )
            ),

        "reward_mode":
            EXPECTED_REWARD_MODE,

        "terrain":
            "rough_perlin",

        "evaluation_splits":
            {
                "val":
                    list(VAL),

                "test":
                    list(TEST),

                "hard":
                    list(HARD),
            },

        "seed_order":
            list(
                ALL_SEEDS
            ),

        "beta_points":
            beta_points,

        "expected_contexts":
            EXPECTED_CONTEXTS,

        "expected_beta_points":
            EXPECTED_BETAS,

        "expected_episodes":
            EXPECTED_EPISODES,

        "settling_steps":
            EXPECTED_SETTLING_STEPS,

        "policy_horizon":
            EXPECTED_POLICY_HORIZON,

        "feasibility_contract":
            (
                "success AND no M4 AND "
                "positive progress"
            ),

        "infeasible_candidate_scoring_contract":
            (
                "In T6.12f, an infeasible beta "
                "candidate has true Q=+inf. "
                "If the precommitted selector chose "
                "an infeasible beta, it remains a "
                "selector failure and no alternative "
                "beta may be substituted."
            ),

        "zero_feasible_context_contract":
            (
                "If no beta is feasible for a context, "
                "the context is reported as a "
                "lattice-infeasible evaluation case; "
                "oracle Q/rank are undefined and the "
                "context is not silently replaced or "
                "removed from the reported population."
            ),

        "physical_objective_source":
            str(
                f3.CONTRACT.relative_to(
                    ROOT
                )
            ),

        "raw_log_pairing":
            (
                "Within each complete beta block, "
                "compact seed-order rows are zipped "
                "to sorted episode_*.jsonl files, "
                "matching the frozen T5.5c/f3 "
                "provenance convention."
            ),

        "resume_semantics":
            (
                "Only beta blocks with a valid "
                "block_complete.json are reused. "
                "An interrupted beta block is never "
                "overwritten; resume creates a fresh "
                "attempt_NNN directory."
            ),
    }


    if out_dir.exists():

        if not args.resume:
            raise RuntimeError(
                "Output directory already exists. "
                "Refusing overwrite. Use --resume "
                "only for this exact T6.12e run:\n"
                f"{out_dir}"
            )

        if not plan_path.exists():
            raise RuntimeError(
                "Cannot resume: atlas_plan.json "
                "is missing."
            )


        previous_plan = json.loads(
            plan_path.read_text()
        )

        for key in (
            "schema",
            "selector_precommit_sha256",
            "checkpoint",
            "reward_mode",
            "seed_order",
            "beta_points",
            "feasibility_contract",
            "infeasible_candidate_scoring_contract",
        ):
            if previous_plan[
                key
            ] != plan[
                key
            ]:
                raise RuntimeError(
                    f"Resume plan drift at key={key}"
                )


        if (
            manifest_path.exists()
            and final_compact.exists()
            and final_physical.exists()
        ):
            manifest = json.loads(
                manifest_path.read_text()
            )

            if manifest.get(
                "status"
            ) == "FREEZE_PASS":
                print()
                print(
                    "[ICRA27] OS-T6.12e already "
                    "completed; refusing to rerun."
                )
                return

    else:

        if args.resume:
            raise RuntimeError(
                "Cannot --resume because output "
                "directory does not exist."
            )

        out_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        # This is written BEFORE the first heldout
        # physical outcome is generated.
        write_json_atomic(
            plan_path,
            plan,
        )


    # ========================================================
    # Exact frozen evaluator setup.
    # ========================================================

    phase = f2.load_module(
        f2.T49_PATH,
        "os_t6p12e_t49_v4_base",
    )


    if str(
        phase.EVAL_REWARD_MODE
    ) != EXPECTED_REWARD_MODE:
        raise RuntimeError(
            "Unexpected T4.9b reward mode."
        )


    historical_base_mode = str(
        phase.base.EVAL_REWARD_MODE
    )

    if historical_base_mode == (
        "tracer_cost_v2"
    ):
        phase.base.EVAL_REWARD_MODE = (
            EXPECTED_REWARD_MODE
        )

    elif historical_base_mode != (
        EXPECTED_REWARD_MODE
    ):
        raise RuntimeError(
            "Unexpected common evaluator "
            f"reward mode: "
            f"{historical_base_mode!r}"
        )


    if str(
        phase.base.EVAL_REWARD_MODE
    ) != EXPECTED_REWARD_MODE:
        raise RuntimeError(
            "Could not promote common "
            "evaluator to tracer_cost_v4."
        )


    if int(
        phase.base.SETTLING_STEPS
    ) != EXPECTED_SETTLING_STEPS:
        raise RuntimeError(
            "Settling-step contract drift."
        )

    if int(
        phase.base.POLICY_HORIZON
    ) != EXPECTED_POLICY_HORIZON:
        raise RuntimeError(
            "Policy-horizon contract drift."
        )


    policy = phase.base.load_policy(
        name="beta_conditioned",
        path=checkpoint,
        expected_reward_mode=(
            EXPECTED_REWARD_MODE
        ),
    )

    phase.validate_checkpoint_beta_bank(
        policy[
            "payload"
        ]
    )


    # ========================================================
    # Beta-block execution.
    # ========================================================

    all_compact = []
    all_physical = []


    for beta_index, point in enumerate(
        beta_points
    ):

        beta_name = point[
            "name"
        ]

        beta = point[
            "beta"
        ]


        block_dir = (
            out_dir
            / "blocks"
            / (
                f"{beta_index:02d}_"
                f"{beta_name}"
            )
        )

        block_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        completed = load_complete_block(
            block_dir=block_dir,
            beta_index=beta_index,
            beta_name=beta_name,
        )


        if completed is not None:

            compact_rows, physical_rows = (
                completed
            )

            all_compact.extend(
                compact_rows
            )

            all_physical.extend(
                physical_rows
            )

            print()
            print(
                f"BETA {beta_index + 1:02d}/"
                f"{EXPECTED_BETAS}: "
                f"{beta_name} — REUSE COMPLETE BLOCK"
            )

            continue


        attempt_dir = next_attempt_dir(
            block_dir
        )

        attempt_dir.mkdir(
            parents=False,
            exist_ok=False,
        )


        log_dir = (
            attempt_dir
            / "env_logs"
        )

        compact_partial = (
            attempt_dir
            / "episodes_partial.csv"
        )

        compact_done = (
            attempt_dir
            / "episodes.csv"
        )

        physical_done = (
            attempt_dir
            / "physical.csv"
        )


        command_port = (
            int(
                args.base_port
            )
            + 100
            * beta_index
        )


        print()
        print("=" * 118)
        print(
            f"BETA {beta_index + 1:02d}/"
            f"{EXPECTED_BETAS}: "
            f"{beta_name} "
            f"[{beta[0]:.3f},"
            f"{beta[1]:.3f},"
            f"{beta[2]:.3f}]"
        )

        print(
            "attempt:",
            attempt_dir.name,
            "port:",
            command_port,
        )

        print("=" * 118)


        env = phase.make_env(
            terrain="rough_perlin",
            beta=beta,
            command_port=command_port,
            log_dir=log_dir,
        )


        block_compact = []


        try:
            for seed in ALL_SEEDS:

                row = phase.base.run_episode(
                    env,
                    policy_name=(
                        "beta_conditioned"
                    ),
                    checkpoint=(
                        checkpoint
                    ),
                    trained_reward_mode=(
                        policy[
                            "trained_reward_mode"
                        ]
                    ),
                    model=(
                        policy[
                            "model"
                        ]
                    ),
                    group="heldout",
                    terrain="rough_perlin",
                    seed=int(
                        seed
                    ),
                )


                row = phase.annotate_row(
                    row,
                    beta_name=(
                        beta_name
                    ),
                    beta=beta,
                )


                row[
                    "beta_index"
                ] = beta_index

                row[
                    "lambda_motion"
                ] = point[
                    "lambda_motion"
                ]

                row[
                    "lambda_stability"
                ] = point[
                    "lambda_stability"
                ]

                row[
                    "lambda_energy"
                ] = point[
                    "lambda_energy"
                ]

                row[
                    "context_id"
                ] = context_id_for_seed(
                    seed
                )

                row[
                    "split_group"
                ] = split_name(
                    seed
                )


                block_compact.append(
                    row
                )


                write_csv_atomic(
                    compact_partial,
                    block_compact,
                )


                print(
                    f"  {row['context_id']:<16} "
                    f"{row['split_group']:<4} "
                    f"status="
                    f"{row['status']:<18} "
                    f"steps="
                    f"{row['policy_steps']:<3} "
                    f"progress="
                    f"{row.get('progress_m')} "
                    f"M4="
                    f"{row.get('m4_interventions', 0)}"
                )


        finally:
            env.close()


        if len(
            block_compact
        ) != EXPECTED_CONTEXTS:
            raise RuntimeError(
                f"{beta_name}: beta block "
                f"has {len(block_compact)} rows; "
                f"expected {EXPECTED_CONTEXTS}."
            )


        raw_files = sorted(
            log_dir.glob(
                "episode_*.jsonl"
            )
        )


        if len(
            raw_files
        ) != EXPECTED_CONTEXTS:
            raise RuntimeError(
                f"{beta_name}: "
                f"{len(raw_files)} raw logs "
                f"vs {EXPECTED_CONTEXTS} "
                "compact rows."
            )


        # Frozen provenance convention:
        # compact seed-order rows zip to sorted raw logs.
        for row, raw_path in zip(
            block_compact,
            raw_files,
        ):
            row[
                "raw_log_relpath"
            ] = str(
                raw_path.relative_to(
                    ROOT
                )
            )


        write_csv_atomic(
            compact_done,
            block_compact,
        )


        block_physical = []


        for row, raw_path in zip(
            block_compact,
            raw_files,
        ):

            physical = extract_physical(
                compact=row,
                raw_path=raw_path,
                f3=f3,
                roll_unsafe=roll_unsafe,
                pitch_unsafe=pitch_unsafe,
                slip_deadband=slip_deadband,
            )

            block_physical.append(
                physical
            )


        write_csv_atomic(
            physical_done,
            block_physical,
        )


        block_marker = {
            "schema":
                "icra27_os_t6p12e_beta_block_complete_v0",

            "status":
                "BLOCK_COMPLETE",

            "beta_index":
                beta_index,

            "beta_name":
                beta_name,

            "attempt":
                attempt_dir.name,

            "compact_csv":
                str(
                    compact_done.relative_to(
                        ROOT
                    )
                ),

            "physical_csv":
                str(
                    physical_done.relative_to(
                        ROOT
                    )
                ),

            "compact_sha256":
                sha256(
                    compact_done
                ),

            "physical_sha256":
                sha256(
                    physical_done
                ),

            "raw_log_count":
                len(
                    raw_files
                ),

            "contexts":
                EXPECTED_CONTEXTS,

            "feasible_rows":
                sum(
                    int(
                        row[
                            "feasible"
                        ]
                    )
                    for row in block_physical
                ),
        }


        write_json_atomic(
            block_dir
            / "block_complete.json",
            block_marker,
        )


        all_compact.extend(
            block_compact
        )

        all_physical.extend(
            block_physical
        )


        write_csv_atomic(
            partial_compact,
            all_compact,
        )

        write_csv_atomic(
            partial_physical,
            all_physical,
        )


        print(
            f"  block complete: "
            f"feasible="
            f"{block_marker['feasible_rows']}/"
            f"{EXPECTED_CONTEXTS}"
        )


    # ========================================================
    # Final aggregate checks.
    # ========================================================

    if len(
        all_compact
    ) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Final compact count "
            f"{len(all_compact)} != "
            f"{EXPECTED_EPISODES}"
        )

    if len(
        all_physical
    ) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Final physical count "
            f"{len(all_physical)} != "
            f"{EXPECTED_EPISODES}"
        )


    pairs = {
        (
            row[
                "context_id"
            ],
            row[
                "beta_name"
            ],
        )
        for row in all_physical
    }


    if len(
        pairs
    ) != EXPECTED_EPISODES:
        raise RuntimeError(
            "Duplicate/missing context-beta pairs."
        )


    # Deterministic final ordering:
    # beta-major, then frozen seed order.
    beta_order = {
        point[
            "name"
        ]:
            index
        for index, point
        in enumerate(
            beta_points
        )
    }

    seed_order = {
        seed:
            index
        for index, seed
        in enumerate(
            ALL_SEEDS
        )
    }


    all_compact.sort(
        key=lambda row:
            (
                beta_order[
                    row[
                        "beta_name"
                    ]
                ],
                seed_order[
                    as_int(
                        row[
                            "seed"
                        ]
                    )
                ],
            )
    )

    all_physical.sort(
        key=lambda row:
            (
                beta_order[
                    row[
                        "beta_name"
                    ]
                ],
                seed_order[
                    as_int(
                        row[
                            "seed"
                        ]
                    )
                ],
            )
    )


    write_csv_atomic(
        final_compact,
        all_compact,
    )

    write_csv_atomic(
        final_physical,
        all_physical,
    )


    # ========================================================
    # Context feasibility summary.
    # ========================================================

    context_summary = []

    full_grid = []
    partial_grid = []
    zero_feasible = []


    for seed in ALL_SEEDS:

        context_id = (
            context_id_for_seed(
                seed
            )
        )

        rows = [
            row
            for row in all_physical
            if row[
                "context_id"
            ]
            == context_id
        ]


        if len(
            rows
        ) != EXPECTED_BETAS:
            raise RuntimeError(
                f"{context_id}: expected "
                "21 beta outcomes."
            )


        feasible_count = sum(
            int(
                row[
                    "feasible"
                ]
            )
            for row in rows
        )

        m4_count = sum(
            as_int(
                row[
                    "m4_interventions"
                ]
            )
            > 0
            for row in rows
        )

        success_count = sum(
            row[
                "status"
            ]
            == "success"
            for row in rows
        )


        if feasible_count == (
            EXPECTED_BETAS
        ):
            grid_class = (
                "full_21_beta_feasible"
            )

            full_grid.append(
                context_id
            )

        elif feasible_count == 0:
            grid_class = (
                "zero_feasible_beta"
            )

            zero_feasible.append(
                context_id
            )

        else:
            grid_class = (
                "partial_beta_feasible"
            )

            partial_grid.append(
                context_id
            )


        context_summary.append(
            {
                "context_id":
                    context_id,

                "seed":
                    seed,

                "split_group":
                    split_name(
                        seed
                    ),

                "beta_rows":
                    EXPECTED_BETAS,

                "feasible_beta_rows":
                    feasible_count,

                "infeasible_beta_rows":
                    (
                        EXPECTED_BETAS
                        - feasible_count
                    ),

                "success_rows":
                    success_count,

                "m4_rows":
                    m4_count,

                "grid_class":
                    grid_class,
            }
        )


    context_summary_path = (
        out_dir
        / "context_feasibility_summary.csv"
    )

    write_csv_atomic(
        context_summary_path,
        context_summary,
    )


    dominance = Counter(
        row[
            "J_stability_dominant"
        ]
        for row in all_physical
        if (
            int(
                row[
                    "feasible"
                ]
            )
            == 1
        )
    )


    feasible_total = sum(
        int(
            row[
                "feasible"
            ]
        )
        for row in all_physical
    )

    success_total = sum(
        row[
            "status"
        ]
        == "success"
        for row in all_physical
    )

    m4_total = sum(
        as_int(
            row[
                "m4_interventions"
            ]
        )
        for row in all_physical
    )


    manifest = {
        "schema":
            "icra27_os_t6p12e_heldout_physical_atlas_v0",

        "status":
            "FREEZE_PASS",

        "scientific_result_interpreted":
            False,

        "selector_precommitted_before_outcomes":
            True,

        "selector_precommit_sha256":
            sha256(
                PRECOMMIT_MANIFEST
            ),

        "checkpoint":
            str(
                checkpoint.relative_to(
                    ROOT
                )
            ),

        "reward_mode":
            EXPECTED_REWARD_MODE,

        "contexts":
            EXPECTED_CONTEXTS,

        "beta_points":
            EXPECTED_BETAS,

        "episodes":
            EXPECTED_EPISODES,

        "success_rows":
            success_total,

        "m4_interventions_total":
            m4_total,

        "feasible_rows":
            feasible_total,

        "infeasible_rows":
            (
                EXPECTED_EPISODES
                - feasible_total
            ),

        "full_21_beta_feasible_contexts":
            full_grid,

        "full_21_beta_feasible_count":
            len(
                full_grid
            ),

        "partial_beta_feasible_contexts":
            partial_grid,

        "partial_beta_feasible_count":
            len(
                partial_grid
            ),

        "zero_feasible_contexts":
            zero_feasible,

        "zero_feasible_count":
            len(
                zero_feasible
            ),

        "stability_dominance_feasible_rows":
            dict(
                dominance
            ),

        "settling_steps":
            EXPECTED_SETTLING_STEPS,

        "policy_horizon":
            EXPECTED_POLICY_HORIZON,

        "physical_contract":
            str(
                f3.CONTRACT.relative_to(
                    ROOT
                )
            ),

        "feasibility_contract":
            (
                "success AND no M4 AND "
                "positive progress"
            ),

        "infeasible_candidate_scoring_contract":
            (
                "T6.12f assigns true Q=+inf to "
                "infeasible beta candidates. "
                "A precommitted infeasible selection "
                "is retained as selector failure; "
                "no post-hoc reselection."
            ),

        "zero_feasible_context_contract":
            (
                "Oracle/rank undefined; context "
                "reported explicitly as lattice "
                "infeasible and never replaced."
            ),

        "artifact_hashes": {
            "episodes.csv":
                sha256(
                    final_compact
                ),

            "heldout_physical_beta_response_atlas.csv":
                sha256(
                    final_physical
                ),

            "context_feasibility_summary.csv":
                sha256(
                    context_summary_path
                ),
        },

        "next_stage":
            (
                "T6.12f evaluates the already-"
                "precommitted selector and TRAIN-only "
                "baselines against this frozen "
                "heldout physical atlas."
            ),
    }


    write_json_atomic(
        manifest_path,
        manifest,
    )


    print()
    print("=" * 132)
    print(
        "ICRA27 OS-T6.12e HELDOUT "
        "PHYSICAL ATLAS"
    )
    print("=" * 132)

    print(
        "episodes                       :",
        len(
            all_physical
        ),
    )

    print(
        "success rows                   :",
        success_total,
    )

    print(
        "M4 interventions total         :",
        m4_total,
    )

    print(
        "feasible / infeasible rows     :",
        feasible_total,
        "/",
        EXPECTED_EPISODES
        - feasible_total,
    )

    print(
        "full-21 feasible contexts      :",
        len(
            full_grid
        ),
        full_grid,
    )

    print(
        "partial feasible contexts      :",
        len(
            partial_grid
        ),
        partial_grid,
    )

    print(
        "zero-feasible contexts         :",
        len(
            zero_feasible
        ),
        zero_feasible,
    )

    print(
        "stability dominance            :",
        dict(
            dominance
        ),
    )

    print()
    print(
        "[ICRA27] OS-T6.12e heldout "
        "physical atlas: FREEZE PASS"
    )

    print("=" * 132)


if __name__ == "__main__":
    main()
