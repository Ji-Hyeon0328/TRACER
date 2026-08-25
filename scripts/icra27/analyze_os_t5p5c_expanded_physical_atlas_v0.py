from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Frozen / completed upstream artifacts
# ============================================================

T53B_HELPER_PATH = (
    ROOT
    / "scripts/icra27"
    / "analyze_os_t5p3b_physical_atlas_normalization_v0.py"
)

T53B_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p3b_physical_atlas_normalization_v0"
)

T53B_PHYSICAL_CSV = (
    T53B_ROOT
    / "physical_beta_response_atlas.csv"
)

T53B_MANIFEST = (
    T53B_ROOT
    / "physical_atlas_manifest.json"
)


T55B_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t5p5b_remaining_rough_train_atlas_v0"
)

T55B_EPISODES = (
    T55B_ROOT
    / "episodes.csv"
)

T55B_RAW_ROOT = (
    T55B_ROOT
    / "env_logs"
)

T55B_PLAN = (
    T55B_ROOT
    / "atlas_plan.json"
)

T55B_MANIFEST = (
    T55B_ROOT
    / "atlas_manifest.json"
)


T55A_CONTEXT_CSV = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v1"
    / "oracle_context_descriptors.csv"
)

T55A_CONTEXT_CONTRACT = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v1"
    / "oracle_context_contract.json"
)


T52_CONTRACT = (
    ROOT
    / "results/icra27"
    / "os_t5p2b_physical_preference_contract_v0"
    / "physical_preference_contract.json"
)


# ============================================================
# Outputs
# ============================================================

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5c_expanded_physical_atlas_v0"
)

OUT_ATLAS = (
    OUT_DIR
    / "physical_beta_response_atlas.csv"
)

OUT_CONTEXTS = (
    OUT_DIR
    / "context_inventory.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "expanded_physical_atlas_manifest.json"
)


# ============================================================
# Frozen protocol
# ============================================================

EXPECTED_BETAS = 21

EXPECTED_CONTEXTS = 20

EXPECTED_ROWS = (
    EXPECTED_BETAS
    * EXPECTED_CONTEXTS
)

EXPECTED_OLD_SELECTED_ROWS = 105

EXPECTED_NEW_ROWS = 315

EXPECTED_SETTLING_STEPS = 5


FLAT_REPRESENTATIVE_SEED = 27100

LOW_FRICTION_REPRESENTATIVE_SEED = 27200


OLD_ROUGH_SEEDS = (
    13,
    7,
    15,
)

NEW_ROUGH_SEEDS = (
    0,
    5,
    11,
    4,
    25,
    12,
    6,
    29,
    18,
    10,
    8,
    22,
    20,
    28,
    17,
)

ALL_ROUGH_SEEDS = (
    OLD_ROUGH_SEEDS
    + NEW_ROUGH_SEEDS
)


TOL = 1.0e-10


PHYSICAL_COLUMNS = (
    "J_motion_s_per_m",
    "J_stability_attitude",
    "J_stability_slip",
    "J_stability",
    "J_energy_j_per_m",
)


# ============================================================
# Utilities
# ============================================================

def load_module(
    path: Path,
    module_name: str,
):
    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


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


def seed_int(
    row: dict[str, Any],
) -> int:
    return int(
        float(
            row["seed"]
        )
    )


def context_id_for(
    *,
    terrain: str,
    seed: int | None,
) -> str:
    if terrain == "flat":
        return "flat"

    if terrain == "low_friction":
        return "low_friction"

    if terrain == "rough_perlin":
        if seed is None:
            raise ValueError(
                "rough_perlin requires seed"
            )

        return (
            f"rough_seed_{int(seed)}"
        )

    raise ValueError(
        f"Unexpected terrain: {terrain!r}"
    )


def add_context_features(
    row: dict[str, Any],
    descriptor: dict[str, str],
) -> dict[str, Any]:
    out = dict(row)

    out[
        "context_friction_mu"
    ] = float(
        descriptor[
            "friction_mu"
        ]
    )

    out[
        "context_height_std_m"
    ] = float(
        descriptor[
            "height_std_m"
        ]
    )

    out[
        "context_height_relief_p95_p05_m"
    ] = float(
        descriptor[
            "height_relief_p95_p05_m"
        ]
    )

    out[
        "context_slope_rms"
    ] = float(
        descriptor[
            "slope_rms"
        ]
    )

    out[
        "context_slope_q95"
    ] = float(
        descriptor[
            "slope_q95"
        ]
    )

    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    required = (
        T53B_HELPER_PATH,
        T53B_PHYSICAL_CSV,
        T53B_MANIFEST,
        T55B_EPISODES,
        T55B_PLAN,
        T55B_MANIFEST,
        T55A_CONTEXT_CSV,
        T55A_CONTEXT_CONTRACT,
        T52_CONTRACT,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    helper = load_module(
        T53B_HELPER_PATH,
        "os_t5p5c_t53b_helper",
    )

    finite = helper.finite
    as_bool = helper.as_bool
    load_steps = helper.load_steps
    policy_tail_fraction = (
        helper.policy_tail_fraction
    )
    rms = helper.rms


    # ========================================================
    # Upstream manifest validation
    # ========================================================

    t53_manifest = json.loads(
        T53B_MANIFEST.read_text()
    )

    if t53_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T5.3b physical atlas is "
            "not FREEZE_PASS."
        )


    t55b_manifest = json.loads(
        T55B_MANIFEST.read_text()
    )

    if t55b_manifest.get(
        "schema"
    ) != (
        "icra27_os_t5p5b_"
        "remaining_rough_train_atlas_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.5b schema."
        )

    if t55b_manifest.get(
        "status"
    ) != "COMPUTE_PASS":
        raise RuntimeError(
            "T5.5b is not COMPUTE_PASS."
        )

    if int(
        t55b_manifest[
            "episodes"
        ]
    ) != EXPECTED_NEW_ROWS:
        raise RuntimeError(
            "T5.5b episode count mismatch."
        )

    if int(
        t55b_manifest[
            "successes"
        ]
    ) != EXPECTED_NEW_ROWS:
        raise RuntimeError(
            "T5.5b success count mismatch."
        )

    if int(
        t55b_manifest[
            "m4_interventions_total"
        ]
    ) != 0:
        raise RuntimeError(
            "T5.5b contains M4 intervention."
        )

    if bool(
        t55b_manifest[
            "heldout_used"
        ]
    ):
        raise RuntimeError(
            "T5.5b reports held-out leakage."
        )

    if list(
        t55b_manifest[
            "existing_t5p3_rough_seeds"
        ]
    ) != list(
        OLD_ROUGH_SEEDS
    ):
        raise RuntimeError(
            "Old rough seed contract mismatch."
        )

    if list(
        t55b_manifest[
            "remaining_rough_train_seeds"
        ]
    ) != list(
        NEW_ROUGH_SEEDS
    ):
        raise RuntimeError(
            "New rough seed contract mismatch."
        )


    # ========================================================
    # Frozen physical preference contract
    # ========================================================

    contract = json.loads(
        T52_CONTRACT.read_text()
    )

    if contract.get(
        "schema"
    ) != (
        "icra27_os_t5p2b_"
        "physical_preference_contract_v0"
    ):
        raise RuntimeError(
            "Unexpected T5.2b contract."
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
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Frozen slip threshold mismatch."
        )


    # ========================================================
    # 21-beta order from completed T5.5b plan
    # ========================================================

    plan = json.loads(
        T55B_PLAN.read_text()
    )

    beta_points = (
        plan[
            "beta_points"
        ]
    )

    if len(
        beta_points
    ) != EXPECTED_BETAS:
        raise RuntimeError(
            "Expected 21 beta points."
        )

    beta_order = [
        str(
            x["name"]
        )
        for x in beta_points
    ]

    if len(
        set(beta_order)
    ) != EXPECTED_BETAS:
        raise RuntimeError(
            "Duplicate beta names."
        )

    beta_index = {
        name:
            i
        for i, name in enumerate(
            beta_order
        )
    }


    # ========================================================
    # Objective-Selector physical context descriptors
    # ========================================================

    context_rows = read_csv(
        T55A_CONTEXT_CSV
    )

    if len(
        context_rows
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "T5.5a context count mismatch: "
            f"{len(context_rows)}"
        )

    context_lookup: dict[
        tuple[str, int | None],
        dict[str, str],
    ] = {}

    for row in context_rows:
        terrain = str(
            row["terrain"]
        )

        raw_seed = str(
            row.get(
                "seed",
                "",
            )
        ).strip()

        seed = (
            int(
                float(raw_seed)
            )
            if raw_seed
            else None
        )

        key = (
            terrain,
            seed,
        )

        if key in context_lookup:
            raise RuntimeError(
                f"Duplicate context descriptor: "
                f"{key}"
            )

        context_lookup[
            key
        ] = row


    expected_context_keys = {
        (
            "flat",
            None,
        ),
        (
            "low_friction",
            None,
        ),
    }

    expected_context_keys.update(
        (
            "rough_perlin",
            int(seed),
        )
        for seed in ALL_ROUGH_SEEDS
    )

    if set(
        context_lookup
    ) != expected_context_keys:
        raise RuntimeError(
            "T5.5a context coverage mismatch."
        )


    # ========================================================
    # Existing frozen T5.3b physical rows
    # ========================================================

    old_all = read_csv(
        T53B_PHYSICAL_CSV
    )

    if len(
        old_all
    ) != 189:
        raise RuntimeError(
            "Unexpected T5.3b row count."
        )


    # --------------------------------------------------------
    # Verify flat/LF deterministic replicate semantics before
    # selecting a single representative context.
    # --------------------------------------------------------

    max_deterministic_replica_error = 0.0

    for group in (
        "flat",
        "low_friction",
    ):
        for beta_name in beta_order:
            rows = [
                row
                for row in old_all
                if (
                    row[
                        "group"
                    ] == group
                    and row[
                        "beta_name"
                    ] == beta_name
                )
            ]

            if len(rows) != 3:
                raise RuntimeError(
                    f"{group}/{beta_name}: "
                    f"expected 3 deterministic "
                    f"replicates, got {len(rows)}"
                )

            for column in (
                PHYSICAL_COLUMNS
            ):
                values = [
                    finite(
                        row[column]
                    )
                    for row in rows
                ]

                spread = (
                    max(values)
                    - min(values)
                )

                max_deterministic_replica_error = max(
                    max_deterministic_replica_error,
                    spread,
                )

                if spread > 1e-9:
                    raise RuntimeError(
                        f"{group}/{beta_name}/{column}: "
                        "deterministic replicate "
                        f"spread={spread}"
                    )


    selected_old = []

    for row in old_all:
        group = row[
            "group"
        ]

        seed = seed_int(
            row
        )

        keep = False

        if (
            group == "flat"
            and seed
            == FLAT_REPRESENTATIVE_SEED
        ):
            keep = True

        elif (
            group == "low_friction"
            and seed
            == LOW_FRICTION_REPRESENTATIVE_SEED
        ):
            keep = True

        elif (
            group == "rough"
            and seed
            in OLD_ROUGH_SEEDS
        ):
            keep = True

        if not keep:
            continue

        # T5.3b normalized values are intentionally not
        # propagated. T5.5d uses context-local ideal/nadir
        # regret instead.
        out = {
            key:
                value
            for key, value in row.items()
            if key not in {
                "J_motion_normalized",
                "J_stability_normalized",
                "J_energy_normalized",
            }
        }

        if group == "flat":
            terrain = "flat"
            context_key = (
                "flat",
                None,
            )

        elif group == "low_friction":
            terrain = "low_friction"
            context_key = (
                "low_friction",
                None,
            )

        elif group == "rough":
            terrain = "rough_perlin"
            context_key = (
                "rough_perlin",
                seed,
            )

        else:
            raise RuntimeError(
                f"Unexpected old group: {group}"
            )

        out[
            "context_id"
        ] = context_id_for(
            terrain=terrain,
            seed=(
                seed
                if terrain
                == "rough_perlin"
                else None
            ),
        )

        out[
            "source_stage"
        ] = "OS-T5.3b"

        out[
            "source_role"
        ] = (
            "frozen_existing_physical_row"
        )

        out = add_context_features(
            out,
            context_lookup[
                context_key
            ],
        )

        selected_old.append(
            out
        )


    if len(
        selected_old
    ) != EXPECTED_OLD_SELECTED_ROWS:
        raise RuntimeError(
            "Old selected-row mismatch: "
            f"{len(selected_old)}"
        )


    # ========================================================
    # Reconstruct physical metrics for NEW T5.5b 315 rows
    # ========================================================

    source_new = read_csv(
        T55B_EPISODES
    )

    if len(
        source_new
    ) != EXPECTED_NEW_ROWS:
        raise RuntimeError(
            "T5.5b source-row mismatch."
        )

    if any(
        not as_bool(
            row.get(
                "success",
                False,
            )
        )
        for row in source_new
    ):
        raise RuntimeError(
            "T5.5c requires all T5.5b "
            "episodes successful."
        )

    # --------------------------------------------------------
    # Match T5.5b compact rows to raw episode logs using the
    # exact frozen T5.3b convention:
    #
    #   group source rows by (beta_name, group),
    #   sort episode_*.jsonl,
    #   zip rows and raw files in deterministic order.
    #
    # T5.5b episodes.csv intentionally does NOT contain an
    # episode_index column, so raw-log provenance must not
    # depend on such a field.
    # --------------------------------------------------------

    source_indices_by_group = {}

    for source_index, row in enumerate(
        source_new
    ):
        key = (
            str(
                row["beta_name"]
            ),
            str(
                row["group"]
            ),
        )

        source_indices_by_group.setdefault(
            key,
            [],
        ).append(
            source_index
        )

    raw_path_by_source_index = {}

    for (
        beta_name,
        group,
    ), source_indices in (
        source_indices_by_group.items()
    ):
        raw_dir = (
            T55B_RAW_ROOT
            / beta_name
            / group
        )

        raw_files = sorted(
            raw_dir.glob(
                "episode_*.jsonl"
            )
        )

        if len(raw_files) != len(
            source_indices
        ):
            raise RuntimeError(
                f"{beta_name}/{group}: "
                f"{len(raw_files)} raw files "
                f"vs {len(source_indices)} "
                "source rows."
            )

        for source_index, raw_path in zip(
            source_indices,
            raw_files,
        ):
            raw_path_by_source_index[
                source_index
            ] = raw_path

    if len(
        raw_path_by_source_index
    ) != EXPECTED_NEW_ROWS:
        raise RuntimeError(
            "T5.5b raw-log pairing count "
            "mismatch: "
            f"{len(raw_path_by_source_index)} "
            f"vs {EXPECTED_NEW_ROWS}"
        )

    settling_counts = set()

    max_time_error = 0.0
    max_energy_error = 0.0
    max_progress_error = 0.0

    reconstructed_new = []

    new_dominance_counts = Counter()

    for source_index, row in enumerate(
        source_new
    ):
        beta_name = str(
            row[
                "beta_name"
            ]
        )

        if beta_name not in (
            beta_index
        ):
            raise RuntimeError(
                f"Unknown beta: {beta_name}"
            )

        group = str(
            row[
                "group"
            ]
        )

        terrain = str(
            row[
                "terrain"
            ]
        )

        seed = seed_int(
            row
        )

        if group != "rough":
            raise RuntimeError(
                f"T5.5b unexpected group: {group}"
            )

        if terrain != "rough_perlin":
            raise RuntimeError(
                "T5.5b unexpected terrain: "
                f"{terrain}"
            )

        if seed not in (
            NEW_ROUGH_SEEDS
        ):
            raise RuntimeError(
                f"Unexpected T5.5b seed: {seed}"
            )

        if int(
            float(
                row[
                    "m4_interventions"
                ]
            )
        ) != 0:
            raise RuntimeError(
                "T5.5b row contains M4."
            )

        raw_path = (
            raw_path_by_source_index[
                source_index
            ]
        )

        if not raw_path.exists():
            raise FileNotFoundError(
                raw_path
            )

        stem = raw_path.stem

        prefix = "episode_"

        if not stem.startswith(
            prefix
        ):
            raise RuntimeError(
                "Unexpected raw-log filename: "
                f"{raw_path.name}"
            )

        episode_index = int(
            stem[
                len(prefix):
            ]
        )

        steps = load_steps(
            raw_path
        )

        policy_steps = int(
            float(
                row[
                    "policy_steps"
                ]
            )
        )

        settling_steps = (
            len(steps)
            - policy_steps
        )

        settling_counts.add(
            settling_steps
        )

        if settling_steps != (
            EXPECTED_SETTLING_STEPS
        ):
            raise RuntimeError(
                "Unexpected settling steps: "
                f"{settling_steps}"
            )

        boundary = steps[
            settling_steps - 1
        ]

        policy = steps[
            settling_steps:
        ]

        if len(policy) != (
            policy_steps
        ):
            raise RuntimeError(
                "Policy-window mismatch."
            )

        if not policy:
            raise RuntimeError(
                "Empty policy window."
            )

        final = policy[-1]


        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        decision_time = sum(
            finite(
                x[
                    "decision_dt_s"
                ]
            )
            for x in policy
        )

        compact_time = finite(
            row[
                "decision_time_s"
            ]
        )

        time_error = abs(
            decision_time
            - compact_time
        )

        max_time_error = max(
            max_time_error,
            time_error,
        )

        if time_error > 1e-7:
            raise RuntimeError(
                "Decision-time reconstruction "
                f"mismatch: {time_error}"
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

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

        if progress_m <= 0.0:
            raise RuntimeError(
                "Physical contract requires "
                "positive progress."
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

        progress_error = abs(
            progress_m
            - progress_integrated
        )

        max_progress_error = max(
            max_progress_error,
            progress_error,
        )

        if progress_error > 1e-7:
            raise RuntimeError(
                "Progress reconstruction "
                f"mismatch: {progress_error}"
            )


        # ----------------------------------------------------
        # Mechanical energy
        # ----------------------------------------------------

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
            row[
                "energy_abs_j"
            ]
        )

        energy_error = abs(
            energy_abs_j
            - compact_energy
        )

        max_energy_error = max(
            max_energy_error,
            energy_error,
        )

        if energy_error > 1e-6:
            raise RuntimeError(
                "Energy reconstruction "
                f"mismatch: {energy_error}"
            )


        # ----------------------------------------------------
        # Physical attitude
        # ----------------------------------------------------

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

        roll_rms = rms(
            abs_roll
        )

        pitch_rms = rms(
            abs_pitch
        )

        max_abs_roll = max(
            abs_roll
        )

        max_abs_pitch = max(
            abs_pitch
        )


        # ----------------------------------------------------
        # Established-contact physical slip
        # ----------------------------------------------------

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
            policy_tail_fraction(
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
            policy_tail_fraction(
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
            policy_tail_fraction(
                boundary_slip=(
                    boundary_slip
                ),
                final_slip=(
                    final_slip
                ),
                threshold=0.10,
            )
        )


        # ----------------------------------------------------
        # Frozen T5.2b physical preference contract
        # ----------------------------------------------------

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

        new_dominance_counts[
            dominant
        ] += 1


        out = {
            "beta_name":
                beta_name,

            "beta_motion":
                finite(
                    row[
                        "beta_motion"
                    ]
                ),

            "beta_stability":
                finite(
                    row[
                        "beta_stability"
                    ]
                ),

            "beta_energy":
                finite(
                    row[
                        "beta_energy"
                    ]
                ),

            "lambda_motion":
                finite(
                    row[
                        "lambda_motion"
                    ]
                ),

            "lambda_stability":
                finite(
                    row[
                        "lambda_stability"
                    ]
                ),

            "lambda_energy":
                finite(
                    row[
                        "lambda_energy"
                    ]
                ),

            "group":
                group,

            "terrain":
                terrain,

            "seed":
                seed,

            "episode_index":
                episode_index,

            "raw_path":
                str(
                    raw_path.relative_to(
                        ROOT
                    )
                ),

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

            "context_id":
                context_id_for(
                    terrain=terrain,
                    seed=seed,
                ),

            "source_stage":
                "OS-T5.5b",

            "source_role":
                (
                    "new_physical_metric_"
                    "reconstruction"
                ),
        }

        out = add_context_features(
            out,
            context_lookup[
                (
                    "rough_perlin",
                    seed,
                )
            ],
        )

        reconstructed_new.append(
            out
        )


    if len(
        reconstructed_new
    ) != EXPECTED_NEW_ROWS:
        raise RuntimeError(
            "New reconstructed-row mismatch."
        )


    if settling_counts != {
        EXPECTED_SETTLING_STEPS
    }:
        raise RuntimeError(
            "Settling-step contract mismatch."
        )


    # ========================================================
    # Merge
    # ========================================================

    all_rows = (
        selected_old
        + reconstructed_new
    )

    if len(
        all_rows
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} total rows; "
            f"got {len(all_rows)}"
        )


    # ========================================================
    # Context order
    # ========================================================

    context_order = [
        "flat",
        "low_friction",
    ]

    context_order.extend(
        f"rough_seed_{seed}"
        for seed in ALL_ROUGH_SEEDS
    )

    if len(
        context_order
    ) != EXPECTED_CONTEXTS:
        raise RuntimeError(
            "Context-order count mismatch."
        )

    context_index = {
        name:
            i
        for i, name in enumerate(
            context_order
        )
    }


    # ========================================================
    # Coverage audit
    # ========================================================

    context_counts = Counter(
        row[
            "context_id"
        ]
        for row in all_rows
    )

    beta_counts = Counter(
        row[
            "beta_name"
        ]
        for row in all_rows
    )

    pair_counts = Counter(
        (
            row[
                "context_id"
            ],
            row[
                "beta_name"
            ],
        )
        for row in all_rows
    )


    if set(
        context_counts
    ) != set(
        context_order
    ):
        raise RuntimeError(
            "Context coverage mismatch."
        )

    if any(
        count != EXPECTED_BETAS
        for count in (
            context_counts.values()
        )
    ):
        raise RuntimeError(
            "Each context must contain "
            "exactly 21 beta rows."
        )

    if set(
        beta_counts
    ) != set(
        beta_order
    ):
        raise RuntimeError(
            "Beta coverage mismatch."
        )

    if any(
        count != EXPECTED_CONTEXTS
        for count in (
            beta_counts.values()
        )
    ):
        raise RuntimeError(
            "Each beta must contain "
            "exactly 20 context rows."
        )

    if len(
        pair_counts
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            "Duplicate context-beta pair."
        )

    if any(
        count != 1
        for count in (
            pair_counts.values()
        )
    ):
        raise RuntimeError(
            "Context-beta multiplicity failure."
        )


    # ========================================================
    # Sort context-major, beta-minor
    # ========================================================

    all_rows.sort(
        key=lambda row: (
            context_index[
                row[
                    "context_id"
                ]
            ],
            beta_index[
                row[
                    "beta_name"
                ]
            ],
        )
    )


    # ========================================================
    # Context inventory
    # ========================================================

    context_inventory = []

    for cid in context_order:
        rows = [
            row
            for row in all_rows
            if row[
                "context_id"
            ] == cid
        ]

        first = rows[0]

        terrain = str(
            first[
                "terrain"
            ]
        )

        if terrain == "rough_perlin":
            seed_value: Any = int(
                first[
                    "seed"
                ]
            )
        else:
            seed_value = ""

        context_inventory.append(
            {
                "context_id":
                    cid,

                "terrain":
                    terrain,

                "seed":
                    seed_value,

                "beta_rows":
                    len(rows),

                "friction_mu":
                    first[
                        "context_friction_mu"
                    ],

                "height_std_m":
                    first[
                        "context_height_std_m"
                    ],

                "height_relief_p95_p05_m":
                    first[
                        "context_height_relief_p95_p05_m"
                    ],

                "slope_rms":
                    first[
                        "context_slope_rms"
                    ],

                "slope_q95":
                    first[
                        "context_slope_q95"
                    ],
            }
        )


    # ========================================================
    # Diagnostics
    # ========================================================

    dominance_counts = Counter(
        str(
            row[
                "J_stability_dominant"
            ]
        )
        for row in all_rows
    )

    objective_ranges = {}

    for column in (
        PHYSICAL_COLUMNS
    ):
        values = [
            finite(
                row[
                    column
                ]
            )
            for row in all_rows
        ]

        objective_ranges[
            column
        ] = [
            min(values),
            max(values),
        ]


    rough_feature_ranges = {}

    # context_inventory stores the compact descriptor
    # column names, whereas atlas rows use a
    # "context_" prefix. Preserve the explicit
    # manifest names while reading the correct
    # inventory columns.
    for (
        manifest_name,
        inventory_column,
    ) in (
        (
            "context_height_std_m",
            "height_std_m",
        ),
        (
            "context_height_relief_p95_p05_m",
            "height_relief_p95_p05_m",
        ),
        (
            "context_slope_rms",
            "slope_rms",
        ),
        (
            "context_slope_q95",
            "slope_q95",
        ),
    ):
        values = [
            finite(
                row[
                    inventory_column
                ]
            )
            for row in context_inventory
            if row[
                "terrain"
            ] == "rough_perlin"
        ]

        if len(values) != len(
            ALL_ROUGH_SEEDS
        ):
            raise RuntimeError(
                "Rough context feature coverage "
                "mismatch for "
                f"{inventory_column}: "
                f"{len(values)} vs "
                f"{len(ALL_ROUGH_SEEDS)}"
            )

        rough_feature_ranges[
            manifest_name
        ] = [
            min(values),
            max(values),
        ]


    # ========================================================
    # Write outputs
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_ATLAS,
        all_rows,
    )

    write_csv(
        OUT_CONTEXTS,
        context_inventory,
    )


    manifest = {
        "schema":
            (
                "icra27_os_t5p5c_"
                "expanded_physical_atlas_v0"
            ),

        "status":
            "COMPUTE_PASS",

        "rows":
            len(
                all_rows
            ),

        "contexts":
            len(
                context_order
            ),

        "beta_points_per_context":
            EXPECTED_BETAS,

        "context_beta_pairs_unique":
            True,

        "context_order":
            context_order,

        "rough_train_seeds":
            list(
                ALL_ROUGH_SEEDS
            ),

        "old_rough_seeds":
            list(
                OLD_ROUGH_SEEDS
            ),

        "new_rough_seeds":
            list(
                NEW_ROUGH_SEEDS
            ),

        "flat_representative_seed":
            FLAT_REPRESENTATIVE_SEED,

        "low_friction_representative_seed":
            LOW_FRICTION_REPRESENTATIVE_SEED,

        "old_selected_rows":
            len(
                selected_old
            ),

        "new_reconstructed_rows":
            len(
                reconstructed_new
            ),

        "physical_objectives":
            {
                "motion":
                    "J_motion_s_per_m",

                "stability":
                    "J_stability",

                "energy":
                    "J_energy_j_per_m",
            },

        "objective_direction":
            "all_lower_is_better",

        "stability_definition":
            (
                "max(J_stability_attitude, "
                "J_stability_slip)"
            ),

        "stability_dominance":
            dict(
                dominance_counts
            ),

        "objective_ranges":
            objective_ranges,

        "rough_context_feature_ranges":
            rough_feature_ranges,

        "new_reconstruction_validation":
            {
                "settling_steps":
                    sorted(
                        settling_counts
                    ),

                "max_time_error_s":
                    max_time_error,

                "max_progress_error_m":
                    max_progress_error,

                "max_energy_error_j":
                    max_energy_error,
            },

        "deterministic_replica_validation":
            {
                "groups":
                    [
                        "flat",
                        "low_friction",
                    ],

                "representative_policy":
                    (
                        "keep seed 27100 for flat "
                        "and 27200 for low_friction"
                    ),

                "max_physical_metric_spread":
                    (
                        max_deterministic_replica_error
                    ),
            },

        "normalization_policy":
            (
                "No global normalization is "
                "defined or refrozen in OS-T5.5c. "
                "Downstream selector labels use "
                "context-local ideal/nadir regret "
                "over each 21-point Pareto response."
            ),

        "selector_row_semantics":
            (
                "Each atlas row contains the "
                "5D OS physical terrain context, "
                "one beta probe, and frozen "
                "physical J_M/J_S/J_E outcomes."
            ),

        "t5p2b_contract":
            str(
                T52_CONTRACT.relative_to(
                    ROOT
                )
            ),

        "t5p2b_contract_sha256":
            sha256_file(
                T52_CONTRACT
            ),

        "t5p5a_context_source":
            str(
                T55A_CONTEXT_CSV.relative_to(
                    ROOT
                )
            ),

        "t5p5a_context_sha256":
            sha256_file(
                T55A_CONTEXT_CSV
            ),

        "t5p3b_physical_source":
            str(
                T53B_PHYSICAL_CSV.relative_to(
                    ROOT
                )
            ),

        "t5p3b_physical_sha256":
            sha256_file(
                T53B_PHYSICAL_CSV
            ),

        "t5p5b_episode_source":
            str(
                T55B_EPISODES.relative_to(
                    ROOT
                )
            ),

        "t5p5b_episode_sha256":
            sha256_file(
                T55B_EPISODES
            ),

        "heldout_used":
            False,

        "next_stage":
            (
                "OS-T5.5d: compute the 20 "
                "context-wise Pareto fronts, "
                "context-local ideal/nadir regrets, "
                "and 25 mission-preference "
                "augmented-Tchebycheff beta labels "
                "for 500 (c,w)->beta* examples."
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
    print("=" * 112)
    print(
        "ICRA27 OS-T5.5c "
        "EXPANDED PHYSICAL BETA-RESPONSE ATLAS"
    )
    print("=" * 112)

    print(
        "rows              :",
        len(
            all_rows
        ),
    )

    print(
        "contexts          :",
        len(
            context_order
        ),
    )

    print(
        "beta/context      :",
        EXPECTED_BETAS,
    )

    print(
        "old frozen rows   :",
        len(
            selected_old
        ),
    )

    print(
        "new physical rows :",
        len(
            reconstructed_new
        ),
    )

    print(
        "unique pairs      :",
        len(
            pair_counts
        ),
        "/",
        EXPECTED_ROWS,
    )

    print()
    print("CONTEXT COVERAGE")

    for cid in context_order:
        print(
            f"  {cid:<20} "
            f"{context_counts[cid]:2d}"
        )

    print()
    print("STABILITY DOMINANCE")

    for key in (
        "attitude",
        "slip",
        "tie",
    ):
        print(
            f"  {key:<10}: "
            f"{dominance_counts.get(key, 0)}"
        )

    print()
    print("PHYSICAL OBJECTIVE RANGES")

    for column in (
        PHYSICAL_COLUMNS
    ):
        lo, hi = (
            objective_ranges[
                column
            ]
        )

        print(
            f"  {column:<30} "
            f"[{lo:.9f}, {hi:.9f}]"
        )

    print()
    print("NEW-ROW RECONSTRUCTION ERRORS")

    print(
        "  time    :",
        max_time_error,
    )

    print(
        "  progress:",
        max_progress_error,
    )

    print(
        "  energy  :",
        max_energy_error,
    )

    print()
    print(
        "flat/LF deterministic "
        "replica max spread:",
        max_deterministic_replica_error,
    )

    print()
    print("outputs:")
    print(" ", OUT_ATLAS)
    print(" ", OUT_CONTEXTS)
    print(" ", OUT_MANIFEST)

    print()
    print(
        "[ICRA27] OS-T5.5c "
        "expanded physical atlas: COMPUTE PASS"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
