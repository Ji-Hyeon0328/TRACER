from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

T55_PATH = (
    ROOT
    / "scripts/icra27"
    / "evaluate_os_t5p5b_remaining_rough_train_atlas_v0.py"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p5a_exact_repeat_determinism_v0"
)

OUT_ROWS = (
    OUT_DIR
    / "repeat_episode_rows.csv"
)

OUT_CASES = (
    OUT_DIR
    / "repeat_case_summary.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "exact_repeat_determinism_manifest.json"
)


SEEDS = (
    13,
    4,
    22,
)

BETA_NAMES = (
    "lm100_ls000_le000",
    "lm000_ls100_le000",
    "lm000_ls000_le100",
    "lm040_ls040_le020",
)

REPEATS = 3

BASE_PORT = 62110

EXPECTED_REWARD_MODE = "tracer_cost_v4"

VOLATILE_LOG_KEYS = {
    "runner_pid",
}


def load_module(
    path: Path,
    name: str,
):
    spec = (
        importlib.util.spec_from_file_location(
            name,
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

    spec.loader.exec_module(
        module
    )

    return module


def normalize_json(
    x: Any,
):
    if isinstance(
        x,
        dict,
    ):
        return {
            str(k):
                normalize_json(v)
            for k, v in x.items()
        }

    if isinstance(
        x,
        (list, tuple),
    ):
        return [
            normalize_json(v)
            for v in x
        ]

    if isinstance(
        x,
        np.ndarray,
    ):
        return [
            normalize_json(v)
            for v in x.tolist()
        ]

    if isinstance(
        x,
        np.generic,
    ):
        return normalize_json(
            x.item()
        )

    if isinstance(
        x,
        Path,
    ):
        return str(x)

    if isinstance(
        x,
        float,
    ):
        if not math.isfinite(x):
            raise RuntimeError(
                f"Non-finite value: {x}"
            )

        return float(x)

    return x


def sha256_json(
    obj,
):
    payload = json.dumps(
        normalize_json(obj),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def strip_volatile(
    x,
):
    if isinstance(
        x,
        dict,
    ):
        return {
            key:
                strip_volatile(value)
            for key, value in x.items()
            if key not in VOLATILE_LOG_KEYS
        }

    if isinstance(
        x,
        list,
    ):
        return [
            strip_volatile(value)
            for value in x
        ]

    return x


def canonical_episode_log_hash(
    log_dir: Path,
):
    paths = sorted(
        log_dir.glob(
            "episode_*.jsonl"
        )
    )

    if len(paths) != 1:
        raise RuntimeError(
            "Expected exactly one episode JSONL "
            f"in {log_dir}; found {paths}"
        )

    path = paths[0]

    canonical_rows = []

    with path.open(
        "r",
    ) as f:
        for line_index, line in enumerate(
            f,
            1,
        ):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(
                line
            )

            obj = strip_volatile(
                obj
            )

            canonical_rows.append(
                obj
            )

    if not canonical_rows:
        raise RuntimeError(
            f"Empty episode log: {path}"
        )

    return (
        sha256_json(
            canonical_rows
        ),
        len(
            canonical_rows
        ),
        path,
    )


def scalar_or_blank(
    row,
    key,
):
    value = row.get(
        key
    )

    if value is None:
        return ""

    if isinstance(
        value,
        (bool, str, int),
    ):
        return value

    if isinstance(
        value,
        (float, np.floating),
    ):
        return float(
            value
        )

    return json.dumps(
        normalize_json(
            value
        ),
        sort_keys=True,
    )


def numeric_range(
    rows,
    key,
):
    values = []

    for row in rows:
        value = row.get(
            key
        )

        if value is None:
            continue

        try:
            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if math.isfinite(
            value
        ):
            values.append(
                value
            )

    if not values:
        return None

    return float(
        max(values)
        - min(values)
    )


def write_csv(
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


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    if not T55_PATH.exists():
        raise FileNotFoundError(
            T55_PATH
        )


    t55 = load_module(
        T55_PATH,
        "os_t6p5a_t55",
    )

    phase = t55.load_module(
        t55.T49_PATH,
        "os_t6p5a_t49",
    )


    # ========================================================
    # Frozen reward/evaluator contract
    # ========================================================

    if str(
        phase.EVAL_REWARD_MODE
    ) != EXPECTED_REWARD_MODE:
        raise RuntimeError(
            "Unexpected T4.9 reward mode."
        )

    historical = str(
        phase.base.EVAL_REWARD_MODE
    )

    if historical == "tracer_cost_v2":
        phase.base.EVAL_REWARD_MODE = (
            EXPECTED_REWARD_MODE
        )

    elif historical != EXPECTED_REWARD_MODE:
        raise RuntimeError(
            "Unexpected base evaluator "
            f"reward mode: {historical!r}"
        )

    if str(
        phase.base.EVAL_REWARD_MODE
    ) != EXPECTED_REWARD_MODE:
        raise RuntimeError(
            "Could not promote evaluator "
            "to tracer_cost_v4."
        )


    checkpoint = (
        t55.DEFAULT_CHECKPOINT.resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    if checkpoint.name != (
        "checkpoint_update_0030.pt"
    ):
        raise RuntimeError(
            "T6.5a requires frozen "
            "TRAIN-selected u30 checkpoint."
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
    # Frozen 21-point beta lattice
    # ========================================================

    beta_points = (
        t55.build_beta_lattice()
    )

    beta_lookup = {
        point[
            "name"
        ]:
            point
        for point in beta_points
    }

    missing = [
        name
        for name in BETA_NAMES
        if name not in beta_lookup
    ]

    if missing:
        raise RuntimeError(
            "Requested diagnostic beta points "
            f"are absent: {missing}"
        )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


    episode_rows = []

    case_counter = 0

    for seed in SEEDS:
        for beta_name in BETA_NAMES:
            case_counter += 1

            point = beta_lookup[
                beta_name
            ]

            beta = tuple(
                float(x)
                for x in point[
                    "beta"
                ]
            )

            print()
            print(
                "=" * 108
            )
            print(
                f"CASE {case_counter:02d}/"
                f"{len(SEEDS) * len(BETA_NAMES)} "
                f"seed={seed} "
                f"beta={beta_name} "
                f"{beta}"
            )
            print(
                "=" * 108
            )


            for repeat in range(
                REPEATS
            ):
                # Fresh environment and fresh runtime
                # for every exact repeat.
                command_port = (
                    BASE_PORT
                    + case_counter * 100
                    + repeat * 10
                )

                log_dir = (
                    OUT_DIR
                    / "env_logs"
                    / f"seed_{seed}"
                    / beta_name
                    / f"repeat_{repeat}"
                )

                env = phase.make_env(
                    terrain="rough_perlin",
                    beta=beta,
                    command_port=(
                        command_port
                    ),
                    log_dir=log_dir,
                )

                try:
                    row = (
                        phase.base.run_episode(
                            env,
                            policy_name=(
                                "beta_conditioned"
                            ),
                            checkpoint=checkpoint,
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
                            group=(
                                "t6p5a_exact_repeat"
                            ),
                            terrain=(
                                "rough_perlin"
                            ),
                            seed=int(
                                seed
                            ),
                        )
                    )

                finally:
                    env.close()


                row = phase.annotate_row(
                    row,
                    beta_name=beta_name,
                    beta=beta,
                )

                row_hash = sha256_json(
                    row
                )

                (
                    log_hash,
                    log_rows,
                    log_path,
                ) = canonical_episode_log_hash(
                    log_dir
                )


                compact = {
                    "seed":
                        int(seed),

                    "beta_name":
                        beta_name,

                    "repeat":
                        int(repeat),

                    "row_sha256":
                        row_hash,

                    "canonical_log_sha256":
                        log_hash,

                    "episode_log_rows":
                        int(
                            log_rows
                        ),

                    "episode_log":
                        str(
                            log_path.relative_to(
                                ROOT
                            )
                        ),
                }


                for key in (
                    "status",
                    "success",
                    "m4_terminal",
                    "m4_interventions",
                    "policy_steps",
                    "initial_goal_distance_m",
                    "final_goal_distance_m",
                    "progress_m",
                    "decision_time_s",
                    "energy_abs_j",
                    "energy_bracket_time_s",
                    "roll_rms_rad",
                    "pitch_rms_rad",
                    "max_abs_roll_rad",
                    "max_abs_pitch_rad",
                    "mean_applied_vx_mps",
                    "mean_abs_applied_yaw_rps",
                    "mean_applied_height_m",
                    "mean_applied_clearance_m",
                    "mean_action_vx_norm",
                    "mean_abs_action_yaw_norm",
                    "mean_action_height_norm",
                ):
                    compact[
                        key
                    ] = scalar_or_blank(
                        row,
                        key,
                    )


                episode_rows.append(
                    compact
                )

                print(
                    f"  repeat={repeat} "
                    f"status={compact['status']} "
                    f"steps={compact['policy_steps']} "
                    f"E={compact['energy_abs_j']} "
                    f"roll={compact['roll_rms_rad']} "
                    f"pitch={compact['pitch_rms_rad']} "
                    f"row={row_hash[:12]} "
                    f"log={log_hash[:12]}"
                )


    # ========================================================
    # Case-level exact-repeat checks
    # ========================================================

    case_rows = []

    for seed in SEEDS:
        for beta_name in BETA_NAMES:
            rows = [
                row
                for row in episode_rows
                if (
                    int(
                        row[
                            "seed"
                        ]
                    )
                    == int(
                        seed
                    )
                    and row[
                        "beta_name"
                    ]
                    == beta_name
                )
            ]

            if len(
                rows
            ) != REPEATS:
                raise RuntimeError(
                    "Repeat count mismatch."
                )


            row_hash_equal = (
                len(
                    {
                        row[
                            "row_sha256"
                        ]
                        for row in rows
                    }
                )
                == 1
            )

            log_hash_equal = (
                len(
                    {
                        row[
                            "canonical_log_sha256"
                        ]
                        for row in rows
                    }
                )
                == 1
            )

            status_equal = (
                len(
                    {
                        str(
                            row[
                                "status"
                            ]
                        )
                        for row in rows
                    }
                )
                == 1
            )


            case_row = {
                "seed":
                    int(seed),

                "beta_name":
                    beta_name,

                "row_hash_equal":
                    int(
                        row_hash_equal
                    ),

                "canonical_log_hash_equal":
                    int(
                        log_hash_equal
                    ),

                "status_equal":
                    int(
                        status_equal
                    ),

                "energy_abs_j_range":
                    numeric_range(
                        rows,
                        "energy_abs_j",
                    ),

                "roll_rms_rad_range":
                    numeric_range(
                        rows,
                        "roll_rms_rad",
                    ),

                "pitch_rms_rad_range":
                    numeric_range(
                        rows,
                        "pitch_rms_rad",
                    ),

                "progress_m_range":
                    numeric_range(
                        rows,
                        "progress_m",
                    ),

                "decision_time_s_range":
                    numeric_range(
                        rows,
                        "decision_time_s",
                    ),
            }

            case_rows.append(
                case_row
            )


    row_exact_fraction = float(
        np.mean(
            [
                row[
                    "row_hash_equal"
                ]
                for row in case_rows
            ]
        )
    )

    log_exact_fraction = float(
        np.mean(
            [
                row[
                    "canonical_log_hash_equal"
                ]
                for row in case_rows
            ]
        )
    )


    max_energy_range = float(
        max(
            (
                row[
                    "energy_abs_j_range"
                ]
                or 0.0
            )
            for row in case_rows
        )
    )

    max_roll_range = float(
        max(
            (
                row[
                    "roll_rms_rad_range"
                ]
                or 0.0
            )
            for row in case_rows
        )
    )

    max_pitch_range = float(
        max(
            (
                row[
                    "pitch_rms_rad_range"
                ]
                or 0.0
            )
            for row in case_rows
        )
    )


    aggregate = {
        "cases":
            len(
                case_rows
            ),

        "repeats_per_case":
            REPEATS,

        "episodes":
            len(
                episode_rows
            ),

        "row_exact_fraction":
            row_exact_fraction,

        "canonical_log_exact_fraction":
            log_exact_fraction,

        "max_energy_abs_j_range":
            max_energy_range,

        "max_roll_rms_rad_range":
            max_roll_range,

        "max_pitch_rms_rad_range":
            max_pitch_range,
    }


    write_csv(
        OUT_ROWS,
        episode_rows,
    )

    write_csv(
        OUT_CASES,
        case_rows,
    )


    manifest = {
        "schema":
            "icra27_os_t6p5a_exact_repeat_determinism_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "checkpoint":
            str(
                checkpoint.relative_to(
                    ROOT
                )
            ),

        "terrain":
            "rough_perlin",

        "terrain_seeds":
            list(
                SEEDS
            ),

        "beta_names":
            list(
                BETA_NAMES
            ),

        "repeats_per_case":
            REPEATS,

        "fresh_environment_each_repeat":
            True,

        "policy_action":
            "deterministic tanh(mu)",

        "seed_semantics":
            (
                "The same episode seed is propagated "
                "to the terrain-seeded PyMPC runner; "
                "terrain geometry is therefore held "
                "exactly fixed across repeats."
            ),

        "canonical_log_hash":
            (
                "SHA256 of the episode JSONL after "
                "removing only runner_pid."
            ),

        "aggregate":
            aggregate,

        "decision_rule":
            (
                "If row and canonical episode-log "
                "hashes are identical across repeats, "
                "single-rollout stochastic/numerical "
                "repeatability is not a plausible "
                "explanation for the weak cross-terrain "
                "J_stability/J_energy transfer observed "
                "in T6.4. If exact repeats differ, "
                "quantify runtime/contact nondeterminism "
                "before modifying the Objective Selector."
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
        "ICRA27 OS-T6.5a EXACT-REPEAT "
        "DETERMINISM AUDIT"
    )
    print("=" * 118)

    for key, value in aggregate.items():
        print(
            f"  {key:<38}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.5a exact-repeat "
        "determinism: COMPUTE PASS"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
