from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

T55_PATH = (
    ROOT
    / "scripts/icra27"
    / "evaluate_os_t5p5b_remaining_rough_train_atlas_v0.py"
)

PATCH_MANIFEST = (
    ROOT
    / "results/icra27"
    / "os_t6p3a_highres_height_patches_v0"
    / "highres_height_patch_manifest.json"
)

T68A_CSV = (
    ROOT
    / "results/icra27"
    / "os_t6p8a_probe_beta_independence_v0"
    / "probe_beta_independence.csv"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p8b_prepolicy_probe_response_atlas_v0"
)

OUT_CSV = (
    OUT_DIR
    / "prepolicy_probe_response_atlas.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "prepolicy_probe_response_manifest.json"
)


PROBE_STEPS = 5
DECISION_DT_S = 0.20

BASE_PORT = 63110

# Beta is irrelevant to the fixed physical probe,
# but make_env() requires one.
FIXED_BETA = (
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
)

EXPECTED_OBS_DIM = 24
BASE_OBS_DIM = 21

EPS = 1.0e-12


def load_module(
    path: Path,
    name: str,
):
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


def write_csv(
    path,
    rows,
):
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


def base_obs(
    obs,
):
    obs = np.asarray(
        obs,
        dtype=np.float64,
    )

    if obs.shape != (
        EXPECTED_OBS_DIM,
    ):
        raise RuntimeError(
            "Expected beta-conditioned 24D "
            f"observation, got {obs.shape}"
        )

    return obs[
        :BASE_OBS_DIM
    ].copy()


def finite(
    name,
    x,
):
    value = float(
        x
    )

    if not math.isfinite(
        value
    ):
        raise RuntimeError(
            f"{name} is not finite: {value}"
        )

    return value


def state_row(
    *,
    context_id,
    seed,
    horizon,
    obs0,
    obs,
    info,
    initial_goal_distance,
):
    # Frozen base observation:
    #
    # 0:3   oracle terrain context
    # 3     goal dx
    # 4     goal dy
    # 5     goal distance
    # 6     heading error
    # 7     vx
    # 8     vy
    # 9     yaw rate
    # 10    base z
    # 11    roll
    # 12    pitch
    # 13:17 applied physical reference
    # 17:21 previous normalized action

    progress = (
        initial_goal_distance
        - finite(
            "goal_distance",
            info[
                "goal_distance"
            ],
        )
    )

    return {
        "context_id":
            context_id,

        "terrain":
            "rough_perlin",

        "seed":
            int(seed),

        "probe_step":
            int(horizon),

        "probe_time_s":
            float(
                horizon
                * DECISION_DT_S
            ),

        "probe_progress_m":
            float(progress),

        "vx_mps":
            float(obs[7]),

        "vy_mps":
            float(obs[8]),

        "yaw_rate_rps":
            float(obs[9]),

        "base_z_m":
            float(obs[10]),

        "roll_rad":
            float(obs[11]),

        "pitch_rad":
            float(obs[12]),

        "delta_vx_mps":
            float(
                obs[7]
                - obs0[7]
            ),

        "delta_vy_mps":
            float(
                obs[8]
                - obs0[8]
            ),

        "delta_yaw_rate_rps":
            float(
                obs[9]
                - obs0[9]
            ),

        "delta_base_z_m":
            float(
                obs[10]
                - obs0[10]
            ),

        "delta_roll_rad":
            float(
                obs[11]
                - obs0[11]
            ),

        "delta_pitch_rad":
            float(
                obs[12]
                - obs0[12]
            ),

        "goal_distance_m":
            finite(
                "goal_distance",
                info[
                    "goal_distance"
                ],
            ),

        "heading_error_rad":
            finite(
                "heading_error",
                (
                    info.get(
                        "heading_error",
                        obs[6],
                    )
                ),
            ),

        "safety_state":
            str(
                info.get(
                    "safety_state",
                    "",
                )
            ),

        "override_active":
            int(
                bool(
                    info.get(
                        "override_active",
                        False,
                    )
                )
            ),

        "m4_intervention":
            int(
                bool(
                    info.get(
                        "m4_intervention",
                        False,
                    )
                )
            ),

        "eval_stance_slip":
            json.dumps(
                info.get(
                    "eval_stance_slip",
                    None,
                ),
                sort_keys=True,
            ),

        "traction_interval_contact_dt_s":
            (
                ""
                if info.get(
                    "traction_interval_contact_dt_s"
                ) is None
                else finite(
                    "traction_interval_contact_dt_s",
                    info[
                        "traction_interval_contact_dt_s"
                    ],
                )
            ),

        "traction_interval_cost":
            (
                ""
                if info.get(
                    "traction_interval_cost"
                ) is None
                else finite(
                    "traction_interval_cost",
                    info[
                        "traction_interval_cost"
                    ],
                )
            ),
    }


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_DIR}"
        )

    for path in (
        T55_PATH,
        PATCH_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )


    patch_manifest = json.loads(
        PATCH_MANIFEST.read_text()
    )

    if patch_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.3a patch manifest is "
            "not FREEZE_PASS."
        )

    if bool(
        patch_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Patch manifest reports heldout use."
        )


    original_seeds = [
        int(x)
        for x in patch_manifest[
            "original_seeds"
        ]
    ]

    extension_seeds = [
        int(x)
        for x in patch_manifest[
            "extension_seeds"
        ]
    ]

    if len(
        original_seeds
    ) != 18:
        raise RuntimeError(
            "Expected 18 original seeds."
        )

    if len(
        extension_seeds
    ) != 15:
        raise RuntimeError(
            "Expected 15 extension seeds."
        )


    seeds = (
        original_seeds
        + extension_seeds
    )

    if len(
        set(seeds)
    ) != 33:
        raise RuntimeError(
            "Expected 33 unique TRAIN contexts."
        )


    t55 = load_module(
        T55_PATH,
        "os_t6p8b_t55",
    )

    phase = t55.load_module(
        t55.T49_PATH,
        "os_t6p8b_t49",
    )


    rows = []

    safety_counts = {
        step: {}
        for step in range(
            PROBE_STEPS + 1
        )
    }


    for context_index, seed in enumerate(
        seeds
    ):
        context_id = (
            f"rough_seed_{seed}"
        )

        command_port = (
            BASE_PORT
            + 10
            * context_index
        )

        log_dir = (
            OUT_DIR
            / "env_logs"
            / context_id
        )


        env = phase.make_env(
            terrain="rough_perlin",
            beta=FIXED_BETA,
            command_port=command_port,
            log_dir=log_dir,
        )


        print()
        print(
            "=" * 104
        )
        print(
            f"{context_index + 1:02d}/"
            f"{len(seeds):02d} "
            f"{context_id}"
        )
        print(
            "=" * 104
        )


        try:
            obs, info = env.reset(
                seed=int(seed)
            )

            obs0 = base_obs(
                obs
            )

            initial_goal_distance = (
                finite(
                    "initial_goal_distance",
                    info[
                        "goal_distance"
                    ],
                )
            )


            row0 = state_row(
                context_id=context_id,
                seed=seed,
                horizon=0,
                obs0=obs0,
                obs=obs0,
                info=info,
                initial_goal_distance=(
                    initial_goal_distance
                ),
            )

            rows.append(
                row0
            )


            state = row0[
                "safety_state"
            ]

            safety_counts[
                0
            ][
                state
            ] = (
                safety_counts[
                    0
                ].get(
                    state,
                    0,
                )
                + 1
            )


            zero = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

            if zero.shape != (3,):
                raise RuntimeError(
                    "Expected 3D learned action space."
                )


            for step in range(
                1,
                PROBE_STEPS + 1,
            ):
                (
                    obs,
                    _reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    zero
                )

                if (
                    terminated
                    or truncated
                ):
                    raise RuntimeError(
                        f"{context_id}: probe "
                        f"terminated at step {step}."
                    )


                requested = np.asarray(
                    info[
                        "requested_normalized"
                    ],
                    dtype=np.float64,
                )

                if not np.array_equal(
                    requested,
                    np.zeros_like(
                        requested
                    ),
                ):
                    raise RuntimeError(
                        f"{context_id}: fixed probe "
                        f"is not zero normalized "
                        f"at step {step}: "
                        f"{requested}"
                    )


                obs_base = base_obs(
                    obs
                )

                row = state_row(
                    context_id=context_id,
                    seed=seed,
                    horizon=step,
                    obs0=obs0,
                    obs=obs_base,
                    info=info,
                    initial_goal_distance=(
                        initial_goal_distance
                    ),
                )

                rows.append(
                    row
                )


                state = row[
                    "safety_state"
                ]

                safety_counts[
                    step
                ][
                    state
                ] = (
                    safety_counts[
                        step
                    ].get(
                        state,
                        0,
                    )
                    + 1
                )


                print(
                    f"  t={step * DECISION_DT_S:.1f}s "
                    f"progress="
                    f"{row['probe_progress_m']:+.4f} "
                    f"vx={row['vx_mps']:+.4f} "
                    f"vy={row['vy_mps']:+.4f} "
                    f"yaw={row['yaw_rate_rps']:+.4f} "
                    f"dz={row['delta_base_z_m']:+.4f} "
                    f"roll={row['roll_rad']:+.4f} "
                    f"pitch={row['pitch_rad']:+.4f} "
                    f"safety={row['safety_state']}"
                )

        finally:
            env.close()


    expected_rows = (
        33
        * (
            PROBE_STEPS
            + 1
        )
    )

    if len(
        rows
    ) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"got {len(rows)}"
        )


    # ========================================================
    # Optional regression against T6.8a for the three
    # previously audited seeds.
    # ========================================================

    regression = {
        "available":
            False,
    }

    if T68A_CSV.exists():
        old = read_csv(
            T68A_CSV
        )

        comparisons = []

        for seed in (
            13,
            4,
            22,
        ):
            old_candidates = [
                row
                for row in old
                if (
                    int(
                        row[
                            "seed"
                        ]
                    )
                    == seed
                    and row[
                        "beta"
                    ]
                    == "lm100_ls000_le000"
                    and int(
                        row[
                            "repeat"
                        ]
                    )
                    == 0
                )
            ]

            if len(
                old_candidates
            ) != 1:
                raise RuntimeError(
                    f"T6.8a regression row "
                    f"missing for seed {seed}."
                )

            new_candidates = [
                row
                for row in rows
                if (
                    int(
                        row[
                            "seed"
                        ]
                    )
                    == seed
                    and int(
                        row[
                            "probe_step"
                        ]
                    )
                    == PROBE_STEPS
                )
            ]

            if len(
                new_candidates
            ) != 1:
                raise RuntimeError(
                    f"T6.8b boundary row "
                    f"missing for seed {seed}."
                )

            old_row = old_candidates[
                0
            ]

            new_row = new_candidates[
                0
            ]

            differences = {
                "vx":
                    abs(
                        float(
                            old_row[
                                "boundary_vx_mps"
                            ]
                        )
                        - float(
                            new_row[
                                "vx_mps"
                            ]
                        )
                    ),

                "vy":
                    abs(
                        float(
                            old_row[
                                "boundary_vy_mps"
                            ]
                        )
                        - float(
                            new_row[
                                "vy_mps"
                            ]
                        )
                    ),

                "yaw":
                    abs(
                        float(
                            old_row[
                                "boundary_yaw_rate_rps"
                            ]
                        )
                        - float(
                            new_row[
                                "yaw_rate_rps"
                            ]
                        )
                    ),

                "roll":
                    abs(
                        float(
                            old_row[
                                "boundary_roll_rad"
                            ]
                        )
                        - float(
                            new_row[
                                "roll_rad"
                            ]
                        )
                    ),

                "pitch":
                    abs(
                        float(
                            old_row[
                                "boundary_pitch_rad"
                            ]
                        )
                        - float(
                            new_row[
                                "pitch_rad"
                            ]
                        )
                    ),
            }

            max_diff = max(
                differences.values()
            )

            comparisons.append(
                {
                    "seed":
                        seed,

                    "max_abs_diff":
                        max_diff,
                }
            )


        regression = {
            "available":
                True,

            "comparisons":
                comparisons,

            "max_abs_diff":
                max(
                    x[
                        "max_abs_diff"
                    ]
                    for x in comparisons
                ),

            "pass":
                all(
                    x[
                        "max_abs_diff"
                    ]
                    <= EPS
                    for x in comparisons
                ),
        }

        if not regression[
            "pass"
        ]:
            raise RuntimeError(
                "T6.8a/T6.8b deterministic "
                f"regression failed: "
                f"{regression}"
            )


    # env log directories are created under OUT_DIR during
    # rollout, so OUT_DIR is expected to exist by this point.
    # The start-of-run guard still prevents reuse of an
    # existing output directory.
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUT_CSV.exists():
        raise RuntimeError(
            f"REFUSING TO OVERWRITE: {OUT_CSV}"
        )

    if OUT_MANIFEST.exists():
        raise RuntimeError(
            f"REFUSING TO OVERWRITE: {OUT_MANIFEST}"
        )

    write_csv(
        OUT_CSV,
        rows,
    )


    safety_summary = {}

    for step in range(
        PROBE_STEPS + 1
    ):
        safety_summary[
            str(step)
        ] = {
            "time_s":
                step
                * DECISION_DT_S,

            "counts":
                safety_counts[
                    step
                ],
        }


    manifest = {
        "schema":
            "icra27_os_t6p8b_prepolicy_probe_response_atlas_v0",

        "status":
            "FREEZE_PASS",

        "heldout_used":
            False,

        "contexts":
            33,

        "original_contexts":
            18,

        "extension_contexts":
            15,

        "probe_steps":
            PROBE_STEPS,

        "decision_dt_s":
            DECISION_DT_S,

        "probe_horizons_s":
            [
                step
                * DECISION_DT_S
                for step in range(
                    PROBE_STEPS + 1
                )
            ],

        "fixed_beta":
            list(
                FIXED_BETA
            ),

        "fixed_beta_semantics":
            (
                "Only required by the beta-conditioned "
                "environment wrapper. The five-step "
                "zero-normalized-action physical probe "
                "is beta-independent; T6.8a verified "
                "this empirically on three representative "
                "TRAIN terrains."
            ),

        "candidate_causal_probe_features":
            [
                "probe_progress_m",
                "delta_vx_mps",
                "delta_vy_mps",
                "delta_yaw_rate_rps",
                "delta_base_z_m",
                "delta_roll_rad",
                "delta_pitch_rad",
            ],

        "diagnostic_only_fields":
            [
                "safety_state",
                "override_active",
                "m4_intervention",
                "eval_stance_slip",
                "traction_interval_contact_dt_s",
                "traction_interval_cost",
            ],

        "safety_by_horizon":
            safety_summary,

        "t6p8a_regression":
            regression,

        "interpretation":
            (
                "This atlas freezes the robot response "
                "to a fixed nominal-motion probe before "
                "beta-conditioned policy action. It is "
                "for identifiability diagnosis only; "
                "it does not yet establish that active "
                "probing is safe or appropriate for the "
                "final Objective Selector."
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
        "ICRA27 OS-T6.8b PRE-POLICY "
        "PROBE RESPONSE ATLAS"
    )
    print("=" * 118)

    print(
        "contexts                              :",
        33,
    )

    print(
        "rows                                  :",
        len(rows),
    )

    print(
        "horizons_s                            :",
        manifest[
            "probe_horizons_s"
        ],
    )

    print(
        "T6.8a deterministic regression        :",
        regression,
    )

    print()
    print(
        "safety states by horizon:"
    )

    for step in range(
        PROBE_STEPS + 1
    ):
        print(
            f"  t={step * DECISION_DT_S:.1f}s "
            f"{safety_counts[step]}"
        )

    print()
    print(
        "[ICRA27] OS-T6.8b pre-policy "
        "probe response atlas: FREEZE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
