from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SOURCE_SCRIPT = (
    ROOT
    / "scripts/icra27"
    / "freeze_os_t6p8b_prepolicy_probe_response_atlas_v0.py"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p8b_prepolicy_probe_response_atlas_v0"
)

ENV_LOG_DIR = (
    OUT_DIR
    / "env_logs"
)

OUT_CSV = (
    OUT_DIR
    / "prepolicy_probe_response_atlas.csv"
)

OUT_MANIFEST = (
    OUT_DIR
    / "prepolicy_probe_response_manifest.json"
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
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_jsonl(path):
    rows = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def write_csv_new(path, rows):
    if path.exists():
        raise RuntimeError(
            f"REFUSING TO OVERWRITE: {path}"
        )

    if not rows:
        raise RuntimeError(
            "No recovered rows."
        )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    if tmp.exists():
        raise RuntimeError(
            f"Temporary file already exists: {tmp}"
        )

    with tmp.open(
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
        writer.writerows(rows)

    tmp.rename(path)


def main():
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(
            SOURCE_SCRIPT
        )

    if not OUT_DIR.exists():
        raise RuntimeError(
            "Expected existing T6.8b output "
            f"directory: {OUT_DIR}"
        )

    if not ENV_LOG_DIR.exists():
        raise RuntimeError(
            f"Missing env_logs: {ENV_LOG_DIR}"
        )

    if OUT_CSV.exists():
        raise RuntimeError(
            f"REFUSING TO OVERWRITE: {OUT_CSV}"
        )

    if OUT_MANIFEST.exists():
        raise RuntimeError(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_MANIFEST}"
        )


    src = load_module(
        SOURCE_SCRIPT,
        "os_t6p8b_recovery_source",
    )


    patch_manifest = json.loads(
        src.PATCH_MANIFEST.read_text()
    )

    if patch_manifest.get(
        "status"
    ) != "FREEZE_PASS":
        raise RuntimeError(
            "T6.3a patch manifest is not "
            "FREEZE_PASS."
        )

    if bool(
        patch_manifest.get(
            "heldout_used",
            False,
        )
    ):
        raise RuntimeError(
            "Patch source reports heldout use."
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

    if len(original_seeds) != 18:
        raise RuntimeError(
            "Expected 18 original contexts."
        )

    if len(extension_seeds) != 15:
        raise RuntimeError(
            "Expected 15 extension contexts."
        )


    seeds = (
        original_seeds
        + extension_seeds
    )

    if len(seeds) != 33:
        raise RuntimeError(
            "Expected 33 contexts."
        )

    if len(set(seeds)) != 33:
        raise RuntimeError(
            "Context seeds are not unique."
        )


    all_episode_logs = sorted(
        ENV_LOG_DIR.glob(
            "rough_seed_*/episode_*.jsonl"
        )
    )

    if len(all_episode_logs) != 33:
        raise RuntimeError(
            "Expected exactly 33 episode logs, "
            f"found {len(all_episode_logs)}."
        )


    recovered = []

    safety_counts = {
        step: {}
        for step in range(
            src.PROBE_STEPS + 1
        )
    }


    for index, seed in enumerate(
        seeds
    ):
        context_id = (
            f"rough_seed_{seed}"
        )

        log_dir = (
            ENV_LOG_DIR
            / context_id
        )

        paths = sorted(
            log_dir.glob(
                "episode_*.jsonl"
            )
        )

        if len(paths) != 1:
            raise RuntimeError(
                f"{context_id}: expected exactly "
                f"one episode log, got {paths}"
            )


        log_rows = load_jsonl(
            paths[0]
        )

        if len(log_rows) != (
            src.PROBE_STEPS
            + 1
        ):
            raise RuntimeError(
                f"{context_id}: expected "
                f"{src.PROBE_STEPS + 1} JSONL rows "
                f"(reset + 5 probe), "
                f"got {len(log_rows)}"
            )


        reset = log_rows[0]

        if reset.get(
            "event"
        ) != "reset":
            raise RuntimeError(
                f"{context_id}: first row "
                "is not reset."
            )


        obs0 = np.asarray(
            reset[
                "observation"
            ],
            dtype=np.float64,
        )

        if obs0.shape != (
            src.BASE_OBS_DIM,
        ):
            raise RuntimeError(
                f"{context_id}: reset base "
                f"observation shape={obs0.shape}"
            )


        initial_goal_distance = (
            src.finite(
                "initial_goal_distance",
                reset[
                    "goal_distance"
                ],
            )
        )


        row0 = src.state_row(
            context_id=context_id,
            seed=seed,
            horizon=0,
            obs0=obs0,
            obs=obs0,
            info=reset,
            initial_goal_distance=(
                initial_goal_distance
            ),
        )

        recovered.append(
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


        for step in range(
            1,
            src.PROBE_STEPS + 1,
        ):
            log_row = log_rows[
                step
            ]

            if log_row.get(
                "event"
            ) != "step":
                raise RuntimeError(
                    f"{context_id}: row {step} "
                    "is not a step."
                )


            requested = np.asarray(
                log_row[
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
                    f"{context_id}: probe step "
                    f"{step} was not zero normalized: "
                    f"{requested}"
                )


            obs = np.asarray(
                log_row[
                    "observation"
                ],
                dtype=np.float64,
            )

            if obs.shape != (
                src.BASE_OBS_DIM,
            ):
                raise RuntimeError(
                    f"{context_id}: step {step} "
                    f"base obs shape={obs.shape}"
                )


            row = src.state_row(
                context_id=context_id,
                seed=seed,
                horizon=step,
                obs0=obs0,
                obs=obs,
                info=log_row,
                initial_goal_distance=(
                    initial_goal_distance
                ),
            )

            recovered.append(
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
            f"{index + 1:02d}/33 "
            f"{context_id:<16} "
            f"boundary_progress="
            f"{recovered[-1]['probe_progress_m']:.4f} "
            f"pitch="
            f"{recovered[-1]['pitch_rad']:+.4f} "
            f"safety="
            f"{recovered[-1]['safety_state']}"
        )


    expected_rows = (
        33
        * (
            src.PROBE_STEPS
            + 1
        )
    )

    if len(recovered) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} recovered "
            f"rows, got {len(recovered)}"
        )


    # ========================================================
    # Deterministic regression to T6.8a.
    # ========================================================

    regression = {
        "available":
            False,
    }

    if src.T68A_CSV.exists():

        with src.T68A_CSV.open(
            "r",
            newline="",
        ) as f:
            old = list(
                csv.DictReader(f)
            )


        comparisons = []

        for seed in (
            13,
            4,
            22,
        ):
            old_rows = [
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

            if len(old_rows) != 1:
                raise RuntimeError(
                    f"T6.8a reference missing "
                    f"for seed {seed}"
                )


            new_rows = [
                row
                for row in recovered
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
                    == src.PROBE_STEPS
                )
            ]

            if len(new_rows) != 1:
                raise RuntimeError(
                    f"Recovered boundary missing "
                    f"for seed {seed}"
                )


            old_row = old_rows[0]
            new_row = new_rows[0]

            diffs = [
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
            ]

            comparisons.append(
                {
                    "seed":
                        seed,

                    "max_abs_diff":
                        max(
                            diffs
                        ),
                }
            )


        regression = {
            "available":
                True,

            "comparisons":
                comparisons,

            "max_abs_diff":
                max(
                    row[
                        "max_abs_diff"
                    ]
                    for row in comparisons
                ),

            "pass":
                all(
                    row[
                        "max_abs_diff"
                    ]
                    <= src.EPS
                    for row in comparisons
                ),
        }

        if not regression[
            "pass"
        ]:
            raise RuntimeError(
                "T6.8a deterministic regression "
                f"failed: {regression}"
            )


    safety_summary = {}

    for step in range(
        src.PROBE_STEPS + 1
    ):
        safety_summary[
            str(step)
        ] = {
            "time_s":
                step
                * src.DECISION_DT_S,

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
            src.PROBE_STEPS,

        "decision_dt_s":
            src.DECISION_DT_S,

        "probe_horizons_s":
            [
                step
                * src.DECISION_DT_S
                for step in range(
                    src.PROBE_STEPS + 1
                )
            ],

        "fixed_beta":
            list(
                src.FIXED_BETA
            ),

        "fixed_beta_semantics":
            (
                "Only required by the beta-conditioned "
                "environment wrapper. T6.8a verified "
                "that the fixed five-step physical "
                "probe is beta-independent on the "
                "audited representative TRAIN terrains."
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

        "recovery":
            {
                "performed":
                    True,

                "reason":
                    (
                        "Original T6.8b rollout completed "
                        "all 33 contexts but failed at the "
                        "final OUT_DIR.mkdir because "
                        "environment logging had already "
                        "created the directory."
                    ),

                "simulation_rerun":
                    False,

                "source_logs":
                    33,
            },

        "interpretation":
            (
                "This atlas freezes the robot response "
                "to a fixed nominal-motion probe before "
                "beta-conditioned learned-policy action. "
                "It is an identifiability diagnostic; "
                "it does not establish that active "
                "probing is safe or appropriate for the "
                "final Objective Selector."
            ),
    }


    write_csv_new(
        OUT_CSV,
        recovered,
    )


    if OUT_MANIFEST.exists():
        raise RuntimeError(
            f"REFUSING TO OVERWRITE: "
            f"{OUT_MANIFEST}"
        )

    tmp_manifest = (
        OUT_MANIFEST.with_suffix(
            OUT_MANIFEST.suffix
            + ".tmp"
        )
    )

    if tmp_manifest.exists():
        raise RuntimeError(
            f"Temporary manifest exists: "
            f"{tmp_manifest}"
        )

    tmp_manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    tmp_manifest.rename(
        OUT_MANIFEST
    )


    print()
    print("=" * 118)
    print(
        "ICRA27 OS-T6.8b RECOVERED "
        "PRE-POLICY PROBE RESPONSE ATLAS"
    )
    print("=" * 118)

    print(
        "contexts                              :",
        33,
    )

    print(
        "rows                                  :",
        len(
            recovered
        ),
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
        src.PROBE_STEPS + 1
    ):
        print(
            f"  t="
            f"{step * src.DECISION_DT_S:.1f}s "
            f"{safety_counts[step]}"
        )

    print()
    print(
        "[ICRA27] OS-T6.8b recovered probe "
        "response atlas: FREEZE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
