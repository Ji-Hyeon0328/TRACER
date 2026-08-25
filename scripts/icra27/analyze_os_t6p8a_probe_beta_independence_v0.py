from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "results/icra27"
    / "os_t6p5a_exact_repeat_determinism_v0"
    / "env_logs"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t6p8a_probe_beta_independence_v0"
)

OUT_ROWS = OUT_DIR / "probe_beta_independence.csv"
OUT_MANIFEST = (
    OUT_DIR
    / "probe_beta_independence_manifest.json"
)


SEEDS = (13, 4, 22)

BETAS = (
    "lm100_ls000_le000",
    "lm000_ls100_le000",
    "lm000_ls000_le100",
    "lm040_ls040_le020",
)

REPEATS = 3
PROBE_STEPS = 5


def normalize(x):
    if isinstance(x, dict):
        return {
            str(k): normalize(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [
            normalize(v)
            for v in x
        ]

    if isinstance(x, np.generic):
        return normalize(x.item())

    return x


def digest(x):
    payload = json.dumps(
        normalize(x),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def load_log(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def probe_signature(rows):
    if len(rows) < PROBE_STEPS + 2:
        raise RuntimeError(
            "Episode log is too short."
        )

    if rows[0].get("event") != "reset":
        raise RuntimeError(
            "First row is not reset."
        )

    steps = [
        row
        for row in rows[1:]
        if row.get("event") == "step"
    ]

    if len(steps) < PROBE_STEPS + 1:
        raise RuntimeError(
            "Insufficient step rows."
        )

    probe = steps[:PROBE_STEPS]

    # Fixed pre-policy phase must really be zero normalized
    # command.  This is more robust than assuming from name.
    for i, row in enumerate(
        probe,
        1,
    ):
        requested = np.asarray(
            row["requested_normalized"],
            dtype=np.float64,
        )

        if not np.array_equal(
            requested,
            np.zeros_like(
                requested
            ),
        ):
            raise RuntimeError(
                f"Probe step {i} is not zero action: "
                f"{requested}"
            )

    # Exclude reward/objective scalarization fields.
    # Keep only physical/state/transport diagnostics.
    reset = rows[0]

    signature = {
        "reset": {
            "observation":
                reset["observation"],

            "goal_distance":
                reset["goal_distance"],

            "goal_world":
                reset["goal_world"],

            "oracle_context":
                reset["oracle_context"],

            "safety_state":
                reset["safety_state"],

            "override_active":
                reset["override_active"],
        },

        "probe_steps": [],
    }

    physical_keys = (
        "episode_step",
        "observation",

        "requested_normalized",
        "requested_physical",

        "applied_normalized",
        "applied_physical",

        "goal_distance",
        "goal_dx_body",
        "goal_dy_body",
        "heading_error",

        "safety_state",
        "override_active",
        "override_reasons",

        "eval_stance_slip",

        "traction_interval_contact_dt_s",
        "traction_interval_cost",
        "traction_interval_cost_integral_s",

        "pympc_robot_height_estimate",
        "pympc_terrain_height_estimate_world",
        "pympc_height_estimate_phase",

        "m4_intervention",
        "m4_terminal",
        "terminated",
        "truncated",
        "success",
    )

    for row in probe:
        signature[
            "probe_steps"
        ].append(
            {
                key: row.get(key)
                for key in physical_keys
            }
        )

    return (
        signature,
        probe[-1],
        steps[PROBE_STEPS],
    )


def write_csv(path, rows):
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
        writer.writerows(rows)


def main():
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    if not SOURCE.exists():
        raise FileNotFoundError(
            SOURCE
        )

    output = []

    for seed in SEEDS:
        reference_hash = None
        reference_boundary = None

        seed_rows = []

        for beta in BETAS:
            for repeat in range(
                REPEATS
            ):
                log_dir = (
                    SOURCE
                    / f"seed_{seed}"
                    / beta
                    / f"repeat_{repeat}"
                )

                paths = list(
                    log_dir.glob(
                        "episode_*.jsonl"
                    )
                )

                if len(paths) != 1:
                    raise RuntimeError(
                        f"Expected one log in {log_dir}, "
                        f"got {paths}"
                    )

                rows = load_log(
                    paths[0]
                )

                (
                    signature,
                    boundary,
                    first_policy,
                ) = probe_signature(
                    rows
                )

                h = digest(
                    signature
                )

                obs = np.asarray(
                    boundary[
                        "observation"
                    ],
                    dtype=np.float64,
                )

                if obs.shape != (21,):
                    raise RuntimeError(
                        f"Expected 21D base observation; "
                        f"got {obs.shape}"
                    )

                # 21D base-observation contract:
                # 0:3 terrain context
                # 3:7 goal
                # 7:10 body velocity
                # 10 z
                # 11 roll
                # 12 pitch
                # 13:17 applied command
                # 17:21 previous normalized action

                if reference_hash is None:
                    reference_hash = h
                    reference_boundary = (
                        obs.copy()
                    )

                max_boundary_diff = float(
                    np.max(
                        np.abs(
                            obs
                            - reference_boundary
                        )
                    )
                )

                probe_progress = float(
                    rows[0][
                        "goal_distance"
                    ]
                    - boundary[
                        "goal_distance"
                    ]
                )

                seed_rows.append(
                    {
                        "seed":
                            seed,

                        "beta":
                            beta,

                        "repeat":
                            repeat,

                        "probe_sha256":
                            h,

                        "matches_seed_reference":
                            int(
                                h
                                == reference_hash
                            ),

                        "boundary_max_abs_diff":
                            max_boundary_diff,

                        "probe_progress_m":
                            probe_progress,

                        "boundary_vx_mps":
                            float(obs[7]),

                        "boundary_vy_mps":
                            float(obs[8]),

                        "boundary_yaw_rate_rps":
                            float(obs[9]),

                        "boundary_base_z_m":
                            float(obs[10]),

                        "boundary_roll_rad":
                            float(obs[11]),

                        "boundary_pitch_rad":
                            float(obs[12]),

                        "boundary_safety_state":
                            str(
                                boundary[
                                    "safety_state"
                                ]
                            ),

                        "first_policy_requested_normalized":
                            json.dumps(
                                first_policy[
                                    "requested_normalized"
                                ]
                            ),
                    }
                )

        hashes = {
            row["probe_sha256"]
            for row in seed_rows
        }

        exact = (
            len(hashes)
            == 1
        )

        print()
        print(
            f"seed={seed} "
            f"unique physical probe hashes={len(hashes)} "
            f"beta-independent={exact}"
        )

        for row in seed_rows[
            ::REPEATS
        ]:
            print(
                f"  {row['beta']:<20} "
                f"hash={row['probe_sha256'][:12]} "
                f"progress={row['probe_progress_m']:.4f} "
                f"vx={row['boundary_vx_mps']:+.4f} "
                f"vy={row['boundary_vy_mps']:+.4f} "
                f"yaw={row['boundary_yaw_rate_rps']:+.4f} "
                f"roll={row['boundary_roll_rad']:+.4f} "
                f"pitch={row['boundary_pitch_rad']:+.4f} "
                f"safety={row['boundary_safety_state']}"
            )

        output.extend(
            seed_rows
        )


    per_seed_exact = {}

    for seed in SEEDS:
        rows = [
            row
            for row in output
            if row["seed"] == seed
        ]

        per_seed_exact[
            str(seed)
        ] = (
            len(
                {
                    row[
                        "probe_sha256"
                    ]
                    for row in rows
                }
            )
            == 1
        )


    aggregate = {
        "seeds":
            len(SEEDS),

        "beta_points":
            len(BETAS),

        "repeats_per_beta":
            REPEATS,

        "episodes_checked":
            len(output),

        "probe_steps":
            PROBE_STEPS,

        "probe_duration_s":
            1.0,

        "all_seeds_beta_independent":
            bool(
                all(
                    per_seed_exact.values()
                )
            ),

        "per_seed_beta_independent":
            per_seed_exact,

        "max_boundary_abs_diff":
            float(
                max(
                    row[
                        "boundary_max_abs_diff"
                    ]
                    for row in output
                )
            ),

        "mean_probe_progress_m":
            float(
                np.mean(
                    [
                        row[
                            "probe_progress_m"
                        ]
                        for row in output
                    ]
                )
            ),

        "max_probe_progress_m":
            float(
                max(
                    row[
                        "probe_progress_m"
                    ]
                    for row in output
                )
            ),
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        OUT_ROWS,
        output,
    )

    manifest = {
        "schema":
            "icra27_os_t6p8a_probe_beta_independence_v0",

        "status":
            "COMPUTE_PASS",

        "heldout_used":
            False,

        "source":
            (
                "T6.5a exact-repeat TRAIN-only "
                "diagnostic episodes."
            ),

        "probe_definition":
            (
                "Five fixed zero-normalized-action "
                "steps before the first learned-policy "
                "action. Because zero normalized action "
                "maps to nominal forward locomotion, "
                "this is an active nominal-motion probe, "
                "not passive settling."
            ),

        "excluded_from_probe_hash":
            (
                "Reward/objective scalarization fields, "
                "which legitimately depend on beta."
            ),

        "aggregate":
            aggregate,

        "interpretation":
            (
                "If physical probe trajectories are "
                "identical across beta for each terrain "
                "seed, probe-derived state is causally "
                "available before beta selection. This "
                "establishes only an identifiability "
                "diagnostic and does not yet justify "
                "deploying a one-second active probe."
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
        "ICRA27 OS-T6.8a PRE-POLICY PROBE "
        "BETA-INDEPENDENCE AUDIT"
    )
    print("=" * 118)

    for key, value in aggregate.items():
        print(
            f"  {key:<38}: {value}"
        )

    print()
    print(
        "[ICRA27] OS-T6.8a probe "
        "beta-independence: COMPUTE PASS"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
