#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import socket
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


PHASE2B_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_phase2b_beta_train_archive_v0.py"
)

T3P4_PATH = (
    ROOT
    / "scripts"
    / "icra27"
    / "evaluate_os_t3p4_balanced_slip_v0.py"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v2_seed27027"
    / "phase1_beta_identifiable_selected_u20.pt"
)

DEFAULT_OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t3p5_beta_slip_intervention_v0"
    / "u0020_train36"
)


GROUPS = (
    {
        "name": "flat",
        "terrain": "flat",
        "seeds": (
            27100,
            27101,
            27102,
        ),
    },
    {
        "name": "low_friction",
        "terrain": "low_friction",
        "seeds": (
            27200,
            27201,
            27202,
        ),
    },
    {
        "name": "rough",
        "terrain": "rough_perlin",
        "seeds": (
            13,
            7,
            15,
        ),
    },
)


FIELDNAMES = (
    "beta_name",
    "beta_motion",
    "beta_stability",
    "beta_energy",
    "group",
    "terrain",
    "seed",

    "status",
    "success",
    "m4_terminal",
    "m4_interventions",
    "policy_steps",

    "decision_time_s",
    "energy_abs_j",

    "mean_cost_motion",
    "mean_cost_stability",
    "mean_cost_energy",

    "roll_rms_rad",
    "pitch_rms_rad",

    "mean_applied_vx_mps",

    "slip_speed_mean_mps",
    "slip_speed_rms_mps",
    "slip_speed_max_mps",
    "slip_contact_time_s",
    "slip_contact_samples",
    "slip_physics_steps_with_contact",
    "slip_per_foot_contact_samples",
    "slip_foot_geom_names",
)


def load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def udp_bindable(
    port: int,
) -> bool:
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        sock.bind(
            (
                "127.0.0.1",
                int(port),
            )
        )
        return True

    except OSError:
        return False

    finally:
        sock.close()


def required_port_offsets(
    beta_count: int,
):
    offsets = []

    for beta_index in range(
        beta_count
    ):
        for group_index in range(
            len(GROUPS)
        ):
            base = (
                100 * beta_index
                + 10 * group_index
            )

            offsets.extend(
                (
                    base,
                    base + 1,
                    base + 2,
                )
            )

    return tuple(offsets)


def port_layout_available(
    base_port: int,
    *,
    beta_count: int,
):
    occupied = []

    for offset in required_port_offsets(
        beta_count
    ):
        port = (
            int(base_port)
            + int(offset)
        )

        if not udp_bindable(
            port
        ):
            occupied.append(
                port
            )

    return occupied


def choose_base_port(
    requested: int,
    *,
    beta_count: int,
):
    if requested > 0:
        occupied = port_layout_available(
            requested,
            beta_count=beta_count,
        )

        if occupied:
            raise RuntimeError(
                "Requested UDP layout is occupied: "
                f"{occupied}"
            )

        return int(
            requested
        )

    # Automatic shared-workstation-safe search.
    #
    # Maximum offset is about +322 for 4 betas.
    for candidate in range(
        60000,
        64501,
        400,
    ):
        occupied = port_layout_available(
            candidate,
            beta_count=beta_count,
        )

        if not occupied:
            return int(
                candidate
            )

    raise RuntimeError(
        "Could not find a free UDP port layout."
    )


def write_rows(
    path: Path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in FIELDNAMES
                }
            )


def median(
    values,
):
    values = [
        float(x)
        for x in values
    ]

    if not values:
        raise RuntimeError(
            "Cannot compute median of empty values."
        )

    return float(
        statistics.median(
            values
        )
    )


def make_key(
    row,
):
    return (
        row["group"],
        int(row["seed"]),
        row["beta_name"],
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=0,
        help=(
            "0 = automatically find a free "
            "36-port evaluation layout."
        ),
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    args = ap.parse_args()

    checkpoint = (
        args.checkpoint.resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    phase2b = load_module(
        PHASE2B_PATH,
        "os_t3p5_phase2b",
    )

    t3p4 = load_module(
        T3P4_PATH,
        "os_t3p5_t3p4",
    )

    beta_bank = tuple(
        (
            str(name),
            tuple(
                float(x)
                for x in beta
            ),
        )
        for name, beta
        in phase2b.BETA_BANK
    )

    beta_names = tuple(
        name
        for name, _beta
        in beta_bank
    )

    expected = {
        "balanced",
        "motion",
        "stability",
        "energy",
    }

    if set(beta_names) != expected:
        raise RuntimeError(
            "Unexpected Phase-2B beta bank: "
            f"{beta_names}"
        )

    policy = phase2b.base.load_policy(
        name="beta_conditioned",
        path=checkpoint,
        expected_reward_mode=(
            "tracer_cost_v2"
        ),
    )

    phase2b.validate_checkpoint_beta_bank(
        policy["payload"]
    )

    out_dir = (
        args.out_dir.resolve()
    )

    episodes_path = (
        out_dir
        / "episodes.csv"
    )

    summary_path = (
        out_dir
        / "summary.json"
    )

    # Avoid silently mixing two runs.
    if (
        episodes_path.exists()
        or summary_path.exists()
    ):
        raise RuntimeError(
            "Output already contains diagnostic "
            "results; use a new --out-dir: "
            f"{out_dir}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_port = choose_base_port(
        args.base_port,
        beta_count=len(beta_bank),
    )

    print("=" * 100)
    print(
        "ICRA27 OS-T3.5 BETA × TERRAIN "
        "STANCE-SLIP INTERVENTION DIAGNOSTIC"
    )
    print("=" * 100)

    print(
        "checkpoint :",
        checkpoint,
    )

    print(
        "beta bank  :",
        beta_bank,
    )

    print(
        "scope      : TRAIN-only, "
        "4 beta x 3 terrain x 3 seed = 36"
    )

    print(
        "base port  :",
        base_port,
    )

    print(
        "primary    : paired delta "
        "stance-slip RMS vs balanced"
    )

    print(
        "output     :",
        out_dir,
    )

    print()

    rows = []

    for beta_index, (
        beta_name,
        beta,
    ) in enumerate(
        beta_bank
    ):
        print(
            f"BETA {beta_name}: "
            f"[{beta[0]:.3f}, "
            f"{beta[1]:.3f}, "
            f"{beta[2]:.3f}]"
        )

        for group_index, group in enumerate(
            GROUPS
        ):
            group_name = group[
                "name"
            ]

            terrain = group[
                "terrain"
            ]

            seeds = group[
                "seeds"
            ]

            command_port = (
                base_port
                + 100 * beta_index
                + 10 * group_index
            )

            log_dir = (
                out_dir
                / "env_logs"
                / beta_name
                / group_name
            )

            env = phase2b.make_env(
                terrain=terrain,
                beta=beta,
                command_port=command_port,
                log_dir=log_dir,
            )

            print(
                f"  [{group_name}] "
                f"terrain={terrain} "
                f"seeds={list(seeds)} "
                f"ports="
                f"{command_port}:"
                f"{command_port + 2}"
            )

            try:
                for seed in seeds:
                    base_row = (
                        phase2b.base.run_episode(
                            env,
                            policy_name=(
                                "beta_conditioned"
                            ),
                            checkpoint=(
                                policy["path"]
                            ),
                            trained_reward_mode=(
                                policy[
                                    "trained_reward_mode"
                                ]
                            ),
                            model=(
                                policy["model"]
                            ),
                            group=(
                                f"os_t3p5_"
                                f"{group_name}"
                            ),
                            terrain=terrain,
                            seed=seed,
                        )
                    )

                    slip = (
                        t3p4.extract_slip(
                            env
                        )
                    )

                    row = {
                        "beta_name":
                            beta_name,

                        "beta_motion":
                            float(beta[0]),

                        "beta_stability":
                            float(beta[1]),

                        "beta_energy":
                            float(beta[2]),

                        "group":
                            group_name,

                        "terrain":
                            terrain,

                        "seed":
                            int(seed),

                        "status":
                            base_row.get(
                                "status"
                            ),

                        "success":
                            bool(
                                base_row.get(
                                    "success",
                                    False,
                                )
                            ),

                        "m4_terminal":
                            bool(
                                base_row.get(
                                    "m4_terminal",
                                    False,
                                )
                            ),

                        "m4_interventions":
                            base_row.get(
                                "m4_interventions"
                            ),

                        "policy_steps":
                            base_row.get(
                                "policy_steps"
                            ),

                        "decision_time_s":
                            base_row.get(
                                "decision_time_s"
                            ),

                        "energy_abs_j":
                            base_row.get(
                                "energy_abs_j"
                            ),

                        "mean_cost_motion":
                            base_row.get(
                                "mean_cost_motion"
                            ),

                        "mean_cost_stability":
                            base_row.get(
                                "mean_cost_stability"
                            ),

                        "mean_cost_energy":
                            base_row.get(
                                "mean_cost_energy"
                            ),

                        "roll_rms_rad":
                            base_row.get(
                                "roll_rms_rad"
                            ),

                        "pitch_rms_rad":
                            base_row.get(
                                "pitch_rms_rad"
                            ),

                        "mean_applied_vx_mps":
                            base_row.get(
                                "mean_applied_vx_mps"
                            ),

                        **slip,
                    }

                    rows.append(
                        row
                    )

                    # Incremental snapshot: preserve completed
                    # episodes if a later runtime fails.
                    write_rows(
                        episodes_path,
                        rows,
                    )

                    print(
                        f"    seed={seed:<6} "
                        f"success={row['success']} "
                        f"Cs="
                        f"{float(row['mean_cost_stability']):.6f} "
                        f"vx="
                        f"{float(row['mean_applied_vx_mps']):.4f} "
                        f"slip_rms="
                        f"{float(row['slip_speed_rms_mps']):.6f} "
                        f"slip_mean="
                        f"{float(row['slip_speed_mean_mps']):.6f}"
                    )

            finally:
                env.close()

        print()

    if len(rows) != 36:
        raise RuntimeError(
            f"Expected 36 episodes, got {len(rows)}"
        )

    row_map = {
        make_key(row): row
        for row in rows
    }

    if len(row_map) != 36:
        raise RuntimeError(
            "Duplicate diagnostic context rows."
        )

    summary = {
        "schema":
            "icra27_os_t3p5_beta_slip_intervention_v0",

        "scope":
            "TRAIN-only paired diagnostic",

        "checkpoint":
            str(checkpoint),

        "base_port":
            int(base_port),

        "beta_bank":
            {
                name: list(beta)
                for name, beta
                in beta_bank
            },

        "primary_metric":
            "slip_speed_rms_mps",

        "reference_beta":
            "balanced",

        "all_episode_success":
            all(
                bool(row["success"])
                for row in rows
            ),

        "terrain_beta_medians":
            {},

        "paired_interventions_vs_balanced":
            {},
    }

    # --------------------------------------------------------
    # Raw median behavior for each beta / terrain.
    # --------------------------------------------------------
    for group in GROUPS:
        group_name = group[
            "name"
        ]

        summary[
            "terrain_beta_medians"
        ][group_name] = {}

        for beta_name, _beta in beta_bank:
            subset = [
                row
                for row in rows
                if (
                    row["group"]
                    == group_name
                    and row["beta_name"]
                    == beta_name
                )
            ]

            if len(subset) != 3:
                raise RuntimeError(
                    "Expected three rows for "
                    f"{group_name}/{beta_name}, "
                    f"got {len(subset)}"
                )

            summary[
                "terrain_beta_medians"
            ][group_name][
                beta_name
            ] = {
                "slip_rms_mps":
                    median(
                        row[
                            "slip_speed_rms_mps"
                        ]
                        for row in subset
                    ),

                "slip_mean_mps":
                    median(
                        row[
                            "slip_speed_mean_mps"
                        ]
                        for row in subset
                    ),

                "attitude_cost":
                    median(
                        row[
                            "mean_cost_stability"
                        ]
                        for row in subset
                    ),

                "motion_cost":
                    median(
                        row[
                            "mean_cost_motion"
                        ]
                        for row in subset
                    ),

                "energy_cost":
                    median(
                        row[
                            "mean_cost_energy"
                        ]
                        for row in subset
                    ),

                "energy_abs_j":
                    median(
                        row[
                            "energy_abs_j"
                        ]
                        for row in subset
                    ),

                "applied_vx_mps":
                    median(
                        row[
                            "mean_applied_vx_mps"
                        ]
                        for row in subset
                    ),
            }

    # --------------------------------------------------------
    # Paired beta intervention relative to balanced.
    #
    # delta = intervention - balanced
    #
    # Negative slip delta      -> less slip
    # Negative attitude delta  -> better posture metric
    # Negative energy delta    -> lower normalized energy cost
    # Positive vx delta        -> faster applied command
    # --------------------------------------------------------
    intervention_names = (
        "motion",
        "stability",
        "energy",
    )

    for group in GROUPS:
        group_name = group[
            "name"
        ]

        summary[
            "paired_interventions_vs_balanced"
        ][group_name] = {}

        for beta_name in intervention_names:
            paired = []

            for seed in group[
                "seeds"
            ]:
                ref = row_map[
                    (
                        group_name,
                        int(seed),
                        "balanced",
                    )
                ]

                test = row_map[
                    (
                        group_name,
                        int(seed),
                        beta_name,
                    )
                ]

                paired.append(
                    {
                        "seed":
                            int(seed),

                        "delta_slip_rms_mps":
                            float(
                                test[
                                    "slip_speed_rms_mps"
                                ]
                                - ref[
                                    "slip_speed_rms_mps"
                                ]
                            ),

                        "delta_slip_mean_mps":
                            float(
                                test[
                                    "slip_speed_mean_mps"
                                ]
                                - ref[
                                    "slip_speed_mean_mps"
                                ]
                            ),

                        "delta_attitude_cost":
                            float(
                                test[
                                    "mean_cost_stability"
                                ]
                                - ref[
                                    "mean_cost_stability"
                                ]
                            ),

                        "delta_motion_cost":
                            float(
                                test[
                                    "mean_cost_motion"
                                ]
                                - ref[
                                    "mean_cost_motion"
                                ]
                            ),

                        "delta_energy_cost":
                            float(
                                test[
                                    "mean_cost_energy"
                                ]
                                - ref[
                                    "mean_cost_energy"
                                ]
                            ),

                        "delta_energy_abs_j":
                            float(
                                test[
                                    "energy_abs_j"
                                ]
                                - ref[
                                    "energy_abs_j"
                                ]
                            ),

                        "delta_applied_vx_mps":
                            float(
                                test[
                                    "mean_applied_vx_mps"
                                ]
                                - ref[
                                    "mean_applied_vx_mps"
                                ]
                            ),
                    }
                )

            def med(name):
                return median(
                    item[name]
                    for item in paired
                )

            summary[
                "paired_interventions_vs_balanced"
            ][group_name][
                beta_name
            ] = {
                "pairs":
                    paired,

                "median_delta_slip_rms_mps":
                    med(
                        "delta_slip_rms_mps"
                    ),

                "median_delta_slip_mean_mps":
                    med(
                        "delta_slip_mean_mps"
                    ),

                "median_delta_attitude_cost":
                    med(
                        "delta_attitude_cost"
                    ),

                "median_delta_motion_cost":
                    med(
                        "delta_motion_cost"
                    ),

                "median_delta_energy_cost":
                    med(
                        "delta_energy_cost"
                    ),

                "median_delta_energy_abs_j":
                    med(
                        "delta_energy_abs_j"
                    ),

                "median_delta_applied_vx_mps":
                    med(
                        "delta_applied_vx_mps"
                    ),

                "slip_rms_lower_count":
                    sum(
                        int(
                            item[
                                "delta_slip_rms_mps"
                            ] < 0.0
                        )
                        for item in paired
                    ),

                "attitude_cost_lower_count":
                    sum(
                        int(
                            item[
                                "delta_attitude_cost"
                            ] < 0.0
                        )
                        for item in paired
                    ),

                "energy_cost_lower_count":
                    sum(
                        int(
                            item[
                                "delta_energy_cost"
                            ] < 0.0
                        )
                        for item in paired
                    ),

                "motion_cost_lower_count":
                    sum(
                        int(
                            item[
                                "delta_motion_cost"
                            ] < 0.0
                        )
                        for item in paired
                    ),
            }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=" * 100)
    print(
        "OS-T3.5 PAIRED INTERVENTION SUMMARY"
    )
    print("=" * 100)

    for group in GROUPS:
        group_name = group[
            "name"
        ]

        print()
        print(
            f"TERRAIN: {group_name}"
        )

        print(
            "  beta       "
            "dSlipRMS       "
            "dAttCost       "
            "dEnergyCost    "
            "dVx            "
            "slip<bal att<bal"
        )

        for beta_name in intervention_names:
            item = summary[
                "paired_interventions_vs_balanced"
            ][group_name][beta_name]

            print(
                f"  {beta_name:<10} "
                f"{item['median_delta_slip_rms_mps']:+.6f}  "
                f"{item['median_delta_attitude_cost']:+.6f}  "
                f"{item['median_delta_energy_cost']:+.6f}  "
                f"{item['median_delta_applied_vx_mps']:+.6f}   "
                f"{item['slip_rms_lower_count']}/3      "
                f"{item['attitude_cost_lower_count']}/3"
            )

    print()
    print(
        "all success :",
        summary[
            "all_episode_success"
        ],
    )

    print(
        "episodes.csv:",
        episodes_path,
    )

    print(
        "summary.json:",
        summary_path,
    )


if __name__ == "__main__":
    main()
