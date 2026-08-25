from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SOURCE_CSV = (
    ROOT
    / "results/icra27"
    / "os_t4p8d_v4_lockstep_checkpoint_sweep_v0"
    / "train9"
    / "episodes.csv"
)

RAW_ROOT = (
    ROOT
    / "results/icra27"
    / "os_t4p8d_v4_lockstep_checkpoint_sweep_v0"
    / "train9"
    / "u30"
    / "env_logs"
)

OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p2a2_train_raw_physical_alignment_v0"
)

EPISODE_CSV = OUT_DIR / "episode_physical_metrics.csv"
PAIRED_CSV = OUT_DIR / "paired_physical_alignment.csv"
MANIFEST = OUT_DIR / "raw_physical_alignment_manifest.json"

TOL = 1.0e-10


def f(x: Any) -> float:
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"Non-finite value: {x!r}")
    return y


def read_compact_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open(newline="") as fp:
        rows = list(csv.DictReader(fp))

    rows = [
        row
        for row in rows
        if int(float(row["update"])) == 30
    ]

    if len(rows) != 36:
        raise RuntimeError(
            f"Expected 36 u30 compact rows, got {len(rows)}"
        )

    return rows


def load_steps(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open() as fp:
        for line in fp:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if obj.get("event") == "step":
                rows.append(obj)

    return rows


def histogram_tail_time(
    histogram: list[dict[str, Any]],
    threshold: float,
) -> float:
    total = 0.0

    for item in histogram:
        low = item.get("low_mps")

        if low is None:
            continue

        if f(low) + TOL >= threshold:
            total += f(
                item.get(
                    "contact_time_s",
                    0.0,
                )
            )

    return total


def subtract_hist_tail(
    final_hist: list[dict[str, Any]],
    boundary_hist: list[dict[str, Any]],
    threshold: float,
) -> float:
    return (
        histogram_tail_time(
            final_hist,
            threshold,
        )
        - histogram_tail_time(
            boundary_hist,
            threshold,
        )
    )


def rms(values: list[float]) -> float:
    if not values:
        raise ValueError("Empty RMS input")

    return math.sqrt(
        sum(x * x for x in values)
        / len(values)
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    keys: list[str] = []

    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=keys,
        )
        writer.writeheader()
        writer.writerows(rows)


def sign_summary(
    target: list[float],
    physical: list[float],
) -> dict[str, Any]:
    if len(target) != len(physical):
        raise ValueError("Length mismatch")

    target_improved = 0
    physical_improved = 0
    both = 0
    conflict = 0

    for td, pd in zip(target, physical):
        if td < -TOL:
            target_improved += 1

        if pd < -TOL:
            physical_improved += 1

        if td < -TOL and pd < -TOL:
            both += 1

        if td < -TOL and pd > TOL:
            conflict += 1

    return {
        "target_improved":
            target_improved,

        "physical_improved":
            physical_improved,

        "both_improved":
            both,

        "target_improved_physical_worsened":
            conflict,
    }


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE: {OUT_DIR}"
        )

    compact = read_compact_rows()

    # --------------------------------------------------------
    # Map compact rows to raw episode indices.
    #
    # Source CSV preserves evaluator append order within each
    # (beta_name, terrain_label) group. episode_0000, 0001, ...
    # are generated in that same evaluator order.
    #
    # Mapping is independently verified below by requiring
    # reconstructed policy-window energy to match compact
    # episode energy.
    # --------------------------------------------------------

    grouped: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = {}

    for row in compact:
        key = (
            row["beta_name"],
            row["terrain_label"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    episode_rows: list[dict[str, Any]] = []

    settling_counts = set()
    energy_errors = []
    progress_errors = []

    for (beta, terrain_label), group_rows in grouped.items():
        raw_dir = (
            RAW_ROOT
            / beta
            / terrain_label
        )

        raw_files = sorted(
            raw_dir.glob("episode_*.jsonl")
        )

        if len(raw_files) != len(group_rows):
            raise RuntimeError(
                f"{beta}/{terrain_label}: "
                f"{len(raw_files)} raw files vs "
                f"{len(group_rows)} compact rows"
            )

        for episode_index, (
            compact_row,
            raw_path,
        ) in enumerate(
            zip(
                group_rows,
                raw_files,
            )
        ):
            steps = load_steps(raw_path)

            policy_steps = int(
                float(
                    compact_row[
                        "policy_steps"
                    ]
                )
            )

            if len(steps) <= policy_steps:
                raise RuntimeError(
                    f"Need pre-policy boundary row: "
                    f"{raw_path}; "
                    f"raw={len(steps)}, "
                    f"policy={policy_steps}"
                )

            settling_steps = (
                len(steps)
                - policy_steps
            )

            settling_counts.add(
                settling_steps
            )

            boundary = steps[
                settling_steps - 1
            ]

            policy = steps[
                settling_steps:
            ]

            if len(policy) != policy_steps:
                raise RuntimeError(
                    "Policy slicing mismatch"
                )

            final = policy[-1]

            # ------------------------------------------------
            # Exact policy-window time and progress.
            # ------------------------------------------------

            decision_time = sum(
                f(row["decision_dt_s"])
                for row in policy
            )

            compact_time = f(
                compact_row[
                    "decision_time_s"
                ]
            )

            if not math.isclose(
                decision_time,
                compact_time,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise RuntimeError(
                    "Decision-time mismatch: "
                    f"{raw_path}: "
                    f"{decision_time} vs {compact_time}"
                )

            progress_geometry = (
                f(boundary["goal_distance"])
                - f(final["goal_distance"])
            )

            progress_integrated = sum(
                f(
                    row[
                        "reward_components"
                    ][
                        "progress_rate_mps"
                    ]
                )
                * f(row["decision_dt_s"])
                for row in policy
            )

            progress_errors.append(
                abs(
                    progress_geometry
                    - progress_integrated
                )
            )

            # The geometric definition is retained as the
            # primary physical episode progress.
            progress_m = progress_geometry

            if progress_m <= 0.0:
                raise RuntimeError(
                    f"Non-positive progress: {raw_path}"
                )

            progress_rate_episode = (
                progress_m
                / decision_time
            )

            seconds_per_meter = (
                decision_time
                / progress_m
            )

            # ------------------------------------------------
            # Exact policy-window mechanical energy.
            # energy_abs_j in each raw row is an interval
            # quantity, not cumulative.
            # ------------------------------------------------

            energy_abs_j = sum(
                f(
                    row[
                        "reward_components"
                    ][
                        "energy_abs_j"
                    ]
                )
                for row in policy
            )

            compact_energy = f(
                compact_row[
                    "energy_abs_j"
                ]
            )

            energy_error = abs(
                energy_abs_j
                - compact_energy
            )

            energy_errors.append(
                energy_error
            )

            if not math.isclose(
                energy_abs_j,
                compact_energy,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Energy reconstruction mismatch: "
                    f"{raw_path}: raw={energy_abs_j}, "
                    f"compact={compact_energy}"
                )

            energy_j_per_meter = (
                energy_abs_j
                / progress_m
            )

            mean_power_w = (
                energy_abs_j
                / decision_time
            )

            # ------------------------------------------------
            # Attitude reconstruction.
            #
            # reward telemetry contract:
            #
            #   fraction = abs(angle) / unsafe_threshold
            #
            # Therefore magnitude is exactly recoverable.
            # Sign is unnecessary for RMS and max magnitude.
            # ------------------------------------------------

            abs_roll = []
            abs_pitch = []

            for row in policy:
                rc = row[
                    "reward_components"
                ]

                roll_mag = (
                    f(
                        rc[
                            "roll_fraction_of_unsafe"
                        ]
                    )
                    * f(
                        rc[
                            "m4_roll_unsafe_rad"
                        ]
                    )
                )

                pitch_mag = (
                    f(
                        rc[
                            "pitch_fraction_of_unsafe"
                        ]
                    )
                    * f(
                        rc[
                            "m4_pitch_unsafe_rad"
                        ]
                    )
                )

                abs_roll.append(
                    roll_mag
                )
                abs_pitch.append(
                    pitch_mag
                )

            roll_rms = rms(abs_roll)
            pitch_rms = rms(abs_pitch)

            attitude_rms_norm = math.sqrt(
                roll_rms * roll_rms
                + pitch_rms * pitch_rms
            )

            max_abs_roll = max(abs_roll)
            max_abs_pitch = max(abs_pitch)

            attitude_peak_max = max(
                max_abs_roll,
                max_abs_pitch,
            )

            # ------------------------------------------------
            # Slip evaluation.
            #
            # eval_stance_slip is cumulative from episode
            # start. Subtract the settling-boundary sufficient
            # statistics from the final cumulative statistics
            # to obtain exact policy-window all-contact
            # mean/RMS slip.
            # ------------------------------------------------

            slip0 = boundary[
                "eval_stance_slip"
            ]

            slip1 = final[
                "eval_stance_slip"
            ]

            slip_contact_time = (
                f(slip1["contact_time_s"])
                - f(slip0["contact_time_s"])
            )

            slip_speed_time_sum = (
                f(slip1["speed_time_sum_m"])
                - f(slip0["speed_time_sum_m"])
            )

            slip_speed_sq_time_sum = (
                f(
                    slip1[
                        "speed_sq_time_sum_m2ps"
                    ]
                )
                - f(
                    slip0[
                        "speed_sq_time_sum_m2ps"
                    ]
                )
            )

            if slip_contact_time <= 0.0:
                raise RuntimeError(
                    f"No policy-window contact time: {raw_path}"
                )

            slip_mean_mps = (
                slip_speed_time_sum
                / slip_contact_time
            )

            slip_rms_mps = math.sqrt(
                max(
                    0.0,
                    slip_speed_sq_time_sum
                    / slip_contact_time,
                )
            )

            established_contact_time = (
                f(
                    slip1[
                        "established_contact_time_s"
                    ]
                )
                - f(
                    slip0[
                        "established_contact_time_s"
                    ]
                )
            )

            # Exact tail fractions from cumulative established
            # slip histogram differences. Thresholds coincide
            # with existing histogram bin boundaries.
            tail_fractions = {}

            hist0 = slip0[
                "established_slip_histogram"
            ]

            hist1 = slip1[
                "established_slip_histogram"
            ]

            for threshold in (
                0.002,
                0.01,
                0.02,
                0.05,
                0.10,
            ):
                tail_time = subtract_hist_tail(
                    hist1,
                    hist0,
                    threshold,
                )

                fraction = (
                    tail_time
                    / established_contact_time
                    if established_contact_time > 0.0
                    else float("nan")
                )

                name = (
                    "established_slip_fraction_ge_"
                    + str(threshold).replace(
                        ".",
                        "p",
                    )
                )

                tail_fractions[
                    name
                ] = fraction

            out = {
                "beta_name":
                    beta,

                "terrain_label":
                    terrain_label,

                "terrain":
                    compact_row[
                        "terrain"
                    ],

                "seed":
                    int(
                        float(
                            compact_row[
                                "seed"
                            ]
                        )
                    ),

                "episode_index":
                    episode_index,

                "raw_path":
                    str(
                        raw_path.relative_to(
                            ROOT
                        )
                    ),

                "raw_step_rows":
                    len(steps),

                "settling_steps":
                    settling_steps,

                "policy_steps":
                    policy_steps,

                "decision_time_s":
                    decision_time,

                "progress_m":
                    progress_m,

                "progress_integrated_m":
                    progress_integrated,

                "progress_reconstruction_error_m":
                    abs(
                        progress_m
                        - progress_integrated
                    ),

                "episode_progress_rate_mps":
                    progress_rate_episode,

                "seconds_per_meter":
                    seconds_per_meter,

                "energy_abs_j":
                    energy_abs_j,

                "energy_j_per_meter":
                    energy_j_per_meter,

                "mean_abs_mechanical_power_w":
                    mean_power_w,

                "roll_rms_rad":
                    roll_rms,

                "pitch_rms_rad":
                    pitch_rms,

                "attitude_rms_norm":
                    attitude_rms_norm,

                "max_abs_roll_rad":
                    max_abs_roll,

                "max_abs_pitch_rad":
                    max_abs_pitch,

                "attitude_peak_max":
                    attitude_peak_max,

                "slip_all_contact_mean_mps":
                    slip_mean_mps,

                "slip_all_contact_rms_mps":
                    slip_rms_mps,

                "policy_contact_time_s":
                    slip_contact_time,

                "policy_established_contact_time_s":
                    established_contact_time,

                "mean_cost_motion":
                    f(
                        compact_row[
                            "mean_cost_motion"
                        ]
                    ),

                "mean_cost_posture":
                    f(
                        compact_row[
                            "mean_cost_posture"
                        ]
                    ),

                "mean_cost_traction":
                    f(
                        compact_row[
                            "mean_cost_traction"
                        ]
                    ),

                "mean_cost_stability":
                    f(
                        compact_row[
                            "mean_cost_stability"
                        ]
                    ),

                "mean_cost_energy":
                    f(
                        compact_row[
                            "mean_cost_energy"
                        ]
                    ),
            }

            out.update(
                tail_fractions
            )

            episode_rows.append(
                out
            )

    if len(episode_rows) != 36:
        raise RuntimeError(
            f"Expected 36 extracted episodes, "
            f"got {len(episode_rows)}"
        )

    # --------------------------------------------------------
    # Pair objective-specific beta against Balanced using
    # identical TRAIN context (terrain label + seed).
    # --------------------------------------------------------

    index = {
        (
            row["terrain_label"],
            row["seed"],
            row["beta_name"],
        ):
            row
        for row in episode_rows
    }

    objective_specs = {
        "motion": {
            "candidate":
                "motion",

            "target":
                "mean_cost_motion",

            "physical": {
                "seconds_per_meter":
                    True,

                "episode_progress_rate_mps":
                    False,

                "progress_m":
                    False,
            },
        },

        "stability": {
            "candidate":
                "stability",

            "target":
                "mean_cost_stability",

            "physical": {
                "pitch_rms_rad":
                    True,

                "roll_rms_rad":
                    True,

                "attitude_rms_norm":
                    True,

                "max_abs_pitch_rad":
                    True,

                "max_abs_roll_rad":
                    True,

                "attitude_peak_max":
                    True,

                "slip_all_contact_mean_mps":
                    True,

                "slip_all_contact_rms_mps":
                    True,

                "established_slip_fraction_ge_0p002":
                    True,

                "established_slip_fraction_ge_0p01":
                    True,

                "established_slip_fraction_ge_0p02":
                    True,

                "established_slip_fraction_ge_0p05":
                    True,

                "established_slip_fraction_ge_0p1":
                    True,
            },
        },

        "energy": {
            "candidate":
                "energy",

            "target":
                "mean_cost_energy",

            "physical": {
                "energy_abs_j":
                    True,

                "energy_j_per_meter":
                    True,

                "mean_abs_mechanical_power_w":
                    True,
            },
        },
    }

    contexts = sorted({
        (
            row["terrain_label"],
            row["seed"],
        )
        for row in episode_rows
        if row["beta_name"] == "balanced"
    })

    if len(contexts) != 9:
        raise RuntimeError(
            f"Expected 9 balanced contexts, got {len(contexts)}"
        )

    paired_rows = []
    summary = {}

    print("=" * 100)
    print(
        "ICRA27 OS-T5.2a-2 "
        "TRAIN RAW PHYSICAL ALIGNMENT"
    )
    print("=" * 100)
    print("episodes        :", len(episode_rows))
    print("contexts        :", len(contexts))
    print("settling steps  :", sorted(settling_counts))
    print(
        "max energy reconstruction error:",
        max(energy_errors),
    )
    print(
        "max progress geometry/integration difference:",
        max(progress_errors),
    )

    for objective, spec in objective_specs.items():
        candidate_name = spec["candidate"]
        target_name = spec["target"]

        target_deltas = []
        physical_deltas: dict[
            str,
            list[float],
        ] = {
            name: []
            for name in spec["physical"]
        }

        print()
        print("-" * 100)
        print(objective.upper())
        print("-" * 100)

        for terrain_label, seed in contexts:
            ref = index[
                (
                    terrain_label,
                    seed,
                    "balanced",
                )
            ]

            cand = index[
                (
                    terrain_label,
                    seed,
                    candidate_name,
                )
            ]

            dt = (
                f(cand[target_name])
                - f(ref[target_name])
            )

            target_deltas.append(dt)

            out = {
                "objective":
                    objective,

                "terrain_label":
                    terrain_label,

                "seed":
                    seed,

                "candidate":
                    candidate_name,

                "reference":
                    "balanced",

                "target_metric":
                    target_name,

                "delta_target_cost":
                    dt,
            }

            for metric_name, lower_is_better in (
                spec["physical"].items()
            ):
                raw_delta = (
                    f(cand[metric_name])
                    - f(ref[metric_name])
                )

                cost_delta = (
                    raw_delta
                    if lower_is_better
                    else -raw_delta
                )

                physical_deltas[
                    metric_name
                ].append(
                    cost_delta
                )

                out[
                    metric_name
                    + "__reference"
                ] = ref[
                    metric_name
                ]

                out[
                    metric_name
                    + "__candidate"
                ] = cand[
                    metric_name
                ]

                out[
                    metric_name
                    + "__delta_cost_oriented"
                ] = cost_delta

            paired_rows.append(out)

        objective_summary = {
            "target_improved":
                sum(
                    d < -TOL
                    for d in target_deltas
                ),

            "physical":
                {},
        }

        print(
            "target:",
            objective_summary[
                "target_improved"
            ],
            "/9 improved",
        )

        for metric_name, deltas in (
            physical_deltas.items()
        ):
            s = sign_summary(
                target_deltas,
                deltas,
            )

            objective_summary[
                "physical"
            ][
                metric_name
            ] = s

            wins = sum(
                d < -TOL
                for d in deltas
            )

            losses = sum(
                d > TOL
                for d in deltas
            )

            ties = (
                9 - wins - losses
            )

            print(
                f"  {metric_name:<42}"
                f" physical={wins}/{losses}/{ties}"
                f" both={s['both_improved']}/9"
                f" conflict="
                f"{s['target_improved_physical_worsened']}/9"
            )

        summary[
            objective
        ] = objective_summary

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        EPISODE_CSV,
        episode_rows,
    )

    write_csv(
        PAIRED_CSV,
        paired_rows,
    )

    manifest = {
        "schema":
            "icra27_os_t5p2a2_train_raw_physical_alignment_v0",

        "scope":
            "TRAIN-only u30",

        "source_compact_csv":
            str(
                SOURCE_CSV.relative_to(
                    ROOT
                )
            ),

        "source_raw_root":
            str(
                RAW_ROOT.relative_to(
                    ROOT
                )
            ),

        "episodes":
            len(episode_rows),

        "contexts":
            len(contexts),

        "settling_steps_observed":
            sorted(
                settling_counts
            ),

        "aggregation_contract": {
            "official_window":
                "last policy_steps raw step rows",

            "pre_policy_boundary":
                (
                    "raw row immediately preceding "
                    "the official policy window"
                ),

            "progress":
                (
                    "boundary goal_distance minus "
                    "final goal_distance"
                ),

            "energy":
                (
                    "sum interval energy_abs_j over "
                    "official policy window"
                ),

            "attitude":
                (
                    "abs(angle)=fraction_of_unsafe "
                    "* M4 unsafe threshold; RMS and "
                    "peak over policy window"
                ),

            "slip":
                (
                    "subtract cumulative sufficient "
                    "statistics at settling boundary "
                    "from final cumulative statistics"
                ),
        },

        "validation": {
            "max_energy_reconstruction_error_j":
                max(energy_errors),

            "max_progress_geometry_vs_integrated_error_m":
                max(progress_errors),
        },

        "summary":
            summary,
    }

    with MANIFEST.open("w") as fp:
        json.dump(
            manifest,
            fp,
            indent=2,
            sort_keys=True,
        )
        fp.write("\n")

    print()
    print("=" * 100)
    print("outputs:")
    print(" ", EPISODE_CSV)
    print(" ", PAIRED_CSV)
    print(" ", MANIFEST)
    print()
    print(
        "[ICRA27] OS-T5.2a-2 "
        "TRAIN raw physical alignment: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
