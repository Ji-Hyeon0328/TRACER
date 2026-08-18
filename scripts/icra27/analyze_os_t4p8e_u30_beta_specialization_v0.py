#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]

SWEEP = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p8d_v4_lockstep_checkpoint_sweep_v0"
    / "train9"
)

EPISODES_CSV = SWEEP / "episodes.csv"
U30 = SWEEP / "u30"

OUT_DIR = (
    ROOT
    / "results"
    / "icra27"
    / "os_t4p8e_u30_beta_specialization_v0"
)

UPDATE = 30

BETA_ORDER = (
    "balanced",
    "motion",
    "stability",
    "energy",
)

TARGETS = {
    "motion": (
        "motion",
        "mean_cost_motion",
    ),
    "stability": (
        "stability",
        "mean_cost_stability",
    ),
    "energy": (
        "energy",
        "mean_cost_energy",
    ),
}


def load_episode_rows():
    with EPISODES_CSV.open(
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    rows = [
        row
        for row in rows
        if int(row["update"]) == UPDATE
    ]

    if len(rows) != 36:
        raise RuntimeError(
            f"Expected 36 u30 episodes, got {len(rows)}"
        )

    return rows


def read_step_rows(path):
    result = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("event") != "step":
            continue

        if (
            not isinstance(
                row.get("applied_physical"),
                list,
            )
            or len(
                row["applied_physical"]
            ) < 3
        ):
            raise RuntimeError(
                f"Missing applied_physical in {path}"
            )

        if (
            not isinstance(
                row.get("requested_normalized"),
                list,
            )
            or len(
                row["requested_normalized"]
            ) < 3
        ):
            raise RuntimeError(
                f"Missing requested_normalized in {path}"
            )

        result.append(row)

    if not result:
        raise RuntimeError(
            f"No step rows in {path}"
        )

    return result


def mean_vector(rows, key):
    vectors = [
        row[key][:3]
        for row in rows
    ]

    return tuple(
        mean(
            float(v[i])
            for v in vectors
        )
        for i in range(3)
    )


def l2(a, b):
    return math.sqrt(
        sum(
            (
                float(x)
                - float(y)
            ) ** 2
            for x, y in zip(a, b)
        )
    )


def rank_of_target(
    values,
    target,
):
    ordered = sorted(
        values.items(),
        key=lambda kv: (
            float(kv[1]),
            kv[0],
        ),
    )

    return (
        1
        + next(
            i
            for i, (name, _value)
            in enumerate(ordered)
            if name == target
        )
    )


def main():
    if OUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists; "
            f"refusing overwrite: {OUT_DIR}"
        )

    OUT_DIR.mkdir(
        parents=True
    )

    rows = load_episode_rows()

    # --------------------------------------------------------
    # Map each CSV episode to its historical JSONL.
    #
    # Within each beta/terrain directory, episodes were
    # generated in the same seed order as episodes.csv.
    # --------------------------------------------------------

    grouped = {}

    for row in rows:
        key = (
            row["beta_name"],
            row["terrain_label"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    enriched = []

    for (
        beta_name,
        terrain_label,
    ), group_rows in grouped.items():

        for episode_index, row in enumerate(
            group_rows
        ):
            path = (
                U30
                / "env_logs"
                / beta_name
                / terrain_label
                / (
                    f"episode_"
                    f"{episode_index:04d}.jsonl"
                )
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            steps = read_step_rows(path)

            policy_steps = int(
                row["policy_steps"]
            )

            if (
                policy_steps <= 0
                or policy_steps > len(steps)
            ):
                raise RuntimeError(
                    "Invalid policy_steps: "
                    f"{policy_steps} for {path}"
                )

            # Same convention as the T4.3c/T4.8d reward
            # analyzer: fixed settling transitions precede
            # policy transitions, so retain the final
            # policy_steps rows.
            selected = steps[
                -policy_steps:
            ]

            applied = mean_vector(
                selected,
                "applied_physical",
            )

            requested_norm = mean_vector(
                selected,
                "requested_normalized",
            )

            csv_vx = float(
                row["mean_applied_vx_mps"]
            )

            if not math.isclose(
                applied[0],
                csv_vx,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise RuntimeError(
                    "Raw-log/CSV vx mismatch: "
                    f"{path}: "
                    f"raw={applied[0]:.12f} "
                    f"csv={csv_vx:.12f}"
                )

            item = dict(row)

            item.update(
                {
                    "episode_jsonl":
                        str(path),

                    "mean_applied_vx":
                        applied[0],

                    "mean_applied_yaw_rate":
                        applied[1],

                    "mean_applied_body_height":
                        applied[2],

                    "mean_requested_norm_vx":
                        requested_norm[0],

                    "mean_requested_norm_yaw":
                        requested_norm[1],

                    "mean_requested_norm_height":
                        requested_norm[2],
                }
            )

            enriched.append(item)

    # --------------------------------------------------------
    # Context-wise specialization.
    # --------------------------------------------------------

    contexts = {}

    for row in enriched:
        key = (
            row["terrain_label"],
            int(row["seed"]),
        )

        contexts.setdefault(
            key,
            {},
        )[
            row["beta_name"]
        ] = row

    if len(contexts) != 9:
        raise RuntimeError(
            f"Expected 9 contexts, got {len(contexts)}"
        )

    context_rows = []

    strict_counts = {
        name: 0
        for name in TARGETS
    }

    balanced_down_counts = {
        name: 0
        for name in TARGETS
    }

    action_distances = []

    for (
        terrain,
        seed,
    ), by_beta in contexts.items():

        if set(by_beta) != set(BETA_ORDER):
            raise RuntimeError(
                "Incomplete beta bank for "
                f"{terrain}/{seed}"
            )

        record = {
            "terrain":
                terrain,
            "seed":
                seed,
        }

        for semantic, (
            target_beta,
            metric,
        ) in TARGETS.items():

            values = {
                beta:
                    float(
                        by_beta[
                            beta
                        ][metric]
                    )
                for beta
                in BETA_ORDER
            }

            target_value = values[
                target_beta
            ]

            balanced_value = values[
                "balanced"
            ]

            best_other = min(
                value
                for beta, value
                in values.items()
                if beta != target_beta
            )

            rank = rank_of_target(
                values,
                target_beta,
            )

            delta_balanced = (
                target_value
                - balanced_value
            )

            margin_best_other = (
                target_value
                - best_other
            )

            record[
                f"{semantic}_rank"
            ] = rank

            record[
                f"{semantic}_delta_vs_balanced"
            ] = delta_balanced

            record[
                f"{semantic}_margin_vs_best_other"
            ] = margin_best_other

            if rank == 1:
                strict_counts[
                    semantic
                ] += 1

            if delta_balanced < 0.0:
                balanced_down_counts[
                    semantic
                ] += 1

        # ----------------------------------------------------
        # Mean action vectors.
        # Physical:
        #   [vx, yaw_rate, body_height]
        #
        # Normalized:
        #   compare policy-output separation without unit
        #   scaling across action dimensions.
        # ----------------------------------------------------

        norm_vectors = {}

        for beta in BETA_ORDER:
            row = by_beta[beta]

            record[
                f"{beta}_vx"
            ] = float(
                row["mean_applied_vx"]
            )

            record[
                f"{beta}_yaw"
            ] = float(
                row["mean_applied_yaw_rate"]
            )

            record[
                f"{beta}_height"
            ] = float(
                row["mean_applied_body_height"]
            )

            norm_vectors[beta] = (
                float(
                    row[
                        "mean_requested_norm_vx"
                    ]
                ),
                float(
                    row[
                        "mean_requested_norm_yaw"
                    ]
                ),
                float(
                    row[
                        "mean_requested_norm_height"
                    ]
                ),
            )

        pair_dists = {}

        for i, a in enumerate(
            BETA_ORDER
        ):
            for b in BETA_ORDER[
                i + 1:
            ]:
                d = l2(
                    norm_vectors[a],
                    norm_vectors[b],
                )

                pair_dists[
                    f"{a}__{b}"
                ] = d

                action_distances.append(d)

        record[
            "min_pairwise_normalized_action_distance"
        ] = min(
            pair_dists.values()
        )

        record[
            "max_pairwise_normalized_action_distance"
        ] = max(
            pair_dists.values()
        )

        record[
            "motion_vs_balanced_action_distance"
        ] = pair_dists[
            "balanced__motion"
        ]

        record[
            "stability_vs_balanced_action_distance"
        ] = pair_dists[
            "balanced__stability"
        ]

        record[
            "energy_vs_balanced_action_distance"
        ] = pair_dists[
            "balanced__energy"
        ]

        context_rows.append(record)

    # --------------------------------------------------------
    # Outputs.
    # --------------------------------------------------------

    context_csv = (
        OUT_DIR
        / "u30_context_specialization.csv"
    )

    with context_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                context_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            context_rows
        )

    action_csv = (
        OUT_DIR
        / "u30_episode_action_means.csv"
    )

    action_fields = (
        "beta_name",
        "terrain_label",
        "seed",
        "success",
        "policy_steps",
        "mean_cost_motion",
        "mean_cost_stability",
        "mean_cost_energy",
        "mean_cost_traction",
        "mean_applied_vx",
        "mean_applied_yaw_rate",
        "mean_applied_body_height",
        "mean_requested_norm_vx",
        "mean_requested_norm_yaw",
        "mean_requested_norm_height",
        "episode_jsonl",
    )

    with action_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=action_fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in enriched:
            writer.writerow(row)

    summary = {
        "schema":
            "icra27_os_t4p8e_u30_beta_specialization_v0",

        "source_sweep":
            str(SWEEP),

        "update":
            UPDATE,

        "contexts":
            len(contexts),

        "episodes":
            len(enriched),

        "strict_target_rank1": {
            key:
                f"{value}/9"
            for key, value
            in strict_counts.items()
        },

        "target_better_than_balanced": {
            key:
                f"{value}/9"
            for key, value
            in balanced_down_counts.items()
        },

        "normalized_action_separation": {
            "min_pairwise_distance":
                min(action_distances),

            "mean_pairwise_distance":
                mean(action_distances),

            "max_pairwise_distance":
                max(action_distances),
        },

        "interpretation_contract": {
            "rank1":
                "strong objective specialization",

            "better_than_balanced":
                "directional beta semantics",

            "action_distance":
                (
                    "beta-conditioned policy-output "
                    "separation; diagnostic only"
                ),
        },
    }

    manifest = (
        OUT_DIR
        / "specialization_manifest.json"
    )

    manifest.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=" * 104)
    print(
        "ICRA27 OS-T4.8e u30 "
        "BETA SPECIALIZATION / ACTION-MARGIN AUDIT"
    )
    print("=" * 104)

    print()
    print("TARGET METRIC SEMANTICS")

    for semantic in (
        "motion",
        "stability",
        "energy",
    ):
        print(
            f"  {semantic:<10} "
            f"better-than-balanced="
            f"{balanced_down_counts[semantic]}/9 "
            f"strict-rank1="
            f"{strict_counts[semantic]}/9"
        )

    print()
    print(
        "NORMALIZED POLICY-ACTION SEPARATION"
    )
    print(
        "  min  : "
        f"{min(action_distances):.8f}"
    )
    print(
        "  mean : "
        f"{mean(action_distances):.8f}"
    )
    print(
        "  max  : "
        f"{max(action_distances):.8f}"
    )

    print()
    print("CONTEXT ACTION MEANS")

    for row in context_rows:
        print(
            f"  {row['terrain']:<12} "
            f"seed={row['seed']:<6} "
            f"vx[B/M/S/E]="
            f"{row['balanced_vx']:.5f}/"
            f"{row['motion_vx']:.5f}/"
            f"{row['stability_vx']:.5f}/"
            f"{row['energy_vx']:.5f} "
            f"h[B/M/S/E]="
            f"{row['balanced_height']:.5f}/"
            f"{row['motion_height']:.5f}/"
            f"{row['stability_height']:.5f}/"
            f"{row['energy_height']:.5f}"
        )

    print()
    print("outputs:")
    print(" ", context_csv)
    print(" ", action_csv)
    print(" ", manifest)

    print()
    print(
        "[ICRA27] OS-T4.8e "
        "u30 beta specialization audit: COMPUTE PASS"
    )


if __name__ == "__main__":
    main()
