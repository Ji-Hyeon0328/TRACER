from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

T49_PATH = (
    ROOT
    / "scripts/icra27"
    / "evaluate_os_t4p9b_v4_primary_heldout_v0.py"
)

TRAIN_CONTEXT_SOURCE = (
    ROOT
    / "results/icra27"
    / "os_t4p8d_v4_lockstep_checkpoint_sweep_v0"
    / "train9"
    / "episodes.csv"
)

PERLIN_SPLIT_SOURCE = (
    ROOT
    / "configs/icra27"
    / "m7_perlin_seed_split_v0.json"
)

OS_CONTEXT_SOURCE = (
    ROOT
    / "results/icra27"
    / "os_t5p5a_oracle_context_descriptor_v1"
    / "oracle_context_descriptors.csv"
)

EXISTING_T53_ROUGH_SEEDS = (
    13,
    7,
    15,
)

EXPECTED_REMAINING_ROUGH_SEEDS = (
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


DEFAULT_CHECKPOINT = (
    ROOT
    / "results/icra27"
    / "phase1a_beta_conditioned_v4_seed27027"
    / "checkpoint_update_0030.pt"
)

DEFAULT_OUT_DIR = (
    ROOT
    / "results/icra27"
    / "os_t5p5b_remaining_rough_train_atlas_v0"
)

EXPECTED_REWARD_MODE = "tracer_cost_v4"

MOTION_ANCHOR = (
    0.70,
    0.15,
    0.15,
)

STABILITY_ANCHOR = (
    0.15,
    0.70,
    0.15,
)

ENERGY_ANCHOR = (
    0.15,
    0.15,
    0.70,
)

GRID_DENOMINATOR = 5
EXPECTED_BETA_POINTS = 21
EXPECTED_CONTEXTS = 15
EXPECTED_EPISODES = (
    EXPECTED_BETA_POINTS
    * EXPECTED_CONTEXTS
)

TOL = 1.0e-10


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fields: list[str] = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

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

        for row in rows:
            writer.writerow(
                row
            )

    os.replace(
        tmp,
        path,
    )


def build_beta_lattice():
    points = []

    # Integer barycentric lattice:
    #
    #   i_M + i_S + i_E = 5
    #
    # lambda = i / 5.
    #
    # Deterministic order:
    # motion-heavy -> stability-heavy -> energy-heavy.
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

            lam_m = (
                i_m
                / GRID_DENOMINATOR
            )

            lam_s = (
                i_s
                / GRID_DENOMINATOR
            )

            lam_e = (
                i_e
                / GRID_DENOMINATOR
            )

            beta = tuple(
                lam_m
                * MOTION_ANCHOR[k]
                + lam_s
                * STABILITY_ANCHOR[k]
                + lam_e
                * ENERGY_ANCHOR[k]
                for k in range(3)
            )

            name = (
                f"lm{i_m * 20:03d}_"
                f"ls{i_s * 20:03d}_"
                f"le{i_e * 20:03d}"
            )

            points.append(
                {
                    "name":
                        name,

                    "lambda_motion":
                        lam_m,

                    "lambda_stability":
                        lam_s,

                    "lambda_energy":
                        lam_e,

                    "beta":
                        beta,
                }
            )

    if len(points) != EXPECTED_BETA_POINTS:
        raise RuntimeError(
            "Expected 21 beta lattice points; "
            f"got {len(points)}"
        )

    names = {
        x["name"]
        for x in points
    }

    if len(names) != len(points):
        raise RuntimeError(
            "Duplicate beta lattice names."
        )

    for point in points:
        beta = point["beta"]

        if not math.isclose(
            sum(beta),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Beta does not sum to one: "
                f"{point}"
            )

        for value in beta:
            if not (
                0.15 - TOL
                <= value
                <= 0.70 + TOL
            ):
                raise RuntimeError(
                    "Beta leaves trained-anchor "
                    f"convex hull: {point}"
                )

    # Verify the three hull corners are exactly
    # the three trained biased anchors.
    beta_set = {
        tuple(
            round(float(x), 12)
            for x in point["beta"]
        )
        for point in points
    }

    for anchor in (
        MOTION_ANCHOR,
        STABILITY_ANCHOR,
        ENERGY_ANCHOR,
    ):
        key = tuple(
            round(float(x), 12)
            for x in anchor
        )

        if key not in beta_set:
            raise RuntimeError(
                f"Missing anchor corner: {anchor}"
            )

    # Balanced is inside the hull, but intentionally
    # not a 0.2 barycentric-grid point.
    balanced = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )

    if any(
        all(
            math.isclose(
                point["beta"][k],
                balanced[k],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for k in range(3)
        )
        for point in points
    ):
        raise RuntimeError(
            "Balanced unexpectedly appears "
            "in the 0.2 barycentric lattice."
        )

    return points


def load_train_context_groups():
    # --------------------------------------------------------
    # OS-T5.5b:
    #
    # Expand the frozen T5.3 rough TRAIN atlas from
    # seeds [13, 7, 15] to the remaining 15 seeds in
    # the already-frozen TRAIN split.
    #
    # No validation/test/hard seed is allowed here.
    # --------------------------------------------------------

    for path in (
        PERLIN_SPLIT_SOURCE,
        OS_CONTEXT_SOURCE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    split = json.loads(
        PERLIN_SPLIT_SOURCE.read_text()
    )

    if split.get("schema") != (
        "icra27_m7_perlin_seed_split_v0"
    ):
        raise RuntimeError(
            "Unexpected Perlin split schema."
        )

    train_seeds = [
        int(x)
        for x in (
            split[
                "splits"
            ][
                "train"
            ]
        )
    ]

    if len(train_seeds) != 18:
        raise RuntimeError(
            "Expected exactly 18 rough TRAIN "
            f"seeds; got {len(train_seeds)}"
        )

    existing = list(
        EXISTING_T53_ROUGH_SEEDS
    )

    if not all(
        seed in train_seeds
        for seed in existing
    ):
        raise RuntimeError(
            "Frozen T5.3 rough seeds are "
            "not all in TRAIN split."
        )

    remaining = [
        seed
        for seed in train_seeds
        if seed not in set(
            existing
        )
    ]

    expected_remaining = list(
        EXPECTED_REMAINING_ROUGH_SEEDS
    )

    if remaining != expected_remaining:
        raise RuntimeError(
            "Remaining rough TRAIN seed "
            "order mismatch:\n"
            f"  actual  ={remaining}\n"
            f"  expected={expected_remaining}"
        )

    if len(remaining) != (
        EXPECTED_CONTEXTS
    ):
        raise RuntimeError(
            "Expected exactly 15 remaining "
            f"TRAIN contexts; got {len(remaining)}"
        )

    # --------------------------------------------------------
    # Held-out leakage audit.
    # --------------------------------------------------------

    forbidden = set()

    for key in (
        "validation",
        "test",
    ):
        forbidden.update(
            int(x)
            for x in (
                split[
                    "splits"
                ][key]
            )
        )

    forbidden.update(
        int(x)
        for x in (
            split[
                "stress_banks"
            ][
                "hard"
            ]
        )
    )

    leaked = sorted(
        set(remaining)
        & forbidden
    )

    if leaked:
        raise RuntimeError(
            "Held-out/hard seed leakage "
            f"detected: {leaked}"
        )

    # --------------------------------------------------------
    # T5.5a context-descriptor coverage audit.
    #
    # T5.5b does NOT feed this descriptor into the
    # frozen PPO. It merely verifies that every
    # expanded TRAIN context has the frozen selector
    # context representation needed downstream.
    # --------------------------------------------------------

    with OS_CONTEXT_SOURCE.open(
        newline="",
    ) as f:
        context_rows = list(
            csv.DictReader(f)
        )

    rough_context_seeds = {
        int(
            float(
                row["seed"]
            )
        )
        for row in context_rows
        if (
            row["terrain"]
            == "rough_perlin"
            and str(
                row["seed"]
            ).strip()
        )
    }

    if rough_context_seeds != set(
        train_seeds
    ):
        raise RuntimeError(
            "T5.5a rough TRAIN descriptor "
            "coverage mismatch:\n"
            f"  descriptor={sorted(rough_context_seeds)}\n"
            f"  split={sorted(train_seeds)}"
        )

    return [
        {
            "name":
                "rough",

            "terrain":
                "rough_perlin",

            "seeds":
                remaining,
        }
    ]

def print_plan(
    beta_points,
    groups,
):
    print("=" * 108)
    print(
        "ICRA27 OS-T5.5b TRAIN "
        "BETA-RESPONSE ATLAS PLAN"
    )
    print("=" * 108)

    print()
    print(
        "beta support : "
        "trained biased-anchor convex hull"
    )
    print(
        "probe type   : "
        "unseen interpolation lattice"
    )
    print(
        "lambda step  : 0.20"
    )
    print(
        "beta points  :",
        len(beta_points),
    )
    print(
        "contexts     :",
        sum(
            len(g["seeds"])
            for g in groups
        ),
    )
    print(
        "episodes     :",
        len(beta_points)
        * sum(
            len(g["seeds"])
            for g in groups
        ),
    )

    print()
    print("TRAIN CONTEXTS")

    for group in groups:
        print(
            f"  {group['name']:<14} "
            f"terrain={group['terrain']:<14} "
            f"seeds={group['seeds']}"
        )

    print()
    print("BETA LATTICE")

    for index, point in enumerate(
        beta_points
    ):
        b = point["beta"]

        print(
            f"  {index:02d} "
            f"{point['name']} "
            f"lambda=["
            f"{point['lambda_motion']:.1f},"
            f"{point['lambda_stability']:.1f},"
            f"{point['lambda_energy']:.1f}] "
            f"beta=["
            f"{b[0]:.3f},"
            f"{b[1]:.3f},"
            f"{b[2]:.3f}]"
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

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
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    beta_points = (
        build_beta_lattice()
    )

    groups = (
        load_train_context_groups()
    )

    print_plan(
        beta_points,
        groups,
    )

    if args.dry_run:
        print()
        print(
            "[ICRA27] OS-T5.5b atlas plan: "
            "DRY-RUN PASS"
        )
        return

    checkpoint = (
        args.checkpoint.resolve()
    )

    out_dir = (
        args.out_dir.resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    if (
        "checkpoint_update_0030.pt"
        != checkpoint.name
    ):
        raise RuntimeError(
            "OS-T5.5b is frozen to the "
            "TRAIN-selected u30 checkpoint; "
            f"got {checkpoint.name!r}"
        )

    if out_dir.exists():
        raise RuntimeError(
            "Output directory already exists; "
            "refusing overwrite: "
            f"{out_dir}"
        )

    if not T49_PATH.exists():
        raise FileNotFoundError(
            T49_PATH
        )

    phase = load_module(
        T49_PATH,
        "os_t5p3_t49_v4_base",
    )

    if str(
        phase.EVAL_REWARD_MODE
    ) != EXPECTED_REWARD_MODE:
        raise RuntimeError(
            "Unexpected T4.9b evaluation "
            "reward mode: "
            f"{phase.EVAL_REWARD_MODE!r}"
        )

    # T4.9b main() performs this promotion
    # before calling the common run_episode().
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
            "reward mode: "
            f"{historical_base_mode!r}"
        )

    if str(
        phase.base.EVAL_REWARD_MODE
    ) != EXPECTED_REWARD_MODE:
        raise RuntimeError(
            "Failed to promote common "
            "evaluator to tracer_cost_v4."
        )

    policy = phase.base.load_policy(
        name="beta_conditioned",
        path=checkpoint,
        expected_reward_mode=(
            EXPECTED_REWARD_MODE
        ),
    )

    # This validates checkpoint TRAIN provenance:
    # the policy was trained with the original
    # Balanced/Motion/Stability/Energy bank.
    phase.validate_checkpoint_beta_bank(
        policy["payload"]
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    plan_path = (
        out_dir
        / "atlas_plan.json"
    )

    episodes_partial = (
        out_dir
        / "episodes_partial.csv"
    )

    episodes_final = (
        out_dir
        / "episodes.csv"
    )

    manifest_path = (
        out_dir
        / "atlas_manifest.json"
    )

    plan = {
        "schema":
            "icra27_os_t5p5b_remaining_rough_train_atlas_plan_v0",

        "checkpoint":
            str(
                checkpoint.relative_to(
                    ROOT
                )
            ),

        "reward_mode":
            EXPECTED_REWARD_MODE,

        "transition_contract":
            "lockstep 100 x 0.002 s = 0.200 s",

        "beta_probe_scope":
            (
                "unseen interpolation inside "
                "the convex hull of the three "
                "trained biased beta anchors"
            ),

        "balanced_in_hull_but_not_grid_point":
            True,

        "lambda_grid_denominator":
            GRID_DENOMINATOR,

        "beta_points":
            [
                {
                    "name":
                        p["name"],

                    "lambda_motion":
                        p["lambda_motion"],

                    "lambda_stability":
                        p[
                            "lambda_stability"
                        ],

                    "lambda_energy":
                        p["lambda_energy"],

                    "beta":
                        list(
                            p["beta"]
                        ),
                }
                for p in beta_points
            ],

        "train_contexts":
            groups,

        "perlin_split_source":
            str(
                PERLIN_SPLIT_SOURCE.relative_to(
                    ROOT
                )
            ),

        "os_context_source":
            str(
                OS_CONTEXT_SOURCE.relative_to(
                    ROOT
                )
            ),

        "existing_t5p3_rough_seeds":
            list(
                EXISTING_T53_ROUGH_SEEDS
            ),

        "remaining_rough_train_seeds":
            list(
                EXPECTED_REMAINING_ROUGH_SEEDS
            ),

        "heldout_used":
            False,

        "expected_episodes":
            EXPECTED_EPISODES,

        "metric_contract":
            (
                "OS-T5.2b physical preference "
                "contract is frozen and must "
                "not be changed from atlas results."
            ),
    }

    plan_path.write_text(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    all_rows: list[
        dict[str, Any]
    ] = []

    success_count = 0
    m4_total = 0

    print()
    print("=" * 108)
    print(
        "STARTING OS-T5.5b "
        "315-EPISODE REMAINING ROUGH TRAIN ATLAS"
    )
    print("=" * 108)

    try:
        for beta_index, point in enumerate(
            beta_points
        ):
            beta_name = point["name"]
            beta = point["beta"]

            print()
            print(
                f"BETA {beta_index + 1:02d}/"
                f"{len(beta_points)}: "
                f"{beta_name} "
                f"[{beta[0]:.3f}, "
                f"{beta[1]:.3f}, "
                f"{beta[2]:.3f}]"
            )

            for group_index, group in enumerate(
                groups
            ):
                group_name = group["name"]
                terrain = group["terrain"]
                seeds = group["seeds"]

                command_port = (
                    int(args.base_port)
                    + 100 * beta_index
                    + 10 * group_index
                )

                log_dir = (
                    out_dir
                    / "env_logs"
                    / beta_name
                    / group_name
                )

                env = phase.make_env(
                    terrain=terrain,
                    beta=beta,
                    command_port=(
                        command_port
                    ),
                    log_dir=log_dir,
                )

                try:
                    for seed in seeds:
                        row = (
                            phase.base.run_episode(
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
                                    policy["model"]
                                ),
                                group=(
                                    group_name
                                ),
                                terrain=terrain,
                                seed=int(seed),
                            )
                        )

                        row = (
                            phase.annotate_row(
                                row,
                                beta_name=(
                                    beta_name
                                ),
                                beta=beta,
                            )
                        )

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

                        all_rows.append(
                            row
                        )

                        if bool(
                            row.get(
                                "success",
                                False,
                            )
                        ):
                            success_count += 1

                        m4_total += int(
                            row.get(
                                "m4_interventions",
                                0,
                            )
                            or 0
                        )

                        # Preserve progress after every
                        # completed episode.
                        write_csv(
                            episodes_partial,
                            all_rows,
                        )

                        print(
                            f"    {group_name:<14} "
                            f"seed={int(seed):<6} "
                            f"status="
                            f"{row['status']:<18} "
                            f"steps="
                            f"{row['policy_steps']:<3} "
                            f"time="
                            f"{row['decision_time_s']} "
                            f"E="
                            f"{row['energy_abs_j']}"
                        )

                finally:
                    env.close()

    except Exception:
        print()
        print(
            "ATLAS INTERRUPTED/FAILED. "
            "Partial CSV and raw logs preserved:"
        )
        print(
            " ",
            episodes_partial,
        )
        raise

    if len(all_rows) != (
        EXPECTED_EPISODES
    ):
        raise RuntimeError(
            "Atlas episode-count mismatch: "
            f"{len(all_rows)} vs "
            f"{EXPECTED_EPISODES}"
        )

    # Atomic promotion from partial to final.
    os.replace(
        episodes_partial,
        episodes_final,
    )

    manifest = {
        "schema":
            "icra27_os_t5p5b_remaining_rough_train_atlas_v0",

        "status":
            "COMPUTE_PASS",

        "checkpoint":
            str(
                checkpoint.relative_to(
                    ROOT
                )
            ),

        "reward_mode":
            EXPECTED_REWARD_MODE,

        "beta_points":
            len(beta_points),

        "train_contexts":
            EXPECTED_CONTEXTS,

        "terrain":
            "rough_perlin",

        "perlin_split_source":
            str(
                PERLIN_SPLIT_SOURCE.relative_to(
                    ROOT
                )
            ),

        "os_context_source":
            str(
                OS_CONTEXT_SOURCE.relative_to(
                    ROOT
                )
            ),

        "existing_t5p3_rough_seeds":
            list(
                EXISTING_T53_ROUGH_SEEDS
            ),

        "remaining_rough_train_seeds":
            list(
                EXPECTED_REMAINING_ROUGH_SEEDS
            ),

        "heldout_used":
            False,

        "episodes":
            len(all_rows),

        "successes":
            success_count,

        "m4_interventions_total":
            m4_total,

        "beta_probe_scope":
            (
                "convex-hull interpolation "
                "of trained biased anchors"
            ),

        "balanced_reference":
            (
                "not rerun in T5.3 lattice; "
                "reuse frozen T5.2/u30 "
                "Balanced reference later"
            ),

        "next_stage":
            (
                "extract frozen physical "
                "J_M/J_S/J_E from T5.3 raw "
                "logs and freeze TRAIN-only "
                "normalization statistics"
            ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 108)
    print(
        "[ICRA27] OS-T5.5b "
        "TRAIN beta-response atlas: COMPUTE PASS"
    )
    print("=" * 108)
    print(
        "episodes :", len(all_rows)
    )
    print(
        "success  :",
        success_count,
        "/",
        len(all_rows),
    )
    print(
        "M4 total :",
        m4_total,
    )
    print()
    print("outputs:")
    print(" ", episodes_final)
    print(" ", manifest_path)
    print(
        " ",
        out_dir / "env_logs",
    )


if __name__ == "__main__":
    main()
