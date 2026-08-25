#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from characterize_m7_friction_v0 import (
    run_episode,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
    )

    parser.add_argument(
        "--fixed-vx",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--rough-height-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--goal-distance",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/icra27/"
            "m7_rough_characterization_v0"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not (
        0.0
        <= args.fixed_vx
        <= 0.40
    ):
        raise ValueError(
            "--fixed-vx must be in [0, 0.4]"
        )

    normalized_vx = (
        args.fixed_vx - 0.20
    ) / 0.20

    fixed_action = np.array(
        [
            normalized_vx,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    output_dir = (
        ROOT
        / args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "ICRA27 M7 ROUGH-BOXES "
        "CHARACTERIZATION"
    )
    print("=" * 78)

    print(
        "terrain       : rough_boxes"
    )

    print(
        "scene         : random_boxes"
    )

    print(
        "friction      : 0.8"
    )

    print(
        "height scale  : "
        f"{args.rough_height_scale:.3f}"
    )

    print(
        "fixed vx      : "
        f"{args.fixed_vx:.3f} m/s"
    )

    print(
        "action        : "
        f"{fixed_action.tolist()}"
    )

    print(
        "terrain seeds : "
        f"{args.seeds}"
    )

    print("=" * 78)

    rows = []

    for seed in args.seeds:
        env = PyMPCM7Env(
            terrain="rough_boxes",
            rough_height_scale=(
                args.rough_height_scale
            ),
            terminate_on_m4_unsafe=True,
            goal_distance_m=(
                args.goal_distance
            ),
            max_episode_steps=(
                args.max_episode_steps
            ),
        )

        try:
            result = run_episode(
                env=env,
                seed=seed,
                fixed_action=fixed_action,
            )

        finally:
            env.close()

        result[
            "terrain"
        ] = "rough_boxes"

        result[
            "terrain_seed"
        ] = int(seed)

        result[
            "fixed_vx_mps"
        ] = float(
            args.fixed_vx
        )

        result[
            "rough_height_scale"
        ] = float(
            args.rough_height_scale
        )

        rows.append(
            result
        )

        print(
            f"seed={seed:02d} "
            f"outcome="
            f"{result['outcome']:<13} "
            f"steps="
            f"{result['steps']:02d} "
            f"goal="
            f"{result['final_goal_distance_m']:.3f} "
            f"return="
            f"{result['episode_return']:+.3f} "
            f"watch="
            f"{result['watch_count']:02d} "
            f"m4="
            f"{result['m4_intervention_count']:02d} "
            f"roll="
            f"{np.degrees(result['max_abs_roll_rad']):.1f}deg "
            f"pitch="
            f"{np.degrees(result['max_abs_pitch_rad']):.1f}deg "
            f"vx="
            f"{result['mean_measured_vx_mps']:.3f} "
            f"rmse="
            f"{result['vx_tracking_rmse_mps']:.3f}"
        )

    csv_path = (
        output_dir
        / "episodes.csv"
    )

    if rows:
        fieldnames = list(
            rows[0].keys()
        )

        with csv_path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    n = len(rows)

    success_count = sum(
        r["outcome"] == "success"
        for r in rows
    )

    m4_count = sum(
        r["outcome"] == "m4_terminal"
        for r in rows
    )

    runtime_count = sum(
        r["outcome"]
        in (
            "state_timeout",
            "runner_exit",
        )
        for r in rows
    )

    summary = {
        "terrain":
            "rough_boxes",

        "scene":
            "random_boxes",

        "friction":
            0.8,

        "rough_height_scale":
            float(
                args.rough_height_scale
            ),

        "fixed_vx_mps":
            float(args.fixed_vx),

        "episodes":
            n,

        "success_rate":
            (
                success_count / n
                if n
                else 0.0
            ),

        "m4_terminal_rate":
            (
                m4_count / n
                if n
                else 0.0
            ),

        "runtime_failure_rate":
            (
                runtime_count / n
                if n
                else 0.0
            ),

        "mean_final_goal_distance_m":
            float(
                np.mean(
                    [
                        r[
                            "final_goal_distance_m"
                        ]
                        for r in rows
                    ]
                )
            ),

        "mean_vx_tracking_rmse_mps":
            float(
                np.mean(
                    [
                        r[
                            "vx_tracking_rmse_mps"
                        ]
                        for r in rows
                    ]
                )
            ),

        "max_roll_deg":
            float(
                max(
                    np.degrees(
                        r[
                            "max_abs_roll_rad"
                        ]
                    )
                    for r in rows
                )
            ),

        "max_pitch_deg":
            float(
                max(
                    np.degrees(
                        r[
                            "max_abs_pitch_rad"
                        ]
                    )
                    for r in rows
                )
            ),
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        f"episodes={n} "
        f"success={summary['success_rate']:.2f} "
        f"m4={summary['m4_terminal_rate']:.2f} "
        f"runtime_failure="
        f"{summary['runtime_failure_rate']:.2f} "
        f"goal="
        f"{summary['mean_final_goal_distance_m']:.3f} "
        f"vx_rmse="
        f"{summary['mean_vx_tracking_rmse_mps']:.3f} "
        f"max_roll="
        f"{summary['max_roll_deg']:.1f}deg "
        f"max_pitch="
        f"{summary['max_pitch_deg']:.1f}deg"
    )

    print()
    print(
        "episodes :",
        csv_path,
    )

    print(
        "summary  :",
        summary_path,
    )


if __name__ == "__main__":
    main()
