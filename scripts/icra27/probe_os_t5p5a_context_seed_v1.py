#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import sys

import freeze_os_t5p5a_oracle_context_descriptor_v1 as freezer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    robot_name = (
        freezer.active_robot_name()
    )

    hip_height = freezer.finite(
        freezer.cfg.hip_height
    )

    # QuadrupedEnv / Gym may print informational
    # messages. Keep stdout machine-readable by
    # redirecting all incidental prints to stderr.
    with contextlib.redirect_stdout(
        sys.stderr
    ):
        row = freezer.rough_row(
            seed=int(args.seed),
            robot_name=robot_name,
            hip_height=hip_height,
        )

    print(
        json.dumps(
            row,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
