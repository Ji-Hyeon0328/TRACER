#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import sys

import freeze_os_t5p5a_oracle_height_patch_v0 as freezer


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
        freezer.BASE.active_robot_name()
    )

    hip_height = freezer.finite(
        freezer.BASE.cfg.hip_height
    )

    with contextlib.redirect_stdout(
        sys.stderr
    ):
        row = freezer.rough_patch_row(
            seed=int(
                args.seed
            ),
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
