#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import sys

import freeze_os_t5p5a_oracle_height_patch_v0 as freezer


N_LONG = 16
N_LAT = 12


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

    # ========================================================
    # Resolution-only controlled modification.
    #
    # Keep:
    #   - same compiled hfield
    #   - same reset semantics
    #   - same start pose / yaw frame
    #   - same longitudinal/lateral domain
    #   - same relative-height reference
    #   - same bilinear interpolation
    #
    # Change ONLY:
    #   8 x 6 -> 16 x 12
    # ========================================================

    freezer.N_LONGITUDINAL = N_LONG
    freezer.N_LATERAL = N_LAT
    freezer.PATCH_DIM = (
        N_LONG
        * N_LAT
    )

    freezer.PATCH_FEATURES = tuple(
        f"h_l{i:02d}_r{j:02d}_m"
        for i in range(N_LONG)
        for j in range(N_LAT)
    )

    if len(
        freezer.PATCH_FEATURES
    ) != 192:
        raise RuntimeError(
            "Expected 192 high-resolution "
            "patch features."
        )

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

    feature_count = sum(
        1
        for key in row
        if key.startswith(
            "h_l"
        )
        and key.endswith(
            "_m"
        )
    )

    if feature_count != 192:
        raise RuntimeError(
            f"Expected 192 patch values; "
            f"got {feature_count}."
        )

    print(
        json.dumps(
            row,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
