#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
ICRA27 = ROOT / "scripts" / "icra27"

sys.path.insert(
    0,
    str(ROOT),
)

sys.path.insert(
    0,
    str(ICRA27),
)


import run_m7_pympc_state_tap_v0 as state_tap

from tracer_core.highlevel_rl.terrain import (
    get_terrain_preset,
    terrain_names,
)


ORIGINAL_RUN_SIMULATION = (
    state_tap.standalone
    .sim
    .run_simulation
)

ACTIVE_PRESET = None


def terrain_run_simulation(
    *args,
    **kwargs,
):
    """
    Thin environment-side terrain injection.

    The original M5 call still owns:
      - controller
      - runtime supervisor
      - command transport
      - reset semantics
      - PyMPC execution

    This wrapper changes only:
      1. qpympc_cfg.simulation_params["scene"]
      2. run_simulation(... friction_coeff=...)
    """

    if ACTIVE_PRESET is None:
        raise RuntimeError(
            "terrain preset not configured"
        )

    qpympc_cfg = kwargs.get(
        "qpympc_cfg",
        None,
    )

    if qpympc_cfg is None:
        raise RuntimeError(
            "run_simulation did not receive "
            "qpympc_cfg as keyword argument"
        )

    simulation_params = (
        qpympc_cfg.simulation_params
    )

    old_scene = (
        simulation_params.get(
            "scene",
            None,
        )
    )

    old_friction = kwargs.get(
        "friction_coeff",
        None,
    )

    simulation_params[
        "scene"
    ] = ACTIVE_PRESET.scene

    kwargs[
        "friction_coeff"
    ] = float(
        ACTIVE_PRESET.friction_coeff
    )

    print("=" * 72)

    print(
        "ICRA27 M7 TERRAIN SIDECAR"
    )

    print("=" * 72)

    print(
        "terrain preset : "
        f"{ACTIVE_PRESET.name}"
    )

    print(
        "scene          : "
        f"{ACTIVE_PRESET.scene}"
    )

    print(
        "friction       : "
        f"{ACTIVE_PRESET.friction_coeff:.6f}"
    )

    print(
        "oracle context : "
        f"{list(ACTIVE_PRESET.oracle_context)}"
    )

    print(
        "original scene : "
        f"{old_scene}"
    )

    print(
        "original friction arg : "
        f"{old_friction}"
    )

    print(
        "NOTE: M4/M5/PyMPC source semantics "
        "are unchanged."
    )

    print("=" * 72)

    try:
        return ORIGINAL_RUN_SIMULATION(
            *args,
            **kwargs,
        )

    finally:
        if old_scene is None:
            simulation_params.pop(
                "scene",
                None,
            )
        else:
            simulation_params[
                "scene"
            ] = old_scene

        print(
            "[M7 terrain] scene config restored "
            f"to {old_scene!r}"
        )


def parse_terrain_args():
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    parser.add_argument(
        "--terrain",
        choices=terrain_names(),
        default="flat",
    )

    parser.add_argument(
        "--terrain-friction",
        type=float,
        default=None,
        help=(
            "Optional friction override. "
            "Use for characterization only."
        ),
    )

    terrain_args, remaining = (
        parser.parse_known_args()
    )

    return (
        terrain_args,
        remaining,
    )


def main():
    global ACTIVE_PRESET

    (
        terrain_args,
        remaining,
    ) = parse_terrain_args()

    ACTIVE_PRESET = (
        get_terrain_preset(
            terrain_args.terrain,
            friction_override=(
                terrain_args.terrain_friction
            ),
        )
    )

    # Remove terrain-sidecar-only arguments before
    # handing argv to the already validated M5 parser.
    original_argv = list(
        sys.argv
    )

    sys.argv = [
        original_argv[0],
        *remaining,
    ]

    (
        state_tap.standalone
        .sim
        .run_simulation
    ) = terrain_run_simulation

    print(
        "[M7 terrain] selected "
        f"{ACTIVE_PRESET.name!r}"
    )

    try:
        state_tap.main()

    finally:
        (
            state_tap.standalone
            .sim
            .run_simulation
        ) = ORIGINAL_RUN_SIMULATION

        sys.argv = original_argv

        hook_restored = (
            state_tap.standalone.sim.run_simulation
            is ORIGINAL_RUN_SIMULATION
        )

        print(
            "[M7 terrain] "
            "run_simulation hook restored: "
            f"{hook_restored}"
        )


if __name__ == "__main__":
    main()
