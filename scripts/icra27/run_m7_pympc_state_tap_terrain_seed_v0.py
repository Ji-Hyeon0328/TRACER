#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


def _seed_from_argv(
    argv: list[str],
) -> int:
    for index, arg in enumerate(argv):
        if arg == "--seed":
            if index + 1 >= len(argv):
                raise ValueError(
                    "--seed requires an integer value"
                )

            return int(
                argv[index + 1]
            )

        if arg.startswith("--seed="):
            return int(
                arg.split(
                    "=",
                    1,
                )[1]
            )

    # Preserve the existing M7 convention:
    # reset(seed=None) resolves to seed=0 before
    # the runner is spawned. Reaching this fallback
    # therefore indicates a direct/manual invocation.
    return 0


def _pop_float_option(
    argv: list[str],
    name: str,
    default: float,
) -> tuple[float, list[str]]:
    cleaned = []
    value = float(default)

    index = 0

    while index < len(argv):
        arg = argv[index]

        if arg == name:
            if index + 1 >= len(argv):
                raise ValueError(
                    f"{name} requires a value"
                )

            value = float(
                argv[index + 1]
            )

            index += 2
            continue

        prefix = name + "="

        if arg.startswith(prefix):
            value = float(
                arg[len(prefix):]
            )

            index += 1
            continue

        cleaned.append(arg)
        index += 1

    return value, cleaned


def main() -> None:
    terrain_seed = _seed_from_argv(
        sys.argv[1:]
    )

    (
        rough_height_scale,
        cleaned_argv,
    ) = _pop_float_option(
        sys.argv[1:],
        "--rough-height-scale",
        1.0,
    )

    if rough_height_scale <= 0.0:
        raise ValueError(
            "--rough-height-scale must be > 0"
        )

    # Remove this sidecar-only argument before
    # forwarding argv to the already-validated
    # terrain/state-tap runner.
    sys.argv = [
        sys.argv[0],
        *cleaned_argv,
    ]

    import gym_quadruped.quadruped_env as gym_quadruped_env
    from gym_quadruped.utils.mujoco import terrain as gym_terrain

    # Import the already-validated M7 terrain
    # sidecar. The current script directory is
    # present in sys.path when this file is run.
    import run_m7_pympc_state_tap_terrain_v0 as terrain_runner

    original_env_generate_terrain = (
        gym_quadruped_env.generate_terrain
    )

    original_module_generate_terrain = (
        gym_terrain.generate_terrain
    )

    original_add_world_of_boxes = (
        gym_terrain.add_world_of_boxes
    )

    original_pnoise2 = (
        gym_terrain.noise.pnoise2
    )

    def seeded_pnoise2(
        *args,
        **kwargs,
    ):
        # gym_quadruped's upstream Perlin generator
        # accepts a seed at generate_terrain(), but
        # does not consume that seed inside pnoise2().
        #
        # Bind Perlin geometry deterministically to
        # the M7 episode/terrain seed without editing
        # site-packages.
        kwargs.setdefault(
            "base",
            int(terrain_seed),
        )

        return original_pnoise2(
            *args,
            **kwargs,
        )

    def scaled_add_world_of_boxes(
        *args,
        **kwargs,
    ):
        if rough_height_scale != 1.0:
            if "box_size" not in kwargs:
                raise RuntimeError(
                    "Expected box_size keyword in "
                    "random_boxes generator"
                )

            if "box_size_rand" not in kwargs:
                raise RuntimeError(
                    "Expected box_size_rand keyword "
                    "in random_boxes generator"
                )

            box_size = list(
                kwargs["box_size"]
            )

            box_size_rand = list(
                kwargs["box_size_rand"]
            )

            # Change only the vertical roughness.
            # XY geometry, spacing and orientation
            # randomness remain untouched.
            box_size[2] *= (
                rough_height_scale
            )

            box_size_rand[2] *= (
                rough_height_scale
            )

            kwargs["box_size"] = (
                box_size
            )

            kwargs["box_size_rand"] = (
                box_size_rand
            )

        return (
            original_add_world_of_boxes(
                *args,
                **kwargs,
            )
        )

    def seeded_generate_terrain(
        *args,
        **kwargs,
    ):
        # QuadrupedEnv currently calls:
        #
        #   generate_terrain(..., seed=10)
        #
        # Ignore that package-internal fixed value and
        # bind the procedural geometry to the M7 runner
        # seed instead.
        kwargs["seed"] = int(
            terrain_seed
        )

        return (
            original_module_generate_terrain(
                *args,
                **kwargs,
            )
        )

    gym_quadruped_env.generate_terrain = (
        seeded_generate_terrain
    )

    gym_terrain.generate_terrain = (
        seeded_generate_terrain
    )

    gym_terrain.add_world_of_boxes = (
        scaled_add_world_of_boxes
    )

    gym_terrain.noise.pnoise2 = (
        seeded_pnoise2
    )

    print(
        "[M7 terrain seed] "
        f"rough height scale={rough_height_scale}"
    )

    print(
        "[M7 terrain seed] procedural terrain "
        f"seed={terrain_seed}"
    )

    try:
        terrain_runner.main()

    finally:
        gym_quadruped_env.generate_terrain = (
            original_env_generate_terrain
        )

        gym_terrain.generate_terrain = (
            original_module_generate_terrain
        )

        gym_terrain.add_world_of_boxes = (
            original_add_world_of_boxes
        )

        gym_terrain.noise.pnoise2 = (
            original_pnoise2
        )

        hooks_restored = (
            gym_quadruped_env.generate_terrain
            is original_env_generate_terrain
            and gym_terrain.generate_terrain
            is original_module_generate_terrain
            and gym_terrain.add_world_of_boxes
            is original_add_world_of_boxes
            and gym_terrain.noise.pnoise2
            is original_pnoise2
        )

        print(
            "[M7 terrain seed] "
            "generate_terrain hooks restored: "
            f"{hooks_restored}"
        )


if __name__ == "__main__":
    main()
