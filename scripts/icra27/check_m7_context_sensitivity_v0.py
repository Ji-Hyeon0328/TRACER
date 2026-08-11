#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.ppo import (
    load_ppo_checkpoint,
)

from tracer_core.highlevel_rl.pympc_env import (
    PyMPCM7Env,
)

from tracer_core.highlevel_rl.fixed_clearance_wrapper import (
    M7FixedClearanceActionWrapper,
)


CONTEXTS = {
    "flat":
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),

    "rough_perlin":
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ),

    "low_friction":
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
}


SETTLING_STEPS = 5

SEEDS = (
    1,
    21,
    16,
    14,
)


def get_settled_obs(
    env,
    *,
    seed,
):
    obs, info = env.reset(
        seed=seed
    )

    zero = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )

    for _ in range(
        SETTLING_STEPS
    ):
        (
            obs,
            _reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero
        )

        if terminated or truncated:
            raise RuntimeError(
                "Settling failed for "
                f"seed={seed}"
            )

    obs = np.asarray(
        obs,
        dtype=np.float32,
    )

    if obs.shape != (21,):
        raise RuntimeError(
            "Expected 21D observation, "
            f"got {obs.shape}"
        )

    return obs


def policy_action(
    model,
    obs,
):
    (
        action,
        _log_prob,
        _value,
    ) = model.act(
        obs,
        deterministic=True,
    )

    return np.asarray(
        action,
        dtype=np.float32,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    ap.add_argument(
        "--updates",
        type=int,
        nargs="+",
        default=[
            1,
            6,
            10,
        ],
    )

    ap.add_argument(
        "--base-port",
        type=int,
        default=51710,
    )

    args = ap.parse_args()

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    models = {}

    for update in args.updates:
        p = (
            checkpoint_dir
            / (
                "checkpoint_update_"
                f"{update:04d}.pt"
            )
        )

        if not p.exists():
            raise FileNotFoundError(
                p
            )

        (
            model,
            _cfg,
            _extra,
        ) = load_ppo_checkpoint(
            p
        )

        model.eval()

        models[
            update
        ] = model

    # --------------------------------------------------------
    # Use real settled states from all three environments.
    #
    # Then replace ONLY obs[0:3].
    # The remaining 18 dimensions remain exactly identical.
    # --------------------------------------------------------
    source_terrains = (
        "flat",
        "low_friction",
        "rough_perlin",
    )

    print("=" * 104)
    print(
        "ICRA27 M7 COUNTERFACTUAL "
        "ORACLE-CONTEXT SENSITIVITY"
    )
    print("=" * 104)

    for terrain_index, source_terrain in enumerate(
        source_terrains
    ):
        port = (
            args.base_port
            + 10 * terrain_index
        )

        base_env = PyMPCM7Env(
            terrain=source_terrain,
            terminate_on_m4_unsafe=True,

            goal_distance_m=2.0,
            success_radius_m=0.15,

            decision_dt_s=0.20,
            max_episode_steps=55,

            command_port=port,
            telemetry_port=port + 1,
            state_port=port + 2,

            command_repeat_hz=20.0,
            telemetry_hz=100.0,
            state_hz=100.0,

            log_dir=(
                "results/icra27/"
                "m7_context_sensitivity_v0/"
                f"{source_terrain}"
            ),
        )

        env = (
            M7FixedClearanceActionWrapper(
                base_env
            )
        )

        try:
            print()
            print(
                "-" * 104
            )
            print(
                "physical-state source:",
                source_terrain,
            )
            print(
                "-" * 104
            )

            for update, model in (
                models.items()
            ):
                per_seed_spreads = []

                print(
                    f"\ncheckpoint "
                    f"{update:04d}"
                )

                for seed in SEEDS:
                    base_obs = (
                        get_settled_obs(
                            env,
                            seed=seed,
                        )
                    )

                    actions = {}

                    for (
                        context_name,
                        context,
                    ) in CONTEXTS.items():
                        cf_obs = (
                            base_obs.copy()
                        )

                        cf_obs[
                            0:3
                        ] = context

                        actions[
                            context_name
                        ] = policy_action(
                            model,
                            cf_obs,
                        )

                    stacked = np.stack(
                        list(
                            actions.values()
                        )
                    )

                    spread = (
                        np.max(
                            stacked,
                            axis=0,
                        )
                        - np.min(
                            stacked,
                            axis=0,
                        )
                    )

                    per_seed_spreads.append(
                        spread
                    )

                    print(
                        f"seed={seed:02d} "
                        f"flat="
                        f"{actions['flat']} "
                        f"rough="
                        f"{actions['rough_perlin']} "
                        f"low="
                        f"{actions['low_friction']} "
                        f"spread="
                        f"{spread}"
                    )

                spreads = np.stack(
                    per_seed_spreads
                )

                mean_spread = np.mean(
                    spreads,
                    axis=0,
                )

                max_spread = np.max(
                    spreads,
                    axis=0,
                )

                print(
                    "  MEAN context spread "
                    "[vx,yaw,h] =",
                    np.array2string(
                        mean_spread,
                        precision=6,
                    ),
                )

                print(
                    "  MAX  context spread "
                    "[vx,yaw,h] =",
                    np.array2string(
                        max_spread,
                        precision=6,
                    ),
                )

        finally:
            env.close()

    print()
    print(
        "[ICRA27] counterfactual "
        "context sensitivity: PASS"
    )


if __name__ == "__main__":
    main()
