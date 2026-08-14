#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tracer_core.highlevel_rl.beta_conditioning_wrapper import (
    M7BetaConditioningWrapper,
)


BETAS = {
    "balanced": (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ),

    "motion": (
        0.70,
        0.15,
        0.15,
    ),

    "stability": (
        0.15,
        0.70,
        0.15,
    ),

    "energy": (
        0.15,
        0.15,
        0.70,
    ),
}


class DummyM7Env(
    gym.Env
):
    def __init__(
        self,
    ):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=np.full(
                21,
                -np.inf,
                dtype=np.float32,
            ),
            high=np.full(
                21,
                np.inf,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=np.full(
                3,
                -1.0,
                dtype=np.float32,
            ),
            high=np.full(
                3,
                1.0,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.tracer_beta = None

    def set_tracer_beta(
        self,
        beta,
    ):
        self.tracer_beta = tuple(
            float(x)
            for x in beta
        )


def main():
    base = DummyM7Env()

    env = M7BetaConditioningWrapper(
        base
    )

    assert env.observation_space.shape == (
        24,
    )

    raw = np.arange(
        21,
        dtype=np.float32,
    )

    for name, beta in BETAS.items():
        env.set_beta(
            beta
        )

        obs = env.observation(
            raw
        )

        assert obs.shape == (
            24,
        )

        np.testing.assert_allclose(
            obs[:21],
            raw,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            obs[21:24],
            beta,
            rtol=0.0,
            atol=1e-7,
        )

        np.testing.assert_allclose(
            base.tracer_beta,
            beta,
            rtol=0.0,
            atol=1e-8,
        )

        info = {
            "reward_components": {
                "beta_motion":
                    beta[0],

                "beta_stability":
                    beta[1],

                "beta_energy":
                    beta[2],
            }
        }

        env.assert_reward_beta(
            info
        )

        print(
            f"{name:<10} "
            f"obs_beta="
            f"{obs[21:24].tolist()} "
            f"reward_beta={list(beta)} "
            "PASS"
        )

    # Invalid simplex checks.
    bad = (
        (0.5, 0.5, 0.5),
        (-0.1, 0.5, 0.6),
        (1.0, 0.0),
    )

    for beta in bad:
        try:
            env.set_beta(
                beta
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid beta accepted: {beta}"
            )

    print()
    print(
        "[ICRA27] beta-conditioned observation "
        "contract: PASS"
    )


if __name__ == "__main__":
    main()
