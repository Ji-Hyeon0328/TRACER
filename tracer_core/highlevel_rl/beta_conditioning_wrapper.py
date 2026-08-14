from __future__ import annotations

import gymnasium as gym
import numpy as np

from tracer_core.highlevel_rl.reward_v2 import (
    validate_beta,
)


class M7BetaConditioningWrapper(
    gym.ObservationWrapper
):
    """
    Append the semantic objective weight beta to the frozen
    M7 policy observation.

    Input observation:
        obs21

    Policy observation:
        [obs21, beta_motion, beta_stability, beta_energy]

    Output shape:
        obs24

    The wrapper also synchronizes the selected beta with the
    underlying PyMPCM7Env reward path.

    Beta is intended to remain fixed within a physical episode
    during Phase-1A identifiability experiments.
    """

    BASE_OBS_DIM = 21
    BETA_DIM = 3
    POLICY_OBS_DIM = 24

    def __init__(
        self,
        env: gym.Env,
        beta=(
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ),
    ) -> None:
        super().__init__(
            env
        )

        downstream_shape = tuple(
            self.env.observation_space.shape
        )

        if downstream_shape != (
            self.BASE_OBS_DIM,
        ):
            raise ValueError(
                "M7BetaConditioningWrapper expects "
                "a downstream observation shape (21,), "
                f"got {downstream_shape}"
            )

        self.observation_space = gym.spaces.Box(
            low=np.full(
                self.POLICY_OBS_DIM,
                -np.inf,
                dtype=np.float32,
            ),
            high=np.full(
                self.POLICY_OBS_DIM,
                np.inf,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._beta = None

        self.set_beta(
            beta
        )

    @property
    def beta(
        self,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        return tuple(
            self._beta
        )

    def _base_env(
        self,
    ):
        base = self.env.unwrapped

        if not hasattr(
            base,
            "set_tracer_beta",
        ):
            raise RuntimeError(
                "Underlying env does not expose "
                "set_tracer_beta()."
            )

        return base

    def set_beta(
        self,
        beta,
    ) -> None:
        validated = validate_beta(
            beta
        )

        self._beta = tuple(
            validated
        )

        self._base_env().set_tracer_beta(
            self._beta
        )

    def observation(
        self,
        observation,
    ) -> np.ndarray:
        obs = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        if obs.shape != (
            self.BASE_OBS_DIM,
        ):
            raise RuntimeError(
                "Unexpected base observation shape: "
                f"{obs.shape}"
            )

        beta = np.asarray(
            self._beta,
            dtype=np.float32,
        )

        conditioned = np.concatenate(
            (
                obs,
                beta,
            ),
            axis=0,
        )

        if conditioned.shape != (
            self.POLICY_OBS_DIM,
        ):
            raise RuntimeError(
                "Unexpected conditioned observation "
                f"shape: {conditioned.shape}"
            )

        if not np.all(
            np.isfinite(
                conditioned
            )
        ):
            raise RuntimeError(
                "Conditioned observation contains "
                "non-finite values."
            )

        return conditioned

    def step(
        self,
        action,
    ):
        result = super().step(
            action
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = result

        # Phase-1A invariant:
        # policy-conditioning beta and reward beta must
        # agree on every physical transition, including
        # settling and terminal transitions.
        self.assert_reward_beta(
            info
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def assert_reward_beta(
        self,
        info,
        *,
        atol: float = 1e-8,
    ) -> None:
        """
        Verify that observation conditioning and the reward
        backend used the same beta.
        """
        components = info.get(
            "reward_components",
            {},
        )

        keys = (
            "beta_motion",
            "beta_stability",
            "beta_energy",
        )

        if not all(
            key in components
            for key in keys
        ):
            raise RuntimeError(
                "reward_components does not expose "
                "the TRACER beta contract."
            )

        reward_beta = np.array(
            [
                components[key]
                for key in keys
            ],
            dtype=np.float64,
        )

        expected = np.asarray(
            self._beta,
            dtype=np.float64,
        )

        if not np.allclose(
            reward_beta,
            expected,
            rtol=0.0,
            atol=float(atol),
        ):
            raise RuntimeError(
                "Policy-conditioning beta and reward beta "
                "do not match: "
                f"obs_beta={expected.tolist()}, "
                f"reward_beta={reward_beta.tolist()}"
            )
