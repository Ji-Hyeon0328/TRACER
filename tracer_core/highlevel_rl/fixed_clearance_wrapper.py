from __future__ import annotations

import gymnasium as gym
import numpy as np


class M7FixedClearanceActionWrapper(
    gym.ActionWrapper
):
    """
    Expose the ICRA27 M7 learned policy as a 3-D action:

        [vx, yaw_rate, body_height]

    while preserving the frozen downstream M7/M5 4-D action
    contract:

        [vx, yaw_rate, body_height, swing_clearance]

    The normalized clearance component is fixed to zero, which
    maps to the characterized nominal physical clearance of
    0.060 m in the existing M7 action contract.

    The wrapped PyMPCM7Env remains unchanged and therefore keeps:
      - its 4-D M5 transport contract,
      - its 4-D previous-action observation,
      - existing characterization tools,
      - existing reward/telemetry semantics.
    """

    POLICY_ACTION_DIM = 3
    DOWNSTREAM_ACTION_DIM = 4

    def __init__(
        self,
        env: gym.Env,
    ) -> None:
        super().__init__(env)

        downstream_shape = tuple(
            self.env.action_space.shape
        )

        if downstream_shape != (
            self.DOWNSTREAM_ACTION_DIM,
        ):
            raise ValueError(
                "M7FixedClearanceActionWrapper expects "
                "a downstream action space of shape (4,), "
                f"got {downstream_shape}"
            )

        self.action_space = gym.spaces.Box(
            low=np.full(
                self.POLICY_ACTION_DIM,
                -1.0,
                dtype=np.float32,
            ),
            high=np.full(
                self.POLICY_ACTION_DIM,
                1.0,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def expand_policy_action(
        action,
    ) -> np.ndarray:
        policy_action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if policy_action.shape != (3,):
            raise ValueError(
                "policy action must have shape (3,), "
                f"got {policy_action.shape}"
            )

        if not np.all(
            np.isfinite(policy_action)
        ):
            raise ValueError(
                "policy action contains non-finite "
                "values"
            )

        policy_action = np.clip(
            policy_action,
            -1.0,
            1.0,
        )

        return np.array(
            [
                policy_action[0],
                policy_action[1],
                policy_action[2],
                0.0,
            ],
            dtype=np.float32,
        )

    def action(
        self,
        action,
    ):
        return self.expand_policy_action(
            action
        )
