from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGS = (
    "FL",
    "FR",
    "RL",
    "RR",
)


def _as_vector(
    value,
    *,
    name: str,
) -> np.ndarray:
    out = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if (
        out.size == 0
        or not np.all(
            np.isfinite(out)
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty "
            "finite vector"
        )

    return out


def _leg_indices(
    attr,
    leg: str,
    *,
    name: str,
) -> np.ndarray:
    idx = np.asarray(
        getattr(
            attr,
            leg,
        ),
        dtype=int,
    ).reshape(-1)

    if idx.size != 3:
        raise ValueError(
            f"{name}.{leg} must contain "
            f"3 indices, got {idx.size}"
        )

    return idx


def _all_leg_indices(
    attr,
    *,
    name: str,
) -> np.ndarray:
    result = np.concatenate(
        [
            _leg_indices(
                attr,
                leg,
                name=name,
            )
            for leg in LEGS
        ]
    )

    if (
        result.size != 12
        or np.unique(
            result
        ).size != 12
    ):
        raise ValueError(
            f"{name} must describe 12 "
            "unique actuated joints"
        )

    return result


def _power_summary(
    per_joint_power,
) -> dict[str, float]:
    p = _as_vector(
        per_joint_power,
        name="per_joint_power",
    )

    return {
        "signed_w":
            float(
                np.sum(p)
            ),

        "abs_w":
            float(
                np.sum(
                    np.abs(p)
                )
            ),

        "positive_w":
            float(
                np.sum(
                    np.maximum(
                        p,
                        0.0,
                    )
                )
            ),
    }


def commanded_joint_power(
    env,
    action,
) -> dict[str, float]:
    """
    Mechanical power implied by the torque command that is
    passed into the MuJoCo environment.

    Pairing:
        clipped actuator command
        x
        pre-step actual joint velocity

    Units:
        N*m * rad/s = W
    """

    action_arr = _as_vector(
        action,
        name="action",
    )

    qvel = _as_vector(
        env.mjData.qvel,
        name="mjData.qvel",
    )

    per_joint = []

    for leg in LEGS:
        tau_idx = _leg_indices(
            env.legs_tau_idx,
            leg,
            name="legs_tau_idx",
        )

        qvel_idx = _leg_indices(
            env.legs_qvel_idx,
            leg,
            name="legs_qvel_idx",
        )

        if (
            np.max(tau_idx)
            >= action_arr.size
        ):
            raise ValueError(
                "actuator index exceeds "
                "action vector"
            )

        if (
            np.max(qvel_idx)
            >= qvel.size
        ):
            raise ValueError(
                "joint velocity index exceeds "
                "qvel vector"
            )

        tau = action_arr[
            tau_idx
        ]

        qdot = qvel[
            qvel_idx
        ]

        per_joint.extend(
            (
                tau * qdot
            ).tolist()
        )

    result = _power_summary(
        per_joint
    )

    result[
        "joint_count"
    ] = 12

    return result


def applied_generalized_power(
    env,
) -> dict[str, float]:
    """
    MuJoCo generalized actuator mechanical power.

    Uses only the 12 actuated leg velocity coordinates:
        qfrc_actuator[j] * qvel[j]

    This intentionally excludes floating-base coordinates.
    """

    qvel = _as_vector(
        env.mjData.qvel,
        name="mjData.qvel",
    )

    qfrc = _as_vector(
        env.mjData.qfrc_actuator,
        name="mjData.qfrc_actuator",
    )

    qvel_idx = _all_leg_indices(
        env.legs_qvel_idx,
        name="legs_qvel_idx",
    )

    if (
        np.max(qvel_idx) >= qvel.size
        or np.max(qvel_idx) >= qfrc.size
    ):
        raise ValueError(
            "actuated qvel index exceeds "
            "MuJoCo generalized vectors"
        )

    per_joint = (
        qfrc[qvel_idx]
        * qvel[qvel_idx]
    )

    result = _power_summary(
        per_joint
    )

    result[
        "joint_count"
    ] = 12

    return result


@dataclass
class MechanicalEnergyAccumulator:
    """
    Logging-only mechanical-energy accumulator.

    No reward semantics live here.
    """

    elapsed_s: float = 0.0
    samples: int = 0

    commanded_signed_j: float = 0.0
    commanded_abs_j: float = 0.0
    commanded_positive_j: float = 0.0

    applied_signed_j: float = 0.0
    applied_abs_j: float = 0.0
    applied_positive_j: float = 0.0

    def add(
        self,
        *,
        dt: float,
        commanded: dict[str, float],
        applied: dict[str, float],
    ) -> None:
        dt = float(dt)

        if (
            not np.isfinite(dt)
            or dt <= 0.0
        ):
            raise ValueError(
                "dt must be finite and > 0"
            )

        self.elapsed_s += dt
        self.samples += 1

        self.commanded_signed_j += (
            float(
                commanded["signed_w"]
            )
            * dt
        )

        self.commanded_abs_j += (
            float(
                commanded["abs_w"]
            )
            * dt
        )

        self.commanded_positive_j += (
            float(
                commanded["positive_w"]
            )
            * dt
        )

        self.applied_signed_j += (
            float(
                applied["signed_w"]
            )
            * dt
        )

        self.applied_abs_j += (
            float(
                applied["abs_w"]
            )
            * dt
        )

        self.applied_positive_j += (
            float(
                applied["positive_w"]
            )
            * dt
        )

    def as_dict(
        self,
    ) -> dict[str, float | int]:
        return {
            "samples":
                int(self.samples),

            "elapsed_s":
                float(self.elapsed_s),

            "commanded_signed_j":
                float(
                    self.commanded_signed_j
                ),

            "commanded_abs_j":
                float(
                    self.commanded_abs_j
                ),

            "commanded_positive_j":
                float(
                    self.commanded_positive_j
                ),

            "applied_signed_j":
                float(
                    self.applied_signed_j
                ),

            "applied_abs_j":
                float(
                    self.applied_abs_j
                ),

            "applied_positive_j":
                float(
                    self.applied_positive_j
                ),
        }
