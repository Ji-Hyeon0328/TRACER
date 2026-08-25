#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


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

from tracer_core.highlevel_rl.beta_conditioning_wrapper import (
    M7BetaConditioningWrapper,
)


CHECKPOINT = (
    ROOT
    / "results"
    / "icra27"
    / "phase1a_beta_conditioned_v1_seed27027"
    / "m7_ppo_beta_conditioned_final.pt"
)

BETA_BANK = (
    (
        "balanced",
        (
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ),
    ),
    (
        "motion",
        (
            0.70,
            0.15,
            0.15,
        ),
    ),
    (
        "stability",
        (
            0.15,
            0.70,
            0.15,
        ),
    ),
    (
        "energy",
        (
            0.15,
            0.15,
            0.70,
        ),
    ),
)

SETTLING_STEPS = 5
POLICY_HORIZON = 50


def policy_action(
    model,
    obs21,
    beta,
):
    obs24 = np.concatenate(
        (
            np.asarray(
                obs21,
                dtype=np.float32,
            ),
            np.asarray(
                beta,
                dtype=np.float32,
            ),
        ),
        axis=0,
    )

    if obs24.shape != (24,):
        raise RuntimeError(
            f"Expected obs24, got {obs24.shape}"
        )

    action, _, value = model.act(
        obs24,
        deterministic=True,
    )

    return (
        np.asarray(
            action,
            dtype=np.float64,
        ),
        float(value),
    )


def physical_from_action3(
    action,
):
    a = np.asarray(
        action,
        dtype=np.float64,
    )

    vx = 0.2 * (
        a[0] + 1.0
    )

    yaw = 0.4 * a[1]

    if a[2] < 0.0:
        height = (
            0.30
            + 0.06 * a[2]
        )
    else:
        height = (
            0.30
            + 0.02 * a[2]
        )

    return np.array(
        [
            vx,
            yaw,
            height,
        ],
        dtype=np.float64,
    )


def main():
    model, _cfg, payload = (
        load_ppo_checkpoint(
            CHECKPOINT
        )
    )

    if int(payload["obs_dim"]) != 24:
        raise RuntimeError(
            "Expected beta-conditioned obs_dim=24"
        )

    if int(payload["act_dim"]) != 3:
        raise RuntimeError(
            "Expected action_dim=3"
        )

    model.eval()

    base_env = PyMPCM7Env(
        terrain="rough_perlin",
        reward_mode="tracer_cost_v2",

        tracer_beta=(
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ),

        terminate_on_m4_unsafe=True,

        goal_distance_m=2.0,
        success_radius_m=0.15,

        decision_dt_s=0.20,

        max_episode_steps=(
            POLICY_HORIZON
            + SETTLING_STEPS
        ),

        command_port=59110,
        telemetry_port=59111,
        state_port=59112,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=(
            ROOT
            / "results"
            / "icra27"
            / "phase1a_beta_sensitivity_v0"
        ),
    )

    action_env = (
        M7FixedClearanceActionWrapper(
            base_env
        )
    )

    env = M7BetaConditioningWrapper(
        action_env,
        beta=BETA_BANK[0][1],
    )

    zero = np.zeros(
        3,
        dtype=np.float32,
    )

    records = []

    try:
        obs, _info = env.reset(
            seed=2
        )

        for settling_i in range(
            SETTLING_STEPS
        ):
            (
                obs,
                _reward,
                terminated,
                truncated,
                _info,
            ) = env.step(
                zero
            )

            if terminated or truncated:
                raise RuntimeError(
                    "Unexpected settling terminal: "
                    f"{settling_i + 1}"
                )

        for step in range(
            POLICY_HORIZON
        ):
            obs = np.asarray(
                obs,
                dtype=np.float32,
            )

            obs21 = obs[:21].copy()

            counterfactual = {}

            for name, beta in BETA_BANK:
                (
                    action,
                    value,
                ) = policy_action(
                    model,
                    obs21,
                    beta,
                )

                counterfactual[
                    name
                ] = {
                    "action":
                        action,

                    "physical":
                        physical_from_action3(
                            action
                        ),

                    "value":
                        value,
                }

            records.append(
                counterfactual
            )

            # Advance the physical trajectory using balanced
            # beta only. All four counterfactual actions above
            # were evaluated on this exact same state.
            balanced_action = (
                counterfactual[
                    "balanced"
                ][
                    "action"
                ]
                .astype(
                    np.float32
                )
            )

            (
                obs,
                _reward,
                terminated,
                truncated,
                _info,
            ) = env.step(
                balanced_action
            )

            if terminated or truncated:
                break

    finally:
        env.close()

    if not records:
        raise RuntimeError(
            "No policy states collected."
        )

    print("=" * 96)
    print(
        "ICRA27 PHASE-1A SAME-STATE BETA "
        "POLICY SENSITIVITY"
    )
    print("=" * 96)

    print(
        "trajectory : rough TEST seed 2, "
        "advanced with balanced beta"
    )

    print(
        "states     :",
        len(records),
    )

    print()

    names = [
        name
        for name, _beta
        in BETA_BANK
    ]

    for name in names:
        actions = np.stack(
            [
                row[name]["action"]
                for row in records
            ],
            axis=0,
        )

        physical = np.stack(
            [
                row[name]["physical"]
                for row in records
            ],
            axis=0,
        )

        print(
            f"{name:<10} "
            f"a_mean="
            f"[{actions[:,0].mean():+.5f}, "
            f"{actions[:,1].mean():+.5f}, "
            f"{actions[:,2].mean():+.5f}] "
            f"physical="
            f"[vx={physical[:,0].mean():.5f}, "
            f"yaw={physical[:,1].mean():+.5f}, "
            f"h={physical[:,2].mean():.5f}]"
        )

    print()
    print(
        "COUNTERFACTUAL DELTA VS BALANCED"
    )

    balanced_actions = np.stack(
        [
            row["balanced"]["action"]
            for row in records
        ],
        axis=0,
    )

    balanced_physical = np.stack(
        [
            row["balanced"]["physical"]
            for row in records
        ],
        axis=0,
    )

    for name in (
        "motion",
        "stability",
        "energy",
    ):
        actions = np.stack(
            [
                row[name]["action"]
                for row in records
            ],
            axis=0,
        )

        physical = np.stack(
            [
                row[name]["physical"]
                for row in records
            ],
            axis=0,
        )

        da = (
            actions
            - balanced_actions
        )

        dp = (
            physical
            - balanced_physical
        )

        print(
            f"{name:<10} "
            f"d_action_mean="
            f"[{da[:,0].mean():+.6f}, "
            f"{da[:,1].mean():+.6f}, "
            f"{da[:,2].mean():+.6f}] "
            f"d_phys_mean="
            f"[dvx={dp[:,0].mean():+.6f}, "
            f"dyaw={dp[:,1].mean():+.6f}, "
            f"dh={dp[:,2].mean():+.6f}] "
            f"|da|mean="
            f"{np.linalg.norm(da, axis=1).mean():.6f}"
        )

    # --------------------------------------------------------
    # First-layer input-weight diagnostic.
    # Weight magnitude is diagnostic only, not proof of
    # functional importance.
    # --------------------------------------------------------

    first_linear = None

    for module in model.body:
        if isinstance(
            module,
            torch.nn.Linear,
        ):
            first_linear = module
            break

    if first_linear is None:
        raise RuntimeError(
            "Could not find first actor-critic Linear layer."
        )

    w = (
        first_linear
        .weight
        .detach()
        .cpu()
        .numpy()
    )

    if w.shape[1] != 24:
        raise RuntimeError(
            f"Unexpected first-layer input dim: {w.shape}"
        )

    state_norm = np.linalg.norm(
        w[:, :21],
        axis=0,
    )

    beta_norm = np.linalg.norm(
        w[:, 21:24],
        axis=0,
    )

    print()
    print(
        "FIRST-LAYER INPUT COLUMN NORMS"
    )

    print(
        "state21 mean/std:",
        f"{state_norm.mean():.6f}",
        f"{state_norm.std():.6f}",
    )

    print(
        "beta columns     :",
        [
            float(x)
            for x in beta_norm
        ],
    )

    print(
        "beta/state-mean  :",
        [
            float(
                x / state_norm.mean()
            )
            for x in beta_norm
        ],
    )

    print()
    print(
        "[ICRA27] same-state beta sensitivity "
        "diagnostic: PASS"
    )


if __name__ == "__main__":
    main()
