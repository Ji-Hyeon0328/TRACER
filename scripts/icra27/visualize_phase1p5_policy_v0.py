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


OBS_DIM = 21
ACTION_DIM = 3

SETTLING_STEPS = 5
POLICY_HORIZON = 50

GOAL_DISTANCE_M = 2.0
SUCCESS_RADIUS_M = 0.15
DECISION_DT_S = 0.20

EVAL_REWARD_MODE = "tracer_cost_v2"


CHECKPOINTS = {
    "tracer": (
        ROOT
        / "results"
        / "icra27"
        / "phase1p5_paired_v3"
        / "tracer_cost_v2_seed27027"
        / "m7_ppo_multiterrain_final.pt"
    ),

    "slr": (
        ROOT
        / "results"
        / "icra27"
        / "phase1p5_paired_v3"
        / "slr_hl_v2_seed27027"
        / "m7_ppo_multiterrain_final.pt"
    ),
}


def fmt_vec(values, digits=3):
    return (
        "["
        + ", ".join(
            f"{float(x):+.{digits}f}"
            for x in values
        )
        + "]"
    )


def classify(info, terminated, truncated):
    if bool(info.get("success", False)):
        return "SUCCESS"

    if bool(info.get("m4_intervention", False)):
        return "SAFETY_ABORT"

    if truncated:
        return "TIME_LIMIT"

    if terminated:
        return "TERMINATED"

    return "RUNNING"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--policy",
        choices=("tracer", "slr"),
        default="tracer",
    )

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--terrain",
        choices=(
            "flat",
            "low_friction",
            "rough_perlin",
        ),
        default="rough_perlin",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--command-port",
        type=int,
        default=53910,
    )

    ap.add_argument(
        "--print-every",
        type=int,
        default=1,
    )

    args = ap.parse_args()

    if args.print_every <= 0:
        raise ValueError(
            "--print-every must be positive"
        )

    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else CHECKPOINTS[args.policy]
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    (
        model,
        _cfg,
        payload,
    ) = load_ppo_checkpoint(
        checkpoint
    )

    if int(payload["obs_dim"]) != OBS_DIM:
        raise RuntimeError(
            "Unexpected obs_dim: "
            f"{payload['obs_dim']}"
        )

    if int(payload["act_dim"]) != ACTION_DIM:
        raise RuntimeError(
            "Unexpected act_dim: "
            f"{payload['act_dim']}"
        )

    model.eval()

    log_dir = (
        ROOT
        / "results"
        / "icra27"
        / "phase1p5_visualization_v0"
        / args.policy
        / f"{args.terrain}_seed{args.seed}"
    )

    base_env = PyMPCM7Env(
        terrain=args.terrain,

        reward_mode=EVAL_REWARD_MODE,

        terminate_on_m4_unsafe=True,

        goal_distance_m=GOAL_DISTANCE_M,
        success_radius_m=SUCCESS_RADIUS_M,

        decision_dt_s=DECISION_DT_S,

        max_episode_steps=(
            POLICY_HORIZON
            + SETTLING_STEPS
        ),

        command_port=args.command_port,
        telemetry_port=args.command_port + 1,
        state_port=args.command_port + 2,

        command_repeat_hz=20.0,
        telemetry_hz=100.0,
        state_hz=100.0,

        log_dir=log_dir,

        # Visualization-only opt-in.
        # Normal training/evaluation remains headless.
        render_runner=True,
    )

    env = M7FixedClearanceActionWrapper(
        base_env
    )

    if env.observation_space.shape != (
        OBS_DIM,
    ):
        raise RuntimeError(
            "Unexpected observation space: "
            f"{env.observation_space.shape}"
        )

    if env.action_space.shape != (
        ACTION_DIM,
    ):
        raise RuntimeError(
            "Unexpected action space: "
            f"{env.action_space.shape}"
        )

    print("=" * 88)
    print(
        "ICRA27 PHASE-1.5 FINAL POLICY "
        "MUJOCO VIEWER"
    )
    print("=" * 88)

    print("policy       :", args.policy)
    print("checkpoint   :", checkpoint)
    print("terrain      :", args.terrain)
    print("terrain seed :", args.seed)
    print("viewer       : ON")
    print()

    print(
        "CONTROL FLOW:"
    )

    print(
        "  obs21"
        " -> PPO action3"
        " -> fixed-clearance action4"
        " -> physical command6"
    )

    print(
        "  -> Low-Level Safety Supervisor"
        " -> frozen PyMPC"
        " -> MuJoCo"
    )

    print()

    print(
        "physical command6 = "
        "[vx, yaw_rate, body_height, "
        "swing_clearance, gait_period, duty_factor]"
    )

    print("=" * 88)

    zero_action = np.zeros(
        (ACTION_DIM,),
        dtype=np.float32,
    )

    try:
        obs, info = env.reset(
            seed=args.seed
        )

        # ----------------------------------------------------
        # Settling: excluded from learned policy.
        # ----------------------------------------------------

        for i in range(SETTLING_STEPS):
            (
                obs,
                _reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                zero_action
            )

            print(
                f"[SETTLE {i + 1}/{SETTLING_STEPS}] "
                f"d={float(info['goal_distance']):.3f} m "
                f"safety={info.get('safety_state')} "
                f"override="
                f"{bool(info.get('override_active', False))}"
            )

            if terminated or truncated:
                print(
                    "Settling terminated:",
                    classify(
                        info,
                        terminated,
                        truncated,
                    ),
                )
                return

        print()
        print(
            "================ POLICY CONTROL ================="
        )

        # ----------------------------------------------------
        # Deterministic final-policy replay.
        # ----------------------------------------------------

        for step in range(
            POLICY_HORIZON
        ):
            (
                action,
                _log_prob,
                _value,
            ) = model.act(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            if action.shape != (
                ACTION_DIM,
            ):
                raise RuntimeError(
                    "Unexpected policy action shape: "
                    f"{action.shape}"
                )

            (
                obs,
                _reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            requested = np.asarray(
                info["requested_physical"],
                dtype=np.float64,
            )

            applied = np.asarray(
                info["applied_physical"],
                dtype=np.float64,
            )

            if requested.shape != (6,):
                raise RuntimeError(
                    "Expected requested physical "
                    f"shape (6,), got {requested.shape}"
                )

            if applied.shape != (6,):
                raise RuntimeError(
                    "Expected applied physical "
                    f"shape (6,), got {applied.shape}"
                )

            rc = info.get(
                "reward_components",
                {},
            )

            should_print = (
                step % args.print_every == 0
                or terminated
                or truncated
            )

            if should_print:
                obs_arr = np.asarray(
                    obs,
                    dtype=np.float64,
                )

                roll = float(
                    obs_arr[11]
                )

                pitch = float(
                    obs_arr[12]
                )

                print(
                    f"[HL {step:02d}] "
                    f"d={float(info['goal_distance']):.3f} m "
                    f"a={fmt_vec(action)}"
                )

                print(
                    "        requested="
                    f"{fmt_vec(requested)}"
                )

                print(
                    "        applied  ="
                    f"{fmt_vec(applied)}"
                )

                print(
                    "        safety="
                    f"{info.get('safety_state')} "
                    "override="
                    f"{bool(info.get('override_active', False))}"
                )

                print(
                    "        "
                    f"roll={roll:+.3f} "
                    f"pitch={pitch:+.3f} "
                    f"dt={float(info['decision_dt_s']):.3f}s"
                )

                if rc:
                    print(
                        "        "
                        f"dE="
                        f"{float(rc.get('energy_abs_j', float('nan'))):.3f} J "
                        f"Cm="
                        f"{float(rc.get('cost_motion', float('nan'))):.3f} "
                        f"Cs="
                        f"{float(rc.get('cost_stability', float('nan'))):.3f} "
                        f"CE="
                        f"{float(rc.get('cost_energy', float('nan'))):.3f}"
                    )

                reasons = info.get(
                    "override_reasons",
                    [],
                )

                if reasons:
                    print(
                        "        safety reasons:",
                        reasons,
                    )

            if terminated or truncated:
                print()
                print("=" * 88)

                print(
                    "EPISODE:",
                    classify(
                        info,
                        terminated,
                        truncated,
                    ),
                )

                print(
                    "final goal distance:",
                    f"{float(info['goal_distance']):.3f} m",
                )

                print("=" * 88)
                break

        else:
            print(
                "POLICY HORIZON EXHAUSTED"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()
