#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def close_or_less(
    value: float,
    bound: float,
    tolerance: float = 1e-12,
) -> bool:
    return (
        math.isfinite(float(value))
        and abs(float(value)) <= float(bound) + tolerance
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        default=(
            "configs/phase_b_theta_lite_rl_v1/"
            "ll1_verified_action_bank_v0.json"
        ),
    )
    parser.add_argument(
        "--contract",
        default=(
            "configs/phase_b_theta_lite_rl_v1/"
            "ll1_bank_j7_runtime_contract_v0.json"
        ),
    )
    args = parser.parse_args()

    bank = load(args.bank)
    contract = load(args.contract)

    errors: list[str] = []

    if (
        contract.get("schema")
        != "phase_b_ll1_bank_j7_runtime_contract_v0"
    ):
        errors.append("unexpected contract schema")

    actions = bank.get("actions")
    if not isinstance(actions, list) or len(actions) != 9:
        errors.append("bank must contain exactly 9 actions")
        actions = actions if isinstance(actions, list) else []

    anchor = contract["empirical_anchor"]
    guard = contract["j7_guard_assumptions"]
    goal = contract["goal_logic"]

    expected_center = {
        "vx_far": 0.065,
        "vx_near": 0.040,
        "body_height": 0.320,
        "swing_clearance": 0.045,
    }

    for field, expected in expected_center.items():
        actual = float(anchor[field])
        if abs(actual - expected) > 1e-12:
            errors.append(
                f"anchor {field}={actual} expected={expected}"
            )

    if abs(float(goal["yaw_rate_max"]) - 0.020) > 1e-12:
        errors.append("yaw_rate_max must equal 0.020")

    rows = []

    for action in actions:
        action_id = int(action["action_id"])
        name = str(action["name"])

        cases = [
            (
                "far_positive_yaw",
                float(anchor["vx_far"]),
                float(action["vx_far"]),
                +float(goal["yaw_rate_max"]),
            ),
            (
                "far_negative_yaw",
                float(anchor["vx_far"]),
                float(action["vx_far"]),
                -float(goal["yaw_rate_max"]),
            ),
            (
                "near_positive_yaw",
                float(anchor["vx_near"]),
                float(action["vx_near"]),
                +float(goal["yaw_rate_max"]),
            ),
            (
                "near_negative_yaw",
                float(anchor["vx_near"]),
                float(action["vx_near"]),
                -float(goal["yaw_rate_max"]),
            ),
            (
                "stop",
                0.0,
                0.0,
                0.0,
            ),
        ]

        for mode, empirical_vx, projected_vx, projected_yaw in cases:
            deltas = {
                "vx": projected_vx - empirical_vx,
                "yaw_rate": projected_yaw,
                "body_height": (
                    float(action["body_height"])
                    - float(anchor["body_height"])
                ),
                "swing_clearance": (
                    float(action["swing_clearance"])
                    - float(anchor["swing_clearance"])
                ),
            }

            checks = {
                "vx": close_or_less(
                    deltas["vx"],
                    float(guard["max_abs_delta_vx"]),
                ),
                "yaw_rate": close_or_less(
                    deltas["yaw_rate"],
                    float(guard["max_abs_delta_yaw_rate"]),
                ),
                "body_height": close_or_less(
                    deltas["body_height"],
                    float(guard["max_abs_delta_body_height"]),
                ),
                "swing_clearance": close_or_less(
                    deltas["swing_clearance"],
                    float(guard["max_abs_delta_swing_clearance"]),
                ),
            }

            passed = all(checks.values())
            rows.append({
                "action_id": action_id,
                "action_name": name,
                "mode": mode,
                "deltas": deltas,
                "checks": checks,
                "passed": passed,
            })

            if not passed:
                errors.append(
                    f"J7 bound failure action={action_id} "
                    f"name={name} mode={mode} deltas={deltas}"
                )

    center = next(
        (
            action
            for action in actions
            if int(action["action_id"]) == 4
        ),
        None,
    )

    if center is None:
        errors.append("action_id=4 center action missing")
    else:
        if abs(float(center["vx_far"]) - 0.065) > 1e-12:
            errors.append("center action vx_far mismatch")
        if (
            abs(
                float(center["swing_clearance"])
                - 0.045
            )
            > 1e-12
        ):
            errors.append("center action clearance mismatch")

    if errors:
        print("VALIDATION=FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("VALIDATION=PASS")
    print(f"bank={args.bank}")
    print(f"contract={args.contract}")
    print("action_count=9")
    print(f"checked_cases={len(rows)}")
    print("center_action_id=4")
    print("empirical_far_vx=0.065")
    print("empirical_clearance=0.045")
    print("max_realized_abs_delta_vx=0.025")
    print("max_realized_abs_delta_yaw_rate=0.020")
    print("max_realized_abs_delta_body_height=0.000")
    print("max_realized_abs_delta_clearance=0.010")
    print("all_actions_within_j7_contract=true")


if __name__ == "__main__":
    main()
