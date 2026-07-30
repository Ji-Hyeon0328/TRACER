#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_VX = [0.040, 0.065, 0.090]
EXPECTED_CLEARANCE = [0.035, 0.045, 0.055]
FIXED = {
    "vx_near": 0.040,
    "goal_slow_distance": 0.25,
    "goal_stop_distance": 0.15,
    "body_height": 0.320,
    "enable": 1.0,
}


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isfinite(float(a)) and abs(float(a) - float(b)) <= tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bank",
        nargs="?",
        default=(
            "configs/phase_b_theta_lite_rl_v1/"
            "ll1_verified_action_bank_v0.json"
        ),
    )
    args = parser.parse_args()

    path = Path(args.bank)
    data = json.loads(path.read_text(encoding="utf-8"))

    errors: list[str] = []

    if data.get("schema") != "phase_b_ll1_verified_theta_lite_action_bank_v0":
        errors.append("unexpected schema")

    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) != 9:
        errors.append("expected exactly 9 actions")
        actions = actions if isinstance(actions, list) else []

    expected_pairs = [
        (vx, clearance)
        for vx in EXPECTED_VX
        for clearance in EXPECTED_CLEARANCE
    ]

    names: set[str] = set()
    ids: set[int] = set()
    observed_pairs: list[tuple[float, float]] = []

    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"action {index} is not an object")
            continue

        action_id = action.get("action_id")
        name = action.get("name")

        if action_id != index:
            errors.append(
                f"action index {index} has action_id={action_id!r}"
            )

        if not isinstance(name, str) or not name:
            errors.append(f"action {index} has invalid name")
        elif name in names:
            errors.append(f"duplicate action name: {name}")
        else:
            names.add(name)

        if isinstance(action_id, int):
            if action_id in ids:
                errors.append(f"duplicate action_id: {action_id}")
            ids.add(action_id)

        try:
            vx = float(action["vx_far"])
            clearance = float(action["swing_clearance"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"action {index} missing numeric vx/clearance")
            continue

        observed_pairs.append((vx, clearance))

        for field, expected in FIXED.items():
            try:
                actual = float(action[field])
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"action {index} missing numeric fixed field {field}"
                )
                continue

            if not close(actual, expected):
                errors.append(
                    f"action {index} {field}={actual} expected={expected}"
                )

        if action.get("yaw_policy") != (
            "deterministic_goal_heading_guarded_not_rl_action_v0"
        ):
            errors.append(f"action {index} has unexpected yaw_policy")

    for expected, observed in zip(expected_pairs, observed_pairs):
        if not (
            close(expected[0], observed[0])
            and close(expected[1], observed[1])
        ):
            errors.append(
                f"pair order mismatch expected={expected} observed={observed}"
            )

    if errors:
        print("VALIDATION=FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("VALIDATION=PASS")
    print(f"bank={path}")
    print("action_count=9")
    print("action_ids=0..8")
    print("vx_far_values=0.040,0.065,0.090")
    print("swing_clearance_values=0.035,0.045,0.055")
    print("vx_near=0.040")
    print("body_height=0.320")
    print("rl_yaw_dimension=false")


if __name__ == "__main__":
    main()
