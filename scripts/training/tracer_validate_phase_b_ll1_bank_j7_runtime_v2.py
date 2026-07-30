#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED_J7_SHA = "2b2692af30e164db53d3c125323ff27d42849f7b6a9fc28485f4742b4580811a"


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_le(value: float, bound: float, tol: float = 1.0e-12) -> bool:
    return (
        math.isfinite(float(value))
        and abs(float(value)) <= float(bound) + tol
    )


def source_default(source: str, env_name: str) -> float:
    pattern = re.compile(
        r'env_float\("'
        + re.escape(env_name)
        + r'",\s*([-+]?[0-9]*\.?[0-9]+)\)'
    )
    match = pattern.search(source)
    if not match:
        raise ValueError(f"source default not found: {env_name}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        default=(
            "configs/phase_b_theta_lite_rl_v1/"
            "ll1_j7_compatible_action_bank_v1.json"
        ),
    )
    parser.add_argument(
        "--contract",
        default=(
            "configs/phase_b_theta_lite_rl_v1/"
            "ll1_bank_j7_runtime_contract_v2.json"
        ),
    )
    parser.add_argument(
        "--j7-source",
        default=(
            "scripts/runtime/"
            "tracer_phase_j7_guarded_meta_action_ref_gate_node_v0.py"
        ),
    )
    args = parser.parse_args()

    bank = load_json(args.bank)
    contract = load_json(args.contract)
    j7_path = Path(args.j7_source)

    errors: list[str] = []

    if bank.get("schema") != (
        "phase_b_ll1_j7_compatible_theta_lite_action_bank_v1"
    ):
        errors.append("unexpected bank schema")

    if contract.get("schema") != (
        "phase_b_ll1_bank_j7_runtime_contract_v2"
    ):
        errors.append("unexpected contract schema")

    if not j7_path.is_file():
        errors.append(f"J7 source missing: {j7_path}")
        source = ""
    else:
        source = j7_path.read_text(encoding="utf-8")
        actual_sha = sha256(j7_path)
        if actual_sha != EXPECTED_J7_SHA:
            errors.append(
                f"J7 SHA={actual_sha} expected={EXPECTED_J7_SHA}"
            )

    expected_defaults = {
        "TRACER_PHASE_J7_MAX_ABS_DELTA_VX": 0.030,
        "TRACER_PHASE_J7_MAX_ABS_DELTA_YAW": 0.020,
        "TRACER_PHASE_J7_MAX_ABS_DELTA_BODY_H": 0.006,
        "TRACER_PHASE_J7_MAX_ABS_DELTA_CLEARANCE": 0.006,
        "TRACER_PHASE_J7_MIN_ACTIVE_BODY_H": 0.314,
        "TRACER_PHASE_J7_GOAL_X_THRESHOLD": 7.90,
        "TRACER_PHASE_J7_MAX_INPUT_AGE_S": 1.0,
    }

    source_defaults: dict[str, float] = {}
    if source:
        for name, expected in expected_defaults.items():
            try:
                actual = source_default(source, name)
            except ValueError as error:
                errors.append(str(error))
                continue
            source_defaults[name] = actual
            if abs(actual - expected) > 1.0e-12:
                errors.append(
                    f"source default {name}={actual} expected={expected}"
                )

    frozen = contract.get("frozen_j7_defaults") or {}
    contract_checks = {
        "max_abs_delta_vx": 0.030,
        "max_abs_delta_yaw_rate": 0.020,
        "max_abs_delta_body_height": 0.006,
        "max_abs_delta_swing_clearance": 0.006,
        "min_active_body_height": 0.314,
        "goal_x_threshold": 7.90,
        "max_input_age_s": 1.0,
    }
    for field, expected in contract_checks.items():
        try:
            actual = float(frozen[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"contract missing numeric field {field}")
            continue
        if abs(actual - expected) > 1.0e-12:
            errors.append(
                f"contract {field}={actual} expected={expected}"
            )

    actions = bank.get("actions")
    if not isinstance(actions, list) or len(actions) != 9:
        errors.append("bank must contain exactly 9 actions")
        actions = actions if isinstance(actions, list) else []

    observed_ids = sorted(int(row["action_id"]) for row in actions)
    if observed_ids != list(range(9)):
        errors.append(f"action IDs={observed_ids}, expected 0..8")

    anchor = contract.get("empirical_anchor") or {}
    layout = contract.get("theta_shadow_layout") or {}
    active_token = float(layout.get("j7_active_action_token", math.nan))
    reserved_token = float(
        layout.get("legacy_reserved_rejected_token", math.nan)
    )
    bank_index = int(layout.get("bank_action_id_index", -1))
    hold_index = int(layout.get("hold_override_index", -1))

    if active_token != 1.0:
        errors.append(f"active token={active_token}, expected 1")
    if reserved_token != 8.0:
        errors.append(f"reserved token={reserved_token}, expected 8")
    if active_token == reserved_token:
        errors.append("active token collides with reserved token")
    if bank_index != 1:
        errors.append(f"bank action index={bank_index}, expected 1")
    if hold_index != 8:
        errors.append(f"hold index={hold_index}, expected 8")

    max_seen = {
        "vx": 0.0,
        "yaw": 0.0,
        "body_height": 0.0,
        "clearance": 0.0,
    }
    checked_cases = 0

    for action in actions:
        action_id = int(action["action_id"])

        theta = [0.0] * 9
        theta[0] = active_token
        theta[bank_index] = float(action_id)
        theta[hold_index] = 0.0

        if theta[0] == reserved_token:
            errors.append(f"action {action_id} uses reserved gate token")
        if int(theta[bank_index]) != action_id:
            errors.append(f"action {action_id} identity not preserved")

        for mode, emp_vx, proj_vx, yaw in [
            (
                "far_positive",
                float(anchor["vx_far"]),
                float(action["vx_far"]),
                +0.020,
            ),
            (
                "far_negative",
                float(anchor["vx_far"]),
                float(action["vx_far"]),
                -0.020,
            ),
            (
                "near_positive",
                float(anchor["vx_near"]),
                float(action["vx_near"]),
                +0.020,
            ),
            (
                "near_negative",
                float(anchor["vx_near"]),
                float(action["vx_near"]),
                -0.020,
            ),
            ("stop", 0.0, 0.0, 0.0),
        ]:
            checked_cases += 1
            deltas = {
                "vx": proj_vx - emp_vx,
                "yaw": yaw,
                "body_height": (
                    float(action["body_height"])
                    - float(anchor["body_height"])
                ),
                "clearance": (
                    float(action["swing_clearance"])
                    - float(anchor["swing_clearance"])
                ),
            }

            for key, value in deltas.items():
                max_seen[key] = max(max_seen[key], abs(value))

            checks = {
                "vx": finite_le(
                    deltas["vx"],
                    float(frozen["max_abs_delta_vx"]),
                ),
                "yaw": finite_le(
                    deltas["yaw"],
                    float(frozen["max_abs_delta_yaw_rate"]),
                ),
                "body_height": finite_le(
                    deltas["body_height"],
                    float(frozen["max_abs_delta_body_height"]),
                ),
                "clearance": finite_le(
                    deltas["clearance"],
                    float(
                        frozen[
                            "max_abs_delta_swing_clearance"
                        ]
                    ),
                ),
            }

            if not all(checks.values()):
                errors.append(
                    f"guard failure action={action_id} "
                    f"mode={mode} deltas={deltas} checks={checks}"
                )

        if float(action["body_height"]) < float(
            frozen["min_active_body_height"]
        ):
            errors.append(
                f"action {action_id} body height below J7 minimum"
            )

    if errors:
        print("VALIDATION=FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("VALIDATION=PASS")
    print(f"bank={args.bank}")
    print(f"contract={args.contract}")
    print(f"j7_source={args.j7_source}")
    print(f"j7_sha256={EXPECTED_J7_SHA}")
    print("action_count=9")
    print(f"checked_cases={checked_cases}")
    print("j7_active_action_token=1")
    print("bank_action_id_index=1")
    print("hold_override_index=8")
    print("clearance_values=0.040,0.045,0.050")
    print(f"max_realized_abs_delta_vx={max_seen['vx']:.3f}")
    print(f"max_realized_abs_delta_yaw={max_seen['yaw']:.3f}")
    print(
        "max_realized_abs_delta_body_height="
        f"{max_seen['body_height']:.3f}"
    )
    print(
        "max_realized_abs_delta_clearance="
        f"{max_seen['clearance']:.3f}"
    )
    print("all_actions_within_frozen_j7_defaults=true")


if __name__ == "__main__":
    main()
