#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
            "ll1_bank_j7_runtime_contract_v1.json"
        ),
    )
    args = parser.parse_args()

    bank = load(args.bank)
    contract = load(args.contract)

    errors: list[str] = []

    if contract.get("schema") != "phase_b_ll1_bank_j7_runtime_contract_v1":
        errors.append("unexpected contract schema")

    actions = bank.get("actions")
    if not isinstance(actions, list) or len(actions) != 9:
        errors.append("bank must contain exactly 9 actions")
        actions = actions if isinstance(actions, list) else []

    layout = contract.get("theta_shadow_layout") or {}
    length = int(layout.get("length", -1))
    gate_index = int(layout.get("j7_gate_action_token_index", -1))
    bank_index = int(layout.get("bank_action_id_index", -1))
    hold_index = int(layout.get("hold_override_index", -1))
    active_token = float(layout.get("j7_active_action_token", float("nan")))
    reserved_token = float(
        layout.get("legacy_reserved_rejected_token", float("nan"))
    )

    if length != 9:
        errors.append(f"theta length={length}, expected 9")
    if gate_index != 0:
        errors.append(f"gate token index={gate_index}, expected 0")
    if bank_index != 1:
        errors.append(f"bank action index={bank_index}, expected 1")
    if hold_index != 8:
        errors.append(f"hold override index={hold_index}, expected 8")
    if not math.isfinite(active_token):
        errors.append("active token is not finite")
    if not math.isfinite(reserved_token):
        errors.append("reserved token is not finite")
    if active_token == reserved_token:
        errors.append("active token collides with reserved token")
    if reserved_token != 8.0:
        errors.append(f"reserved token={reserved_token}, expected 8")
    if active_token != 1.0:
        errors.append(f"active token={active_token}, expected 1")

    observed_ids = sorted(int(action["action_id"]) for action in actions)
    if observed_ids != list(range(9)):
        errors.append(f"bank action IDs={observed_ids}, expected 0..8")

    active_theta_rows = []
    hold_theta_rows = []

    for action in actions:
        action_id = int(action["action_id"])

        active_theta = [0.0] * length
        active_theta[gate_index] = active_token
        active_theta[bank_index] = float(action_id)
        active_theta[hold_index] = 0.0

        hold_theta = list(active_theta)
        hold_theta[hold_index] = 1.0

        active_theta_rows.append(active_theta)
        hold_theta_rows.append(hold_theta)

        if active_theta[0] == reserved_token:
            errors.append(
                f"action {action_id}: active theta[0] uses reserved value"
            )
        if int(active_theta[1]) != action_id:
            errors.append(
                f"action {action_id}: theta[1] does not preserve bank ID"
            )
        if active_theta[8] != 0.0:
            errors.append(
                f"action {action_id}: active hold override is nonzero"
            )
        if hold_theta[8] != 1.0:
            errors.append(
                f"action {action_id}: hold override is not one"
            )

    action8 = active_theta_rows[8] if len(active_theta_rows) == 9 else []
    if action8:
        if action8[0] != 1.0 or action8[1] != 8.0:
            errors.append(
                f"action 8 semantic mapping incorrect: {action8}"
            )

    if errors:
        print("VALIDATION=FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("VALIDATION=PASS")
    print(f"bank={args.bank}")
    print(f"contract={args.contract}")
    print("action_count=9")
    print("j7_gate_action_token_index=0")
    print("bank_action_id_index=1")
    print("hold_override_index=8")
    print("j7_active_action_token=1")
    print("legacy_reserved_rejected_token=8")
    print("bank_action_8_active_theta_prefix=[1,8]")
    print("all_bank_actions_avoid_reserved_gate_token=true")


if __name__ == "__main__":
    main()
