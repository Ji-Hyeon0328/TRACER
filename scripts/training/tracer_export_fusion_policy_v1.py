#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

V0 = ROOT / "configs/highlevel_policy/tracer_fusion_policy_v0.json"
OUT = ROOT / "configs/highlevel_policy/tracer_fusion_policy_v1.json"

SAFE_STOP_CMD = {
    "vx": 0.0,
    "yaw_rate": 0.0,
    "body_height": 0.310,
    "swing_clearance": 0.050,
    "enable": 0.0,
}

def require_terrain(data: dict, key: str) -> dict:
    terrains = data.get("terrains", {})
    if key not in terrains:
        available = "\n".join(sorted(terrains.keys()))
        raise KeyError(f"missing terrain key: {key}\nAvailable keys:\n{available}")
    return terrains[key]

def mark_avoid_required(data: dict, key: str, note: str) -> None:
    entry = require_terrain(data, key)

    entry["fused_mode"] = "avoid_required"
    entry["semantic_mode"] = "no_valid_simple_primitive"
    entry["suggested_style"] = "safe_stop"
    entry["runtime_fallback"] = "safe_stop"
    entry["command"] = dict(SAFE_STOP_CMD)

    entry["validated_by_online_sanity"] = True
    entry["decision_note"] = note
    entry["v1_decision"] = {
        "label": "avoid_required",
        "reason": note,
        "tested_primitives": [
            "cautious",
            "slip_crawl_06_lowclear",
            "safe_stop",
            "active_hold",
        ],
        "runtime_behavior": "fallback_to_safe_stop_command_because_current_low_level_interface_has_no_avoid_command",
    }

def mark_validated_locomotion(data: dict, key: str, note: str) -> None:
    entry = require_terrain(data, key)

    entry["semantic_mode"] = "validated_locomotion"
    entry["validated_by_online_sanity"] = True
    entry["decision_note"] = note
    entry["v1_decision"] = {
        "label": "validated_locomotion",
        "reason": note,
    }

def main() -> None:
    if not V0.exists():
        raise FileNotFoundError(V0)

    data = json.loads(V0.read_text())

    data["policy_version"] = "tracer_fusion_policy_v1"
    data["base_policy"] = str(V0.relative_to(ROOT))
    data["v1_notes"] = {
        "summary": (
            "V1 adds semantic labels from online sanity tests. "
            "It distinguishes locomotion-capable terrains from avoid-required or "
            "no-valid-simple-primitive terrains."
        ),
        "important_distinction": {
            "safe_stop": "Robot can stably stop in place.",
            "avoid_required": "Robot should not enter this terrain; stopping on it may still fail.",
            "no_valid_simple_primitive": "Current walking/stop/hold primitives are insufficient.",
        },
    }

    mark_avoid_required(
        data,
        "slippery_mid_downslope_5deg_forward_postfix",
        (
            "Online sanity showed cautious, slip_crawl_06_lowclear, safe_stop, "
            "and active_hold were insufficient. The robot still slid or collapsed. "
            "This terrain requires avoid-before-entry or a new recovery/traversal primitive."
        ),
    )

    mark_validated_locomotion(
        data,
        "sponge_firm_downslope_5deg_forward",
        (
            "Online sanity with sponge_tall_10_c080 produced stable locomotion. "
            "Absolute min z looked low, but terrain-relative min z was about 0.19 "
            "and relative fall heuristic was false."
        ),
    )

    mark_validated_locomotion(
        data,
        "flat_normal",
        (
            "Online sanity confirmed fast command routes correctly and produces valid forward motion."
        ),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))

    print("[TRACER] wrote", OUT)
    print("[TRACER] terrains:", len(data.get("terrains", {})))

    print("\n========== V1 changed/validated entries ==========")
    for key in [
        "flat_normal",
        "sponge_firm_downslope_5deg_forward",
        "slippery_mid_downslope_5deg_forward_postfix",
    ]:
        e = data["terrains"][key]
        cmd = e.get("command", {})
        print(
            f"{key:55s} "
            f"mode={e.get('fused_mode')} "
            f"semantic={e.get('semantic_mode')} "
            f"style={e.get('suggested_style')} "
            f"vx={cmd.get('vx')} "
            f"h={cmd.get('body_height')} "
            f"clr={cmd.get('swing_clearance')} "
            f"enable={cmd.get('enable')}"
        )

if __name__ == "__main__":
    main()
