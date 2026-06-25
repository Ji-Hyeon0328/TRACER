#!/usr/bin/env python3
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

src = ROOT / "configs/highlevel_policy/tracer_fusion_policy_v1.json"
dst = ROOT / "configs/highlevel_policy/tracer_fusion_policy_v2_slippery_recovery.json"

d = json.loads(src.read_text())

d["name"] = "tracer_fusion_policy_v2_slippery_recovery"
d["policy_version"] = "v2_slippery_recovery"
d["base_policy"] = str(src)
d["v2_notes"] = {
    "summary": (
        "Update slippery recovery behavior using direct primitive sweep and repeat validation. "
        "Slippery mid-flat uses active stabilization fallback. Slippery downslope remains "
        "avoid_required because no tested high-level reference primitive is robust."
    ),
    "slippery_mid_flat_result": {
        "baseline": {
            "style": "slip_crawl_06_lowclear",
            "runs": 3,
            "fall_rate": 1.0,
            "valid_rate": 0.0,
        },
        "selected": {
            "style": "slip_active_hold_h335_c080",
            "runs": 3,
            "fall_rate": 1.0 / 3.0,
            "valid_rate": 2.0 / 3.0,
            "interpretation": (
                "Not a traversal primitive yet, but a better active stabilization fallback "
                "than low-clearance crawl on slippery flat."
            ),
        },
    },
    "slippery_downslope_result": {
        "safe_stop": {
            "runs": 3,
            "fall_rate": 2.0 / 3.0,
            "valid_rate": 0.0,
        },
        "active_hold_h335_c080": {
            "runs": 2,
            "fall_rate": 1.0,
            "valid_rate": 0.0,
        },
        "interpretation": (
            "No tested high-level reference primitive is robust on slippery downslope. "
            "This terrain should be treated as avoid_required until a low-level recovery "
            "primitive such as residual impedance, friction-aware MPC/WBC, CBF-QP, "
            "controlled slide, or crawl/kneel-like traversal is implemented."
        ),
    },
}

terrains = d.setdefault("terrains", {})
style_commands = d.setdefault("style_commands", {})

# Register active hold style.
style_commands["slip_active_hold_h335_c080"] = {
    "vx": 0.0,
    "yaw_rate": 0.0,
    "body_height": 0.335,
    "swing_clearance": 0.08,
    "enable": 1.0,
}

# Update slippery mid-flat.
if "slippery_mid_flat" not in terrains:
    raise KeyError("missing terrain key: slippery_mid_flat")

mid = terrains["slippery_mid_flat"]
mid["fused_mode"] = "recovery_needed"
mid["semantic_mode"] = "active_stabilization_recovery"
mid["suggested_style"] = "slip_active_hold_h335_c080"
mid["command"] = copy.deepcopy(style_commands["slip_active_hold_h335_c080"])
mid["validated_by_online_sanity"] = True
mid["deployable"] = True
mid["v2_decision"] = {
    "label": "active_stabilization_recovery",
    "reason": (
        "Repeat validation showed the original slip_crawl_06_lowclear collapsed in 3/3 runs, "
        "while slip_active_hold_h335_c080 survived in 2/3 runs. This is selected as an interim "
        "active stabilization fallback for slippery flat terrain."
    ),
    "baseline": "slip_crawl_06_lowclear",
    "selected": "slip_active_hold_h335_c080",
    "repeat_validation": {
        "baseline_runs": 3,
        "baseline_fall_rate": 1.0,
        "baseline_valid_rate": 0.0,
        "selected_runs": 3,
        "selected_fall_rate": 1.0 / 3.0,
        "selected_valid_rate": 2.0 / 3.0,
    },
}

# Update slippery downslope.
if "slippery_downslope_5deg_forward" not in terrains:
    raise KeyError("missing terrain key: slippery_downslope_5deg_forward")

down = terrains["slippery_downslope_5deg_forward"]
down["fused_mode"] = "avoid_required"
down["semantic_mode"] = "no_valid_high_level_reference_primitive"
down["suggested_style"] = "no_deployable_high_level_fallback"
down["runtime_fallback"] = "safe_stop_command_only_because_no_avoid_command_exists"
down["deployable"] = False

# Keep safe_stop command as a non-deployable fallback for the current interface.
down["command"] = {
    "vx": 0.0,
    "yaw_rate": 0.0,
    "body_height": 0.31,
    "swing_clearance": 0.05,
    "enable": 0.0,
}

down["validated_by_online_sanity"] = True
down["v2_decision"] = {
    "label": "avoid_required_no_valid_high_level_reference_primitive",
    "reason": (
        "Both passive safe_stop and active_hold_h335_c080 failed repeat validation on slippery "
        "downslope. The current high-level reference interface [vx, yaw, body_height, clearance, enable] "
        "is insufficient for robust recovery here."
    ),
    "tested_primitives": [
        "safe_stop_enable0",
        "active_hold_h305_c055",
        "active_hold_h335_c080",
        "micro_backstep_02",
        "micro_backstep_04",
        "micro_backstep_06",
        "forward_crawl_02",
    ],
    "repeat_validation": {
        "safe_stop_runs": 3,
        "safe_stop_fall_rate": 2.0 / 3.0,
        "safe_stop_valid_rate": 0.0,
        "active_hold_h335_c080_runs": 2,
        "active_hold_h335_c080_fall_rate": 1.0,
        "active_hold_h335_c080_valid_rate": 0.0,
    },
    "required_next_layer": [
        "RAM-conditioned residual impedance",
        "friction-aware MPC/WBC",
        "CBF-QP safety recovery",
        "controlled-slide or crawl/kneel-like traversal primitive",
    ],
}

dst.write_text(json.dumps(d, indent=2))
print("[TRACER] wrote", dst)
print()
for key in ["slippery_mid_flat", "slippery_downslope_5deg_forward"]:
    item = d["terrains"][key]
    print("=" * 80)
    print("terrain:", key)
    print("mode:", item.get("fused_mode"))
    print("semantic:", item.get("semantic_mode"))
    print("style:", item.get("suggested_style"))
    print("deployable:", item.get("deployable"))
    print("command:", item.get("command"))
